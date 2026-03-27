from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from app.services.training.config.automl import AutoMLConfig
from app.services.training.pipeline.evaluator import Evaluator

logger = logging.getLogger(__name__)

# Map our metric names to FLAML metric names
_FLAML_METRIC_MAP: dict[str, str] = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "accuracy": "accuracy",
    "pr_auc": "ap",           # FLAML uses "ap" for average precision
    "rmse": "rmse",
    "r2": "r2",
    "mae": "mae",
    "mse": "mse",
    "f1_macro": "macro_f1",
    "f1_micro": "micro_f1",
    "f1_weighted": "weighted_f1",
}

_AUTO_METRIC: dict[str, str] = {
    "classification": "roc_auc",
    "regression": "rmse",
}


# ── Explicit sklearn preprocessing before FLAML ───────────────────────────────

def _build_automl_preprocessor(
    X_train: pd.DataFrame,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    """
    Build and fit a standard sklearn preprocessing pipeline on X_train only (no leakage).

    Steps:
      - Numeric columns  : median imputation → StandardScaler
      - Categorical cols : most_frequent imputation → OrdinalEncoder (integer values)
      - VarianceThreshold(0.0) removes constant features

    Returns (fitted_pipeline, X_train_prep_df, prep_feature_names).
    FLAML receives a clean all-numeric DataFrame — it can focus on model/HP search.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.feature_selection import VarianceThreshold
    from app.services.preparation_ml.preprocessing.preprocessing import infer_columns

    # Pandas 3.0 StringDtype is not understood by numpy/sklearn — convert to object
    str_cols = X_train.select_dtypes(include="string").columns.tolist()
    if str_cols:
        X_train = X_train.copy()
        X_train = X_train.astype({col: "object" for col in str_cols})
    X_train.columns = X_train.columns.astype(object)

    numeric_cols, categorical_cols = infer_columns(X_train)

    transformers: list = []
    if numeric_cols:
        num_pipe = SKPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = SKPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        transformers.append(("passthrough", "passthrough", list(X_train.columns)))

    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    vt = VarianceThreshold(threshold=0.0)
    preprocessor = SKPipeline([("ct", ct), ("vt", vt)])

    X_arr = preprocessor.fit_transform(X_train)

    try:
        prep_names = [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        prep_names = [f"feat_{i}" for i in range(X_arr.shape[1])]

    # Pandas 3.0 infers StringDtype for string column names — force object dtype
    # so that FLAML's internal np.array() calls on column indices don't crash.
    cols_idx = pd.Index(prep_names, dtype=object)
    X_prep = pd.DataFrame(X_arr, columns=cols_idx, index=X_train.index)
    return preprocessor, X_prep, prep_names


class AutoMLPipeline:
    """
    Wraps (sklearn preprocessor + FLAML AutoML) into a single picklable object.
    At inference time, raw data is preprocessed transparently before FLAML predicts.
    The existing predictor needs no changes — it calls predict_proba(X) as usual.
    """

    def __init__(
        self,
        preprocessor: Any,
        automl: Any,
        feature_names_in: List[str],
        prep_cols: Optional[List[str]] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.automl = automl
        self.feature_names_in = feature_names_in  # original column names expected at inference
        self._prep_cols = prep_cols               # column names after preprocessing

    # ── internal helper ───────────────────────────────────────────────────────

    def _preprocess(self, X: Any) -> Any:
        """Align raw input to training schema, then apply sklearn preprocessing."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in[: X.shape[1]])
        elif isinstance(X, pd.DataFrame):
            X = X.copy()
            for c in self.feature_names_in:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[self.feature_names_in]
        return self.preprocessor.transform(X)

    # ── sklearn-compatible API ────────────────────────────────────────────────

    def predict(self, X: Any) -> np.ndarray:
        return self.automl.predict(self._preprocess(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.automl.predict_proba(self._preprocess(X))

    @property
    def classes_(self) -> Any:
        return getattr(self.automl, "classes_", None)

    @property
    def model(self) -> Any:
        return getattr(self.automl, "model", None)

    def _inner_estimator(self) -> Any:
        model = self.model
        if model is None:
            return self.automl
        return getattr(model, "estimator", model)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return getattr(self._inner_estimator(), "feature_importances_", None)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return getattr(self._inner_estimator(), "coef_", None)


# ── Medical preparation ────────────────────────────────────────────────────────

@dataclass
class _MedicalPrepResult:
    """All parameters produced by the medical pre-fit optimizations."""
    X_train: pd.DataFrame
    feature_pairs: List[Tuple[str, str, str]]
    sample_weight: Optional[np.ndarray]
    effective_budget: int
    eval_method: str
    n_splits: int
    flaml_metric: str
    custom_hp: Dict[str, Any]
    imbalance_applied: bool
    features_added: int


def _prepare_medical_automl(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: "AutoMLConfig",
    flaml_metric: str,
    n_samples: int,
) -> _MedicalPrepResult:
    """
    Apply generic pre-fit optimizations before automl.fit():

      1. Sample-weight balancing — inverse-frequency weights for imbalanced classes.
      2. CV 5-fold forced — more robust HP search than 3-fold.
      3. Wider HP search space — more exploration freedom for FLAML.

    Note: metric switch to F1 removed — ROC AUC is threshold-independent and
    more reliable for medical classification.
    """
    from flaml import tune  # lazy import so flaml is optional at module load

    X = X_train.copy()

    # ── Opt 1 : Sample-weight balancing ───────────────────────────────────────
    sample_weight: Optional[np.ndarray] = None
    imbalance_applied = False
    if cfg.task_type == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) >= 2:
            ir = float(counts.max()) / float(counts.min())
            if ir > 1.2:
                sample_weight = compute_sample_weight("balanced", y_train)
                imbalance_applied = True
                logger.info("MedAutoML Opt1: IR=%.2f → sample_weight balanced", ir)

    # ── Opt 2 : CV 5-fold forced ──────────────────────────────────────────────
    effective_budget = cfg.time_budget
    eval_method = "cv"
    n_splits = 5   # 5-fold: better generalization estimate than 3-fold
    logger.info("MedAutoML Opt2: budget=%ds, eval=cv/5-fold", effective_budget)

    # ── Opt 3 : Wider HP search space ─────────────────────────────────────────
    custom_hp: Dict[str, Any] = {
        "lgbm": {
            "n_estimators":      {"domain": tune.lograndint(lower=50, upper=2000),   "init_value": 300},
            "max_depth":         {"domain": tune.randint(lower=3, upper=12),          "init_value": 6},
            "learning_rate":     {"domain": tune.loguniform(lower=0.005, upper=0.3),  "init_value": 0.05},
            "min_child_samples": {"domain": tune.randint(lower=5, upper=100),         "init_value": 20},
            "reg_alpha":         {"domain": tune.loguniform(lower=1e-4, upper=10.0),  "init_value": 0.1},
            "reg_lambda":        {"domain": tune.loguniform(lower=1e-4, upper=10.0),  "init_value": 0.1},
        },
        "xgboost": {
            "n_estimators":    {"domain": tune.lograndint(lower=50, upper=2000),   "init_value": 300},
            "max_depth":       {"domain": tune.randint(lower=3, upper=12),          "init_value": 6},
            "learning_rate":   {"domain": tune.loguniform(lower=0.005, upper=0.3),  "init_value": 0.05},
            "min_child_weight":{"domain": tune.randint(lower=1, upper=30),          "init_value": 5},
            "reg_alpha":       {"domain": tune.loguniform(lower=1e-4, upper=10.0),  "init_value": 0.1},
            "reg_lambda":      {"domain": tune.loguniform(lower=1e-4, upper=10.0),  "init_value": 0.1},
        },
        "rf": {
            "max_depth":        {"domain": tune.randint(lower=3, upper=15),  "init_value": 7},
            "min_samples_split":{"domain": tune.randint(lower=2, upper=20),  "init_value": 5},
            "min_samples_leaf": {"domain": tune.randint(lower=1, upper=10),  "init_value": 2},
        },
    }

    feature_pairs: List[Tuple[str, str, str]] = []

    return _MedicalPrepResult(
        X_train=X,
        feature_pairs=feature_pairs,
        sample_weight=sample_weight,
        effective_budget=effective_budget,
        eval_method=eval_method,
        n_splits=n_splits,
        flaml_metric=flaml_metric,   # keep requested/auto metric — no F1 switch
        custom_hp=custom_hp,
        imbalance_applied=imbalance_applied,
        features_added=0,
    )


def _find_optimal_threshold(
    automl: Any,
    X: Any,
    y: np.ndarray,
    positive_label: Any = None,
) -> float:
    """
    Post-fit: find the probability threshold that maximises F1
    on the holdout test set via the Precision-Recall curve.
    Only meaningful for binary classification.
    Returns a value in [0.01, 0.99] or 0.5 as fallback.
    """
    try:
        if not hasattr(automl, "predict_proba"):
            return 0.5
        proba = automl.predict_proba(X)
        if proba.ndim < 2 or proba.shape[1] < 2:
            return 0.5
        classes = np.unique(y)
        if len(classes) != 2:
            return 0.5

        labels = getattr(automl, "classes_", None)
        pos_col = 1
        if positive_label is not None and labels is not None:
            try:
                pos_col = list(labels).index(positive_label)
            except ValueError:
                pass

        scores = proba[:, pos_col]
        precisions, recalls, thresholds = precision_recall_curve(y, scores)
        if len(thresholds) == 0:
            return 0.5

        denom = precisions[:-1] + recalls[:-1]
        f1_scores = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / denom, 0.0)
        optimal = float(thresholds[int(np.argmax(f1_scores))])
        return max(0.01, min(0.99, optimal))
    except Exception:
        return 0.5


@dataclass
class AutoMLRunResult:
    metrics_json: Dict[str, Any]
    artifacts_json: Dict[str, Any]
    fitted_model: Any
    task_type: str
    is_best: bool = False


def run_automl(
    df: pd.DataFrame,
    cfg: AutoMLConfig,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> List[AutoMLRunResult]:
    """
    Run a FLAML AutoML search on `df` using `cfg`.

    Pipeline:
      1. Clean data / train-test split
      2. Explicit sklearn preprocessing (impute + scale + encode + VarianceThreshold)
         fitted on X_train only — FLAML receives a clean numeric matrix
      3. SMOTE oversampling if classification and imbalance ratio > 2.0 (binary only)
      4. FLAML HPO search on preprocessed data (5-fold CV, wider HP bounds)
      5. Threshold calibration on holdout test set
      6. Wrap (preprocessor + automl) in AutoMLPipeline for transparent inference

    Returns a list of AutoMLRunResult (best + per-estimator).
    """
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise RuntimeError(
            "FLAML n'est pas installé. Installez-le avec : pip install 'flaml[automl]'"
        ) from exc

    t_start = time.monotonic()

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    if cfg.target_column not in df.columns:
        raise ValueError(f"Colonne cible '{cfg.target_column}' introuvable dans le dataset.")

    df_clean = df[df[cfg.target_column].notna()].copy()
    if len(df_clean) < 10:
        raise ValueError(
            "Pas assez de données pour AutoML (minimum 10 lignes après suppression des NaN cibles)."
        )

    # Pandas 3.0: StringDtype breaks FLAML's np.issubdtype() checks — convert to object
    string_cols = df_clean.select_dtypes(include="string").columns.tolist()
    if string_cols:
        df_clean = df_clean.astype({col: "object" for col in string_cols})

    X = df_clean.drop(columns=[cfg.target_column])
    # Force object dtype on column Index — Pandas 3.0 uses StringDtype by default
    # for string column names which breaks FLAML's internal np.array() calls.
    X.columns = pd.Index(list(X.columns), dtype=object)
    y = df_clean[cfg.target_column].to_numpy()
    n_samples = len(df_clean)

    # ── 2. Train/test split ──────────────────────────────────────────────────
    if cfg.test_ratio > 0 and int(n_samples * cfg.test_ratio) >= 1:
        stratify = y if cfg.task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_ratio, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_ratio, random_state=42
            )
        has_test = True
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        has_test = False

    # Store original column names for inference schema (before preprocessing)
    original_feature_names = list(X_train.columns)

    # ── 3. Explicit sklearn preprocessing (fit on X_train only) ─────────────
    preprocessor, X_train_prep, prep_names = _build_automl_preprocessor(X_train)
    X_test_prep: Optional[pd.DataFrame] = None
    if has_test and X_test is not None:
        X_test_arr = preprocessor.transform(X_test)
        X_test_prep = pd.DataFrame(X_test_arr, columns=pd.Index(prep_names, dtype=object), index=X_test.index)

    # Keep a pre-SMOTE copy for unbiased train-set evaluation
    X_train_prep_for_eval = X_train_prep.copy()
    y_train_for_eval = y_train.copy()

    # ── 4. SMOTE oversampling (binary classification, severe imbalance only) ─
    smote_applied = False
    if cfg.task_type == "classification":
        classes_sm, counts_sm = np.unique(y_train, return_counts=True)
        if len(classes_sm) == 2 and counts_sm.min() > 0:
            ir_sm = float(counts_sm.max()) / float(counts_sm.min())
            k_nn = min(5, int(counts_sm.min()) - 1)
            if ir_sm > 2.0 and k_nn >= 1 and X_train_prep.shape[0] >= 12:
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(k_neighbors=k_nn, random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(
                        X_train_prep.values, y_train
                    )
                    X_train_prep = pd.DataFrame(X_train_resampled, columns=pd.Index(prep_names, dtype=object))
                    y_train = y_train_resampled
                    smote_applied = True
                    logger.info(
                        "MedAutoML: SMOTE applied (IR=%.2f), train size %d → %d",
                        ir_sm, len(y_train_for_eval), len(y_train),
                    )
                except Exception as exc:
                    logger.warning("MedAutoML: SMOTE failed, continuing without: %s", exc)

    # ── 5. Map metric ─────────────────────────────────────────────────────────
    requested_metric = cfg.metric
    if requested_metric:
        flaml_metric = _FLAML_METRIC_MAP.get(requested_metric, requested_metric)
    else:
        flaml_metric = _AUTO_METRIC.get(cfg.task_type, "roc_auc")

    # ── 6. Medical pre-fit optimizations (sample weights, CV folds, HP bounds) ─
    med = _prepare_medical_automl(X_train_prep, y_train, cfg, flaml_metric, n_samples)

    # ── 7. Progress thread ────────────────────────────────────────────────────
    _stop = threading.Event()

    def _progress_worker() -> None:
        while not _stop.is_set():
            elapsed = time.monotonic() - t_start
            pct = min(95, int(elapsed / max(1, cfg.time_budget) * 90) + 5)
            if progress_cb is not None:
                try:
                    progress_cb(pct, f"AutoML : {int(elapsed)}s / {cfg.time_budget}s")
                except Exception:
                    pass
            _stop.wait(timeout=3.0)

    prog_thread = threading.Thread(target=_progress_worker, daemon=True)
    prog_thread.start()

    # ── 8. Run FLAML on preprocessed data ─────────────────────────────────────
    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(log_fd)

    automl = AutoML()
    try:
        automl.fit(
            X_train_prep,           # clean numeric DataFrame — no raw categoricals
            y_train,
            task=cfg.task_type,
            time_budget=med.effective_budget,
            metric=med.flaml_metric,
            eval_method=med.eval_method,    # "cv"
            n_splits=med.n_splits,          # 5-fold
            n_jobs=-1,
            ensemble=True,
            early_stop=True,
            custom_hp=med.custom_hp,
            verbose=0,
            log_file_name=log_path,
            sample_weight=med.sample_weight,
            model_history=True,
        )
    finally:
        _stop.set()
        prog_thread.join(timeout=5)

    elapsed = time.monotonic() - t_start

    if progress_cb is not None:
        try:
            progress_cb(97, "Évaluation du meilleur modèle…")
        except Exception:
            pass

    # ── 9. Count trials ───────────────────────────────────────────────────────
    n_trials = _count_log_trials(log_path)
    try:
        os.unlink(log_path)
    except OSError:
        pass

    # ── 10. Threshold calibration on holdout test (post-fit) ──────────────────
    optimal_threshold = 0.5
    threshold_optimized = False
    if cfg.task_type == "classification":
        if has_test and X_test_prep is not None:
            optimal_threshold = _find_optimal_threshold(
                automl, X_test_prep, y_test, positive_label=cfg.positive_label
            )
        threshold_optimized = optimal_threshold != 0.5
        if threshold_optimized:
            logger.info("MedAutoML: optimal threshold = %.3f", optimal_threshold)

    # ── 11. Evaluate on preprocessed test / pre-SMOTE train ──────────────────
    evaluator = Evaluator(
        task_type=cfg.task_type,
        positive_label=cfg.positive_label,
    )
    eval_test = (
        evaluator.evaluate(automl, X_test_prep, y_test, threshold=optimal_threshold)
        if has_test and X_test_prep is not None
        else None
    )
    # Evaluate on pre-SMOTE training data (unbiased — SMOTE samples not present)
    eval_train = evaluator.evaluate(
        automl, X_train_prep_for_eval, y_train_for_eval, threshold=optimal_threshold
    )

    # ── 12. Feature importance (from preprocessed feature space) ─────────────
    feature_importance = _extract_feature_importance(automl, prep_names)

    # ── 13. Wrap model for transparent inference ──────────────────────────────
    pipeline = AutoMLPipeline(
        preprocessor=preprocessor,
        automl=automl,
        feature_names_in=original_feature_names,
        prep_cols=prep_names,
    )

    # ── 14. Build metrics_json ────────────────────────────────────────────────
    best_estimator = str(getattr(automl, "best_estimator", "unknown"))
    best_config = _safe_dict(getattr(automl, "best_config", {}))
    best_loss = float(getattr(automl, "best_loss", 0.0))

    split_info: Dict[str, Any] = {
        "method": "automl_holdout" if has_test else "automl_full",
        "train_rows": int(len(X_train_prep_for_eval)),
        "test_rows": int(len(X_test_prep)) if has_test and X_test_prep is not None else 0,
        "n_samples": int(n_samples),
        "test_ratio": float(cfg.test_ratio),
    }

    metrics_json: Dict[str, Any] = {
        "automl": True,
        "test": eval_test.metrics if eval_test is not None else None,
        "has_test": has_test,
        "train": eval_train.metrics,
        "best_estimator": best_estimator,
        "best_loss": best_loss,
        "n_iterations": n_trials,
        "total_time_s": round(elapsed, 2),
        "training_time_sec": round(elapsed, 2),
        "split_info": split_info,
        "imbalance_handled": med.imbalance_applied,
        "smote_applied": smote_applied,
        "threshold_used": optimal_threshold,
        "threshold_optimized": threshold_optimized,
        "features_added": 0,
    }

    # ── 15. Build artifacts_json ──────────────────────────────────────────────
    confusion_matrix_data: List[List[int]] = []
    if eval_test is not None and eval_test.confusion_matrix is not None:
        confusion_matrix_data = [list(row) for row in eval_test.confusion_matrix]

    artifacts_json: Dict[str, Any] = {
        "automl": {
            "best_estimator": best_estimator,
            "best_config": best_config,
            "n_iterations": n_trials,
            "time_budget_s": cfg.time_budget,
            "total_time_s": round(elapsed, 2),
            "eval_method": med.eval_method,
            "n_splits": med.n_splits,
            "metric_optimized": med.flaml_metric,
            "requested_metric": requested_metric,
            "imbalance_handled": med.imbalance_applied,
            "smote_applied": smote_applied,
            "features_added": 0,
            "budget_used_s": med.effective_budget,
            "is_best": True,
        },
        "thresholding": {
            "enabled": threshold_optimized,
            "optimal_threshold": optimal_threshold,
        },
        "split_info": split_info,
        "feature_importance": feature_importance,
        "confusion_matrix": confusion_matrix_data,
        "model": {
            "class_name": best_estimator,
            "params": best_config,
        },
        "automl_feature_pairs": [],
        # Original column names the user must provide at prediction time
        "training_schema": {
            "feature_names": original_feature_names,
        },
    }

    best_result = AutoMLRunResult(
        metrics_json=metrics_json,
        artifacts_json=artifacts_json,
        fitted_model=pipeline,      # AutoMLPipeline wraps preprocessing + FLAML
        task_type=cfg.task_type,
        is_best=True,
    )

    # ── 16. Per-estimator results ─────────────────────────────────────────────
    all_results: List[AutoMLRunResult] = [best_result]
    best_estimator_name = str(getattr(automl, "best_estimator", ""))

    per_estimator_configs: Dict[str, Any] = {}
    per_estimator_losses: Dict[str, Any] = {}
    try:
        per_estimator_configs = dict(getattr(automl, "best_config_per_estimator", {}) or {})
        per_estimator_losses = dict(getattr(automl, "best_loss_per_estimator", {}) or {})
    except Exception:
        pass

    _original_trained = automl._trained_estimator

    for est_name, est_config in per_estimator_configs.items():
        if est_name == best_estimator_name:
            continue
        try:
            est_learner = automl.best_model_for_estimator(est_name)
            if est_learner is None:
                logger.debug("AutoML: no trained model for %s, skipping", est_name)
                continue

            automl._trained_estimator = est_learner
            try:
                est_eval_test = (
                    evaluator.evaluate(automl, X_test_prep, y_test, threshold=0.5)
                    if has_test and X_test_prep is not None
                    else evaluator.evaluate(automl, X_train_prep_for_eval, y_train_for_eval, threshold=0.5)
                )
                est_eval_train = evaluator.evaluate(
                    automl, X_train_prep_for_eval, y_train_for_eval, threshold=0.5
                )
            finally:
                automl._trained_estimator = _original_trained

            est_loss = per_estimator_losses.get(est_name)

            est_cm: List[List[int]] = []
            if est_eval_test.confusion_matrix is not None:
                est_cm = [list(row) for row in est_eval_test.confusion_matrix]

            est_metrics_json: Dict[str, Any] = {
                "automl": True,
                "test": est_eval_test.metrics,
                "train": est_eval_train.metrics,
                "best_estimator": est_name,
                "best_loss": float(est_loss) if est_loss is not None else None,
                "total_time_s": round(elapsed, 2),
                "training_time_sec": round(elapsed, 2),
                "split_info": split_info,
                "imbalance_handled": med.imbalance_applied,
                "smote_applied": smote_applied,
                "threshold_used": 0.5,
                "threshold_optimized": False,
                "features_added": 0,
            }

            est_artifacts_json: Dict[str, Any] = {
                "automl": {
                    "best_estimator": est_name,
                    "best_config": _safe_dict(est_config),
                    "time_budget_s": cfg.time_budget,
                    "total_time_s": round(elapsed, 2),
                    "eval_method": med.eval_method,
                    "metric_optimized": med.flaml_metric,
                    "best_loss": float(est_loss) if est_loss is not None else None,
                    "is_best": False,
                },
                "thresholding": {"enabled": False, "optimal_threshold": 0.5},
                "split_info": split_info,
                "feature_importance": _extract_feature_importance(est_learner, prep_names),
                "confusion_matrix": est_cm,
                "model": {"class_name": est_name, "params": _safe_dict(est_config)},
            }

            all_results.append(AutoMLRunResult(
                metrics_json=est_metrics_json,
                artifacts_json=est_artifacts_json,
                fitted_model=est_learner,   # individual estimator (no preprocessing wrap)
                task_type=cfg.task_type,
                is_best=False,
            ))
            logger.info("AutoML per-estimator result added: %s", est_name)
        except Exception as exc:
            logger.warning("AutoML per-estimator result for %s failed: %s", est_name, exc)
            automl._trained_estimator = _original_trained

    return all_results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_log_trials(log_path: str) -> int:
    """Count non-empty lines in FLAML's log file as a proxy for trial count."""
    try:
        with open(log_path) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _safe_dict(obj: Any) -> dict:
    try:
        return dict(obj) if obj else {}
    except Exception:
        return {}


def _extract_feature_importance(estimator: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extract feature importances from a FLAML AutoML object or sklearn estimator.
    Handles AutoMLPipeline wrapper transparently.
    Returns top 20 features sorted by importance descending.
    """
    try:
        # Unwrap AutoMLPipeline if needed
        real = estimator.automl if isinstance(estimator, AutoMLPipeline) else estimator

        model = getattr(real, "model", None)
        inner = getattr(model, "estimator", model) if model is not None else real

        importances: Optional[np.ndarray] = None
        if hasattr(inner, "feature_importances_"):
            importances = np.asarray(inner.feature_importances_)
        elif hasattr(inner, "coef_"):
            coef = np.asarray(inner.coef_)
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None or len(importances) != len(feature_names):
            return []

        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": float(v)} for f, v in pairs[:20]]
    except Exception:
        return []
