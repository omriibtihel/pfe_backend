from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
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
    "f1_weighted": "f1",
}

_AUTO_METRIC: dict[str, str] = {
    "classification": "roc_auc",
    "regression": "rmse",
}


@dataclass
class _MedicalPrepResult:
    """All parameters produced by the medical pre-fit optimizations."""
    X_train: pd.DataFrame
    feature_pairs: List[Tuple[str, str, str]]   # (col1, col2, new_name)
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
      2. CV 3-fold forced — more robust HP search on small datasets.
      3. Constrained HP search space — anti-overfit bounds via custom_hp.
      4. Metric switch to F1 when imbalance detected (threshold optimised post-fit).
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
            if ir > 1.2:   # lower threshold than default (1.5) for medical data
                sample_weight = compute_sample_weight("balanced", y_train)
                imbalance_applied = True
                logger.info("MedAutoML Opt1: IR=%.2f → sample_weight balanced", ir)

    # ── Opt 2 : CV 3-fold forced (budget respecté tel quel) ───────────────────
    effective_budget = cfg.time_budget   
    eval_method = "cv"
    n_splits = 3   # 3-fold: faster than 5
    logger.info(
        "MedAutoML Opt2: budget=%ds, eval=cv/3-fold", effective_budget
    )

    # ── Opt 3 : Medical-constrained HP search space ────────────────────────────
    custom_hp: Dict[str, Any] = {
        "lgbm": {
            "n_estimators":      {"domain": tune.lograndint(lower=100, upper=1000), "init_value": 300},
            "max_depth":         {"domain": tune.randint(lower=3, upper=8),         "init_value": 5},
            "learning_rate":     {"domain": tune.loguniform(lower=0.01, upper=0.2), "init_value": 0.05},
            "min_child_samples": {"domain": tune.randint(lower=20, upper=100),      "init_value": 50},
            "reg_alpha":         {"domain": tune.loguniform(lower=1e-3, upper=1.0), "init_value": 0.1},
            "reg_lambda":        {"domain": tune.loguniform(lower=1e-3, upper=1.0), "init_value": 0.1},
        },
        "xgboost": {
            "n_estimators":    {"domain": tune.lograndint(lower=100, upper=1000), "init_value": 300},
            "max_depth":       {"domain": tune.randint(lower=3, upper=8),         "init_value": 5},
            "learning_rate":   {"domain": tune.loguniform(lower=0.01, upper=0.2), "init_value": 0.05},
            "min_child_weight":{"domain": tune.randint(lower=5, upper=30),        "init_value": 10},
            "reg_alpha":       {"domain": tune.loguniform(lower=1e-3, upper=1.0), "init_value": 0.1},
            "reg_lambda":      {"domain": tune.loguniform(lower=1e-3, upper=1.0), "init_value": 0.1},
        },
        "rf": {
            "max_depth":        {"domain": tune.randint(lower=3, upper=8),  "init_value": 5},
            "min_samples_split":{"domain": tune.randint(lower=5, upper=20), "init_value": 10},
            "min_samples_leaf": {"domain": tune.randint(lower=3, upper=10), "init_value": 5},
        },
    }

    # No column-name-based feature engineering: the app is dataset-agnostic.
    feature_pairs: List[Tuple[str, str, str]] = []

    # ── Opt 4 : Switch metric to F1 when imbalanced ────────────────────────────
    effective_metric = flaml_metric
    if cfg.task_type == "classification" and not cfg.metric and imbalance_applied:
        effective_metric = "f1"
        logger.info("MedAutoML Opt5: imbalance detected → metric switched to f1")

    return _MedicalPrepResult(
        X_train=X,
        feature_pairs=feature_pairs,  # always empty — no column-name-based engineering
        sample_weight=sample_weight,
        effective_budget=effective_budget,
        eval_method=eval_method,
        n_splits=n_splits,
        flaml_metric=effective_metric,
        custom_hp=custom_hp,
        imbalance_applied=imbalance_applied,
        features_added=0,
    )


def _find_optimal_threshold(
    automl: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    positive_label: Any = None,
) -> float:
    """
    Opt 5 (post-fit): find the probability threshold that maximises F1
    on the training set via the Precision-Recall curve.
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
            return 0.5   # multi-class: skip

        # Determine the positive-class column
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
    fitted_model: Any   # FLAML AutoML object (or individual sklearn estimator)
    task_type: str
    is_best: bool = False  # True for the best estimator overall


def run_automl(
    df: pd.DataFrame,
    cfg: AutoMLConfig,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> List[AutoMLRunResult]:
    """
    Run a FLAML AutoML search on `df` using `cfg`.
    FLAML handles preprocessing, feature engineering, model selection, and HPO.
    Returns an AutoMLRunResult compatible with _model_to_front_result().
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

    # Pandas 3.0 uses StringDtype for all string columns (values AND column index).
    # FLAML internally calls np.issubdtype(X.columns.dtype, …) which fails on
    # StringDtype.  Two fixes needed:
    #   1. Convert string-typed column VALUES to object dtype.
    #   2. Convert the column INDEX itself to object (it also gets StringDtype).
    string_cols = df_clean.select_dtypes(include="string").columns.tolist()
    if string_cols:
        df_clean = df_clean.astype({col: "object" for col in string_cols})

    X = df_clean.drop(columns=[cfg.target_column])
    X.columns = X.columns.astype(object)   # fix the index dtype (StringDtype → object)
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
            # Stratification failed (e.g. minority class too small)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_ratio, random_state=42
            )
        has_test = True
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        has_test = False

    # ── 3. Map metric ─────────────────────────────────────────────────────────
    requested_metric = cfg.metric
    if requested_metric:
        flaml_metric = _FLAML_METRIC_MAP.get(requested_metric, requested_metric)
    else:
        flaml_metric = _AUTO_METRIC.get(cfg.task_type, "roc_auc")

    # ── 4. Medical pre-fit optimizations (opts 1–5) ───────────────────────────
    med = _prepare_medical_automl(X_train, y_train, cfg, flaml_metric, n_samples)
    X_train = med.X_train
    # Apply the same interaction features to X_test (no leakage: only products of raw cols)
    for c1, c2, fname in med.feature_pairs:
        X_test[fname] = X_test[c1] * X_test[c2]
    if med.feature_pairs:
        X_test.columns = X_test.columns.astype(object)

    # ── 4. Progress thread (SSE-only, thread-safe) ────────────────────────────
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

    # ── 5. Run FLAML ──────────────────────────────────────────────────────────
    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(log_fd)

    automl = AutoML()
    try:
        automl.fit(
            X_train,
            y_train,
            task=cfg.task_type,
            time_budget=med.effective_budget,   # Opt 2: budget × 3
            metric=med.flaml_metric,            # Opt 5: f1 when imbalanced
            eval_method=med.eval_method,        # Opt 2: cv forced
            n_splits=med.n_splits,              # Opt 2: 3-fold
            n_jobs=-1,                          # all CPU cores
            ensemble=True,                      # stack best models
            early_stop=True,                    # stop if no improvement
            custom_hp=med.custom_hp,            # Opt 3: medical HP bounds
            verbose=0,
            log_file_name=log_path,
            sample_weight=med.sample_weight,    # Opt 1: balanced weights
            model_history=True,                 # keep fitted model per estimator
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

    # ── 6. Count trials ───────────────────────────────────────────────────────
    n_trials = _count_log_trials(log_path)
    try:
        os.unlink(log_path)
    except OSError:
        pass

    # ── 7. Opt 5 (post-fit): find optimal decision threshold ─────────────────
    optimal_threshold = 0.5
    threshold_optimized = False
    if cfg.task_type == "classification":
        optimal_threshold = _find_optimal_threshold(
            automl, X_train, y_train, positive_label=cfg.positive_label
        )
        threshold_optimized = optimal_threshold != 0.5
        if threshold_optimized:
            logger.info("MedAutoML Opt5: optimal threshold = %.3f", optimal_threshold)

    # ── 8. Evaluate on test / train ───────────────────────────────────────────
    evaluator = Evaluator(
        task_type=cfg.task_type,
        positive_label=cfg.positive_label,
    )
    eval_test = evaluator.evaluate(automl, X_test, y_test, threshold=optimal_threshold)
    eval_train = evaluator.evaluate(automl, X_train, y_train, threshold=optimal_threshold)

    # ── 8. Feature importance ─────────────────────────────────────────────────
    feature_importance = _extract_feature_importance(automl, list(X_train.columns))

    # ── 9. Build metrics_json (compatible with _model_to_front_result) ────────
    best_estimator = str(getattr(automl, "best_estimator", "unknown"))
    best_config = _safe_dict(getattr(automl, "best_config", {}))
    best_loss = float(getattr(automl, "best_loss", 0.0))

    split_info: Dict[str, Any] = {
        "method": "automl_holdout" if has_test else "automl_full",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)) if has_test else 0,
        "n_samples": int(n_samples),
        "test_ratio": float(cfg.test_ratio),
    }

    metrics_json: Dict[str, Any] = {
        "automl": True,
        "test": eval_test.metrics,
        "train": eval_train.metrics,
        "best_estimator": best_estimator,
        "best_loss": best_loss,
        "n_iterations": n_trials,
        "total_time_s": round(elapsed, 2),
        "training_time_sec": round(elapsed, 2),
        "split_info": split_info,
        "imbalance_handled": med.imbalance_applied,
        "threshold_used": optimal_threshold,
        "threshold_optimized": threshold_optimized,
        "features_added": med.features_added,
    }

    # ── 10. Build artifacts_json ──────────────────────────────────────────────
    confusion_matrix_data: List[List[int]] = []
    if eval_test.confusion_matrix is not None:
        confusion_matrix_data = [list(row) for row in eval_test.confusion_matrix]

    artifacts_json: Dict[str, Any] = {
        "automl": {
            "best_estimator": best_estimator,
            "best_config": best_config,
            "n_iterations": n_trials,
            "time_budget_s": cfg.time_budget,
            "total_time_s": round(elapsed, 2),
            "eval_method": med.eval_method,
            "metric_optimized": med.flaml_metric,
            "requested_metric": requested_metric,
            "imbalance_handled": med.imbalance_applied,
            "features_added": med.features_added,
            "budget_used_s": med.effective_budget,
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
        "automl_feature_pairs": [
            {"col1": c1, "col2": c2, "name": fname}
            for c1, c2, fname in med.feature_pairs
        ],
    }

    # Store the feature names that the user must provide for manual prediction.
    # X_train contains only the original dataset columns (no engineered interaction
    # features, since feature_pairs is now always empty).
    artifacts_json["training_schema"] = {
        "feature_names": [str(c) for c in X_train.columns.tolist()],
    }

    # Mark best estimator in its artifacts
    if isinstance(artifacts_json.get("automl"), dict):
        artifacts_json["automl"]["is_best"] = True

    best_result = AutoMLRunResult(
        metrics_json=metrics_json,
        artifacts_json=artifacts_json,
        fitted_model=automl,
        task_type=cfg.task_type,
        is_best=True,
    )

    # ── 11. Build per-estimator results ──────────────────────────────────────
    all_results: List[AutoMLRunResult] = [best_result]
    best_estimator_name = str(getattr(automl, "best_estimator", ""))

    per_estimator_configs: Dict[str, Any] = {}
    per_estimator_losses: Dict[str, Any] = {}
    try:
        per_estimator_configs = dict(getattr(automl, "best_config_per_estimator", {}) or {})
        per_estimator_losses = dict(getattr(automl, "best_loss_per_estimator", {}) or {})
    except Exception:
        pass

    # Save the original best trained estimator to restore it after each evaluation.
    _original_trained = automl._trained_estimator

    for est_name, est_config in per_estimator_configs.items():
        if est_name == best_estimator_name:
            continue  # already included above as best_result
        try:
            est_learner = automl.best_model_for_estimator(est_name)
            if est_learner is None:
                logger.debug("AutoML: no trained model for %s (model_history may be off), skipping", est_name)
                continue

            # Temporarily swap _trained_estimator so that automl.predict/predict_proba
            # go through FLAML's preprocessing pipeline but use this estimator's weights.
            automl._trained_estimator = est_learner
            try:
                est_eval_test = evaluator.evaluate(automl, X_test, y_test, threshold=0.5)
                est_eval_train = evaluator.evaluate(automl, X_train, y_train, threshold=0.5)
            finally:
                automl._trained_estimator = _original_trained  # always restore

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
                "threshold_used": 0.5,
                "threshold_optimized": False,
                "features_added": med.features_added,
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
                "feature_importance": _extract_feature_importance(est_learner, list(X_train.columns)),
                "confusion_matrix": est_cm,
                "model": {"class_name": est_name, "params": _safe_dict(est_config)},
            }

            all_results.append(AutoMLRunResult(
                metrics_json=est_metrics_json,
                artifacts_json=est_artifacts_json,
                fitted_model=est_learner,
                task_type=cfg.task_type,
                is_best=False,
            ))
            logger.info("AutoML per-estimator result added: %s", est_name)
        except Exception as exc:
            logger.warning("AutoML per-estimator result for %s failed: %s", est_name, exc)
            automl._trained_estimator = _original_trained  # safety restore on error

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


def _extract_feature_importance(automl: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Try to extract feature importances from FLAML's best estimator.
    Returns top 20 features sorted by importance descending.
    """
    try:
        model = getattr(automl, "model", None)
        if model is None:
            return []
        estimator = getattr(model, "estimator", model)

        importances: Optional[np.ndarray] = None
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
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
