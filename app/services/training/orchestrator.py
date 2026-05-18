from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sqlalchemy.orm import Session

from app.services.training.output.audit import build_and_persist_audit
from app.services.preparation_ml.balancing import (
    BalancingExecutor,
    BalancingDecision,
    DataProfile,
    class_counts,
    minority_ratio,
    profile_binary_dataset,
    resolve,
)
from app.services.training.config.schema import (
    TrainingConfig, PreprocessingConfig, PreprocessingDefaults,
    inject_class_weight_for_imbalance, normalize_model_hyperparams,
)
from app.services.training.pipeline.confidence import compute_bootstrap_cis
from app.services.training.pipeline.evaluator import Evaluator
from app.services.training.pipeline.importance import compute_permutation_importance
from app.services.training.pipeline.learning_curves import compute_learning_curve
from app.services.training.pipeline.metrics import (
    _is_zero_one_label_set,
    get_class_labels,
    get_proba_or_score,
)
from app.services.training.pipeline.residuals import compute_residuals
from app.services.training.pipeline.models import build_model, get_model_capabilities
from app.services.shap import compute_global_shap
from app.services.preparation_ml.preprocessing.preprocessing import build_preprocessor
from app.services.preparation_ml.feature_engineering import FeatureEngineeringTransformer
from app.services.preparation_ml.feature_engineering.transformer import validate_feature_defs
from app.services.training.output.reporter import Reporter, build_training_schema
from app.services.preparation_ml.splitters import (
    make_holdout_split,
    iter_kfold_splits,
    iter_repeated_kfold_splits,
    iter_group_kfold_splits,
    iter_loo_splits,
)
from app.services.training.pipeline.trainer import Trainer
from app.services.training.pipeline.cv_utils import _build_cv_splitter, _choose_refit_metric
from sklearn.model_selection import GridSearchCV as _GridSearchCV
from app.services.preparation_ml.preprocessing.transformers import (
    ColumnAligner,
    get_clip_warnings,
    clear_clip_warnings,
)

logger = logging.getLogger(__name__)
_DENSE_REQUIRED_MODELS = {"naivebayes", "mlp"}
_RESAMPLE_STRATEGIES = {"smote", "smote_tomek", "random_undersampling"}
# Models that require feature scaling to converge.  When the user selects
# "none" for numeric scaling we silently upgrade to "standard" and log a
# warning rather than letting the model produce nonsensical results (e.g.
# MLP with R² = -12 on unscaled data).
_SCALING_SENSITIVE_MODELS = {"mlp", "svm", "knn", "logisticregression", "logreg"}


def _ensure_scaling_for_model(
    model_type: str,
    preprocessing: "PreprocessingConfig",
) -> "PreprocessingConfig":
    """Return a (possibly patched) PreprocessingConfig that guarantees scaling
    for models that cannot converge without it (MLP, SVM, KNN, logistic regression).

    When the user explicitly set ``numericScaling="none"`` we silently upgrade
    to ``"standard"``.  The original config is never mutated (frozen dataclass).
    A structured log event is emitted so the audit trail shows the override.
    """
    if model_type not in _SCALING_SENSITIVE_MODELS:
        return preprocessing
    current_scaling = preprocessing.defaults.numeric_scaling
    if current_scaling != "none":
        return preprocessing  # already has a scaler — nothing to do

    _log_event(
        "training.preprocessing.scaling_override",
        model_type=model_type,
        original="none",
        override="standard",
        reason=(
            f"{model_type} requires feature scaling to converge. "
            "numericScaling automatically upgraded from 'none' to 'standard'."
        ),
    )
    new_defaults = PreprocessingDefaults(
        numeric_imputation=preprocessing.defaults.numeric_imputation,
        categorical_imputation=preprocessing.defaults.categorical_imputation,
        categorical_encoding=preprocessing.defaults.categorical_encoding,
        numeric_scaling="standard",
        numeric_power_transform=preprocessing.defaults.numeric_power_transform,
    )
    import dataclasses as _dc
    return _dc.replace(preprocessing, defaults=new_defaults)


def _build_resampler_for_gs(decision: Any) -> Any:
    """Build an imblearn resampler instance matching decision.strategy.

    Returns None for non-resampling strategies (class_weight, sample_weight, none …).
    The resampler is meant to be embedded *inside* a GridSearchCV pipeline so that
    resampling is applied per-fold, which prevents leakage of synthetic samples into
    GridSearch validation folds.
    """
    strategy = str(decision.strategy or "none").strip().lower()
    random_state = int(getattr(decision, "random_state", 42))
    k = int(getattr(decision, "smote_k_neighbors", None) or 5)

    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            return None
        return SMOTE(random_state=random_state, k_neighbors=k)

    if strategy == "smote_tomek":
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
        except ImportError:
            return None
        return SMOTETomek(random_state=random_state, smote=SMOTE(random_state=random_state, k_neighbors=k))

    if strategy == "random_undersampling":
        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            return None
        return RandomUnderSampler(random_state=random_state)

    return None


@dataclass
class ModelRunResult:
    model_type: str
    task_type: str
    metrics_json: Dict[str, Any]
    artifacts_json: Dict[str, Any]
    fitted_pipeline: Any  # pipeline fitted (align + preprocess + model)


def _log_event(event: str, **payload: Any) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


def _ensure_dense_matrix(X: Any) -> Any:
    return X.toarray() if hasattr(X, "toarray") else X


def _log_variance_threshold(
    model_type: str,
    n_before: int,
    n_after: int,
    threshold: float,
    feature_selector: Any = None,
    feature_names_before: Optional[list] = None,
) -> list[str]:
    n_removed = n_before - n_after
    if n_removed == 0:
        logger.debug(
            "VarianceThreshold(thr=%.4g) model=%s: no features removed (%d kept).",
            threshold, model_type, n_before,
        )
        return []
    dropped_names: list = []
    if feature_selector is not None and feature_names_before:
        try:
            mask = feature_selector.get_support()
            dropped_names = [n for n, kept in zip(feature_names_before, mask) if not kept]
        except Exception:
            pass
    if dropped_names:
        names_str = ", ".join(str(n) for n in dropped_names[:20])
        if len(dropped_names) > 20:
            names_str += f", … (+{len(dropped_names) - 20} more)"
        msg = (
            f"[VarianceThreshold] Removed {n_removed} constant feature(s) for model={model_type}: "
            f"{names_str}. Set variance_threshold=0.0 to keep all non-constant features."
        )
    else:
        msg = (
            f"[VarianceThreshold] Removed {n_removed}/{n_before} feature(s) for model={model_type} "
            f"(variance < {threshold:.4g}). Set variance_threshold=0.0 to keep all non-constant features."
        )
    logger.warning(msg)
    return [msg]


def _build_inference_pipeline(
    aligner: ColumnAligner,
    preprocessor: Any,
    model: Any,
    model_type: str,
    feature_selector: Any = None,
    fe_transformer: Any = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("align", aligner)]
    if fe_transformer is not None and not fe_transformer.is_noop():
        steps.append(("fe", fe_transformer))
    steps.append(("prep", preprocessor))
    if feature_selector is not None:
        steps.append(("select", feature_selector))
    if model_type in _DENSE_REQUIRED_MODELS:
        steps.append(("dense", FunctionTransformer(_ensure_dense_matrix, accept_sparse=True)))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def _is_debug_enabled(cfg: TrainingConfig) -> bool:
    if bool(getattr(cfg, "debug", False)):
        return True
    import os
    raw = str(os.getenv("TRAINING_DEBUG", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_positive_label_for_run(y_all: np.ndarray, requested_positive_label: Any) -> tuple[Any, Optional[str]]:
    if requested_positive_label is not None:
        return requested_positive_label, None

    unique_vals, counts = np.unique(y_all, return_counts=True)
    if len(unique_vals) != 2:
        return None, None
    if _is_zero_one_label_set(unique_vals):
        return None, None

    ranked = sorted(
        ((int(count), str(label), label) for label, count in zip(unique_vals, counts)),
        key=lambda item: (item[0], item[1]),
    )
    resolved = ranked[0][2]
    warning_msg = (
        "positiveLabel is missing for non {0,1} binary labels; backend resolved positive class to "
        f"'{resolved}'. Set positiveLabel explicitly to ensure stable semantics."
    )
    return resolved, warning_msg


def _default_decision_for_non_classification() -> BalancingDecision:
    return BalancingDecision(
        strategy="none",
        apply_threshold=False,
        threshold_strategy="maximize_f1",
        rationale="Balancing is skipped for non-classification tasks.",
        refit_metric="r2",
        smote_k_neighbors=None,
        class_weight_mode="balanced",
        audit_flags=["none", "task_not_classification"],
        optimal_threshold=0.5,
    )


def _smote_minority_guard(
    decision: BalancingDecision,
    y_train_fold: np.ndarray,
) -> tuple[BalancingDecision, Optional[str]]:
    """Downgrade a SMOTE decision to "no resampling" when the minority class
    is too small for ``k_neighbors``-based interpolation.

    SMOTE / SMOTETomek require at least ``k_neighbors + 1`` minority samples
    to draw nearest neighbours from; on smaller folds the resampler will
    either raise (failing the fold) or generate degenerate synthetic samples
    that all collapse onto the same point. The LOO and K-Fold pipelines
    share this guard so behaviour is consistent across CV strategies.

    Returns ``(decision, warning)``:
      * ``warning is None`` — no guard triggered, original decision returned.
      * ``warning`` is a structured string suitable for logging and for the
        per-fold ``warnings`` list surfaced to the frontend.
    """
    strategy = str(getattr(decision, "strategy", "") or "").lower()
    if strategy not in {"smote", "smote_tomek"}:
        return decision, None
    y_arr = np.asarray(y_train_fold)
    if y_arr.size == 0:
        return decision, None
    _, counts = np.unique(y_arr, return_counts=True)
    if counts.size == 0:
        return decision, None
    min_count = int(counts.min())
    k = int(getattr(decision, "smote_k_neighbors", None) or 5)
    if min_count <= k:
        warning = (
            f"smote_skipped_minority_too_small "
            f"(strategy={strategy}, min_count={min_count}, k_neighbors={k})"
        )
        return _default_decision_for_non_classification(), warning
    return decision, None


# ──────────────────────────────────────────────────────────────────────────────
# CV helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten nested classification or regression metrics to a dict of str→float.
    Used for CV aggregation across folds.
    """
    result: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return result

    _NON_SCALAR = {"warnings", "meta", "confusion_matrix", "per_class", "averaged",
                   "binary", "global", "legacy_flat", "score_shapes"}

    # Direct scalars (regression, or legacy flat)
    for k, v in metrics.items():
        if k in _NON_SCALAR:
            continue
        try:
            result[k] = float(v)
        except (TypeError, ValueError):
            pass

    # Nested classification sub-dicts
    for section in ("global", "legacy_flat"):
        sub = metrics.get(section)
        if isinstance(sub, dict):
            for k, v in sub.items():
                if v is None:
                    continue
                try:
                    result.setdefault(k, float(v))
                except (TypeError, ValueError):
                    pass

    binary = metrics.get("binary")
    if isinstance(binary, dict):
        for k, v in binary.items():
            if k == "positive_label" or v is None:
                continue
            try:
                result.setdefault(k, float(v))
            except (TypeError, ValueError):
                pass

    return result


def _aggregate_cv_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std/min/max across successful folds.

    Metrics that are None (e.g. AUC when a fold has a single class) are excluded
    from aggregation for that metric only.  ``n_folds_per_metric`` reports the
    actual number of folds that contributed to each metric so the caller can
    detect partial coverage.
    """
    successful = [fr for fr in fold_results if fr.get("status") == "ok"]
    if not successful:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "n_folds_ok": 0, "n_folds_per_metric": {}}

    collected: Dict[str, List[float]] = {}
    for fr in successful:
        flat = _extract_scalar_metrics(fr.get("metrics", {}))
        for k, v in flat.items():
            if not np.isnan(v):
                collected.setdefault(k, []).append(v)

    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    mn: Dict[str, float] = {}
    mx: Dict[str, float] = {}
    n_folds_per_metric: Dict[str, int] = {}
    n_ok = len(successful)
    for k, vals in collected.items():
        arr = np.asarray(vals, dtype=float)
        mean[k] = float(np.mean(arr))
        std[k] = float(np.std(arr))
        mn[k] = float(np.min(arr))
        mx[k] = float(np.max(arr))
        n_folds_per_metric[k] = len(vals)
        if len(vals) < n_ok:
            _log_event(
                "training.cv.metric_partial_folds",
                metric=k,
                n_contributing=len(vals),
                n_ok=n_ok,
                reason="metric was None/NaN in some folds (e.g. single-class validation fold)",
            )

    instability_warnings: list[str] = []
    for k, s in std.items():
        if s > 0.15 and mean.get(k, 0.0) > 0.0:
            instability_warnings.append(
                f"High CV variance on '{k}': std={s:.3f} (mean={mean[k]:.3f}). "
                "Results may be unreliable — consider more data or a simpler model."
            )
            _log_event(
                "training.cv.high_variance",
                metric=k,
                mean=mean[k],
                std=s,
                n_folds_ok=n_ok,
            )

    return {
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "n_folds_ok": n_ok,
        "n_folds_per_metric": n_folds_per_metric,
        "instability_warnings": instability_warnings,
    }


def _compute_oof_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics on pooled out-of-fold predictions.

    Aggregating RMSE per fold via the arithmetic mean is biased by Jensen's
    inequality: ``mean(sqrt(MSE_i)) < sqrt(mean(MSE_i))``. R² aggregated per
    fold also diverges from the pooled OOF R² when fold sizes or target
    variances differ. This helper recomputes both on the concatenated OOF
    predictions, which is the statistically correct CV estimate.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mse":  mse,
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Holdout pipeline (original, preserved)
# ──────────────────────────────────────────────────────────────────────────────

def run_one_model(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    """
    Entry point for a single model run.
    Dispatches to holdout or k-fold CV based on cfg.split_method.
    """
    _CV_METHODS = {
        "kfold", "stratified_kfold",
        "repeated_stratified_kfold",
        "group_kfold", "stratified_group_kfold",
    }
    if cfg.split_method == "loo":
        return _run_loo(df, cfg, model_type, db=db, session_id=session_id, kind_overrides=kind_overrides)
    if cfg.split_method in _CV_METHODS:
        return _run_kfold_cv(df, cfg, model_type, db=db, session_id=session_id, kind_overrides=kind_overrides)
    return _run_holdout(df, cfg, model_type, db=db, session_id=session_id, kind_overrides=kind_overrides)


def _run_holdout(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    debug_mode = _is_debug_enabled(cfg)
    _log_event("training.start_model", model_type=model_type_norm, task_type=cfg.task_type)

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    if len(df2) < 10:
        raise RuntimeError("Not enough rows after dropping target NaNs")

    X = df2.drop(columns=[cfg.target_column])
    y = df2[cfg.target_column].values
    if debug_mode and cfg.task_type == "classification":
        _log_event(
            "training.debug.global_distribution",
            model_type=model_type_norm,
            class_distribution=class_counts(np.asarray(y)),
            minority_ratio=minority_ratio(np.asarray(y)),
        )

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        np.asarray(y),
        cfg.positive_label,
    )
    if positive_label_warning is not None:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    split = make_holdout_split(
        X,
        y,
        task_type=cfg.task_type,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_state=cfg.random_state,
    )
    if split.warnings:
        for warning_msg in split.warnings:
            _log_event("training.split.warning", model_type=model_type_norm, message=warning_msg)
    if debug_mode and cfg.task_type == "classification":
        _log_event(
            "training.debug.split_distribution",
            model_type=model_type_norm,
            attempts=split.attempts,
            random_state_used=split.random_state_used,
            train=class_counts(np.asarray(split.y_train)),
            val=class_counts(np.asarray(split.y_val)) if split.y_val is not None else {},
            test=class_counts(np.asarray(split.y_test)) if split.y_test is not None else {},
        )

    if split.X_train is None or len(split.X_train) == 0:
        raise RuntimeError("Preprocessing requires a valid split with a non-empty train set.")

    effective_preprocessing = _ensure_scaling_for_model(model_type_norm, cfg.preprocessing)
    fe_defs = [f.as_dict() for f in cfg.feature_engineering.features]
    fe_validation_errors = validate_feature_defs(fe_defs, list(split.X_train.columns))
    if fe_validation_errors:
        raise RuntimeError("Feature engineering config errors:\n" + "\n".join(fe_validation_errors))
    requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})
    if not isinstance(requested_hyperparams_raw, dict):
        requested_hyperparams_raw = {}
    hp_normalized = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=bool(cfg.use_grid_search),
        task_type=cfg.task_type,
    )
    hp_errors = [str(msg) for msg in (hp_normalized.get("errors") or []) if str(msg).strip()]
    if hp_errors:
        raise RuntimeError(f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}")
    hp_warnings: list[str] = [str(w) for w in (hp_normalized.get("warnings") or []) if str(w).strip()]
    for hp_warning in hp_warnings:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=hp_warning)

    estimator_hyperparams = dict(hp_normalized.get("estimator_params") or {})
    param_grid = dict(hp_normalized.get("param_grid") or {})
    param_distributions = dict(hp_normalized.get("param_distributions") or {})
    training_schema = build_training_schema(
        X=split.X_train,
        target=cfg.target_column,
        preprocessing_config=effective_preprocessing.as_dict(),
    )
    aligner = ColumnAligner(
        feature_names=training_schema["feature_names"],
        dtypes=training_schema["dtypes"],
    )

    # Fit preprocessing on train split once (train-only policy), then execute balancing on transformed train data.
    X_train_aligned = aligner.fit_transform(split.X_train)
    # Feature engineering: fit stats on train only (anti-leakage), add new columns before preprocessor.
    fe_transformer = FeatureEngineeringTransformer(fe_defs)
    X_train_fe = fe_transformer.fit_transform(X_train_aligned)
    spec = build_preprocessor(X_train_fe, effective_preprocessing, kind_overrides=kind_overrides)
    clear_clip_warnings()
    X_train_prepared = spec.preprocessor.fit_transform(X_train_fe, np.asarray(split.y_train))
    _prep_clip_warnings = get_clip_warnings()
    clear_clip_warnings()

    # VarianceThreshold after StandardScaler: threshold=0.0 removes only
    # perfectly constant features. Higher values (e.g. 0.01) would
    # incorrectly drop rare binary features (prevalence ~1%) whose
    # clinical relevance may be high.
    _vt_holdout_w: list[str] = []
    if spec.feature_selector is not None:
        _vt_n_before = X_train_prepared.shape[1] if hasattr(X_train_prepared, "shape") else 0
        _vt_names_before: list = []
        try:
            _vt_names_before = list(spec.preprocessor.get_feature_names_out())
        except Exception:
            pass
        spec.feature_selector.fit(X_train_prepared)
        X_train_prepared = spec.feature_selector.transform(X_train_prepared)
        _vt_holdout_w = _log_variance_threshold(
            model_type_norm, _vt_n_before, X_train_prepared.shape[1],
            cfg.preprocessing.variance_threshold,
            feature_selector=spec.feature_selector,
            feature_names_before=_vt_names_before,
        )
        if X_train_prepared.shape[1] == 0:
            raise RuntimeError(
                f"Preprocessing a supprimé toutes les features pour le modèle '{model_type_norm}'. "
                "Vérifiez votre configuration de preprocessing (VarianceThreshold trop strict, "
                "ou toutes les colonnes ont une variance nulle)."
            )

    if model_type_norm in _DENSE_REQUIRED_MODELS:
        X_train_prepared = _ensure_dense_matrix(X_train_prepared)

    # Use registry capabilities instead of instantiating a throwaway model.
    capabilities = get_model_capabilities(model_type_norm)

    profile: DataProfile | None = None
    decision: BalancingDecision = _default_decision_for_non_classification()
    if cfg.task_type == "classification":
        profile = profile_binary_dataset(np.asarray(split.y_train), split.X_train.shape)
        decision = resolve(
            profile=profile,
            config=cfg.balancing,
            model_supports_class_weight=capabilities["supports_class_weight"],
            model_supports_predict_proba=capabilities["supports_predict_proba"],
            model_supports_sample_weight=capabilities["supports_sample_weight"],
            random_state=cfg.random_state,
        )
        # Threshold optimization is binary-only: disable it for multiclass with explicit warning.
        _multiclass_threshold_disabled = False
        if decision.apply_threshold and len(np.unique(np.asarray(split.y_train))) > 2:
            decision.apply_threshold = False
            _multiclass_threshold_disabled = True
    else:
        _multiclass_threshold_disabled = False

    executor = BalancingExecutor()
    X_f = X_train_prepared
    y_f = np.asarray(split.y_train)
    fit_params: dict[str, Any] = {}

    # ── SMOTE + GridSearch anti-leakage fix ──────────────────────────────────
    # When GridSearch is active AND the balancing strategy resamples the data
    # (smote / smote_tomek / random_undersampling), applying the resampler
    # *before* GridSearchCV causes leakage: GS's internal validation folds will
    # contain synthetic samples whose neighbours came from those same folds.
    # Fix: embed the resampler inside the GridSearch pipeline (imblearn.Pipeline)
    # so resampling is applied independently within each GS fold.
    # For non-resampling strategies (class_weight, sample_weight, none) the
    # existing apply_prefit path is unchanged — no leakage there.
    resampler_for_gs: Any | None = None
    use_gs_with_resampling = (
        cfg.search_type != "none"
        and cfg.task_type == "classification"
        and decision.strategy in _RESAMPLE_STRATEGIES
    )

    if cfg.task_type == "classification":
        if use_gs_with_resampling:
            # Build resampler to inject inside GS; do NOT pre-resample X_f.
            resampler_for_gs = _build_resampler_for_gs(decision)
            # Dense conversion is still required for imblearn resamplers.
            if resampler_for_gs is not None and hasattr(X_train_prepared, "toarray"):
                X_train_prepared = X_train_prepared.toarray()
                X_f = X_train_prepared
            # fit_params stays {} — no class_weight for SMOTE strategies
        else:
            # Original flow: pre-apply resampling before fitting (no GS, no leakage risk).
            if decision.strategy in _RESAMPLE_STRATEGIES:
                if hasattr(X_train_prepared, "toarray"):
                    X_train_prepared = X_train_prepared.toarray()
            X_f, y_f, fit_params = executor.apply_prefit(
                X_train_prepared, np.asarray(split.y_train), decision
            )

    smote_samples_added: int | None = None
    if cfg.task_type == "classification" and decision.strategy in {"smote", "smote_tomek"}:
        if use_gs_with_resampling and resampler_for_gs is not None:
            # Estimate augmented sample count for audit (fit a clone, don't mutate the GS resampler).
            try:
                from sklearn.base import clone as _clone
                _, y_counted = _clone(resampler_for_gs).fit_resample(X_f, np.asarray(split.y_train))
                smote_samples_added = int(len(y_counted) - len(np.asarray(split.y_train)))
            except Exception:
                smote_samples_added = None
        else:
            smote_samples_added = int(len(np.asarray(y_f)) - len(np.asarray(split.y_train)))

    if "class_weight" in fit_params:
        estimator_hyperparams["class_weight"] = fit_params["class_weight"]
    model = build_model(model_type_norm, cfg.task_type, estimator_hyperparams)

    fit_sample_weight = fit_params.get("sample_weight")
    if fit_sample_weight is not None:
        _sw_caps = get_model_capabilities(model_type_norm)
        if not _sw_caps.get("supports_sample_weight", True):
            _log_event(
                "training.balancing.sample_weight_skipped",
                model_type=model_type_norm,
                reason="model does not support sample_weight — weight ignored to avoid TypeError",
            )
            fit_sample_weight = None
        else:
            fit_sample_weight = np.asarray(fit_sample_weight)

    # imbalanced=True only when the dataset is skewed AND no balancing mechanism
    # already handles it.  This flag controls whether class_weight is injected
    # into the GridSearch grid.  If SMOTE/undersampling is active the resampler
    # is already embedded in the GS pipeline (anti-leakage path above) and
    # adding class_weight on top would doubly compensate.  If the class_weight
    # strategy was chosen by BalancingExecutor it is already baked into the
    # estimator's params — letting GS re-explore it could silently override
    # the balancing decision with class_weight=None.
    _active_balancing = decision.strategy not in {"none", "threshold_optimization"}
    _is_imbalanced = (
        cfg.task_type == "classification"
        and profile is not None
        and profile.imbalance_ratio is not None
        and profile.imbalance_ratio > 3.0
        and not _active_balancing
    )
    if bool(cfg.use_grid_search):
        param_grid = inject_class_weight_for_imbalance(
            param_grid,
            model_key=model_type_norm,
            task_type=cfg.task_type,
            imbalanced=_is_imbalanced,
        )
    trainer = Trainer()
    fit_result = trainer.fit(
        pipeline=Pipeline(steps=[("model", model)]),
        X_train=X_f,
        y_train=np.asarray(y_f),
        fit_sample_weight=fit_sample_weight,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=param_grid,
        model_param_distributions=param_distributions,
        refit_metric_override=(decision.refit_metric if cfg.task_type == "classification" else None),
        resampler=resampler_for_gs,
        n_samples=int(len(y_f)),
        imbalanced=_is_imbalanced,
        positive_label=resolved_positive_label,
    )

    fitted_model = fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        aligner, spec.preprocessor, fitted_model, model_type_norm,
        feature_selector=spec.feature_selector,
        fe_transformer=fe_transformer,
    )
    if debug_mode and cfg.task_type == "classification":
        _log_event(
            "training.debug.model_classes",
            model_type=model_type_norm,
            model_classes=get_class_labels(fitted_pipe),
        )

    # Threshold calibration: prefer the validation set when available; otherwise
    # fall back to X_train (no test-set leakage, though slightly optimistic).
    # The source is recorded in metrics_json["threshold_source"] so callers can
    # distinguish the two cases ("val_set" vs "train_fallback" vs "disabled").
    has_val = split.X_val is not None and len(split.X_val) > 0
    if has_val:
        threshold_input_X = split.X_val
        threshold_input_y = split.y_val
        _threshold_source_raw = "val_set"
    else:
        threshold_input_X = split.X_train
        threshold_input_y = split.y_train
        _threshold_source_raw = "train_fallback"
    optimal_threshold = 0.5
    threshold_f1_gain: float | None = None
    threshold_source = "disabled"
    if cfg.task_type == "classification":
        optimal_threshold = executor.apply_postfit(
            fitted_pipe,
            threshold_input_X,
            np.asarray(threshold_input_y) if threshold_input_y is not None else None,
            decision,
            positive_label=resolved_positive_label,
        )
        if executor.last_threshold_result is not None:
            threshold_f1_gain = float(executor.last_threshold_result.improvement_delta)
            r = executor.last_threshold_result
            threshold_source = "disabled" if r.strategy_used == "disabled" else _threshold_source_raw

        # Surface informational warnings that the frontend can display.
        if _multiclass_threshold_disabled:
            executor.postfit_warnings.append("threshold_optimization_disabled_multiclass")
        if threshold_source == "train_fallback" and optimal_threshold != 0.5:
            executor.postfit_warnings.append("threshold_calibrated_on_train_data_may_be_optimistic")

    # Build and persist the audit record in a single commit after all pipeline stages complete.
    balancing_audit: dict[str, Any] = {}
    if profile is not None:
        balancing_audit = build_and_persist_audit(
            profile=profile,
            decision=decision,
            session_id=session_id,
            db=db,
            smote_samples_added=smote_samples_added,
            optimal_threshold=optimal_threshold,
            threshold_f1_gain=threshold_f1_gain,
            postfit_warnings=list(executor.postfit_warnings or []),
        )

    evaluator = Evaluator(
        task_type=cfg.task_type,
        requested_metrics=cfg.metrics,
        positive_label=resolved_positive_label,
    )
    # Use the same threshold as the test set so train vs test metrics are
    # directly comparable. Computing train metrics at threshold=0.5 while
    # test metrics use optimal_threshold makes precision/recall look inconsistent.
    train_eval = evaluator.evaluate(fitted_pipe, split.X_train, np.asarray(split.y_train), threshold=optimal_threshold)

    val_metrics = None
    if split.X_val is not None and split.y_val is not None and len(split.X_val) > 0:
        val_eval = evaluator.evaluate(fitted_pipe, split.X_val, np.asarray(split.y_val), threshold=optimal_threshold)
        val_metrics = val_eval.metrics
    else:
        val_eval = None

    test_eval = (
        evaluator.evaluate(fitted_pipe, split.X_test, np.asarray(split.y_test), threshold=optimal_threshold)
        if split.X_test is not None and split.y_test is not None and len(split.X_test) > 0
        else None
    )

    class_distribution = None
    confusion_matrix = None
    if cfg.task_type == "classification":
        class_distribution = {
            "train": class_counts(np.asarray(split.y_train)),
            "val": class_counts(np.asarray(split.y_val)) if split.y_val is not None else {},
            "test": class_counts(np.asarray(split.y_test)) if split.y_test is not None else {},
            "train_minority_ratio": minority_ratio(np.asarray(split.y_train)),
            "val_minority_ratio": minority_ratio(np.asarray(split.y_val)) if split.y_val is not None else None,
            "test_minority_ratio": minority_ratio(np.asarray(split.y_test)) if split.y_test is not None else None,
        }
        confusion_matrix = test_eval.confusion_matrix if test_eval is not None else None
        _log_event(
            "training.class_distribution",
            model_type=model_type_norm,
            class_distribution=class_distribution,
        )

        eval_blocks: Dict[str, Any] = {"train": train_eval.metrics, "test": test_eval.metrics if test_eval is not None else {}}
        if val_metrics is not None:
            eval_blocks["val"] = val_metrics
        for split_name, split_metrics in eval_blocks.items():
            if not isinstance(split_metrics, dict):
                continue
            meta = split_metrics.get("meta") if isinstance(split_metrics.get("meta"), dict) else {}
            score_shapes = meta.get("score_shapes") if isinstance(meta.get("score_shapes"), dict) else {}
            _log_event(
                "training.classification.metrics",
                model_type=model_type_norm,
                split=split_name,
                classification_type=meta.get("classification_type"),
                labels=meta.get("labels"),
                positive_label=meta.get("positive_label"),
                resolved_positive_label=meta.get("resolved_positive_label"),
                model_classes=meta.get("model_classes"),
                auc_score_source=meta.get("auc_score_source"),
                auc_pos_index=meta.get("auc_pos_index"),
                threshold_used=meta.get("threshold_used"),
                proba_shape=score_shapes.get("proba"),
                score_shape=score_shapes.get("y_score"),
                warnings=split_metrics.get("warnings", []),
            )
            if debug_mode:
                cm_payload = split_metrics.get("confusion_matrix")
                if isinstance(cm_payload, dict):
                    cm_matrix = cm_payload.get("matrix")
                    cm_labels = cm_payload.get("labels")
                    if (
                        isinstance(cm_matrix, list)
                        and len(cm_matrix) == 2
                        and isinstance(cm_matrix[0], list)
                        and len(cm_matrix[0]) == 2
                    ):
                        _log_event(
                            "training.debug.confusion_matrix",
                            model_type=model_type_norm,
                            split=split_name,
                            labels=cm_labels,
                            tn=cm_matrix[0][0],
                            fp=cm_matrix[0][1],
                            fn=cm_matrix[1][0],
                            tp=cm_matrix[1][1],
                        )

    reporter = Reporter()
    artifacts = reporter.build_artifacts(
        cfg=cfg,
        split=split,
        spec=spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        training_schema=training_schema,
        tuning_artifacts=fit_result.tuning_artifacts,
        confusion_matrix=confusion_matrix,
        class_distribution=class_distribution,
        resolved_positive_label=resolved_positive_label,
        curves=test_eval.metrics.get("curves") if (test_eval is not None and isinstance(test_eval.metrics, dict)) else None,
    )
    hyperparams_artifacts: Dict[str, Any] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized.get("effective") or {}),
    }
    if bool(cfg.use_grid_search):
        hyperparams_artifacts["param_grid"] = dict(param_grid)
        best_params = fit_result.tuning_artifacts.get("best_params")
        if isinstance(best_params, dict):
            hyperparams_artifacts["best"] = dict(best_params)
    artifacts["hyperparams"] = hyperparams_artifacts

    # Bootstrap confidence intervals on the test set (skipped if test set < 10 samples)
    bootstrap_cis = None
    if test_eval is not None and split.X_test is not None and split.y_test is not None:
        _ci_y_proba, _ci_y_score = get_proba_or_score(fitted_pipe, split.X_test)
        _ci_score_vec: Optional[np.ndarray] = None
        if _ci_y_proba is not None and _ci_y_proba.ndim == 2 and _ci_y_proba.shape[1] == 2:
            _ci_score_vec = _ci_y_proba[:, 1]
        elif _ci_y_score is not None and _ci_y_score.ndim == 1:
            _ci_score_vec = _ci_y_score
        bootstrap_cis = compute_bootstrap_cis(
            split.y_test,
            test_eval.predictions,
            y_score=_ci_score_vec,
            task_type=cfg.task_type,
        )

    artifact_warnings: list[dict] = []

    # Learning curves on preprocessed training data (fresh model, no SMOTE).
    # Skipped when training time is slow (search_type != "none") to avoid doubling runtime.
    _lc_result: Optional[Dict[str, Any]] = None
    if cfg.search_type == "none" and len(np.asarray(split.y_train)) >= 30:
        try:
            _lc_model = build_model(model_type_norm, cfg.task_type, estimator_hyperparams)
            _lc_result = compute_learning_curve(
                _lc_model,
                X_train_prepared,
                np.asarray(split.y_train),
                task_type=cfg.task_type,
                random_state=cfg.random_state,
            )
        except Exception as e:
            _lc_result = None
            artifact_warnings.append({"artifact": "learning_curves", "error": type(e).__name__, "detail": str(e)})

    if _lc_result is not None:
        artifacts["learning_curves"] = _lc_result

    # Permutation importance + residual analysis on the test set.
    # Only computed when the test set is large enough to be meaningful.
    if test_eval is not None and split.X_test is not None and split.y_test is not None and len(np.asarray(split.y_test)) >= 20:
        try:
            _pi_result = compute_permutation_importance(
                fitted_pipe,
                split.X_test,
                np.asarray(split.y_test),
                task_type=cfg.task_type,
                feature_names=list(split.X_test.columns),
            )
            if _pi_result is not None:
                artifacts["permutation_importance"] = _pi_result
        except Exception as e:
            artifact_warnings.append({"artifact": "permutation_importance", "error": type(e).__name__, "detail": str(e)})

        try:
            _ra_result = compute_residuals(
                np.asarray(split.y_test),
                np.asarray(test_eval.predictions),
                task_type=cfg.task_type,
            )
            if _ra_result is not None:
                artifacts["residual_analysis"] = _ra_result
        except Exception as e:
            artifact_warnings.append({"artifact": "residual_analysis", "error": type(e).__name__, "detail": str(e)})

        # Global SHAP — computed on the final refit pipeline.
        # When a search is active, fitted_pipe is best_estimator_ (already refit),
        # so SHAP cost is independent of search runtime. KernelExplainer is
        # capped internally by _KERNEL_MAX_ROWS to bound runtime.
        try:
            _shap_result = compute_global_shap(
                fitted_pipe,
                split.X_test,
                task_type=cfg.task_type,
                feature_names=list(split.X_test.columns),
                positive_label=resolved_positive_label,
            )
            if _shap_result is not None:
                artifacts["shap"] = _shap_result
        except Exception as e:
            artifact_warnings.append({"artifact": "shap", "error": type(e).__name__, "detail": str(e)})

    if artifact_warnings:
        artifacts["artifact_warnings"] = artifact_warnings

    has_real_test = test_eval is not None
    metrics_json: Dict[str, Any] = {
        "train": train_eval.metrics,
        "test": test_eval.metrics if has_real_test else {},
        "training_time_sec": float(time.perf_counter() - t0),
        "threshold_used": optimal_threshold,
        "threshold_source": threshold_source,
        "confidence_intervals": bootstrap_cis,
    }
    metrics_json["has_holdout_test"] = has_real_test
    metrics_json["test_is_cv_mean"] = False
    if has_real_test:
        metrics_json["test_label"] = "Holdout test set"
    else:
        metrics_json["test_label"] = "Entraînement uniquement — aucun jeu de test"
        metrics_json["evaluation_strategy"] = "train_only"
    postfit_w: list[str] = list(executor.postfit_warnings or [])
    _clip_strs = [
        f"CLIP_NEGATIVE in '{w['column']}': {w['n_clipped']} value(s) clipped before "
        f"{w['transform']} transform (min={w['min_observed']:.6g})"
        for w in _prep_clip_warnings
    ]
    all_warnings: list[str] = list(hp_warnings) + postfit_w + _clip_strs + _vt_holdout_w + [
        f"Artefact {w['artifact']} non calculé ({w['error']})" for w in artifact_warnings
    ]
    if all_warnings:
        metrics_json["warnings"] = all_warnings
    if _prep_clip_warnings:
        metrics_json["clip_warnings"] = [
            {"severity": "warning", "code": "CLIP_NEGATIVE", **w} for w in _prep_clip_warnings
        ]
    if val_metrics is not None:
        metrics_json["val"] = val_metrics

    _log_event(
        "training.end_model",
        model_type=model_type_norm,
        task_type=cfg.task_type,
        training_time_sec=metrics_json["training_time_sec"],
        balancing_strategy=decision.strategy,
        threshold_applied=bool(decision.apply_threshold),
        tuned=bool(fit_result.tuning_artifacts.get("enabled", False)),
    )

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )


# [REFACTOR] _run_kfold_cv, _run_loo and _collect_fold_warnings live in
# app/services/training/cv_runner.py since 2026-05-02. These deferred imports
# keep them accessible as orchestrator.<name> for tests and for run_one_model
# dispatch. The imports sit at the bottom so all helpers referenced by
# cv_runner are already defined when cv_runner is loaded.
from app.services.training.cv_runner import (  # noqa: E402,F401
    _collect_fold_warnings,
    _run_kfold_cv,
    _run_loo,
)
