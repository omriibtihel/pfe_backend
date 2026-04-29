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
    TrainingConfig, PreprocessingConfig, PreprocessingDefaults, normalize_model_hyperparams,
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
        refit_metric_override=(decision.refit_metric if cfg.task_type == "classification" else None),
        resampler=resampler_for_gs,
        n_samples=int(len(y_f)),
        imbalanced=_is_imbalanced,
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
    train_eval = evaluator.evaluate(fitted_pipe, split.X_train, np.asarray(split.y_train))

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
        curves=test_eval.metrics.get("curves") if isinstance(test_eval.metrics, dict) else None,
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

        # Global SHAP — computed after permutation importance.
        # KernelExplainer can be slow; skipped for search_type != "none"
        # (already slow from hyperparameter tuning).
        _shap_allowed = cfg.search_type == "none"
        try:
            _shap_result = compute_global_shap(
                fitted_pipe,
                split.X_test,
                task_type=cfg.task_type,
                feature_names=list(split.X_test.columns),
            ) if _shap_allowed else None
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


# ──────────────────────────────────────────────────────────────────────────────
# K-Fold / Stratified K-Fold CV pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _collect_fold_warnings(
    fold_results: list[dict],
    *,
    collapse_threshold: int = 3,
) -> tuple[list[dict], list[str]]:
    """
    Surface per-fold warnings that were previously stored in fold_results but
    never propagated to metrics_json["warnings"].

    Each entry in fold_results["warnings"] is normalised to a dict and enriched
    with "fold" and "context" fields.  When the same (code, message) pair
    appears in more than *collapse_threshold* folds it is collapsed into a
    single FOLD_WARNING_REPEATED summary to prevent warning overload for large K.

    Returns:
        structured  — list[dict] suitable for metrics_json["fold_warnings"]
        string_list — list[str]  to append to the existing all_warnings string list
    """
    raw: list[dict] = []
    n_folds_total = len(fold_results)
    for fr in fold_results:
        fold_num = fr.get("fold", "?")
        for w in fr.get("warnings", []):
            if isinstance(w, str):
                entry: dict[str, Any] = {
                    "severity": "warning",
                    "code": "FOLD_WARNING",
                    "message": w,
                }
            elif isinstance(w, dict):
                entry = dict(w)
            else:
                continue
            entry["fold"] = fold_num
            entry["context"] = "cv_fold"
            raw.append(entry)

    if not raw:
        return [], []

    # Group by (code, message) to detect same-warning repetition across folds.
    groups: dict[tuple, list[dict]] = {}
    for entry in raw:
        key = (entry.get("code", "FOLD_WARNING"), entry.get("message", ""))
        groups.setdefault(key, []).append(entry)

    structured: list[dict] = []
    for (code, message), entries in groups.items():
        if len(entries) > collapse_threshold:
            affected = sorted(
                e["fold"] for e in entries if e.get("fold") is not None
            )
            structured.append({
                "severity": entries[0].get("severity", "warning"),
                "code": "FOLD_WARNING_REPEATED",
                "message": (
                    f"'{code}' appeared in {len(entries)}/{n_folds_total} folds "
                    f"(folds {','.join(str(f) for f in affected)}): {message}"
                ),
                "affected_folds": affected,
                "context": "cv_fold",
            })
        else:
            structured.extend(entries)

    string_list: list[str] = []
    for s in structured:
        if s.get("code") == "FOLD_WARNING_REPEATED":
            string_list.append(s["message"])
        else:
            f_num = s.get("fold", "?")
            string_list.append(f"fold {f_num}: {s.get('message', s.get('code', ''))}")

    return structured, string_list


def _run_kfold_cv(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    """
    Cross-validation pipeline (kfold / stratified_kfold / repeated_stratified_kfold /
    group_kfold / stratified_group_kfold).

    Anti-leakage guarantees
    -----------------------
    • When cfg.test_ratio > 0, a stratified holdout test set is carved out of
      the full data FIRST — before any preprocessing fit, before any CV fold.
      For group_kfold: GroupShuffleSplit ensures no patient spans cv/test boundary.
    • The CV loop (fold fit + val evaluation) runs on the non-test portion only.
    • Preprocessor is fit on train_fold indices ONLY, then transforms both
      train_fold and val_fold.
    • Resampling (SMOTE / undersampling) is applied on train_fold ONLY.
    • val_fold and the holdout test set are NEVER resampled.
    • group_column is excluded from X before any processing (prevents leakage).

    GridSearch + CV interaction
    ---------------------------
    GridSearch (use_grid_search=True) is intentionally DISABLED inside the
    per-fold training loop to avoid unintended double-CV (outer loop × inner
    GridSearchCV). GridSearch is applied once on the FINAL REFIT so the saved
    model benefits from hyperparameter tuning.

    Persistence (Option A — refit-final)
    -------------------------------------
    • test_ratio == 0: refit on ALL data → deployed model trained on 100% of samples.
    • test_ratio >  0: refit on non-test data → test set remains truly independent.
    The saved artifact declares whether refit covered the full dataset.
    """
    from sklearn.model_selection import train_test_split as _train_test_split

    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    _log_event(
        "training.cv.start",
        model_type=model_type_norm,
        split_method=cfg.split_method,
        k_folds=cfg.k_folds,
        task_type=cfg.task_type,
        test_ratio=float(cfg.test_ratio),
        n_repeats=int(getattr(cfg, "n_repeats", 3)),
        group_column=getattr(cfg, "group_column", None),
    )

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    if len(df2) < 10:
        raise RuntimeError("Not enough rows after dropping target NaNs (minimum 10).")

    # ── Group column handling (group_kfold / stratified_group_kfold) ──────────
    # Extract groups BEFORE building X so the group_column is never a feature.
    _is_group_cv = cfg.split_method in ("group_kfold", "stratified_group_kfold")
    groups_all: Optional[np.ndarray] = None
    if _is_group_cv:
        gc = getattr(cfg, "group_column", None)
        if not gc or gc not in df2.columns:
            raise RuntimeError(
                f"splitMethod='{cfg.split_method}' requires groupColumn "
                f"to be a valid column in the dataset. Got: '{gc}'."
            )
        groups_all = np.asarray(df2[gc].values)
        drop_cols = [cfg.target_column, gc]
    else:
        drop_cols = [cfg.target_column]

    X_all = df2.drop(columns=drop_cols)
    y_all = np.asarray(df2[cfg.target_column].values)

    fe_defs = [f.as_dict() for f in cfg.feature_engineering.features]
    fe_validation_errors = validate_feature_defs(fe_defs, list(X_all.columns))
    if fe_validation_errors:
        raise RuntimeError("Feature engineering config errors:\n" + "\n".join(fe_validation_errors))

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        y_all, cfg.positive_label
    )
    if positive_label_warning:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    # ── Optional holdout test split ───────────────────────────────────────────
    # When cfg.test_ratio > 0, we separate a stratified (for classification)
    # test set BEFORE any CV or preprocessing — pure holdout, never touched.
    # For group_kfold: use GroupShuffleSplit to keep patient groups intact.
    has_holdout_test = float(cfg.test_ratio) > 1e-6
    X_test_holdout: Optional[pd.DataFrame] = None
    y_test_holdout: Optional[np.ndarray] = None
    X_cv: pd.DataFrame = X_all
    y_cv: np.ndarray = y_all
    groups_cv: Optional[np.ndarray] = groups_all  # may be None for non-group methods

    if has_holdout_test:
        if _is_group_cv and groups_all is not None:
            # GroupShuffleSplit: ensures no patient spans the cv/test boundary
            from sklearn.model_selection import GroupShuffleSplit as _GSS
            gss = _GSS(n_splits=1, test_size=float(cfg.test_ratio), random_state=cfg.random_state)
            try:
                cv_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))
            except Exception as split_exc:
                _log_event("training.cv.holdout_split_fallback", model_type=model_type_norm, reason=str(split_exc))
                cv_idx = np.arange(int(len(y_all) * (1.0 - float(cfg.test_ratio))))
                test_idx = np.arange(len(cv_idx), len(y_all))
            X_cv = X_all.iloc[cv_idx].copy()
            X_test_holdout = X_all.iloc[test_idx].copy()
            y_cv = y_all[cv_idx]
            y_test_holdout = np.asarray(y_all[test_idx])
            groups_cv = groups_all[cv_idx]
        else:
            stratify_arr = y_all if cfg.task_type == "classification" else None
            try:
                X_cv_arr, X_test_holdout, y_cv_arr, y_test_holdout = _train_test_split(
                    X_all, y_all,
                    test_size=float(cfg.test_ratio),
                    random_state=cfg.random_state,
                    stratify=stratify_arr,
                )
            except Exception as split_exc:
                _log_event(
                    "training.cv.holdout_split_fallback",
                    model_type=model_type_norm,
                    reason=str(split_exc),
                )
                X_cv_arr, X_test_holdout, y_cv_arr, y_test_holdout = _train_test_split(
                    X_all, y_all,
                    test_size=float(cfg.test_ratio),
                    random_state=cfg.random_state,
                )
            X_cv = X_cv_arr
            y_cv = np.asarray(y_cv_arr)
            y_test_holdout = np.asarray(y_test_holdout)
        _log_event(
            "training.cv.holdout_test_split",
            model_type=model_type_norm,
            n_total=int(len(y_all)),
            n_cv=int(len(y_cv)),
            n_test=int(len(y_test_holdout)),
            test_ratio=float(cfg.test_ratio),
        )

    # Normalize hyperparams once — GridSearch disabled in fold loop (see docstring)
    requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})
    if not isinstance(requested_hyperparams_raw, dict):
        requested_hyperparams_raw = {}
    hp_normalized_no_gs = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=False,  # No inner GridSearch per fold (avoid double-CV)
        task_type=cfg.task_type,
    )
    hp_errors = [str(m) for m in (hp_normalized_no_gs.get("errors") or []) if str(m).strip()]
    if hp_errors:
        raise RuntimeError(f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}")
    hp_warnings: list[str] = [str(w) for w in (hp_normalized_no_gs.get("warnings") or []) if str(w).strip()]
    for hp_warn in hp_warnings:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=hp_warn)

    base_estimator_hp = dict(hp_normalized_no_gs.get("estimator_params") or {})
    capabilities = get_model_capabilities(model_type_norm)
    shuffle = bool(getattr(cfg, "shuffle", True))
    effective_preprocessing = _ensure_scaling_for_model(model_type_norm, cfg.preprocessing)

    # ── Nested CV: param_grid for inner GridSearch ────────────────────────────
    # When use_grid_search=True and the model has a param_grid, each outer fold
    # runs an inner GridSearchCV (nested CV). This prevents optimistic bias that
    # arises when the same data is used to both tune and evaluate the model.
    _hp_with_gs = normalize_model_hyperparams(
        model_type_norm, requested_hyperparams_raw,
        use_grid_search=True, task_type=cfg.task_type,
    )
    nested_param_grid = dict(_hp_with_gs.get("param_grid") or {})
    use_nested_cv = bool(cfg.use_grid_search) and bool(nested_param_grid)
    inner_k = int(getattr(cfg, "inner_cv_folds", 3))
    inner_scoring = _choose_refit_metric(cfg.task_type, y_cv, list(cfg.metrics))
    if use_nested_cv:
        _log_event(
            "training.cv.nested_cv_enabled",
            model_type=model_type_norm,
            inner_k=inner_k,
            inner_scoring=inner_scoring,
            n_param_combinations=len(nested_param_grid),
        )

    # ── Generate fold indices (on CV portion only) ────────────────────────────
    try:
        if cfg.split_method == "repeated_stratified_kfold":
            folds_list = list(
                iter_repeated_kfold_splits(
                    X_cv,
                    y_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    n_repeats=int(getattr(cfg, "n_repeats", 3)),
                    random_state=cfg.random_state,
                )
            )
        elif cfg.split_method in ("group_kfold", "stratified_group_kfold"):
            if groups_cv is None:
                raise RuntimeError(
                    f"splitMethod='{cfg.split_method}' requires groups — "
                    "groupColumn missing or not extracted."
                )
            folds_list = list(
                iter_group_kfold_splits(
                    X_cv,
                    y_cv,
                    groups_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    random_state=cfg.random_state,
                )
            )
        else:
            folds_list = list(
                iter_kfold_splits(
                    X_cv,
                    y_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    shuffle=shuffle,
                    random_state=cfg.random_state,
                )
            )
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    actual_k = len(folds_list)
    fold_results: List[Dict[str, Any]] = []
    # OOF predictions accumulated for leakage-free threshold calibration.
    oof_proba_parts: list = []
    oof_true_parts: list = []
    # OOF predicted class labels (classification) for pooled bootstrap CIs.
    oof_y_pred_class_parts: list = []
    # OOF predictions (regression only) for unbiased pooled RMSE/R² recomputation.
    oof_y_true_reg: list = []
    oof_y_pred_reg: list = []

    # ── Per-fold loop ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(folds_list):
        fold_num = fold_idx + 1
        _log_event("training.cv.fold_start", fold=fold_num, k=actual_k, model_type=model_type_norm)
        try:
            X_train_fold = X_cv.iloc[train_idx].copy()
            X_val_fold = X_cv.iloc[val_idx].copy()
            y_train_fold = y_cv[train_idx]
            y_val_fold = y_cv[val_idx]

            # ① Fit preprocessor on train_fold ONLY
            fold_schema = build_training_schema(
                X=X_train_fold,
                target=cfg.target_column,
                preprocessing_config=effective_preprocessing.as_dict(),
            )
            fold_aligner = ColumnAligner(
                feature_names=fold_schema["feature_names"],
                dtypes=fold_schema["dtypes"],
            )
            X_train_aligned = fold_aligner.fit_transform(X_train_fold)
            # Feature engineering: fit stats on train_fold only (anti-leakage)
            fold_fe = FeatureEngineeringTransformer(fe_defs)
            X_train_fe = fold_fe.fit_transform(X_train_aligned)
            fold_spec = build_preprocessor(X_train_fe, effective_preprocessing, kind_overrides=kind_overrides)
            X_train_prep = fold_spec.preprocessor.fit_transform(X_train_fe, y_train_fold)
            # ② Transform val_fold with the train-fitted preprocessor (no leakage)
            X_val_aligned = fold_aligner.transform(X_val_fold)
            X_val_fe = fold_fe.transform(X_val_aligned)
            X_val_prep = fold_spec.preprocessor.transform(X_val_fe)

            # ② b) Fit VarianceThreshold on train_fold ONLY, then apply to both (no leakage)
            # threshold=0.0: removes only perfectly constant features (see PreprocessingConfig).
            _vt_fold_w: list[str] = []
            if fold_spec.feature_selector is not None:
                _vt_n_before = X_train_prep.shape[1] if hasattr(X_train_prep, "shape") else 0
                _vt_fold_names: list = []
                try:
                    _vt_fold_names = list(fold_spec.preprocessor.get_feature_names_out())
                except Exception:
                    pass
                fold_spec.feature_selector.fit(X_train_prep)
                X_train_prep = fold_spec.feature_selector.transform(X_train_prep)
                X_val_prep = fold_spec.feature_selector.transform(X_val_prep)
                _vt_fold_w = _log_variance_threshold(
                    model_type_norm, _vt_n_before, X_train_prep.shape[1],
                    cfg.preprocessing.variance_threshold,
                    feature_selector=fold_spec.feature_selector,
                    feature_names_before=_vt_fold_names,
                )
                if X_train_prep.shape[1] == 0:
                    raise RuntimeError(
                        f"Preprocessing a supprimé toutes les features pour le modèle '{model_type_norm}'. "
                        "Vérifiez votre configuration de preprocessing (VarianceThreshold trop strict, "
                        "ou toutes les colonnes ont une variance nulle)."
                    )

            if model_type_norm in _DENSE_REQUIRED_MODELS:
                X_train_prep = _ensure_dense_matrix(X_train_prep)
                X_val_prep = _ensure_dense_matrix(X_val_prep)

            # ③ Balancing — train_fold ONLY (val never resampled)
            fold_decision: BalancingDecision = _default_decision_for_non_classification()
            fold_profile: Optional[DataProfile] = None
            if cfg.task_type == "classification":
                fold_profile = profile_binary_dataset(y_train_fold, X_train_fold.shape)
                fold_decision = resolve(
                    profile=fold_profile,
                    config=cfg.balancing,
                    model_supports_class_weight=capabilities["supports_class_weight"],
                    model_supports_predict_proba=capabilities["supports_predict_proba"],
                    model_supports_sample_weight=capabilities["supports_sample_weight"],
                    random_state=cfg.random_state,
                )

            fold_executor = BalancingExecutor()
            X_f = X_train_prep
            y_f = np.asarray(y_train_fold)
            fold_fit_params: Dict[str, Any] = {}
            fold_warnings: list[str] = list(_vt_fold_w)
            if cfg.task_type == "classification":
                # SMOTE guard: when the minority class on this fold is too
                # small for k_neighbors interpolation, fall back to "no
                # resampling" instead of letting imblearn raise. Mirrors the
                # LOO guard so behaviour is consistent across CV strategies.
                fold_decision, _smote_warn = _smote_minority_guard(fold_decision, y_train_fold)
                if _smote_warn is not None:
                    fold_warnings.append(_smote_warn)
                    _log_event(
                        "training.cv.fold_smote_skipped",
                        fold=fold_num,
                        model_type=model_type_norm,
                        reason=_smote_warn,
                    )
                # Convert sparse → dense ONLY when resampler requires it
                if fold_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
                    if hasattr(X_train_prep, "toarray"):
                        X_train_prep = X_train_prep.toarray()
                    if hasattr(X_val_prep, "toarray"):
                        X_val_prep = X_val_prep.toarray()
                X_f, y_f, fold_fit_params = fold_executor.apply_prefit(
                    X_train_prep, np.asarray(y_train_fold), fold_decision
                )

            fold_hp = dict(base_estimator_hp)
            if "class_weight" in fold_fit_params:
                fold_hp["class_weight"] = fold_fit_params["class_weight"]

            # ④ Train model — nested CV (inner GridSearch per fold) when enabled,
            #    otherwise plain fit.
            sw = fold_fit_params.get("sample_weight")
            fold_best_inner_params: Optional[Dict[str, Any]] = None
            if use_nested_cv:
                _inner_cv_splitter, _ = _build_cv_splitter(
                    cfg.task_type, y_f, inner_k, cfg.random_state
                )
                _inner_model = build_model(model_type_norm, cfg.task_type, fold_hp)
                _inner_pipe = Pipeline(steps=[("model", _inner_model)])
                _inner_gs = _GridSearchCV(
                    _inner_pipe, nested_param_grid,
                    cv=_inner_cv_splitter, scoring=inner_scoring,
                    refit=True, n_jobs=-1, error_score="raise",
                )
                if sw is not None:
                    _inner_gs.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
                else:
                    _inner_gs.fit(X_f, y_f)
                fold_pipeline = _inner_gs.best_estimator_
                fold_best_inner_params = {
                    k.replace("model__", "", 1): v
                    for k, v in (_inner_gs.best_params_ or {}).items()
                    if k.startswith("model__")
                }
            else:
                fold_model = build_model(model_type_norm, cfg.task_type, fold_hp)
                fold_pipeline = Pipeline(steps=[("model", fold_model)])
                if sw is not None:
                    fold_pipeline.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
                else:
                    fold_pipeline.fit(X_f, y_f)

            # ⑤ Evaluate on val_fold (untouched by resampling)
            evaluator = Evaluator(
                task_type=cfg.task_type,
                requested_metrics=cfg.metrics,
                positive_label=resolved_positive_label,
            )
            val_eval = evaluator.evaluate(fold_pipeline, X_val_prep, np.asarray(y_val_fold))

            # Collect OOF predictions for leakage-free threshold calibration.
            if cfg.task_type == "classification":
                _pp_fn = getattr(fold_pipeline, "predict_proba", None)
                if callable(_pp_fn):
                    try:
                        oof_proba_parts.append(np.asarray(_pp_fn(X_val_prep)))
                        oof_true_parts.append(np.asarray(y_val_fold))
                    except Exception:
                        pass
                # Pooled OOF y_pred (classes) feeds the bootstrap CI step below.
                try:
                    oof_y_pred_class_parts.append(np.asarray(val_eval.predictions))
                except Exception:
                    pass

            # Collect OOF predictions (regression) for unbiased pooled metrics.
            # Skipped for classification to avoid unnecessary memory use.
            if cfg.task_type == "regression":
                try:
                    oof_y_pred_reg.append(np.asarray(val_eval.predictions))
                    oof_y_true_reg.append(np.asarray(y_val_fold))
                except Exception:
                    pass

            smote_added: Optional[int] = None
            if cfg.task_type == "classification" and fold_decision.strategy in {"smote", "smote_tomek"}:
                smote_added = int(len(y_f) - len(y_train_fold))

            fold_class_dist: Optional[Dict[str, Any]] = None
            if cfg.task_type == "classification":
                fold_class_dist = {
                    "train": class_counts(y_train_fold),
                    "val": class_counts(y_val_fold),
                    "train_minority_ratio": minority_ratio(y_train_fold),
                    "val_minority_ratio": minority_ratio(y_val_fold),
                }

            fold_results.append({
                "fold": fold_num,
                "status": "ok",
                "train_size": int(len(y_train_fold)),
                "val_size": int(len(y_val_fold)),
                "metrics": val_eval.metrics,
                "balancing_strategy": str(fold_decision.strategy),
                "balancing_rationale": str(getattr(fold_decision, "rationale", "")),
                "smote_samples_added": smote_added,
                "class_distribution": fold_class_dist,
                "best_inner_params": fold_best_inner_params,
                "warnings": list(fold_warnings),
            })
            _log_event("training.cv.fold_ok", fold=fold_num, model_type=model_type_norm)

        except Exception as fold_exc:
            _log_event(
                "training.cv.fold_error",
                fold=fold_num,
                model_type=model_type_norm,
                reason=str(fold_exc),
            )
            fold_results.append({
                "fold": fold_num,
                "status": "failed",
                "error": str(fold_exc),
            })

    n_ok = sum(1 for fr in fold_results if fr.get("status") == "ok")
    if n_ok == 0:
        errors_summary = "; ".join(
            fr.get("error", "?") for fr in fold_results if fr.get("status") == "failed"
        )
        raise RuntimeError(
            f"All {actual_k} folds failed for model '{model_type_norm}'. "
            f"Errors: {errors_summary}"
        )

    cv_summary = _aggregate_cv_metrics(fold_results)

    # ── Pooled OOF recomputation for regression (unbiased RMSE/MSE/R²) ────────
    # _aggregate_cv_metrics averages each metric across folds — correct for MAE
    # but biased for RMSE (Jensen) and divergent for R².  We recompute these
    # three on the concatenated OOF predictions and override them in
    # cv_summary["mean"]; MAE and all other entries stay untouched.
    if cfg.task_type == "regression" and oof_y_true_reg and oof_y_pred_reg:
        try:
            _oof_y_true = np.concatenate(oof_y_true_reg)
            _oof_y_pred = np.concatenate(oof_y_pred_reg)
            _pooled = _compute_oof_regression_metrics(_oof_y_true, _oof_y_pred)
            _mean_block = cv_summary.get("mean") if isinstance(cv_summary.get("mean"), dict) else {}
            _mean_block["rmse"] = _pooled["rmse"]
            _mean_block["mse"]  = _pooled["mse"]
            _mean_block["r2"]   = _pooled["r2"]
            _mean_block["oof_sample_count"] = int(len(_oof_y_true))
            cv_summary["mean"] = _mean_block
            logger.info(
                "CV regression metrics recomputed on %d OOF predictions (pooled): "
                "RMSE=%.4f, R²=%.4f",
                len(_oof_y_true), _pooled["rmse"], _pooled["r2"],
            )
        except Exception as exc:
            _log_event(
                "training.cv.oof_regression_recompute_failed",
                model_type=model_type_norm,
                reason=str(exc),
            )

    # ── Bootstrap confidence intervals on pooled OOF predictions ──────────────
    # Mirrors the holdout path so the frontend reads the same shape under
    # metrics_json["confidence_intervals"] regardless of split method.
    bootstrap_cis_cv: Optional[Dict[str, Any]] = None
    _ci_y_true: Optional[np.ndarray] = None
    _ci_y_pred: Optional[np.ndarray] = None
    _ci_y_score: Optional[np.ndarray] = None
    if cfg.task_type == "regression" and oof_y_true_reg and oof_y_pred_reg:
        try:
            _ci_y_true = np.concatenate(oof_y_true_reg)
            _ci_y_pred = np.concatenate(oof_y_pred_reg)
        except Exception:
            _ci_y_true = None
            _ci_y_pred = None
    elif cfg.task_type == "classification" and oof_true_parts and oof_y_pred_class_parts:
        try:
            _ci_y_true = np.concatenate([np.asarray(a) for a in oof_true_parts])
            _ci_y_pred = np.concatenate([np.asarray(a) for a in oof_y_pred_class_parts])
        except Exception:
            _ci_y_true = None
            _ci_y_pred = None
        if oof_proba_parts:
            try:
                _proba_pooled = np.vstack(oof_proba_parts)
                if _proba_pooled.ndim == 2 and _proba_pooled.shape[1] == 2:
                    _ci_y_score = _proba_pooled[:, 1]
                elif _proba_pooled.ndim == 1:
                    _ci_y_score = _proba_pooled
            except Exception:
                _ci_y_score = None

    if _ci_y_true is not None and _ci_y_pred is not None:
        _n_ci = int(len(_ci_y_true))
        if _n_ci < 30:
            logger.warning(
                "Skipping bootstrap CIs in CV mode — fewer than 30 OOF samples (%d)",
                _n_ci,
            )
        else:
            try:
                bootstrap_cis_cv = compute_bootstrap_cis(
                    _ci_y_true, _ci_y_pred,
                    y_score=_ci_y_score,
                    task_type=cfg.task_type,
                )
            except Exception as exc:
                _log_event(
                    "training.cv.bootstrap_ci_failed",
                    model_type=model_type_norm,
                    reason=str(exc),
                )

    _log_event(
        "training.cv.end",
        model_type=model_type_norm,
        k=actual_k,
        n_ok=n_ok,
        cv_mean_metrics=cv_summary.get("mean", {}),
    )

    # ── Final refit ───────────────────────────────────────────────────────────
    # • test_ratio == 0 → refit on X_all (all data, same as before).
    # • test_ratio >  0 → refit on X_cv (non-test data only) to keep the
    #   holdout test set genuinely independent for the final evaluation.
    # GridSearch is applied here if cfg.use_grid_search=True (not per-fold).
    X_refit = X_cv
    y_refit = y_cv
    _log_event(
        "training.cv.refit_start",
        model_type=model_type_norm,
        n_samples=int(len(y_refit)),
        has_holdout_test=has_holdout_test,
    )

    final_training_schema = build_training_schema(
        X=X_refit,
        target=cfg.target_column,
        preprocessing_config=effective_preprocessing.as_dict(),
    )
    final_aligner = ColumnAligner(
        feature_names=final_training_schema["feature_names"],
        dtypes=final_training_schema["dtypes"],
    )
    X_refit_aligned = final_aligner.fit_transform(X_refit)
    # Feature engineering: fit on full refit data (anti-leakage: test set transforms via pipeline)
    final_fe = FeatureEngineeringTransformer(fe_defs)
    X_refit_fe = final_fe.fit_transform(X_refit_aligned)
    final_spec = build_preprocessor(X_refit_fe, effective_preprocessing, kind_overrides=kind_overrides)
    clear_clip_warnings()
    X_refit_prep = final_spec.preprocessor.fit_transform(X_refit_fe, y_refit)
    _prep_clip_warnings = get_clip_warnings()
    clear_clip_warnings()

    # VarianceThreshold after StandardScaler: threshold=0.0 removes only
    # perfectly constant features. Higher values (e.g. 0.01) would
    # incorrectly drop rare binary features (prevalence ~1%) whose
    # clinical relevance may be high.
    _vt_refit_w: list[str] = []
    if final_spec.feature_selector is not None:
        _vt_n_before = X_refit_prep.shape[1] if hasattr(X_refit_prep, "shape") else 0
        _vt_refit_names: list = []
        try:
            _vt_refit_names = list(final_spec.preprocessor.get_feature_names_out())
        except Exception:
            pass
        final_spec.feature_selector.fit(X_refit_prep)
        X_refit_prep = final_spec.feature_selector.transform(X_refit_prep)
        _vt_refit_w = _log_variance_threshold(
            model_type_norm, _vt_n_before, X_refit_prep.shape[1],
            cfg.preprocessing.variance_threshold,
            feature_selector=final_spec.feature_selector,
            feature_names_before=_vt_refit_names,
        )

    if model_type_norm in _DENSE_REQUIRED_MODELS:
        X_refit_prep = _ensure_dense_matrix(X_refit_prep)

    # Balancing for final refit
    final_decision: BalancingDecision = _default_decision_for_non_classification()
    final_profile: Optional[DataProfile] = None
    balancing_audit: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        final_profile = profile_binary_dataset(y_refit, X_refit.shape)
        final_decision = resolve(
            profile=final_profile,
            config=cfg.balancing,
            model_supports_class_weight=capabilities["supports_class_weight"],
            model_supports_predict_proba=capabilities["supports_predict_proba"],
            model_supports_sample_weight=capabilities["supports_sample_weight"],
            random_state=cfg.random_state,
        )

    final_executor = BalancingExecutor()
    X_final_f = X_refit_prep
    y_final_f = np.asarray(y_refit)
    final_fit_params: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        if final_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
            if hasattr(X_refit_prep, "toarray"):
                X_refit_prep = X_refit_prep.toarray()
                X_final_f = X_refit_prep
        X_final_f, y_final_f, final_fit_params = final_executor.apply_prefit(
            X_refit_prep, y_refit, final_decision
        )

    final_hp = dict(base_estimator_hp)
    if "class_weight" in final_fit_params:
        final_hp["class_weight"] = final_fit_params["class_weight"]

    # Hyperparams for final refit: re-normalize WITH GridSearch flag to get param_grid
    hp_normalized_final = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=bool(cfg.use_grid_search),
        task_type=cfg.task_type,
    )
    final_param_grid = dict(hp_normalized_final.get("param_grid") or {})
    final_estimator_hp = dict(hp_normalized_final.get("estimator_params") or {})
    if "class_weight" in final_fit_params:
        final_estimator_hp["class_weight"] = final_fit_params["class_weight"]

    _final_active_balancing = final_decision.strategy not in {"none", "threshold_optimization"}
    _final_is_imbalanced = (
        cfg.task_type == "classification"
        and final_profile is not None
        and final_profile.imbalance_ratio is not None
        and final_profile.imbalance_ratio > 3.0
        and not _final_active_balancing
    )
    final_model_raw = build_model(model_type_norm, cfg.task_type, final_estimator_hp)
    final_trainer = Trainer()
    sw_final = final_fit_params.get("sample_weight")
    final_fit_result = final_trainer.fit(
        pipeline=Pipeline(steps=[("model", final_model_raw)]),
        X_train=X_final_f,
        y_train=y_final_f,
        fit_sample_weight=np.asarray(sw_final) if sw_final is not None else None,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=final_param_grid if final_param_grid else None,
        refit_metric_override=(final_decision.refit_metric if cfg.task_type == "classification" else None),
        n_samples=int(len(y_final_f)),
        imbalanced=_final_is_imbalanced,
    )

    final_fitted_model = final_fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        final_aligner, final_spec.preprocessor, final_fitted_model, model_type_norm,
        feature_selector=final_spec.feature_selector,
        fe_transformer=final_fe,
    )

    # Threshold + audit — prefer OOF predictions for leakage-free calibration.
    optimal_threshold = 0.5
    threshold_f1_gain: Optional[float] = None
    threshold_calibration_source = "not_applicable"
    if cfg.task_type == "classification":
        # Threshold optimization is binary-only: disable it for multiclass with explicit warning.
        _cv_multiclass_threshold_disabled = False
        if final_decision.apply_threshold and len(np.unique(np.asarray(y_refit))) > 2:
            final_decision.apply_threshold = False
            _cv_multiclass_threshold_disabled = True

        # Build OOF arrays from per-fold predictions accumulated above.
        oof_proba_arr: Optional[np.ndarray] = None
        oof_true_arr: Optional[np.ndarray] = None
        if oof_proba_parts and oof_true_parts:
            try:
                oof_proba_arr = np.vstack(oof_proba_parts)
                oof_true_arr = np.concatenate(oof_true_parts)
            except Exception:
                oof_proba_arr = None
                oof_true_arr = None

        oof_available = (
            oof_proba_arr is not None
            and oof_true_arr is not None
            and oof_proba_arr.shape[0] > 0
        )
        if oof_available:
            optimal_threshold = final_executor.apply_postfit_from_oof(
                y_true=oof_true_arr,
                y_proba=oof_proba_arr,
                decision=final_decision,
                model_classes=getattr(final_fitted_model, "classes_", None),
            )
            threshold_calibration_source = "oof"
        else:
            optimal_threshold = final_executor.apply_postfit(
                fitted_pipe, X_refit, y_refit, final_decision
            )
            threshold_calibration_source = "train_refit"
            final_executor.postfit_warnings.append(
                "threshold_calibrated_on_train_data_may_be_optimistic"
            )
        if final_executor.last_threshold_result is not None:
            threshold_f1_gain = float(final_executor.last_threshold_result.improvement_delta)
        if _cv_multiclass_threshold_disabled:
            final_executor.postfit_warnings.append("threshold_optimization_disabled_multiclass")
        if final_profile is not None:
            balancing_audit = build_and_persist_audit(
                profile=final_profile,
                decision=final_decision,
                session_id=session_id,
                db=db,
                smote_samples_added=(
                    int(len(y_final_f) - len(y_refit))
                    if final_decision.strategy in {"smote", "smote_tomek"}
                    else None
                ),
                optimal_threshold=optimal_threshold,
                threshold_f1_gain=threshold_f1_gain,
                postfit_warnings=list(final_executor.postfit_warnings or []),
            )

    _log_event("training.cv.refit_end", model_type=model_type_norm)

    # ── Holdout test evaluation ───────────────────────────────────────────────
    # Transform test set with preprocessor fitted on X_refit (no leakage).
    # Never resample the test set.
    holdout_test_metrics: Optional[Dict[str, Any]] = None
    holdout_test_eval = None
    if has_holdout_test and X_test_holdout is not None and y_test_holdout is not None:
        holdout_evaluator = Evaluator(
            task_type=cfg.task_type,
            requested_metrics=cfg.metrics,
            positive_label=resolved_positive_label,
        )
        holdout_test_eval = holdout_evaluator.evaluate(
            fitted_pipe,
            X_test_holdout,
            np.asarray(y_test_holdout),
            threshold=optimal_threshold,
        )
        holdout_test_metrics = holdout_test_eval.metrics
        _log_event(
            "training.cv.holdout_test_eval",
            model_type=model_type_norm,
            n_test=int(len(y_test_holdout)),
        )

    # ── Build CV-specific artifacts ───────────────────────────────────────────
    _cv_curves = (
        holdout_test_eval.metrics.get("curves")
        if holdout_test_eval is not None and isinstance(holdout_test_eval.metrics, dict)
        else None
    )

    reporter = Reporter()
    artifacts = reporter.build_cv_artifacts(
        cfg=cfg,
        fold_results=fold_results,
        cv_summary=cv_summary,
        actual_k=actual_k,
        n_folds_ok=n_ok,
        X_all=X_all,
        y_all=y_all,
        X_cv=X_cv,
        y_cv=y_cv,
        X_test_holdout=X_test_holdout,
        y_test_holdout=y_test_holdout,
        final_spec=final_spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        final_training_schema=final_training_schema,
        tuning_artifacts=final_fit_result.tuning_artifacts,
        resolved_positive_label=resolved_positive_label,
        curves=_cv_curves,
    )

    hyperparams_artifacts: Dict[str, Any] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized_final.get("effective") or {}),
        "note": (
            f"Nested CV active: inner GridSearch ({inner_k} folds, scoring={inner_scoring}) "
            "run per outer fold; GridSearch also applied on final refit."
            if use_nested_cv
            else "GridSearch (if any) applied only on final refit, not per-fold."
        ),
    }
    if bool(cfg.use_grid_search):
        hyperparams_artifacts["param_grid"] = dict(final_param_grid)
        best_params = final_fit_result.tuning_artifacts.get("best_params")
        if isinstance(best_params, dict):
            hyperparams_artifacts["best"] = dict(best_params)
    artifacts["hyperparams"] = hyperparams_artifacts

    # ── Build metrics_json ────────────────────────────────────────────────────
    # "test" key: holdout test metrics when available (honest external score),
    # otherwise CV mean metrics (backward compat for route/frontend).
    # "cv_mean" always holds the aggregated CV validation metrics.
    cv_mean_metrics = cv_summary.get("mean", {})
    metrics_json: Dict[str, Any] = {
        "split_method": cfg.split_method,
        "cv": True,
        "nested_cv": use_nested_cv,
        "k_folds": actual_k,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "has_holdout_test": has_holdout_test,
        "cv_mean": cv_mean_metrics,
        "test": holdout_test_metrics if (has_holdout_test and holdout_test_metrics is not None) else cv_mean_metrics,
        "test_is_cv_mean": not has_holdout_test,
        "test_label": "CV validation (moyenne des folds)" if not has_holdout_test else "Holdout test set",
        "training_time_sec": float(time.perf_counter() - t0),
    }
    if has_holdout_test and holdout_test_metrics is not None:
        metrics_json["holdout_test_metrics"] = holdout_test_metrics
    metrics_json["threshold_used"] = optimal_threshold
    metrics_json["threshold_source"] = threshold_calibration_source
    metrics_json["confidence_intervals"] = bootstrap_cis_cv
    cv_instability = cv_summary.get("instability_warnings", [])
    postfit_w = list(getattr(final_executor, "postfit_warnings", None) or [])
    _clip_strs = [
        f"CLIP_NEGATIVE in '{w['column']}': {w['n_clipped']} value(s) clipped before "
        f"{w['transform']} transform (min={w['min_observed']:.6g})"
        for w in _prep_clip_warnings
    ]
    _fold_structured, _fold_strs = _collect_fold_warnings(fold_results)
    all_warnings: list[str] = (
        list(hp_warnings) + list(cv_instability) + postfit_w + _clip_strs + _vt_refit_w + _fold_strs
    )
    if all_warnings:
        metrics_json["warnings"] = all_warnings
    if _prep_clip_warnings:
        metrics_json["clip_warnings"] = [
            {"severity": "warning", "code": "CLIP_NEGATIVE", **w} for w in _prep_clip_warnings
        ]
    if _fold_structured:
        metrics_json["fold_warnings"] = _fold_structured

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Leave-One-Out (LOO) pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _run_loo(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    """
    Leave-One-Out evaluation pipeline.

    Anti-leakage guarantees
    -----------------------
    • Preprocessor is fit on n-1 samples (train_fold) ONLY, transformed on 1 test sample.
    • Resampling applied to train_fold ONLY; SMOTE guard disables resampling when
      the minority class is too small for k_neighbors.
    • GridSearch is DISABLED per-fold (would create n × k_gs nested fits).

    Aggregation — prediction collection
    ------------------------------------
    Because each fold has 1 test sample, per-fold metrics (AUC, F1) are meaningless.
    Instead: collect y_true and y_pred across ALL n folds, then compute global
    metrics once. This is the statistically correct LOO estimate.

    Persistence (Option A — refit-final)
    -------------------------------------
    After n folds, refit on ALL data (same as kfold). This is the deployed model.
    Guard: refuses if n_samples > 500.
    """
    from app.services.training.pipeline.metrics import classification_metrics, regression_metrics

    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    _log_event("training.loo.start", model_type=model_type_norm, task_type=cfg.task_type)

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    n_samples = len(df2)
    if n_samples < 3:
        raise RuntimeError("LOO nécessite au moins 3 échantillons après suppression des NaN cibles.")
    if n_samples > 500:
        raise RuntimeError(
            f"LOO est limité à 500 échantillons (n={n_samples}). "
            "Utilisez kfold ou stratified_kfold à la place."
        )

    X_all = df2.drop(columns=[cfg.target_column])
    y_all = np.asarray(df2[cfg.target_column].values)

    fe_defs = [f.as_dict() for f in cfg.feature_engineering.features]
    fe_validation_errors = validate_feature_defs(fe_defs, list(X_all.columns))
    if fe_validation_errors:
        raise RuntimeError("Feature engineering config errors:\n" + "\n".join(fe_validation_errors))

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        y_all, cfg.positive_label
    )
    if positive_label_warning:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    # Normalize hyperparams — GridSearch disabled per fold
    requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})
    if not isinstance(requested_hyperparams_raw, dict):
        requested_hyperparams_raw = {}
    hp_normalized_no_gs = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=False,
        task_type=cfg.task_type,
    )
    hp_errors = [str(m) for m in (hp_normalized_no_gs.get("errors") or []) if str(m).strip()]
    if hp_errors:
        raise RuntimeError(f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}")
    hp_warnings: list[str] = [str(w) for w in (hp_normalized_no_gs.get("warnings") or []) if str(w).strip()]
    for hp_warn in hp_warnings:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=hp_warn)

    base_estimator_hp = dict(hp_normalized_no_gs.get("estimator_params") or {})
    capabilities = get_model_capabilities(model_type_norm)
    effective_preprocessing = _ensure_scaling_for_model(model_type_norm, cfg.preprocessing)

    # ── LOO prediction collection ─────────────────────────────────────────────
    y_true_loo: list[Any] = []
    y_pred_loo: list[Any] = []
    y_proba_loo: list[Any] = []
    fold_results: List[Dict[str, Any]] = []
    n_ok = 0

    try:
        loo_splits = list(iter_loo_splits(X_all, y_all))
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    actual_n = len(loo_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(loo_splits):
        fold_num = fold_idx + 1
        try:
            X_train_fold = X_all.iloc[train_idx].copy()
            X_val_fold = X_all.iloc[val_idx].copy()
            y_train_fold = y_all[train_idx]
            y_val_sample = y_all[val_idx]

            # ① Fit preprocessor on train_fold ONLY (n-1 samples)
            fold_schema = build_training_schema(
                X=X_train_fold,
                target=cfg.target_column,
                preprocessing_config=effective_preprocessing.as_dict(),
            )
            fold_aligner = ColumnAligner(
                feature_names=fold_schema["feature_names"],
                dtypes=fold_schema["dtypes"],
            )
            X_train_aligned = fold_aligner.fit_transform(X_train_fold)
            # Feature engineering: fit stats on train_fold only (anti-leakage)
            fold_fe = FeatureEngineeringTransformer(fe_defs)
            X_train_fe = fold_fe.fit_transform(X_train_aligned)
            fold_spec = build_preprocessor(X_train_fe, effective_preprocessing, kind_overrides=kind_overrides)
            X_train_prep = fold_spec.preprocessor.fit_transform(X_train_fe, y_train_fold)
            X_val_aligned = fold_aligner.transform(X_val_fold)
            X_val_fe = fold_fe.transform(X_val_aligned)
            X_val_prep = fold_spec.preprocessor.transform(X_val_fe)

            # VarianceThreshold after StandardScaler: threshold=0.0 removes only
            # perfectly constant features. Higher values (e.g. 0.01) would
            # incorrectly drop rare binary features (prevalence ~1%) whose
            # clinical relevance may be high.
            _vt_loo_fold_w: list[str] = []
            if fold_spec.feature_selector is not None:
                _vt_n_before = X_train_prep.shape[1] if hasattr(X_train_prep, "shape") else 0
                _vt_loo_fold_names: list = []
                try:
                    _vt_loo_fold_names = list(fold_spec.preprocessor.get_feature_names_out())
                except Exception:
                    pass
                fold_spec.feature_selector.fit(X_train_prep)
                X_train_prep = fold_spec.feature_selector.transform(X_train_prep)
                X_val_prep = fold_spec.feature_selector.transform(X_val_prep)
                _vt_loo_fold_w = _log_variance_threshold(
                    model_type_norm, _vt_n_before, X_train_prep.shape[1],
                    cfg.preprocessing.variance_threshold,
                    feature_selector=fold_spec.feature_selector,
                    feature_names_before=_vt_loo_fold_names,
                )

            if model_type_norm in _DENSE_REQUIRED_MODELS:
                X_train_prep = _ensure_dense_matrix(X_train_prep)
                X_val_prep = _ensure_dense_matrix(X_val_prep)

            # ② Balancing — train_fold ONLY; SMOTE guard for small folds
            fold_decision: BalancingDecision = _default_decision_for_non_classification()
            fold_executor = BalancingExecutor()
            X_f = X_train_prep
            y_f = np.asarray(y_train_fold)
            fold_fit_params: Dict[str, Any] = {}

            fold_warnings: list[str] = list(_vt_loo_fold_w)
            if cfg.task_type == "classification":
                fold_profile = profile_binary_dataset(y_train_fold, X_train_fold.shape)
                fold_decision = resolve(
                    profile=fold_profile,
                    config=cfg.balancing,
                    model_supports_class_weight=capabilities["supports_class_weight"],
                    model_supports_predict_proba=capabilities["supports_predict_proba"],
                    model_supports_sample_weight=capabilities["supports_sample_weight"],
                    random_state=cfg.random_state,
                )
                # SMOTE guard: shared with the K-Fold path so behaviour is
                # identical when the minority class is too small for SMOTE.
                fold_decision, _smote_warn = _smote_minority_guard(fold_decision, y_train_fold)
                if _smote_warn is not None:
                    fold_warnings.append(_smote_warn)
                    _log_event(
                        "training.loo.fold_smote_skipped",
                        fold=fold_num,
                        model_type=model_type_norm,
                        reason=_smote_warn,
                    )
                if fold_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
                    if hasattr(X_train_prep, "toarray"):
                        X_train_prep = X_train_prep.toarray()
                    if hasattr(X_val_prep, "toarray"):
                        X_val_prep = X_val_prep.toarray()
                X_f, y_f, fold_fit_params = fold_executor.apply_prefit(
                    X_train_prep, np.asarray(y_train_fold), fold_decision
                )

            fold_hp = dict(base_estimator_hp)
            if "class_weight" in fold_fit_params:
                fold_hp["class_weight"] = fold_fit_params["class_weight"]

            # ③ Train model — no GridSearch per fold
            fold_model = build_model(model_type_norm, cfg.task_type, fold_hp)
            fold_pipeline = Pipeline(steps=[("model", fold_model)])
            sw = fold_fit_params.get("sample_weight")
            if sw is not None:
                fold_pipeline.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
            else:
                fold_pipeline.fit(X_f, y_f)

            # ④ Predict on the single val sample — collect predictions
            y_pred_sample = fold_pipeline.predict(X_val_prep)
            y_true_loo.append(y_val_sample[0])
            y_pred_loo.append(y_pred_sample[0])
            if hasattr(fold_pipeline, "predict_proba"):
                proba = fold_pipeline.predict_proba(X_val_prep)
                y_proba_loo.append(proba[0])

            fold_results.append({
                "fold": fold_num,
                "status": "ok",
                "train_size": int(len(y_train_fold)),
                "val_size": 1,
                "warnings": list(fold_warnings),
            })
            n_ok += 1

        except Exception as fold_exc:
            _log_event("training.loo.fold_error", fold=fold_num, model_type=model_type_norm, reason=str(fold_exc))
            fold_results.append({"fold": fold_num, "status": "failed", "error": str(fold_exc)})

    if n_ok == 0:
        errors_summary = "; ".join(
            fr.get("error", "?") for fr in fold_results if fr.get("status") == "failed"
        )
        raise RuntimeError(f"All {actual_n} LOO folds failed for '{model_type_norm}'. Errors: {errors_summary}")

    # ── Global LOO metrics from collected predictions ─────────────────────────
    y_true_arr = np.asarray(y_true_loo)
    y_pred_arr = np.asarray(y_pred_loo)
    y_proba_arr = np.asarray(y_proba_loo) if y_proba_loo else None

    # Initialise outside the if/else so it remains in scope for the
    # bootstrap CI block below regardless of task type.
    y_score_loo: Optional[np.ndarray] = None
    if cfg.task_type == "classification":
        _labels = np.unique(y_all)

        if y_proba_arr is not None:
            if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == 2:
                y_score_loo = y_proba_arr[:, 1]
            else:
                y_score_loo = y_proba_arr

        loo_metrics = classification_metrics(
            y_true_arr, y_pred_arr,
            y_proba=y_proba_arr,
            y_score=y_score_loo,
            labels=_labels,
            estimator=None,
            positive_label=resolved_positive_label,
            requested_metrics=list(cfg.metrics),
            task_type=cfg.task_type,
        )
    else:
        loo_metrics = regression_metrics(y_true_arr, y_pred_arr)

    # ── Bootstrap confidence intervals on pooled LOO predictions ──────────────
    bootstrap_cis_loo: Optional[Dict[str, Any]] = None
    _n_loo_ci = int(len(y_true_arr))
    if _n_loo_ci < 30:
        logger.warning(
            "Skipping bootstrap CIs in CV mode — fewer than 30 OOF samples (%d)",
            _n_loo_ci,
        )
    else:
        try:
            bootstrap_cis_loo = compute_bootstrap_cis(
                y_true_arr, y_pred_arr,
                y_score=(y_score_loo if cfg.task_type == "classification" else None),
                task_type=cfg.task_type,
            )
        except Exception as exc:
            _log_event(
                "training.loo.bootstrap_ci_failed",
                model_type=model_type_norm,
                reason=str(exc),
            )

    # cv_summary-compatible structure for reporter reuse
    loo_flat = _extract_scalar_metrics(loo_metrics)
    cv_summary = {
        "mean": loo_flat,
        "std": {},
        "min": loo_flat,
        "max": loo_flat,
        "n_folds_ok": n_ok,
        "n_folds_per_metric": {k: n_ok for k in loo_flat},
    }

    _log_event(
        "training.loo.end",
        model_type=model_type_norm,
        n_iterations=actual_n,
        n_ok=n_ok,
        loo_metrics=loo_flat,
    )

    # ── Final refit on ALL data (Option A — same as kfold) ────────────────────
    _log_event("training.loo.refit_start", model_type=model_type_norm, n_samples=n_samples)

    final_training_schema = build_training_schema(
        X=X_all, target=cfg.target_column,
        preprocessing_config=effective_preprocessing.as_dict(),
    )
    final_aligner = ColumnAligner(
        feature_names=final_training_schema["feature_names"],
        dtypes=final_training_schema["dtypes"],
    )
    X_refit_aligned = final_aligner.fit_transform(X_all)
    final_fe = FeatureEngineeringTransformer(fe_defs)
    X_refit_fe = final_fe.fit_transform(X_refit_aligned)
    final_spec = build_preprocessor(X_refit_fe, effective_preprocessing, kind_overrides=kind_overrides)
    clear_clip_warnings()
    X_refit_prep = final_spec.preprocessor.fit_transform(X_refit_fe, y_all)
    _prep_clip_warnings = get_clip_warnings()
    clear_clip_warnings()
    # VarianceThreshold after StandardScaler: threshold=0.0 removes only
    # perfectly constant features. Higher values (e.g. 0.01) would
    # incorrectly drop rare binary features (prevalence ~1%) whose
    # clinical relevance may be high.
    _vt_loo_refit_w: list[str] = []
    if final_spec.feature_selector is not None:
        _vt_n_before = X_refit_prep.shape[1] if hasattr(X_refit_prep, "shape") else 0
        _vt_loo_refit_names: list = []
        try:
            _vt_loo_refit_names = list(final_spec.preprocessor.get_feature_names_out())
        except Exception:
            pass
        final_spec.feature_selector.fit(X_refit_prep)
        X_refit_prep = final_spec.feature_selector.transform(X_refit_prep)
        _vt_loo_refit_w = _log_variance_threshold(
            model_type_norm, _vt_n_before, X_refit_prep.shape[1],
            cfg.preprocessing.variance_threshold,
            feature_selector=final_spec.feature_selector,
            feature_names_before=_vt_loo_refit_names,
        )
    if model_type_norm in _DENSE_REQUIRED_MODELS:
        X_refit_prep = _ensure_dense_matrix(X_refit_prep)

    final_decision: BalancingDecision = _default_decision_for_non_classification()
    final_profile = None
    balancing_audit: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        final_profile = profile_binary_dataset(y_all, X_all.shape)
        final_decision = resolve(
            profile=final_profile,
            config=cfg.balancing,
            model_supports_class_weight=capabilities["supports_class_weight"],
            model_supports_predict_proba=capabilities["supports_predict_proba"],
            model_supports_sample_weight=capabilities["supports_sample_weight"],
            random_state=cfg.random_state,
        )

    final_executor = BalancingExecutor()
    X_final_f = X_refit_prep
    y_final_f = np.asarray(y_all)
    final_fit_params: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        if final_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
            if hasattr(X_refit_prep, "toarray"):
                X_refit_prep = X_refit_prep.toarray()
                X_final_f = X_refit_prep
        X_final_f, y_final_f, final_fit_params = final_executor.apply_prefit(
            X_refit_prep, y_all, final_decision
        )

    final_hp = dict(base_estimator_hp)
    if "class_weight" in final_fit_params:
        final_hp["class_weight"] = final_fit_params["class_weight"]

    hp_normalized_final = normalize_model_hyperparams(
        model_type_norm, requested_hyperparams_raw,
        use_grid_search=bool(cfg.use_grid_search), task_type=cfg.task_type,
    )
    final_param_grid = dict(hp_normalized_final.get("param_grid") or {})
    final_estimator_hp = dict(hp_normalized_final.get("estimator_params") or {})
    if "class_weight" in final_fit_params:
        final_estimator_hp["class_weight"] = final_fit_params["class_weight"]

    final_model_raw = build_model(model_type_norm, cfg.task_type, final_estimator_hp)
    final_trainer = Trainer()
    sw_final = final_fit_params.get("sample_weight")
    final_fit_result = final_trainer.fit(
        pipeline=Pipeline(steps=[("model", final_model_raw)]),
        X_train=X_final_f,
        y_train=y_final_f,
        fit_sample_weight=np.asarray(sw_final) if sw_final is not None else None,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=final_param_grid if final_param_grid else None,
        refit_metric_override=(final_decision.refit_metric if cfg.task_type == "classification" else None),
        n_samples=int(len(y_final_f)),
        imbalanced=False,
    )

    final_fitted_model = final_fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        final_aligner, final_spec.preprocessor, final_fitted_model, model_type_norm,
        feature_selector=final_spec.feature_selector,
        fe_transformer=final_fe,
    )

    optimal_threshold = 0.5
    threshold_f1_gain: Optional[float] = None
    threshold_calibration_source = "disabled"
    if cfg.task_type == "classification":
        if final_decision.apply_threshold and len(np.unique(y_all)) > 2:
            final_decision.apply_threshold = False

        # Prefer leakage-free OOF threshold calibration: y_true_loo and
        # y_proba_loo were collected per fold and contain only out-of-sample
        # predictions — the closest analogue of a validation set available in
        # the LOO regime. Fall back to (X_all, y_all) only when those arrays
        # are unavailable, and surface a warning when the fallback fires
        # (mirrors the holdout pattern at orchestrator.py:673).
        oof_proba_arr = np.asarray(y_proba_loo) if y_proba_loo else None
        oof_true_arr = np.asarray(y_true_loo) if y_true_loo else None
        oof_available = (
            final_decision.apply_threshold
            and oof_proba_arr is not None
            and oof_true_arr is not None
            and oof_true_arr.size > 0
            and oof_proba_arr.size > 0
        )

        if oof_available:
            optimal_threshold = final_executor.apply_postfit_from_oof(
                y_true=oof_true_arr,
                y_proba=oof_proba_arr,
                decision=final_decision,
                model_classes=getattr(final_fitted_model, "classes_", None),
            )
            threshold_calibration_source = "loo_oof"
        else:
            optimal_threshold = final_executor.apply_postfit(
                fitted_pipe, X_all, y_all, final_decision
            )
            if final_decision.apply_threshold:
                threshold_calibration_source = "train_fallback"
                if optimal_threshold != 0.5:
                    final_executor.postfit_warnings.append(
                        "threshold_calibrated_on_train_data_may_be_optimistic"
                    )
        if final_executor.last_threshold_result is not None:
            threshold_f1_gain = float(final_executor.last_threshold_result.improvement_delta)
        if final_profile is not None:
            balancing_audit = build_and_persist_audit(
                profile=final_profile,
                decision=final_decision,
                session_id=session_id,
                db=db,
                smote_samples_added=(
                    int(len(y_final_f) - len(y_all))
                    if final_decision.strategy in {"smote", "smote_tomek"} else None
                ),
                optimal_threshold=optimal_threshold,
                threshold_f1_gain=threshold_f1_gain,
                postfit_warnings=list(final_executor.postfit_warnings or []),
            )

    _log_event(
        "training.loo.refit_end",
        model_type=model_type_norm,
        threshold_calibration_source=threshold_calibration_source,
    )

    # ── Artifacts and metrics_json ────────────────────────────────────────────
    reporter = Reporter()
    artifacts = reporter.build_cv_artifacts(
        cfg=cfg,
        fold_results=fold_results,
        cv_summary=cv_summary,
        actual_k=actual_n,
        n_folds_ok=n_ok,
        X_all=X_all,
        y_all=y_all,
        X_cv=X_all,
        y_cv=y_all,
        X_test_holdout=None,
        y_test_holdout=None,
        final_spec=final_spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        final_training_schema=final_training_schema,
        tuning_artifacts=final_fit_result.tuning_artifacts,
        resolved_positive_label=resolved_positive_label,
        curves=None,
    )
    artifacts["hyperparams"] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized_final.get("effective") or {}),
        "note": "GridSearch disabled per LOO fold; applied only on final refit.",
    }

    metrics_json: Dict[str, Any] = {
        "split_method": "loo",
        "cv": True,
        "loo": True,
        "n_loo_iterations": actual_n,
        "n_loo_ok": n_ok,
        "loo_metrics": loo_metrics,
        "cv_mean": loo_flat,
        "test": loo_metrics,
        "test_is_cv_mean": True,
        "test_label": "LOO validation",
        "evaluation_strategy": "loo",
        "has_holdout_test": False,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "training_time_sec": float(time.perf_counter() - t0),
        "threshold_used": optimal_threshold,
        "threshold_source": threshold_calibration_source,
        "confidence_intervals": bootstrap_cis_loo,
    }
    loo_instability = cv_summary.get("instability_warnings", [])
    # Include postfit_warnings (e.g. threshold_calibrated_on_train_data_may_be_optimistic)
    # so the frontend can surface threshold-related issues alongside the others.
    loo_postfit_warnings = list(final_executor.postfit_warnings or [])
    _loo_clip_strs = [
        f"CLIP_NEGATIVE in '{w['column']}': {w['n_clipped']} value(s) clipped before "
        f"{w['transform']} transform (min={w['min_observed']:.6g})"
        for w in _prep_clip_warnings
    ]
    _loo_fold_structured, _loo_fold_strs = _collect_fold_warnings(fold_results)
    loo_all_warnings: list[str] = (
        list(hp_warnings) + list(loo_instability) + loo_postfit_warnings
        + _loo_clip_strs + _vt_loo_refit_w + _loo_fold_strs
    )
    if loo_all_warnings:
        metrics_json["warnings"] = loo_all_warnings
    if _prep_clip_warnings:
        metrics_json["clip_warnings"] = [
            {"severity": "warning", "code": "CLIP_NEGATIVE", **w} for w in _prep_clip_warnings
        ]
    if _loo_fold_structured:
        metrics_json["fold_warnings"] = _loo_fold_structured

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )
