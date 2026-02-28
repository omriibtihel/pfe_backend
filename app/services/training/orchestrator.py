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

from .audit import build_and_persist_audit
from .balancing import (
    BalancingExecutor,
    BalancingDecision,
    DataProfile,
    class_counts,
    minority_ratio,
    profile_binary_dataset,
    resolve,
)
from .config import TrainingConfig, normalize_model_hyperparams
from .evaluator import Evaluator
from .metrics import get_class_labels
from .models import build_model, get_model_capabilities
from .preprocessing import build_preprocessor
from .reporter import Reporter, build_training_schema
from .splitters import make_holdout_split, iter_kfold_splits
from .trainer import Trainer
from .transformers import ColumnAligner

logger = logging.getLogger(__name__)
_DENSE_REQUIRED_MODELS = {"naivebayes"}


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


def _build_inference_pipeline(aligner: ColumnAligner, preprocessor: Any, model: Any, model_type: str) -> Pipeline:
    steps = [("align", aligner), ("prep", preprocessor)]
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


def _is_zero_one_label_set(labels: np.ndarray) -> bool:
    try:
        normalized = {float(v) for v in labels}
    except Exception:
        return False
    return normalized == {0.0, 1.0}


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

    return {"mean": mean, "std": std, "min": mn, "max": mx, "n_folds_ok": n_ok, "n_folds_per_metric": n_folds_per_metric}


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
) -> ModelRunResult:
    """
    Entry point for a single model run.
    Dispatches to holdout or k-fold CV based on cfg.split_method.
    """
    if cfg.split_method in ("kfold", "stratified_kfold"):
        return _run_kfold_cv(df, cfg, model_type, db=db, session_id=session_id)
    return _run_holdout(df, cfg, model_type, db=db, session_id=session_id)


def _run_holdout(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
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
            test=class_counts(np.asarray(split.y_test)),
        )

    if split.X_train is None or len(split.X_train) == 0:
        raise RuntimeError("Preprocessing requires a valid split with a non-empty train set.")

    spec = build_preprocessor(split.X_train, cfg.preprocessing)
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
    for hp_warning in hp_normalized.get("warnings", []) or []:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=str(hp_warning))

    estimator_hyperparams = dict(hp_normalized.get("estimator_params") or {})
    param_grid = dict(hp_normalized.get("param_grid") or {})
    training_schema = build_training_schema(
        X=split.X_train,
        target=cfg.target_column,
        preprocessing_config=cfg.preprocessing.as_dict(),
    )
    aligner = ColumnAligner(
        feature_names=training_schema["feature_names"],
        dtypes=training_schema["dtypes"],
    )

    # Fit preprocessing on train split once (train-only policy), then execute balancing on transformed train data.
    X_train_aligned = aligner.fit_transform(split.X_train)
    X_train_prepared = spec.preprocessor.fit_transform(X_train_aligned, np.asarray(split.y_train))
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
            random_state=cfg.random_state,
        )

    executor = BalancingExecutor()
    X_f = X_train_prepared
    y_f = np.asarray(split.y_train)
    fit_params: dict[str, Any] = {}
    if cfg.task_type == "classification":
        # SMOTE/SMOTETomek/RandomUnderSampler refuse sparse matrices — convert to dense before resampling.
        if decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
            if hasattr(X_train_prepared, "toarray"):
                X_train_prepared = X_train_prepared.toarray()
        X_f, y_f, fit_params = executor.apply_prefit(X_train_prepared, np.asarray(split.y_train), decision)

    smote_samples_added: int | None = None
    if cfg.task_type == "classification" and decision.strategy in {"smote", "smote_tomek"}:
        smote_samples_added = int(len(np.asarray(y_f)) - len(np.asarray(split.y_train)))

    if "class_weight" in fit_params:
        estimator_hyperparams["class_weight"] = fit_params["class_weight"]
    model = build_model(model_type_norm, cfg.task_type, estimator_hyperparams)

    fit_sample_weight = fit_params.get("sample_weight")
    if fit_sample_weight is not None:
        fit_sample_weight = np.asarray(fit_sample_weight)

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
    )

    fitted_model = fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(aligner, spec.preprocessor, fitted_model, model_type_norm)
    if debug_mode and cfg.task_type == "classification":
        _log_event(
            "training.debug.model_classes",
            model_type=model_type_norm,
            model_classes=get_class_labels(fitted_pipe),
        )

    # Threshold calibration must use the validation set only.
    # If no validation set exists (val_ratio=0), threshold optimization is skipped
    # to prevent test-set leakage: calibrating on X_test then evaluating on X_test
    # would produce overoptimistic metrics.
    threshold_input_X = split.X_val if split.X_val is not None and len(split.X_val) > 0 else None
    threshold_input_y = split.y_val if split.y_val is not None and len(split.y_val) > 0 else None
    optimal_threshold = 0.5
    threshold_f1_gain: float | None = None
    if cfg.task_type == "classification":
        optimal_threshold = executor.apply_postfit(
            fitted_pipe,
            threshold_input_X,
            np.asarray(threshold_input_y) if threshold_input_y is not None else None,
            decision,
        )
        if executor.last_threshold_result is not None:
            threshold_f1_gain = float(executor.last_threshold_result.improvement_delta)

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

    test_eval = evaluator.evaluate(fitted_pipe, split.X_test, np.asarray(split.y_test), threshold=optimal_threshold)

    class_distribution = None
    confusion_matrix = None
    if cfg.task_type == "classification":
        class_distribution = {
            "train": class_counts(np.asarray(split.y_train)),
            "val": class_counts(np.asarray(split.y_val)) if split.y_val is not None else {},
            "test": class_counts(np.asarray(split.y_test)),
            "train_minority_ratio": minority_ratio(np.asarray(split.y_train)),
            "val_minority_ratio": minority_ratio(np.asarray(split.y_val)) if split.y_val is not None else None,
            "test_minority_ratio": minority_ratio(np.asarray(split.y_test)),
        }
        confusion_matrix = test_eval.confusion_matrix
        _log_event(
            "training.class_distribution",
            model_type=model_type_norm,
            class_distribution=class_distribution,
        )

        eval_blocks: Dict[str, Any] = {"train": train_eval.metrics, "test": test_eval.metrics}
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

    metrics_json: Dict[str, Any] = {
        "train": train_eval.metrics,
        "test": test_eval.metrics,
        "training_time_sec": float(time.perf_counter() - t0),
    }
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

def _run_kfold_cv(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
) -> ModelRunResult:
    """
    Cross-validation pipeline (kfold / stratified_kfold).

    Anti-leakage guarantees
    -----------------------
    • When cfg.test_ratio > 0, a stratified holdout test set is carved out of
      the full data FIRST — before any preprocessing fit, before any CV fold.
      This test set is never used during CV (not in train_fold, not in val_fold).
    • The CV loop (fold fit + val evaluation) runs on the non-test portion only.
    • Preprocessor is fit on train_fold indices ONLY, then transforms both
      train_fold and val_fold.
    • Resampling (SMOTE / undersampling) is applied on train_fold ONLY.
    • val_fold and the holdout test set are NEVER resampled.

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
    )

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    if len(df2) < 10:
        raise RuntimeError("Not enough rows after dropping target NaNs (minimum 10).")

    X_all = df2.drop(columns=[cfg.target_column])
    y_all = np.asarray(df2[cfg.target_column].values)

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        y_all, cfg.positive_label
    )
    if positive_label_warning:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    # ── Optional holdout test split ───────────────────────────────────────────
    # When cfg.test_ratio > 0, we separate a stratified (for classification)
    # test set BEFORE any CV or preprocessing — pure holdout, never touched.
    has_holdout_test = float(cfg.test_ratio) > 1e-6
    X_test_holdout: Optional[pd.DataFrame] = None
    y_test_holdout: Optional[np.ndarray] = None
    X_cv: pd.DataFrame = X_all
    y_cv: np.ndarray = y_all

    if has_holdout_test:
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
    for hp_warn in hp_normalized_no_gs.get("warnings", []) or []:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=str(hp_warn))

    base_estimator_hp = dict(hp_normalized_no_gs.get("estimator_params") or {})
    capabilities = get_model_capabilities(model_type_norm)
    shuffle = bool(getattr(cfg, "shuffle", True))

    # ── Generate fold indices (on CV portion only) ────────────────────────────
    try:
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
                preprocessing_config=cfg.preprocessing.as_dict(),
            )
            fold_aligner = ColumnAligner(
                feature_names=fold_schema["feature_names"],
                dtypes=fold_schema["dtypes"],
            )
            fold_spec = build_preprocessor(X_train_fold, cfg.preprocessing)

            X_train_aligned = fold_aligner.fit_transform(X_train_fold)
            X_train_prep = fold_spec.preprocessor.fit_transform(
                X_train_aligned, y_train_fold
            )
            # ② Transform val_fold with the train-fitted preprocessor (no leakage)
            X_val_aligned = fold_aligner.transform(X_val_fold)
            X_val_prep = fold_spec.preprocessor.transform(X_val_aligned)

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
                    random_state=cfg.random_state,
                )

            fold_executor = BalancingExecutor()
            X_f = X_train_prep
            y_f = np.asarray(y_train_fold)
            fold_fit_params: Dict[str, Any] = {}
            if cfg.task_type == "classification":
                # Convert sparse → dense ONLY when resampler requires it
                if fold_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
                    if hasattr(X_train_prep, "toarray"):
                        X_train_prep = X_train_prep.toarray()
                X_f, y_f, fold_fit_params = fold_executor.apply_prefit(
                    X_train_prep, np.asarray(y_train_fold), fold_decision
                )

            fold_hp = dict(base_estimator_hp)
            if "class_weight" in fold_fit_params:
                fold_hp["class_weight"] = fold_fit_params["class_weight"]

            # ④ Train model on (possibly resampled) train_fold
            fold_model = build_model(model_type_norm, cfg.task_type, fold_hp)
            fold_pipeline = Pipeline(steps=[("model", fold_model)])
            sw = fold_fit_params.get("sample_weight")
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
        preprocessing_config=cfg.preprocessing.as_dict(),
    )
    final_aligner = ColumnAligner(
        feature_names=final_training_schema["feature_names"],
        dtypes=final_training_schema["dtypes"],
    )
    final_spec = build_preprocessor(X_refit, cfg.preprocessing)

    X_refit_aligned = final_aligner.fit_transform(X_refit)
    X_refit_prep = final_spec.preprocessor.fit_transform(X_refit_aligned, y_refit)
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
    )

    final_fitted_model = final_fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        final_aligner, final_spec.preprocessor, final_fitted_model, model_type_norm
    )

    # Threshold + audit — calibrated on X_refit (never on test set)
    optimal_threshold = 0.5
    threshold_f1_gain: Optional[float] = None
    if cfg.task_type == "classification":
        optimal_threshold = final_executor.apply_postfit(
            fitted_pipe, X_refit, y_refit, final_decision
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
        X_test_holdout=X_test_holdout,
        final_spec=final_spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        final_training_schema=final_training_schema,
        tuning_artifacts=final_fit_result.tuning_artifacts,
    )

    hyperparams_artifacts: Dict[str, Any] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized_final.get("effective") or {}),
        "note": "GridSearch (if any) applied only on final refit, not per-fold.",
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
        "k_folds": actual_k,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "has_holdout_test": has_holdout_test,
        "cv_mean": cv_mean_metrics,
        "test": holdout_test_metrics if (has_holdout_test and holdout_test_metrics is not None) else cv_mean_metrics,
        "training_time_sec": float(time.perf_counter() - t0),
    }
    if has_holdout_test and holdout_test_metrics is not None:
        metrics_json["holdout_test_metrics"] = holdout_test_metrics

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )
