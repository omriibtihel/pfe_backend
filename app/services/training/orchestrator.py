from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import inspect
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    ImbPipeline = None

from .config import TrainingConfig, normalize_model_hyperparams
from .evaluator import Evaluator
from .imbalance import build_smote_for_train, class_counts, minority_ratio
from .metrics import get_class_labels
from .models import build_model
from .preprocessing import build_preprocessor
from .reporter import Reporter, build_training_schema
from .splitters import make_holdout_split
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
    fitted_pipeline: Any  # pipeline fitted (align + preprocess + model [+ smote])


def _log_event(event: str, **payload: Any) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


def _ensure_dense_matrix(X: Any) -> Any:
    return X.toarray() if hasattr(X, "toarray") else X


def _build_pipeline(aligner: ColumnAligner, preprocessor: Any, model: Any, smote_obj: Any, model_type: str):
    steps = [("align", aligner), ("prep", preprocessor)]
    if model_type in _DENSE_REQUIRED_MODELS:
        steps.append(("dense", FunctionTransformer(_ensure_dense_matrix, accept_sparse=True)))
    if smote_obj is None:
        return Pipeline(steps=[*steps, ("model", model)])
    if ImbPipeline is None:
        raise RuntimeError("SMOTE enabled but imblearn is not installed")
    return ImbPipeline(steps=[*steps, ("smote", smote_obj), ("model", model)])


def _is_debug_enabled(cfg: TrainingConfig) -> bool:
    if bool(getattr(cfg, "debug", False)):
        return True
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


def _supports_estimator_param(estimator: Any, param_name: str) -> bool:
    if estimator is None or not hasattr(estimator, "get_params"):
        return False
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return False
    return param_name in params


def _supports_fit_sample_weight(estimator: Any) -> bool:
    fit_fn = getattr(estimator, "fit", None)
    if fit_fn is None:
        return False
    try:
        signature = inspect.signature(fit_fn)
    except Exception:
        return False
    return "sample_weight" in signature.parameters


def run_one_model(df: pd.DataFrame, cfg: TrainingConfig, model_type: str) -> ModelRunResult:
    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    debug_mode = _is_debug_enabled(cfg)
    _log_event("training.start_model", model_type=model_type_norm, task_type=cfg.task_type)

    if cfg.split_method != "holdout":
        raise RuntimeError("Only holdout is implemented for now in orchestrator.py")

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
            class_distribution=class_counts(y),
            minority_ratio=minority_ratio(y),
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
        random_state=42,
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
            train=class_counts(split.y_train),
            val=class_counts(split.y_val) if split.y_val is not None else {},
            test=class_counts(split.y_test),
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
        raise RuntimeError(
            f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}"
        )
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

    fit_sample_weight = None
    # SMOTE should be enabled whenever explicitly requested by user.
    smote_obj = None
    smote_meta: Dict[str, Any] = {
        "requested_by_user": bool(cfg.use_smote),
        "applied": False,
        "reason": "task_not_classification" if cfg.task_type != "classification" else "user_disabled",
        "smote_applied": False,
        "smote_skipped_reason": None,
        "fallback_applied": None,
    }
    if cfg.task_type == "classification":
        smote_meta["train_class_counts"] = class_counts(split.y_train)
        smote_meta["train_minority_ratio"] = minority_ratio(split.y_train)
        if cfg.use_smote:
            smote_obj, raw_smote_meta = build_smote_for_train(np.asarray(split.y_train))
            smote_meta["applied"] = bool(smote_obj is not None)
            smote_meta["reason"] = "applied" if smote_obj is not None else str(raw_smote_meta.get("reason", "guard_blocked"))
            if isinstance(raw_smote_meta, dict):
                for k, v in raw_smote_meta.items():
                    if k in {"enabled", "reason"}:
                        continue
                    smote_meta[str(k)] = v
            if smote_obj is None:
                smote_meta["smote_skipped_reason"] = smote_meta["reason"]
                fallback_reason = str(smote_meta.get("reason") or "")
                tentative_model = build_model(model_type_norm, cfg.task_type, dict(estimator_hyperparams))
                if fallback_reason in {"minority_too_small", "imblearn_not_installed"}:
                    if _supports_estimator_param(tentative_model, "class_weight"):
                        estimator_hyperparams["class_weight"] = "balanced"
                        smote_meta["fallback_applied"] = "class_weight_balanced"
                    elif _supports_fit_sample_weight(tentative_model):
                        fit_sample_weight = compute_sample_weight(
                            class_weight="balanced",
                            y=np.asarray(split.y_train),
                        )
                        smote_meta["fallback_applied"] = "sample_weight_balanced"
                    else:
                        smote_meta["fallback_applied"] = "none"
                        _log_event(
                            "training.smote.fallback_unavailable",
                            model_type=model_type_norm,
                            reason=fallback_reason,
                        )
                else:
                    smote_meta["fallback_applied"] = "none"
            else:
                smote_meta["fallback_applied"] = "none"
    if debug_mode and cfg.task_type == "classification":
        if smote_obj is not None:
            before_counts = class_counts(split.y_train)
            after_counts = dict(before_counts)
            if len(after_counts) == 2:
                majority = max(after_counts.values())
                after_counts = {str(k): int(majority) for k in after_counts.keys()}
            _log_event(
                "training.debug.smote_distribution",
                model_type=model_type_norm,
                before=before_counts,
                after=after_counts,
            )
        else:
            _log_event(
                "training.debug.smote_distribution",
                model_type=model_type_norm,
                before=class_counts(split.y_train),
                after=None,
                reason=smote_meta.get("reason"),
                fallback=smote_meta.get("fallback_applied"),
            )
    smote_meta["smote_applied"] = bool(smote_obj is not None)
    smote_meta["enabled"] = bool(smote_meta["applied"])

    model = build_model(model_type_norm, cfg.task_type, estimator_hyperparams)
    pipe = _build_pipeline(aligner, spec.preprocessor, model, smote_obj, model_type_norm)

    trainer = Trainer()
    fit_result = trainer.fit(
        pipeline=pipe,
        X_train=split.X_train,
        y_train=np.asarray(split.y_train),
        fit_sample_weight=fit_sample_weight,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=param_grid,
    )
    fitted_pipe = fit_result.fitted_pipeline
    if debug_mode and cfg.task_type == "classification":
        _log_event(
            "training.debug.model_classes",
            model_type=model_type_norm,
            model_classes=get_class_labels(fitted_pipe),
        )

    evaluator = Evaluator(
        task_type=cfg.task_type,
        requested_metrics=cfg.metrics,
        positive_label=resolved_positive_label,
    )
    train_eval = evaluator.evaluate(fitted_pipe, split.X_train, np.asarray(split.y_train))

    val_metrics = None
    if split.X_val is not None and split.y_val is not None and len(split.X_val) > 0:
        val_eval = evaluator.evaluate(fitted_pipe, split.X_val, np.asarray(split.y_val))
        val_metrics = val_eval.metrics
    else:
        val_eval = None

    test_eval = evaluator.evaluate(fitted_pipe, split.X_test, np.asarray(split.y_test))

    class_distribution = None
    confusion_matrix = None
    if cfg.task_type == "classification":
        class_distribution = {
            "train": class_counts(split.y_train),
            "val": class_counts(split.y_val) if split.y_val is not None else {},
            "test": class_counts(split.y_test),
            "train_minority_ratio": minority_ratio(split.y_train),
            "val_minority_ratio": minority_ratio(split.y_val) if split.y_val is not None else None,
            "test_minority_ratio": minority_ratio(split.y_test),
        }
        confusion_matrix = test_eval.confusion_matrix
        _log_event(
            "training.class_distribution",
            model_type=model_type_norm,
            class_distribution=class_distribution,
        )

        eval_blocks: Dict[str, Any] = {
            "train": train_eval.metrics,
            "test": test_eval.metrics,
        }
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
        smote_meta=smote_meta,
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
        smote_applied=smote_meta.get("applied"),
        tuned=bool(fit_result.tuning_artifacts.get("enabled", False)),
    )

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )
