from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PREPROCESSING_CAPABILITIES, PREPROCESSING_EXECUTION_POLICY


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_feature_names(fitted_pipeline: Any, n_fallback: int) -> list[str]:
    named_steps = getattr(fitted_pipeline, "named_steps", {}) or {}
    prep = named_steps.get("prep")
    align = named_steps.get("align")
    align_feature_names = getattr(align, "feature_names", None)

    names: list[str] = []

    if prep is not None and hasattr(prep, "get_feature_names_out"):
        try:
            names = [str(n) for n in prep.get_feature_names_out()]
        except Exception:
            pass

        if not names and isinstance(align_feature_names, (list, tuple)) and align_feature_names:
            try:
                names = [str(n) for n in prep.get_feature_names_out(input_features=list(align_feature_names))]
            except Exception:
                pass

    if not names and isinstance(align_feature_names, (list, tuple)) and align_feature_names:
        names = [str(n) for n in align_feature_names]

    if not names:
        names = [f"f_{idx}" for idx in range(int(max(0, n_fallback)))]

    # Apply VarianceThreshold mask if the selector step exists in the pipeline.
    select_step = named_steps.get("select")
    if select_step is not None and hasattr(select_step, "get_support"):
        try:
            mask = select_step.get_support()
            if len(mask) == len(names):
                names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

    return names


def _extract_importance_values(model: Any) -> np.ndarray:
    if model is None:
        return np.array([], dtype=float)

    # Tree/boosting-style estimators.
    if hasattr(model, "feature_importances_"):
        try:
            return np.asarray(getattr(model, "feature_importances_"), dtype=float).reshape(-1)
        except Exception:
            return np.array([], dtype=float)

    # Linear estimators (e.g. LogisticRegression): use absolute coefficient magnitude.
    if hasattr(model, "coef_"):
        try:
            coefs = np.asarray(getattr(model, "coef_"), dtype=float)
        except Exception:
            return np.array([], dtype=float)

        if coefs.ndim == 0:
            return np.array([], dtype=float)
        if coefs.ndim == 1:
            return np.abs(coefs).reshape(-1)
        return np.mean(np.abs(coefs), axis=0).reshape(-1)

    return np.array([], dtype=float)


def summarize_feature_importance(fitted_pipeline: Any, *, max_items: int = 50) -> list[Dict[str, Any]]:
    named_steps = getattr(fitted_pipeline, "named_steps", {}) or {}
    model = named_steps.get("model")
    values = _extract_importance_values(model)
    if values.size == 0:
        return []

    values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = _extract_feature_names(fitted_pipeline, values.size)

    items: list[Dict[str, Any]] = []
    for idx, imp in enumerate(values):
        feat = feature_names[idx] if idx < len(feature_names) else f"f_{idx}"
        items.append({"feature": str(feat), "importance": float(imp)})

    items = sorted(items, key=lambda d: float(d.get("importance", 0.0)), reverse=True)
    return items[: int(max(1, max_items))]


def _to_builtin_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_value(value: Any, *, depth: int = 0, max_depth: int = 2, max_items: int = 20) -> Any:
    value = _to_builtin_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        if depth >= max_depth:
            return {"type": "dict", "size": int(len(value))}
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                out["__truncated__"] = int(len(value) - max_items)
                break
            out[str(k)] = _safe_value(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if depth >= max_depth:
            return {"type": type(value).__name__, "length": int(len(seq))}
        out = [
            _safe_value(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out.append(f"...(+{len(seq) - max_items} more)")
        return out

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return {"type": value.__class__.__name__, "shape": [int(s) for s in shape]}
        except Exception:
            return {"type": value.__class__.__name__}

    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__)
        return f"<callable:{name}>"

    return f"<{value.__class__.__name__}>"


def _safe_params(params: Dict[str, Any], *, max_keys: int, max_items: int = 20, max_depth: int = 2) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted(str(k) for k in params.keys())
    for i, key in enumerate(keys):
        if i >= max_keys:
            out["__truncated_keys__"] = int(len(keys) - max_keys)
            break
        out[key] = _safe_value(params.get(key), depth=0, max_depth=max_depth, max_items=max_items)
    return out


def _summarize_component(component: Any, *, max_keys: int = 60) -> Dict[str, Any]:
    if component is None:
        return {"class_name": "NoneType", "params": {}}
    if isinstance(component, str):
        return {"class_name": component, "params": {}}

    params: Dict[str, Any] = {}
    if hasattr(component, "get_params"):
        try:
            params = _safe_params(component.get_params(deep=False), max_keys=max_keys, max_items=15, max_depth=2)
        except Exception as e:
            params = {"error": str(e)}

    return {"class_name": component.__class__.__name__, "params": params}


def summarize_preprocessing_applied(prep: Any, X_train: pd.DataFrame) -> Dict[str, Any]:
    summary = _summarize_component(prep, max_keys=60)

    if prep == "passthrough":
        sample_rows = int(min(200, len(X_train))) if X_train is not None else 0
        if sample_rows > 0:
            X_sample = X_train.head(sample_rows)
            summary["sample_input_shape"] = [int(X_sample.shape[0]), int(X_sample.shape[1])]
            summary["sample_transform_shape"] = [int(X_sample.shape[0]), int(X_sample.shape[1])]
            summary["n_features_out"] = int(X_sample.shape[1])
        return summary

    n_features_in = getattr(prep, "n_features_in_", None)
    if n_features_in is not None:
        summary["n_features_in"] = int(n_features_in)

    fitted_transformers = getattr(prep, "transformers_", None)
    if isinstance(fitted_transformers, list):
        tx_summaries = []
        for name, transformer, columns in fitted_transformers:
            col_count: Optional[int] = None
            try:
                col_count = int(len(columns))
            except Exception:
                col_count = None
            tx_summaries.append(
                {
                    "name": str(name),
                    "columns_count": col_count,
                    "transformer": _summarize_component(transformer, max_keys=30),
                }
            )
        summary["transformers"] = tx_summaries

    sample_rows = int(min(200, len(X_train))) if X_train is not None else 0
    if sample_rows > 0:
        X_sample = X_train.head(sample_rows)
        summary["sample_input_shape"] = [int(X_sample.shape[0]), int(X_sample.shape[1])]
        try:
            X_out = prep.transform(X_sample)
            shape = getattr(X_out, "shape", None)
            if shape is not None and len(shape) >= 2:
                summary["sample_transform_shape"] = [int(shape[0]), int(shape[1])]
                summary["n_features_out"] = int(shape[1])
        except Exception as e:
            summary["sample_transform_shape_error"] = str(e)

    return summary


def summarize_model(model: Any) -> Dict[str, Any]:
    class_name = model.__class__.__name__ if model is not None else "NoneType"
    max_keys = 120
    if class_name.startswith("XGB") or class_name.startswith("LGBM"):
        max_keys = 80

    params: Dict[str, Any] = {}
    if model is not None and hasattr(model, "get_params"):
        try:
            params = _safe_params(model.get_params(), max_keys=max_keys, max_items=15, max_depth=2)
        except Exception as e:
            params = {"error": str(e)}

    return {"class_name": class_name, "params": params}


def build_training_schema(*, X: pd.DataFrame, target: str, preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "feature_names": [str(c) for c in X.columns],
        "dtypes": {str(c): str(dt) for c, dt in X.dtypes.items()},
        "target": str(target),
        "preprocessing_config": preprocessing_config,
        "created_at": _utc_now_iso(),
    }


class Reporter:
    def build_artifacts(
        self,
        *,
        cfg: Any,
        split: Any,
        spec: Any,
        fitted_pipeline: Any,
        balancing_audit: Dict[str, Any],
        training_schema: Dict[str, Any],
        tuning_artifacts: Dict[str, Any],
        confusion_matrix: Optional[list[list[int]]] = None,
        class_distribution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        named_steps = getattr(fitted_pipeline, "named_steps", {}) or {}
        fitted_prep = named_steps.get("prep")
        fitted_model = named_steps.get("model")

        preprocessing_applied: Dict[str, Any] = {}
        if fitted_prep is not None:
            preprocessing_applied = summarize_preprocessing_applied(fitted_prep, split.X_train)

        artifacts: Dict[str, Any] = {
            "split_info": {
                "method": "holdout",
                "train_rows": int(len(split.X_train)),
                "val_rows": int(len(split.X_val)) if split.X_val is not None else 0,
                "test_rows": int(len(split.X_test)),
                "train_ratio": float(cfg.train_ratio),
                "val_ratio": float(cfg.val_ratio),
                "test_ratio": float(cfg.test_ratio),
            },
            "columns": {
                "numeric": spec.numeric_cols,
                "categorical": spec.categorical_cols,
            },
            "preprocessing": {
                "availableMethods": PREPROCESSING_CAPABILITIES,
                "selectedMethods": spec.selected_methods,
                "defaults": spec.selected_methods.get("defaults", {}),
                "effectiveByColumn": spec.effective_by_column,
                "droppedColumns": spec.dropped_columns,
                "columnTypes": spec.column_types,
                "execution": PREPROCESSING_EXECUTION_POLICY,
                "applied": preprocessing_applied,
            },
            "training_schema": training_schema,
            "balancing": balancing_audit,
            "thresholding": {
                "enabled": bool(balancing_audit.get("apply_threshold", False)),
                "strategy": balancing_audit.get("threshold_strategy"),
                "optimal_threshold": balancing_audit.get("optimal_threshold"),
                "improvement_delta": balancing_audit.get("threshold_f1_gain"),
                "warnings": balancing_audit.get("warnings", []),
            },
            "model": summarize_model(fitted_model),
            "feature_importance": summarize_feature_importance(fitted_pipeline),
            "grid_search": tuning_artifacts,
        }

        if confusion_matrix is not None:
            artifacts["confusion_matrix"] = confusion_matrix
        if class_distribution is not None:
            artifacts["class_distribution"] = class_distribution

        return artifacts

    def build_cv_artifacts(
        self,
        *,
        cfg: Any,
        fold_results: List[Dict[str, Any]],
        cv_summary: Dict[str, Any],
        actual_k: int,
        n_folds_ok: int,
        X_all: pd.DataFrame,
        y_all: Any,
        X_cv: Optional[pd.DataFrame] = None,
        X_test_holdout: Optional[pd.DataFrame] = None,
        final_spec: Any,
        fitted_pipeline: Any,
        balancing_audit: Dict[str, Any],
        final_training_schema: Dict[str, Any],
        tuning_artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the artifacts dict for a cross-validation run.

        When X_test_holdout is provided (cfg.test_ratio > 0), the CV ran on
        X_cv only and the model was refit on X_cv. The holdout test set is
        genuinely independent.

        When X_test_holdout is None (cfg.test_ratio == 0), original behaviour:
        CV on all data, refit on all data.
        """
        named_steps = getattr(fitted_pipeline, "named_steps", {}) or {}
        fitted_prep = named_steps.get("prep")
        fitted_model = named_steps.get("model")

        # Use X_cv for preprocessing summary (that's what the refit model saw)
        X_refit = X_cv if X_cv is not None else X_all
        has_holdout_test = X_test_holdout is not None
        n_samples_cv = int(len(X_refit))
        n_samples_total = int(len(X_all))

        preprocessing_applied: Dict[str, Any] = {}
        if fitted_prep is not None:
            preprocessing_applied = summarize_preprocessing_applied(fitted_prep, X_refit)

        # Aggregate fold sizes
        ok_folds = [fr for fr in fold_results if fr.get("status") == "ok"]
        train_sizes = [int(fr["train_size"]) for fr in ok_folds if "train_size" in fr]
        val_sizes = [int(fr["val_size"]) for fr in ok_folds if "val_size" in fr]

        # Per-fold class distribution (classification only)
        fold_class_distributions = [
            {"fold": fr["fold"], "class_distribution": fr.get("class_distribution")}
            for fr in ok_folds
            if fr.get("class_distribution") is not None
        ]

        # Balancing summary across folds
        balancing_strategies_used = list({
            fr.get("balancing_strategy", "none")
            for fr in ok_folds
            if fr.get("balancing_strategy")
        })
        smote_added_per_fold = [
            fr["smote_samples_added"]
            for fr in ok_folds
            if fr.get("smote_samples_added") is not None
        ]

        if has_holdout_test:
            refit_note = (
                "The saved model was refit on the non-test CV data only. "
                "CV metrics estimate generalisation on unseen folds; "
                "the holdout test score is the final external evaluation."
            )
        else:
            refit_note = (
                "The saved model was refit on ALL data after cross-validation. "
                "Use cv_summary for the honest generalisation estimate. "
                "The refit model is suitable for deployment/prediction."
            )

        artifacts: Dict[str, Any] = {
            "split_info": {
                "method": str(cfg.split_method),
                "k_folds": int(actual_k),
                "folds_ok": int(n_folds_ok),
                "folds_failed": int(actual_k - n_folds_ok),
                "n_samples": n_samples_total,
                "n_samples_cv": n_samples_cv,
                "has_holdout_test": has_holdout_test,
                "test_rows": int(len(X_test_holdout)) if has_holdout_test else None,
                "shuffle": bool(getattr(cfg, "shuffle", True)),
                "random_state": int(getattr(cfg, "random_state", 42)),
                "avg_train_size": float(np.mean(train_sizes)) if train_sizes else None,
                "avg_val_size": float(np.mean(val_sizes)) if val_sizes else None,
                # Legacy holdout-style keys kept as None for frontend compat
                "train_rows": None,
                "val_rows": None,
            },
            "cv_fold_details": fold_results,
            "cv_summary": cv_summary,
            "columns": {
                "numeric": final_spec.numeric_cols,
                "categorical": final_spec.categorical_cols,
            },
            "preprocessing": {
                "availableMethods": PREPROCESSING_CAPABILITIES,
                "selectedMethods": final_spec.selected_methods,
                "defaults": final_spec.selected_methods.get("defaults", {}),
                "effectiveByColumn": final_spec.effective_by_column,
                "droppedColumns": final_spec.dropped_columns,
                "columnTypes": final_spec.column_types,
                "execution": PREPROCESSING_EXECUTION_POLICY,
                "applied": preprocessing_applied,
            },
            "training_schema": final_training_schema,
            "balancing": balancing_audit,
            "balancing_cv": {
                "strategies_used_in_folds": balancing_strategies_used,
                "smote_samples_added_per_fold": smote_added_per_fold,
                "note": "Balancing was applied per-fold on train_fold only. "
                        "Val folds were never resampled.",
            },
            "thresholding": {
                "enabled": bool(balancing_audit.get("apply_threshold", False)),
                "strategy": balancing_audit.get("threshold_strategy"),
                "optimal_threshold": balancing_audit.get("optimal_threshold"),
                "improvement_delta": balancing_audit.get("threshold_f1_gain"),
                "warnings": balancing_audit.get("warnings", []),
                "note": "Threshold was optimised on the refit data (non-test).",
            },
            "model": summarize_model(fitted_model),
            "feature_importance": summarize_feature_importance(fitted_pipeline),
            "grid_search": tuning_artifacts,
            "refit_info": {
                "refit_on_full_data": not has_holdout_test,
                "refit_n_samples": n_samples_cv,
                "note": refit_note,
            },
            # Fold-level class distributions (classification only)
            "fold_class_distributions": fold_class_distributions if fold_class_distributions else None,
        }

        return artifacts
