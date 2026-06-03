"""
Global SHAP computation for post-training artifacts.

Computes mean |SHAP| per feature on the test set (or a subsample for
KernelExplainer) and returns a summary suitable for storage in
artifacts["shap"].

Output schema
-------------
{
    "summary": [
        {
            "feature":       str,
            "mean_abs_shap": float,   # mean |SHAP| — primary ranking criterion
            "mean_shap":     float,   # signed mean SHAP — shows direction
        },
        ...  # sorted by mean_abs_shap descending, top _MAX_FEATURES
    ],
    "expected_value": float | None,
    "explainer_type": "tree" | "linear" | "kernel",
    "n_samples":      int,
}

Returns None if shap is not installed, test set is too small, or any error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.shap._utils import (
    extract_model_from_pipeline,
    get_positive_class_index,
    shap_values_to_float_array,
    to_json_safe,
)
from app.services.shap._explainer import build_explainer


_MIN_SAMPLES = 10
_MAX_FEATURES = 30
# KernelExplainer is slow — cap the number of rows we explain
_KERNEL_MAX_ROWS = 200
# Rows of preprocessed data to save as background for local SHAP at prediction time
_BACKGROUND_SAVE_N = 100


def compute_global_shap(
    fitted_pipe: Any,
    X_test_raw: pd.DataFrame,
    *,
    task_type: str = "classification",
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    positive_label: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute global SHAP feature importances on the test set.

    Parameters
    ----------
    fitted_pipe:
        Full fitted sklearn Pipeline (align → prep → model).
    X_test_raw:
        Raw (un-preprocessed) test DataFrame — same format as training input.
    task_type:
        "classification" | "regression"
    feature_names:
        Original feature names for display.  Defaults to X_test_raw.columns.
    random_state:
        RNG seed used when subsampling for KernelExplainer.

    Returns
    -------
    dict matching the output schema above, or None on failure.
    """
    try:
        import shap as _shap_pkg  # noqa: F401 — confirms shap is installed
    except ImportError:
        return None

    if len(X_test_raw) < _MIN_SAMPLES:
        return None

    names = list(feature_names) if feature_names is not None else list(X_test_raw.columns)

    try:
        estimator = extract_model_from_pipeline(fitted_pipe)

        # Determine which class column SHAP values should be oriented toward.
        # For binary classification, TreeExplainer returns [shap_class0, shap_class1];
        # we must pick the index matching the agreed positive class, not always index 1.
        binary_class_index = (
            get_positive_class_index(fitted_pipe, positive_label)
            if task_type == "classification"
            else 0
        )

        # ── Preprocessed features for the explainer ──────────────────────────
        # We need a numpy background matrix that has already been transformed
        # by the pipeline's preprocessing steps (align + prep), so the
        # estimator sees the correct input space.
        # We transform X_test_raw through all steps EXCEPT the final "model".
        X_prep = _transform_except_model(fitted_pipe, X_test_raw)

        # Subsample for KernelExplainer (performance guard)
        from app.services.shap._utils import get_explainer_type
        explainer_type_hint = get_explainer_type(estimator)
        X_explain = X_prep
        if explainer_type_hint == "kernel" and len(X_prep) > _KERNEL_MAX_ROWS:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_prep), size=_KERNEL_MAX_ROWS, replace=False)
            X_explain = X_prep[idx]

        explainer, explainer_type = build_explainer(
            estimator,
            X_prep,
            task_type=task_type,
        )

        # ── Compute SHAP values ───────────────────────────────────────────────
        raw_shap = explainer.shap_values(X_explain)
        # For multiclass global SHAP: compute mean(abs) across classes for the
        # overall importance ranking, then also build per-class summaries so
        # users can see which features push toward each specific class.
        _is_multiclass_list = isinstance(raw_shap, list) and len(raw_shap) > 2
        shap_arr = shap_values_to_float_array(raw_shap, binary_class_index=binary_class_index)

        # ── Map preprocessed features back to original feature names ─────────
        # After OHE / scaling the number of columns may differ from len(names).
        # We fall back to raw column indices when the shapes don't match.
        shap_feature_names = _get_preprocessed_feature_names(fitted_pipe, names)
        if len(shap_feature_names) != shap_arr.shape[1]:
            # Shape mismatch — use numeric indices
            shap_feature_names = [f"f_{i}" for i in range(shap_arr.shape[1])]

        # ── Aggregate: mean |SHAP| and mean SHAP per original feature ─────────
        mean_abs = np.abs(shap_arr).mean(axis=0)
        mean_signed = shap_arr.mean(axis=0)

        items: List[Dict[str, Any]] = []
        for i, fname in enumerate(shap_feature_names):
            items.append({
                "feature":       str(fname),
                "mean_abs_shap": float(to_json_safe(mean_abs[i])),
                "mean_shap":     float(to_json_safe(mean_signed[i])),
            })

        items.sort(key=lambda d: float(d["mean_abs_shap"]), reverse=True)
        items = items[:_MAX_FEATURES]

        # ── Per-class summaries for multiclass (Bug 13 fix) ───────────────────
        # mean(abs(SHAP)) hides direction — a feature that strongly predicts
        # class A may appear "unimportant" if it has low absolute values for
        # other classes. Per-class signed summaries let the user ask
        # "which features push toward diabetes vs. healthy?".
        per_class_summary: Optional[List[Dict[str, Any]]] = None
        if _is_multiclass_list:
            try:
                model_classes = getattr(estimator, "classes_", None)
                per_class_summary = []
                for cls_idx, cls_shap in enumerate(raw_shap):
                    cls_arr = np.asarray(cls_shap, dtype=float)
                    if cls_arr.ndim != 2 or cls_arr.shape[1] != len(shap_feature_names):
                        continue
                    cls_mean_abs = np.abs(cls_arr).mean(axis=0)
                    cls_mean_signed = cls_arr.mean(axis=0)
                    cls_items = [
                        {
                            "feature": str(shap_feature_names[i]),
                            "mean_abs_shap": float(to_json_safe(cls_mean_abs[i])),
                            "mean_shap": float(to_json_safe(cls_mean_signed[i])),
                        }
                        for i in range(len(shap_feature_names))
                    ]
                    cls_items.sort(key=lambda d: float(d["mean_abs_shap"]), reverse=True)
                    class_label = (
                        str(model_classes[cls_idx])
                        if model_classes is not None and cls_idx < len(model_classes)
                        else f"class_{cls_idx}"
                    )
                    per_class_summary.append({
                        "class_index": cls_idx,
                        "class_label": class_label,
                        "features": cls_items[:_MAX_FEATURES],
                    })
            except Exception:
                per_class_summary = None

        # ── Expected value ────────────────────────────────────────────────────
        ev = getattr(explainer, "expected_value", None)
        if isinstance(ev, (list, np.ndarray)):
            # Use the same class index as the SHAP values so the baseline
            # probability matches the direction of the explanations.
            ev = ev[binary_class_index] if len(ev) == 2 else float(np.mean(ev))
        expected_value = to_json_safe(float(ev)) if ev is not None else None

        # Save a small background subsample for local SHAP at prediction time.
        # Stored as a list-of-lists (JSON-safe); reloaded in predict_with_shap.
        rng_bg = np.random.default_rng(random_state + 1)
        n_bg = min(_BACKGROUND_SAVE_N, len(X_prep))
        bg_idx = rng_bg.choice(len(X_prep), size=n_bg, replace=False)
        bg_data = X_prep[bg_idx].tolist()

        result: Dict[str, Any] = {
            "summary":         items,
            "expected_value":  expected_value,
            "explainer_type":  explainer_type,
            "n_samples":       int(len(X_explain)),
            "background_data": bg_data,
        }
        if per_class_summary is not None:
            result["per_class_summary"] = per_class_summary
        return result

    except Exception:
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _transform_except_model(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Apply all pipeline steps except the last "model" step.

    Returns a dense numpy float array.
    """
    from sklearn.pipeline import Pipeline as _SKPipeline
    named = getattr(pipeline, "named_steps", None)

    if named is None:
        # Not a sklearn Pipeline — return raw values
        return X.values.astype(float)

    X_out = X
    steps = list(named.items())
    for name, step in steps[:-1]:  # skip last step (the estimator)
        X_out = step.transform(X_out)

    # Convert sparse to dense
    if hasattr(X_out, "toarray"):
        X_out = X_out.toarray()

    return np.asarray(X_out, dtype=float)


def _get_preprocessed_feature_names(pipeline: Any, original_names: List[str]) -> List[str]:
    """
    Try to retrieve feature names in the space the estimator actually sees.

    OHE expands features (handled by ``prep.get_feature_names_out()``), and the
    ``select`` step (VarianceThreshold) may then drop some of them. Because
    ``_transform_except_model`` applies ``select`` too, we must mirror that here
    or the returned names won't line up with the transformed matrix — which
    silently falls back to ``f_0, f_1 …`` for every explainer.

    Returns original_names unchanged if the pipeline does not expose
    get_feature_names_out().
    """
    named = getattr(pipeline, "named_steps", None)
    if named is None:
        return original_names

    # Look for a "prep" step with get_feature_names_out
    prep = named.get("prep")
    if prep is None or not hasattr(prep, "get_feature_names_out"):
        return original_names
    try:
        names = [str(n) for n in prep.get_feature_names_out()]
    except Exception:
        return original_names

    # Apply the post-prep feature-selection mask so names match the model's
    # input space (and the output of _transform_except_model).
    select = named.get("select")
    if select is not None and hasattr(select, "get_support"):
        try:
            mask = select.get_support()
            if len(mask) == len(names):
                names = [n for n, keep in zip(names, mask) if bool(keep)]
        except Exception:
            pass

    return names
