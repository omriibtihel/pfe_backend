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
        shap_arr = shap_values_to_float_array(raw_shap)  # (n, n_preprocessed_features)

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

        # ── Expected value ────────────────────────────────────────────────────
        ev = getattr(explainer, "expected_value", None)
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) == 2 else float(np.mean(ev))
        expected_value = to_json_safe(float(ev)) if ev is not None else None

        # Save a small background subsample for local SHAP at prediction time.
        # Stored as a list-of-lists (JSON-safe); reloaded in predict_with_shap.
        rng_bg = np.random.default_rng(random_state + 1)
        n_bg = min(_BACKGROUND_SAVE_N, len(X_prep))
        bg_idx = rng_bg.choice(len(X_prep), size=n_bg, replace=False)
        bg_data = X_prep[bg_idx].tolist()

        return {
            "summary":         items,
            "expected_value":  expected_value,
            "explainer_type":  explainer_type,
            "n_samples":       int(len(X_explain)),
            "background_data": bg_data,
        }

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
    Try to retrieve feature names after preprocessing (OHE expands features).

    Returns original_names unchanged if the pipeline does not expose
    get_feature_names_out().
    """
    named = getattr(pipeline, "named_steps", None)
    if named is None:
        return original_names

    # Look for a "prep" step with get_feature_names_out
    prep = named.get("prep")
    if prep is not None and hasattr(prep, "get_feature_names_out"):
        try:
            return [str(n) for n in prep.get_feature_names_out()]
        except Exception:
            pass

    return original_names
