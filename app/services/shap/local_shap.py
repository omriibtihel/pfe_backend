"""
Local SHAP computation — per-row explanation for the prediction endpoint.

For a single input row (or small batch), returns:
  [
    {
      "feature":    str,   # original feature name
      "shap_value": float, # SHAP contribution to the prediction
      "data":       any,   # raw input value for that feature
    },
    ...  # sorted by |shap_value| descending
  ]

The explainer is built fresh each call using the training background stored in
artifacts["shap"]["background_data"] (a subsample saved at training time).
Falls back to KernelExplainer without a pre-computed background.

Returns None on any failure (shap not installed, shape mismatch, etc.) so
the prediction endpoint degrades gracefully.
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
from app.services.shap.global_shap import (
    _transform_except_model,
    _get_preprocessed_feature_names,
)


def compute_local_shap(
    fitted_pipe: Any,
    row: pd.DataFrame,
    *,
    task_type: str = "classification",
    background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Compute SHAP values for a single row (or small batch).

    Parameters
    ----------
    fitted_pipe:
        Full fitted sklearn Pipeline.
    row:
        A 1-row (or few-rows) DataFrame in the same format as training input.
    task_type:
        "classification" | "regression"
    background:
        Pre-computed background array (preprocessed).  When None, the
        row itself is used as its own background (fast but less accurate).
    feature_names:
        Original feature names for display.  Defaults to row.columns.

    Returns
    -------
    List of dicts sorted by |shap_value| descending, or None on failure.
    """
    try:
        import shap as _shap_pkg  # noqa: F401
    except ImportError:
        return None

    if row is None or len(row) == 0:
        return None

    names = list(feature_names) if feature_names is not None else list(row.columns)

    try:
        estimator = extract_model_from_pipeline(fitted_pipe)

        # Preprocess the row through all steps except the model
        X_row_prep = _transform_except_model(fitted_pipe, row)

        # Use provided background.
        # IMPORTANT: never use the row itself as its own background — that
        # produces all-zero SHAP values because mean(background) == input.
        # If no background is available, use zeros (neutral reference point).
        if background is not None and len(background) > 1:
            bg = background
        else:
            bg = np.zeros_like(X_row_prep)

        explainer, explainer_type = build_explainer(
            estimator,
            bg,
            task_type=task_type,
        )

        raw_shap = explainer.shap_values(X_row_prep)
        shap_arr = shap_values_to_float_array(raw_shap)  # (n_rows, n_features)

        # Use preprocessed feature names for display
        shap_feature_names = _get_preprocessed_feature_names(fitted_pipe, names)
        if len(shap_feature_names) != shap_arr.shape[1]:
            shap_feature_names = [f"f_{i}" for i in range(shap_arr.shape[1])]

        # Build per-feature explanation for the first row
        shap_row = shap_arr[0]

        # Map original input values (best-effort — may differ after OHE)
        raw_values = row.iloc[0].to_dict()

        items: List[Dict[str, Any]] = []
        for i, fname in enumerate(shap_feature_names):
            # Try to retrieve the original value for features that kept their name
            raw_val = raw_values.get(fname, None)
            items.append({
                "feature":    str(fname),
                "shap_value": float(to_json_safe(shap_row[i])),
                "data":       to_json_safe(raw_val),
            })

        items.sort(key=lambda d: abs(float(d["shap_value"])), reverse=True)
        return items

    except Exception:
        return None
