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
    get_positive_class_index,
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
    positive_label: Optional[Any] = None,
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

        # Same class-index logic as global_shap: always orient SHAP values toward
        # the agreed positive class, not blindly toward index 1.
        binary_class_index = (
            get_positive_class_index(fitted_pipe, positive_label)
            if task_type == "classification"
            else 0
        )

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

        # For multiclass local SHAP: orient toward the predicted class so the
        # explanation answers "why was this sample classified as X?" rather than
        # showing a direction-less mean(abs(SHAP)) across all classes.
        _predicted_class_indices: Optional[np.ndarray] = None
        if isinstance(raw_shap, list) and len(raw_shap) > 2:
            try:
                if hasattr(estimator, "predict_proba"):
                    _proba_mc = estimator.predict_proba(X_row_prep)
                    _predicted_class_indices = np.argmax(_proba_mc, axis=1)
                else:
                    _predicted_class_indices = np.asarray(
                        [int(estimator.predict(X_row_prep)[0])]
                    )
            except Exception:
                _predicted_class_indices = None

        shap_arr = shap_values_to_float_array(
            raw_shap,
            binary_class_index=binary_class_index,
            predicted_class_indices=_predicted_class_indices,
        )

        # Use preprocessed feature names for display
        shap_feature_names = _get_preprocessed_feature_names(fitted_pipe, names)
        if len(shap_feature_names) != shap_arr.shape[1]:
            shap_feature_names = [f"f_{i}" for i in range(shap_arr.shape[1])]

        # Build per-feature explanation for the first row
        shap_row = shap_arr[0]

        # Map preprocessed names back to original input values.
        # For OHE features (cat__sex_male) this returns the original column
        # value ('male') rather than the 0/1 indicator or None.
        raw_values = row.iloc[0].to_dict()

        def _map_to_raw(prep_name: str) -> Any:
            if prep_name in raw_values:
                return raw_values[prep_name]
            base = prep_name.split("__", 1)[-1] if "__" in prep_name else prep_name
            if base in raw_values:
                return raw_values[base]
            parts = base.split("_")
            for n in range(len(parts) - 1, 0, -1):
                candidate = "_".join(parts[:n])
                if candidate in raw_values:
                    return raw_values[candidate]
            return None

        items: List[Dict[str, Any]] = []
        for i, fname in enumerate(shap_feature_names):
            items.append({
                "feature":    str(fname),
                "shap_value": float(to_json_safe(shap_row[i])),
                "data":       to_json_safe(_map_to_raw(fname)),
            })

        items.sort(key=lambda d: abs(float(d["shap_value"])), reverse=True)
        return items

    except Exception:
        return None
