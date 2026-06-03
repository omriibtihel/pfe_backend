"""
Local LIME computation — per-row explanation for the prediction endpoint.

For a single input row, returns:
  [
    {
      "feature":      str,    # original (preprocessed) feature name
      "contribution": float,  # LIME local contribution to the prediction
      "data":         any,    # raw input value for that feature when retrievable
    },
    ...  # sorted by |contribution| descending
  ]

Design choices (kept consistent with app.services.shap.local_shap):
  - Reuses the preprocessed background array saved at training time
    (artifacts["shap"]["background_data"]). We deliberately re-use the SHAP
    background to avoid duplicating storage and to keep both methods explaining
    the same baseline distribution.
  - Operates in the *preprocessed* feature space — the row is transformed
    through every pipeline step except the final estimator, and LIME is built
    on top of the raw estimator's predict_proba/predict.
  - Categorical features are treated as continuous (after one-hot encoding
    they are 0/1 numeric columns, which LIME handles natively without
    special discretisation).

Returns None on any failure (lime not installed, shape mismatch, etc.) so
the prediction endpoint degrades gracefully — never raises to the caller.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.shap._utils import (
    extract_model_from_pipeline,
    get_positive_class_index,
    to_json_safe,
)
from app.services.shap.global_shap import (
    _transform_except_model,
    _get_preprocessed_feature_names,
)


# Maximum number of features returned per explanation (LIME ranks them anyway)
_MAX_FEATURES = 30


def _map_prep_to_raw_value(prep_name: str, raw_values: "Dict[str, Any]") -> "Any":
    """Map a preprocessed feature name back to the closest original input value.

    Handles three naming patterns produced by sklearn's ColumnTransformer:
    1. Direct match:  'age'            → raw_values['age']
    2. CT prefix:     'num__age'       → raw_values['age']  (strip prefix)
    3. OHE expansion: 'cat__sex_male'  → raw_values['sex']  (original column)
       For OHE the *original column value* is returned ('male'), not the 0/1
       indicator, so the user sees the clinically meaningful category.

    Uses progressively shorter prefixes so multi-word category suffixes like
    'cat__diagnosis_high_blood_pressure' correctly resolve to 'diagnosis'.
    """
    # 1. Direct match
    if prep_name in raw_values:
        return raw_values[prep_name]

    # 2. Strip ColumnTransformer prefix (num__, cat__, remainder__, …)
    base = prep_name.split("__", 1)[-1] if "__" in prep_name else prep_name
    if base in raw_values:
        return raw_values[base]

    # 3. OHE: try progressively shorter prefixes (longest match wins)
    parts = base.split("_")
    for n in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:n])
        if candidate in raw_values:
            return raw_values[candidate]

    return None


def compute_local_lime(
    fitted_pipe: Any,
    row: pd.DataFrame,
    *,
    task_type: str = "classification",
    background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    num_samples: int = 1000,
    random_state: int = 42,
    positive_label: Optional[Any] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Compute LIME values for a single row.

    Parameters
    ----------
    fitted_pipe:
        Full fitted sklearn Pipeline (align → prep → model).
    row:
        A 1-row DataFrame in the same format as training input.
    task_type:
        "classification" | "regression"
    background:
        Pre-computed background array in preprocessed feature space — same
        artefact used by local SHAP. When None, falls back to a single-row
        background (degraded but functional).
    feature_names:
        Original feature names for display. Defaults to row.columns.
    num_samples:
        Number of perturbation samples LIME draws around the row. 1000 is
        LIME's default; lower values are faster but noisier.
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    List of dicts sorted by |contribution| descending, or None on failure.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None

    if row is None or len(row) == 0:
        return None

    names = list(feature_names) if feature_names is not None else list(row.columns)

    try:
        estimator = extract_model_from_pipeline(fitted_pipe)

        # ── Preprocess the row through all steps except the final model ─────
        X_row_prep = _transform_except_model(fitted_pipe, row)
        if X_row_prep.ndim != 2 or X_row_prep.shape[0] == 0:
            return None

        # ── Background data for LIME's training_data parameter ──────────────
        # LIME uses this to learn feature distributions for perturbation.
        # When unavailable, fall back to the row itself (degraded but doesn't crash).
        if background is not None and len(background) >= 2:
            training_data = np.asarray(background, dtype=float)
        else:
            training_data = X_row_prep

        # Shape sanity check: training_data must match X_row_prep's column count
        if training_data.shape[1] != X_row_prep.shape[1]:
            return None

        # ── Feature names in preprocessed space (after OHE expansion) ───────
        shap_feature_names = _get_preprocessed_feature_names(fitted_pipe, names)
        if len(shap_feature_names) != X_row_prep.shape[1]:
            shap_feature_names = [f"f_{i}" for i in range(X_row_prep.shape[1])]

        # ── Build the LIME explainer ────────────────────────────────────────
        mode = "classification" if str(task_type).lower() == "classification" else "regression"

        # class_names: only meaningful for classification
        class_names: Optional[List[str]] = None
        if mode == "classification":
            classes = getattr(estimator, "classes_", None)
            if classes is not None:
                class_names = [str(c) for c in classes]

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=shap_feature_names,
            class_names=class_names,
            mode=mode,
            discretize_continuous=False,  # treat all features as continuous
            random_state=random_state,
        )

        # ── Pick the predict function ───────────────────────────────────────
        if mode == "classification" and hasattr(estimator, "predict_proba"):
            predict_fn = estimator.predict_proba
        else:
            predict_fn = estimator.predict

        # ── Explain the row ─────────────────────────────────────────────────
        # LIME expects a 1-D array; we pass the first (and only) row.
        instance = X_row_prep[0]

        # num_features = total columns so we get all weights, then we trim
        n_features_total = int(X_row_prep.shape[1])

        explain_kwargs: Dict[str, Any] = {
            "num_features": n_features_total,
            "num_samples": int(num_samples),
        }

        if mode == "classification":
            # For binary: explain the agreed positive class (same orientation as
            # SHAP via get_positive_class_index) so LIME and SHAP panels never
            # show opposite directions — not blindly index 1.
            # For multiclass: explain the predicted class.
            if class_names is not None and len(class_names) == 2:
                target_label = get_positive_class_index(fitted_pipe, positive_label)
            else:
                # Predicted class index for the row
                proba = predict_fn(X_row_prep)
                target_label = int(np.argmax(proba[0])) if proba is not None and proba.ndim == 2 else 0
            explain_kwargs["labels"] = (target_label,)

        explanation = explainer.explain_instance(instance, predict_fn, **explain_kwargs)

        # ── Extract weights as {feature_idx: weight} ────────────────────────
        # as_map() returns {label: [(feature_idx, weight), ...]}
        weights_map: Dict[int, float] = {}
        if mode == "classification":
            target_label = explain_kwargs["labels"][0]
            pairs = explanation.as_map().get(target_label, [])
        else:
            # Regression uses label 1 internally (lime convention)
            pairs = explanation.as_map().get(1, []) or explanation.as_map().get(0, [])

        for feat_idx, weight in pairs:
            weights_map[int(feat_idx)] = float(weight)

        # ── Build per-feature output, mapping back to original raw values ───
        raw_values = row.iloc[0].to_dict()
        items: List[Dict[str, Any]] = []
        for i, fname in enumerate(shap_feature_names):
            weight = weights_map.get(i, 0.0)
            # Use the robust mapper so OHE features (cat__sex_male) resolve to
            # the original column value ('male') rather than returning None.
            raw_val = _map_prep_to_raw_value(fname, raw_values)
            items.append({
                "feature":      str(fname),
                "contribution": float(to_json_safe(weight)) if to_json_safe(weight) is not None else 0.0,
                "data":         to_json_safe(raw_val),
            })

        items.sort(key=lambda d: abs(float(d["contribution"])), reverse=True)
        return items[:_MAX_FEATURES]

    except Exception:
        return None
