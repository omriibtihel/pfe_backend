"""
Internal utilities for the SHAP module.

Responsibilities:
  - Extract the raw estimator from a sklearn Pipeline
  - Detect which SHAP explainer type to use for a given estimator
  - Safe background-data sampler (kmeans or random subset)
  - JSON-safe conversion for SHAP values
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# ── Estimator class → explainer routing ──────────────────────────────────────

# Models that support TreeExplainer (exact, fast)
_TREE_CLASSES = (
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
)

# Models that support LinearExplainer (exact, fast)
_LINEAR_CLASSES = (
    "LogisticRegression",
    "Ridge",
    "LinearRegression",
    "Lasso",
    "ElasticNet",
    "SGDClassifier",
    "SGDRegressor",
)

# All other estimators fall back to KernelExplainer (slow, sampled)


def extract_model_from_pipeline(pipeline: Any) -> Any:
    """
    Return the raw estimator from the last step of a sklearn Pipeline.

    Works with:
    - sklearn Pipeline  (named_steps["model"])
    - FLAML AutoML      (pipeline.model.estimator)
    - Bare estimator    (returned as-is)
    """
    # sklearn Pipeline
    named = getattr(pipeline, "named_steps", None)
    if named is not None:
        # The last step is always the model; try "model" key first
        model = named.get("model")
        if model is not None:
            return model
        # Fallback: return the last step value
        return list(named.values())[-1]

    # FLAML AutoML wraps estimator in pipeline.model.estimator
    flaml_model = getattr(pipeline, "model", None)
    if flaml_model is not None:
        inner = getattr(flaml_model, "estimator", None)
        if inner is not None:
            return inner

    return pipeline


def get_explainer_type(estimator: Any) -> str:
    """
    Return "tree", "linear", or "kernel" based on estimator class name.
    """
    cls_name = type(estimator).__name__
    if cls_name in _TREE_CLASSES:
        return "tree"
    if cls_name in _LINEAR_CLASSES:
        return "linear"
    return "kernel"


def safe_background(
    X: np.ndarray,
    *,
    max_samples: int = 50,
    use_kmeans: bool = True,
) -> Any:
    """
    Build a background dataset for KernelExplainer.

    - When use_kmeans=True: uses shap.kmeans(X, k) — weighted representative points.
    - Falls back to a random subsample when kmeans fails or use_kmeans=False.
    - Always caps at max_samples rows.
    """
    n = len(X)
    k = min(max_samples, n)

    if use_kmeans:
        try:
            import shap as _shap
            return _shap.kmeans(X, k)
        except Exception:
            pass

    # Fallback: random subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=k, replace=False)
    return X[idx]


def shap_values_to_float_array(
    shap_vals: Any,
    *,
    binary_class_index: int = 1,
    predicted_class_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Normalise SHAP output to a 2-D float array: (n_samples, n_features).

    Different explainers return values in different formats:
    - TreeExplainer binary classification: list of 2 arrays → take binary_class_index
    - TreeExplainer multiclass: list of C arrays
      * When predicted_class_indices is provided: take per-sample predicted class
        (clinically correct — shows why THIS sample was classified as class X)
      * Fallback: mean absolute across classes (overall importance view)
    - LinearExplainer: already (n, p) 2-D
    - KernelExplainer: already (n, p) 2-D (for binary) or list
    """
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            # Binary: return the agreed positive class
            return np.asarray(shap_vals[binary_class_index], dtype=float)
        if len(shap_vals) > 2:
            if predicted_class_indices is not None:
                # Per-sample: extract SHAP values for the predicted class.
                # stacked: (n_classes, n_samples, n_features)
                stacked = np.stack([np.asarray(v, dtype=float) for v in shap_vals], axis=0)
                n_samples = stacked.shape[1]
                indices = np.asarray(predicted_class_indices, dtype=int)
                return np.array([stacked[indices[i], i, :] for i in range(n_samples)])
            # Default: mean absolute across classes (global feature importance view)
            stacked = np.stack([np.abs(np.asarray(v, dtype=float)) for v in shap_vals], axis=0)
            return stacked.mean(axis=0)
        if len(shap_vals) == 1:
            return np.asarray(shap_vals[0], dtype=float)
        return np.array([], dtype=float)

    arr = np.asarray(shap_vals, dtype=float)

    # 3-D (n, p, C): take positive class or mean abs
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[:, :, binary_class_index]
        if predicted_class_indices is not None:
            indices = np.asarray(predicted_class_indices, dtype=int)
            return np.array([arr[i, :, indices[i]] for i in range(arr.shape[0])])
        return np.abs(arr).mean(axis=2)

    return arr  # (n, p) already


def get_positive_class_index(pipeline_or_estimator: Any, positive_label: Any) -> int:
    """Return the column index of positive_label in the fitted model's classes_ array.

    Used so that SHAP values always correspond to the class that was designated
    as "positive" at training time, regardless of sklearn's internal sort order.

    Falls back to 1 (sklearn default for {0,1}) when:
    - positive_label is None  (standard {0,1} encoding — class 1 is always positive)
    - positive_label is not found in classes_
    - classes_ is absent or not exactly 2 classes
    """
    if positive_label is None:
        return 1

    named = getattr(pipeline_or_estimator, "named_steps", None)
    if named is not None:
        estimator = named.get("model") or list(named.values())[-1]
    else:
        estimator = pipeline_or_estimator

    classes_ = getattr(estimator, "classes_", None)
    if classes_ is None or len(classes_) != 2:
        return 1

    pos_str = str(positive_label)
    for i, label in enumerate(classes_):
        if str(label) == pos_str:
            return i
    return 1


def to_json_safe(v: Any) -> Any:
    """Convert numpy scalars / nan / inf to JSON-safe Python primitives."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (v != v or abs(v) == float("inf")):
        return None
    return v
