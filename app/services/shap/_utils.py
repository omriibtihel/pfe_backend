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


def shap_values_to_float_array(shap_vals: Any, *, binary_class_index: int = 1) -> np.ndarray:
    """
    Normalise SHAP output to a 2-D float array: (n_samples, n_features).

    Different explainers return values in different formats:
    - TreeExplainer binary classification: list of 2 arrays → take index 1
    - TreeExplainer multiclass: list of C arrays → stack and average abs
    - LinearExplainer: already (n, p) 2-D
    - KernelExplainer: already (n, p) 2-D (for binary) or list
    """
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            # Binary: return positive class
            return np.asarray(shap_vals[binary_class_index], dtype=float)
        if len(shap_vals) > 2:
            # Multiclass: mean absolute across classes
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
        return np.abs(arr).mean(axis=2)

    return arr  # (n, p) already


def to_json_safe(v: Any) -> Any:
    """Convert numpy scalars / nan / inf to JSON-safe Python primitives."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (v != v or abs(v) == float("inf")):
        return None
    return v
