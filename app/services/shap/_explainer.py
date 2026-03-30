"""
ExplainerFactory — builds the right shap.Explainer for any estimator.

Routing strategy:
  tree   → shap.TreeExplainer(estimator)
  linear → shap.LinearExplainer(estimator, background)
  kernel → shap.KernelExplainer(predict_fn, background)

All methods return (explainer, explainer_type_str) or raise on failure.
Callers should wrap in try/except and treat exceptions as "SHAP unavailable".
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from app.services.shap._utils import get_explainer_type, safe_background


# Maximum background samples for KernelExplainer (performance guard)
_KERNEL_BACKGROUND_K = 50


def build_explainer(
    estimator: Any,
    X_background: np.ndarray,
    *,
    task_type: str = "classification",
) -> Tuple[Any, str]:
    """
    Build a SHAP explainer appropriate for `estimator`.

    Parameters
    ----------
    estimator:
        Raw sklearn/XGB/LGBM estimator (NOT a Pipeline — extract first).
    X_background:
        Background data as a numpy array.  Used by LinearExplainer and
        KernelExplainer.  TreeExplainer ignores it.
    task_type:
        "classification" | "regression" — used to select predict function
        for KernelExplainer.

    Returns
    -------
    (explainer, explainer_type)  where explainer_type ∈ {"tree","linear","kernel"}
    """
    import shap  # lazy import — not installed in all environments

    explainer_type = get_explainer_type(estimator)

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(estimator)
        return explainer, "tree"

    if explainer_type == "linear":
        background = safe_background(X_background, max_samples=_KERNEL_BACKGROUND_K, use_kmeans=False)
        explainer = shap.LinearExplainer(estimator, background)
        return explainer, "linear"

    # KernelExplainer — model-agnostic but slow
    background = safe_background(X_background, max_samples=_KERNEL_BACKGROUND_K, use_kmeans=True)

    # Pick the right predict function
    if task_type == "classification" and hasattr(estimator, "predict_proba"):
        predict_fn = estimator.predict_proba
    else:
        predict_fn = estimator.predict

    explainer = shap.KernelExplainer(predict_fn, background)
    return explainer, "kernel"
