"""
Naive Bayes spec: GaussianNB.
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import loguniform

from sklearn.naive_bayes import GaussianNB

from app.services.training.pipeline.models.registry import ModelSpec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _naivebayes_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "classification":
        raise RuntimeError("naivebayes is classification-only")
    return GaussianNB(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="naivebayes",
            aliases=("nb", "gaussiannb"),
            supports_classification=True,
            supports_regression=False,
            supports_class_weight=False,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "var_smoothing": 1e-9,
                },
            },
            param_grid={
                "classification": {
                    "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
                },
            },
            param_distributions={
                "classification": {
                    "var_smoothing": loguniform(1e-12, 1e-2),
                },
            },
            estimator_factory=_naivebayes_factory,
        ),
    ]
