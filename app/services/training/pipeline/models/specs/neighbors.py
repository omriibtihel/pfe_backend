"""
Nearest-neighbour spec: KNN.
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import randint

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.services.training.pipeline.models.registry import ModelSpec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _knn_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return KNeighborsClassifier(**params) if task_type == "classification" else KNeighborsRegressor(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="knn",
            aliases=(),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=False,
            supports_predict_proba=True,
            default_params={
                "classification": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "minkowski",
                },
                "regression": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "minkowski",
                },
            },
            param_grid={
                "classification": {
                    "n_neighbors": [3, 5, 9, 15],
                    "weights": ["uniform", "distance"],
                },
                "regression": {
                    "n_neighbors": [3, 5, 9, 15],
                    "weights": ["uniform", "distance"],
                },
            },
            param_distributions={
                "classification": {
                    "n_neighbors": randint(3, 25),
                    "weights":     ["uniform", "distance"],
                    "metric":      ["euclidean", "manhattan", "minkowski"],
                    "p":           [1, 2, 3],
                },
                "regression": {
                    "n_neighbors": randint(3, 25),
                    "weights":     ["uniform", "distance"],
                    "metric":      ["euclidean", "manhattan", "minkowski"],
                    "p":           [1, 2, 3],
                },
            },
            estimator_factory=_knn_factory,
        ),
    ]
