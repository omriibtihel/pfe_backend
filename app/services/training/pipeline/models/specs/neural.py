"""
Neural network spec: MLP (Multi-Layer Perceptron).

Includes the _StrLayersMLP mixin that accepts hidden_layer_sizes as a string
(e.g. "100,50") so that GridSearchCV can set it without sklearn iterating over
individual characters. Conversion to tuple happens in fit().
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import loguniform

from sklearn.neural_network import MLPClassifier, MLPRegressor

from app.services.training.pipeline.models.registry import ModelSpec


# ---------------------------------------------------------------------------
# String → tuple conversion for hidden_layer_sizes
# ---------------------------------------------------------------------------

def _parse_hidden_layer_sizes(hls: Any) -> tuple:
    if isinstance(hls, tuple):
        return hls
    if isinstance(hls, (int, float)) and not isinstance(hls, bool):
        return (int(hls),)
    if isinstance(hls, str):
        parts = [p.strip() for p in hls.split(",") if p.strip()]
        try:
            parsed = tuple(int(p) for p in parts)
        except ValueError:
            return (100,)
        return parsed if parsed else (100,)
    return (100,)


class _StrLayersMLP:
    """Mixin: converts hidden_layer_sizes string → tuple before sklearn's fit."""

    def fit(self, X: Any, y: Any, **kw: Any) -> Any:
        self.hidden_layer_sizes = _parse_hidden_layer_sizes(self.hidden_layer_sizes)
        return super().fit(X, y, **kw)  # type: ignore[misc]


class _MLPClassifier(_StrLayersMLP, MLPClassifier):  # type: ignore[misc]
    pass


class _MLPRegressor(_StrLayersMLP, MLPRegressor):  # type: ignore[misc]
    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _mlp_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        return _MLPClassifier(**params)
    return _MLPRegressor(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="mlp",
            aliases=("neural_network", "nn"),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=False,
            supports_predict_proba=True,
            supports_sample_weight=False,
            default_params={
                "classification": {
                    "hidden_layer_sizes": "100",
                    "activation": "relu",
                    "alpha": 0.0001,
                    "learning_rate_init": 0.001,
                    "max_iter": 500,
                    "random_state": 42,
                },
                "regression": {
                    "hidden_layer_sizes": "100",
                    "activation": "relu",
                    "alpha": 0.0001,
                    "learning_rate_init": 0.001,
                    "max_iter": 500,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "hidden_layer_sizes": ["100", "100,50"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "activation": ["relu", "tanh"],
                },
                "regression": {
                    "hidden_layer_sizes": ["100", "100,50"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "activation": ["relu", "tanh"],
                },
            },
            param_distributions={
                "classification": {
                    "hidden_layer_sizes": ["100", "100,50", "50,50,50", "200", "200,100", "64,32"],
                    "alpha": loguniform(1e-5, 1e-1),
                    "activation": ["relu", "tanh"],
                    "learning_rate_init": loguniform(1e-4, 1e-1),
                },
                "regression": {
                    "hidden_layer_sizes": ["100", "100,50", "50,50,50", "200", "200,100", "64,32"],
                    "alpha": loguniform(1e-5, 1e-1),
                    "activation": ["relu", "tanh"],
                    "learning_rate_init": loguniform(1e-4, 1e-1),
                },
            },
            estimator_factory=_mlp_factory,
        ),
    ]
