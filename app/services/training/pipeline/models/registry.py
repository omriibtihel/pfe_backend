"""
ModelSpec, ModelRegistry, adaptive grids, and all public helper functions.

MODEL_REGISTRY is a module-level singleton created here (empty).
It is populated by pipeline/models/__init__.py which registers
specs from each specs/ submodule in canonical order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Log-uniform discrete distribution (for n_estimators etc.)
# ---------------------------------------------------------------------------

def log_randint(low: int, high: int):
    """Log-uniform discrete distribution between low and high (inclusive)."""
    class LogRandInt:
        def __init__(self, lo: int, hi: int) -> None:
            self.lo = lo
            self.hi = hi

        def rvs(self, size=None, random_state=None):
            if isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
            else:
                rng = random_state if random_state is not None else np.random
            log_lo, log_hi = np.log(self.lo), np.log(self.hi)
            values = np.exp(rng.uniform(log_lo, log_hi, size=size))
            return np.clip(np.round(values).astype(int), self.lo, self.hi)

    return LogRandInt(low, high)


# ---------------------------------------------------------------------------
# Search profiles
# ---------------------------------------------------------------------------

SEARCH_PROFILES: Dict[str, Dict[str, int]] = {
    "fast":     {"n_iter": 15, "cv": 3},
    "balanced": {"n_iter": 40, "cv": 5},
    "thorough": {"n_iter": 80, "cv": 5},
}
DEFAULT_SEARCH_LEVEL = "balanced"


# ---------------------------------------------------------------------------
# ModelSpec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: tuple[str, ...]
    supports_classification: bool
    supports_regression: bool
    supports_class_weight: bool
    supports_predict_proba: bool
    default_params: Dict[str, Dict[str, Any]]
    param_grid: Dict[str, Dict[str, list[Any]]]
    estimator_factory: Callable[[str, Dict[str, Any]], Any]
    param_distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    supports_sample_weight: bool = False

    def supports(self, task_type: str) -> bool:
        if task_type == "classification":
            return self.supports_classification
        if task_type == "regression":
            return self.supports_regression
        return False

    def params_for(self, task_type: str) -> Dict[str, Any]:
        return dict(self.default_params.get(task_type, {}))

    def grid_for(self, task_type: str) -> Dict[str, list[Any]]:
        return dict(self.param_grid.get(task_type, {}))

    def distributions_for(self, task_type: str) -> Dict[str, Any]:
        return dict(self.param_distributions.get(task_type, {}))


# ---------------------------------------------------------------------------
# ModelRegistry — lean (no _build_defaults; populated by __init__.py)
# ---------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}
        self._aliases: Dict[str, str] = {}

    def _register(self, spec: ModelSpec) -> None:
        self._specs[spec.name] = spec
        for alias in (spec.name, *spec.aliases):
            self._aliases[alias] = spec.name

    def normalize_name(self, model_type: str) -> str:
        key = str(model_type or "").strip().lower()
        if key in self._aliases:
            return self._aliases[key]
        return key

    def get_spec(self, model_type: str, task_type: str) -> ModelSpec:
        key = self.normalize_name(model_type)
        spec = self._specs.get(key)
        if spec is None:
            raise RuntimeError(f"Unknown model type: {model_type}")
        if not spec.supports(task_type):
            raise RuntimeError(f"Model '{model_type}' does not support task '{task_type}'")
        return spec

    def make_estimator(self, model_type: str, task_type: str, overrides: Dict[str, Any] | None = None) -> Any:
        spec = self.get_spec(model_type, task_type)
        params = spec.params_for(task_type)
        if overrides:
            params.update(overrides)
        return spec.estimator_factory(task_type, params)

    def model_grid(self, model_type: str, task_type: str) -> Dict[str, list[Any]]:
        spec = self.get_spec(model_type, task_type)
        return spec.grid_for(task_type)

    def model_distributions(self, model_type: str, task_type: str) -> Dict[str, Any]:
        spec = self.get_spec(model_type, task_type)
        return spec.distributions_for(task_type)


# ---------------------------------------------------------------------------
# Singleton — populated by pipeline/models/__init__.py
# ---------------------------------------------------------------------------

MODEL_REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# Adaptive parameter grids (dataset-aware: size + imbalance)
# ---------------------------------------------------------------------------

_ADAPTIVE_GRIDS: Dict[str, Dict[str, Dict[str, Dict[str, list]]]] = {
    "randomforest": {
        "classification": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 4], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
        "regression": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [2, 5], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
    },
    "logisticregression": {
        "classification": {
            "micro":  {"C": [0.01, 0.1, 1.0], "l1_ratio": [0, 1]},
            "tiny":   {"C": [0.1, 1.0, 10.0], "l1_ratio": [0, 1]},
            "small":  {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "saga"]},
            "medium": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "saga"], "l1_ratio": [0, 1]},
            "large":  {"C": [0.1, 1.0, 10.0]},
        },
    },
    "svm": {
        "classification": {
            "micro":  {"C": [0.01, 0.1, 1.0]},
            "tiny":   {"C": [0.1, 1.0, 10.0]},
            "small":  {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
            "medium": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
            "large":  {"C": [0.1, 1.0], "kernel": ["rbf", "linear"]},
        },
        "regression": {
            "micro":  {"C": [0.01, 0.1, 1.0]},
            "tiny":   {"C": [0.1, 1.0, 10.0]},
            "small":  {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
            "medium": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1, 0.5]},
            "large":  {"C": [0.1, 1.0], "kernel": ["rbf", "linear"], "epsilon": [0.1, 0.5]},
        },
    },
    "knn": {
        "classification": {
            "micro":  {"n_neighbors": [3, 5, 7]},
            "tiny":   {"n_neighbors": [3, 5, 9]},
            "small":  {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
            "medium": {"n_neighbors": [3, 5, 9, 15, 21], "weights": ["uniform", "distance"]},
            "large":  {"n_neighbors": [5, 9, 15]},
        },
        "regression": {
            "micro":  {"n_neighbors": [3, 5, 7]},
            "tiny":   {"n_neighbors": [3, 5, 9]},
            "small":  {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
            "medium": {"n_neighbors": [3, 5, 9, 15, 21], "weights": ["uniform", "distance"]},
            "large":  {"n_neighbors": [5, 9, 15]},
        },
    },
    "naivebayes": {
        "classification": {
            "micro":  {"var_smoothing": [1e-9, 1e-7, 1e-5]},
            "tiny":   {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "small":  {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "medium": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "large":  {"var_smoothing": [1e-9, 1e-7, 1e-5]},
        },
    },
    "decisiontree": {
        "classification": {
            "micro":  {"max_depth": [2, 3], "min_samples_leaf": [4, 8]},
            "tiny":   {"max_depth": [3, 7, None]},
            "small":  {"max_depth": [3, 7, 15, None], "min_samples_leaf": [1, 4]},
            "medium": {"max_depth": [3, 7, 15, None], "min_samples_leaf": [1, 4, 10], "criterion": ["gini", "entropy"]},
            "large":  {"max_depth": [3, 7, None], "min_samples_leaf": [1, 4]},
        },
        "regression": {
            "micro":  {"max_depth": [2, 3], "min_samples_leaf": [4, 8]},
            "tiny":   {"max_depth": [3, 7, None]},
            "small":  {"max_depth": [3, 7, 15, None], "min_samples_leaf": [2, 5]},
            "medium": {"max_depth": [3, 7, 15, None], "min_samples_leaf": [2, 5, 10]},
            "large":  {"max_depth": [3, 7, None], "min_samples_leaf": [2, 5]},
        },
    },
    "xgboost": {
        "classification": {
            "micro":  {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            "small":  {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [3, 5, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 3]},
            "medium": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3, 5, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 3], "reg_alpha": [0, 0.1, 1.0]},
            "large":  {"n_estimators": [200, 400], "learning_rate": [0.1, 0.2], "max_depth": [3, 6], "min_child_weight": [1, 5]},
        },
        "regression": {
            "micro":  {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            "small":  {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [3, 5, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 3]},
            "medium": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3, 5, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 3], "reg_alpha": [0, 0.1, 1.0]},
            "large":  {"n_estimators": [200, 400], "learning_rate": [0.1, 0.2], "max_depth": [3, 6], "min_child_weight": [1, 5]},
        },
    },
    "lightgbm": {
        "classification": {
            "micro":  {"learning_rate": [0.05, 0.1], "num_leaves": [7, 15]},
            "tiny":   {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31], "reg_lambda": [0.1, 1.0]},
            "small":  {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63], "reg_lambda": [0.1, 1.0]},
            "medium": {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63, 127], "min_child_samples": [10, 30], "reg_alpha": [0, 0.1], "reg_lambda": [0.1, 1.0]},
            "large":  {"n_estimators": [300, 500], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63], "reg_lambda": [0.1, 1.0]},
        },
        "regression": {
            "micro":  {"learning_rate": [0.05, 0.1], "num_leaves": [7, 15]},
            "tiny":   {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31], "reg_lambda": [0.1, 1.0]},
            "small":  {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63], "reg_lambda": [0.1, 1.0]},
            "medium": {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63, 127], "min_child_samples": [10, 30], "reg_alpha": [0, 0.1], "reg_lambda": [0.1, 1.0]},
            "large":  {"n_estimators": [300, 500], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63], "reg_lambda": [0.1, 1.0]},
        },
    },
    "extratrees": {
        "classification": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 4], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
        "regression": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [2, 5], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
    },
    "gradientboosting": {
        "classification": {
            "micro":  {"learning_rate": [0.05, 0.1], "min_samples_leaf": [8, 16]},
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "min_samples_leaf": [1, 4]},
            "small":  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "min_samples_leaf": [1, 4]},
            "medium": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "subsample": [0.7, 1.0], "min_samples_leaf": [1, 4]},
            "large":  {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [3], "min_samples_leaf": [1, 4]},
        },
        "regression": {
            "micro":  {"learning_rate": [0.05, 0.1], "min_samples_leaf": [8, 16]},
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "min_samples_leaf": [1, 4]},
            "small":  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "min_samples_leaf": [1, 4]},
            "medium": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "subsample": [0.7, 1.0], "min_samples_leaf": [1, 4]},
            "large":  {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [3], "min_samples_leaf": [1, 4]},
        },
    },
    "catboost": {
        "classification": {
            "micro":  {"learning_rate": [0.05, 0.1], "l2_leaf_reg": [3.0, 10.0]},
            "tiny":   {"iterations": [200, 400], "learning_rate": [0.05, 0.1]},
            "small":  {"iterations": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6]},
            "medium": {"iterations": [200, 500], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8], "l2_leaf_reg": [1.0, 3.0, 10.0]},
            "large":  {"iterations": [300, 500], "learning_rate": [0.05, 0.1]},
        },
        "regression": {
            "micro":  {"learning_rate": [0.05, 0.1], "l2_leaf_reg": [3.0, 10.0]},
            "tiny":   {"iterations": [200, 400], "learning_rate": [0.05, 0.1]},
            "small":  {"iterations": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6]},
            "medium": {"iterations": [200, 500], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8], "l2_leaf_reg": [1.0, 3.0, 10.0]},
            "large":  {"iterations": [300, 500], "learning_rate": [0.05, 0.1]},
        },
    },
    "ridge": {
        "regression": {
            "micro":  {"alpha": [1.0, 10.0, 100.0]},
            "tiny":   {"alpha": [0.1, 1.0, 10.0]},
            "small":  {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "medium": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "large":  {"alpha": [0.1, 1.0, 10.0, 100.0]},
        },
    },
    "mlp": {
        "classification": {
            "micro":  {"alpha": [0.01, 0.1, 1.0]},
            "tiny":   {"alpha": [0.001, 0.01, 0.1], "activation": ["relu", "tanh"]},
            "small":  {"alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            "medium": {"hidden_layer_sizes": ["100", "100,50"], "alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            "large":  {"hidden_layer_sizes": ["100", "100,50"], "alpha": [0.0001, 0.001]},
        },
        "regression": {
            "micro":  {"alpha": [0.01, 0.1, 1.0]},
            "tiny":   {"alpha": [0.001, 0.01, 0.1], "activation": ["relu", "tanh"]},
            "small":  {"alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            "medium": {"hidden_layer_sizes": ["100", "100,50"], "alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            "large":  {"hidden_layer_sizes": ["100", "100,50"], "alpha": [0.0001, 0.001]},
        },
    },
    "elasticnet": {
        "regression": {
            "micro":  {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
            "tiny":   {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
            "small":  {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.15, 0.5, 0.85]},
            "medium": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "large":  {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
        },
    },
    "lasso": {
        "regression": {
            "micro":  {"alpha": [0.1, 1.0, 10.0, 100.0]},
            "tiny":   {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "small":  {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
            "medium": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            "large":  {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        },
    },
}

# Models that support class_weight — used to inject it in grids for imbalanced data
_CLASS_WEIGHT_MODELS = frozenset({"randomforest", "logisticregression", "svm", "decisiontree", "extratrees"})

# n_iter budget for RandomizedSearchCV per dataset size
_RANDOM_SEARCH_N_ITER: Dict[str, int] = {
    "micro":  10,
    "tiny":   25,
    "small":  40,
    "medium": 60,
    "large":  80,
}

# Initial candidate count for HalvingRandomSearchCV per dataset size
_HALVING_N_CANDIDATES: Dict[str, int] = {
    "micro":  15,
    "tiny":   30,
    "small":  60,
    "medium": 100,
    "large":  50,
}


def _n_samples_to_size_cat(n_samples: int) -> str:
    if n_samples < 100:
        return "micro"
    if n_samples < 500:
        return "tiny"
    if n_samples < 2_000:
        return "small"
    if n_samples < 50_000:
        return "medium"
    return "large"


def get_adaptive_model_grid(
    model_type: str,
    task_type: str,
    *,
    n_samples: int,
    imbalanced: bool = False,
) -> Dict[str, list]:
    """
    Return a context-aware param grid for GridSearchCV.

    Grid size scales with n_samples; class_weight injected for imbalanced
    classification with supporting models. Falls back to the static
    ModelSpec grid when no adaptive grid is defined.
    """
    model_key = MODEL_REGISTRY.normalize_name(model_type)
    size_cat = _n_samples_to_size_cat(n_samples)

    adaptive = _ADAPTIVE_GRIDS.get(model_key, {}).get(task_type, {}).get(size_cat)
    if adaptive is None:
        try:
            grid = dict(MODEL_REGISTRY.model_grid(model_key, task_type))
        except RuntimeError:
            return {}
    else:
        grid = dict(adaptive)

    if (
        imbalanced
        and task_type == "classification"
        and model_key in _CLASS_WEIGHT_MODELS
        and "class_weight" not in grid
    ):
        grid["class_weight"] = ["balanced", None]

    return grid


def get_adaptive_n_iter(n_samples: int) -> int:
    """Return a sensible n_iter for RandomizedSearchCV based on dataset size."""
    return _RANDOM_SEARCH_N_ITER[_n_samples_to_size_cat(n_samples)]


def get_halving_n_candidates(n_samples: int) -> int:
    """Return the initial candidate count for HalvingRandomSearchCV based on dataset size."""
    return _HALVING_N_CANDIDATES[_n_samples_to_size_cat(n_samples)]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_model(model_key: str, task_type: str, hyperparams_clean: Dict[str, Any] | None = None) -> Any:
    model_name = MODEL_REGISTRY.normalize_name(model_key)
    overrides = dict(hyperparams_clean or {})

    if model_name == "decisiontree" and task_type == "regression":
        criterion = overrides.get("criterion")
        if isinstance(criterion, str):
            allowed_regression = {"squared_error", "friedman_mse", "absolute_error", "poisson"}
            if criterion not in allowed_regression:
                overrides.pop("criterion", None)

    return MODEL_REGISTRY.make_estimator(model_name, task_type, overrides=overrides)


def make_estimator(model_type: str, task_type: str, overrides: Dict[str, Any] | None = None) -> Any:
    # Backward-compatible alias.
    return build_model(model_type, task_type, hyperparams_clean=overrides)


def get_model_spec(model_type: str, task_type: str) -> ModelSpec:
    return MODEL_REGISTRY.get_spec(model_type, task_type)


def get_model_grid(model_type: str, task_type: str) -> Dict[str, list[Any]]:
    return MODEL_REGISTRY.model_grid(model_type, task_type)


def get_model_distributions(model_type: str, task_type: str) -> Dict[str, Any]:
    return MODEL_REGISTRY.model_distributions(model_type, task_type)


def model_is_installed(name: str) -> bool:
    key = str(name or "").strip().lower()
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor  # noqa: F401
            return True
        except Exception:
            return False
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # noqa: F401
            return True
        except Exception:
            return False
    return True


def get_model_capabilities(model_type: str) -> Dict[str, bool]:
    """Return static capability flags for a model without instantiating it."""
    key = MODEL_REGISTRY.normalize_name(model_type)
    spec = MODEL_REGISTRY._specs.get(key)
    if spec is None:
        return {
            "supports_class_weight": False,
            "supports_predict_proba": False,
            "supports_sample_weight": False,
        }
    return {
        "supports_class_weight": spec.supports_class_weight,
        "supports_predict_proba": spec.supports_predict_proba,
        "supports_sample_weight": spec.supports_sample_weight,
    }


def list_available_models() -> list[Dict[str, Any]]:
    labels = {
        "randomforest": "Random Forest",
        "logisticregression": "Logistic Regression",
        "svm": "SVM",
        "knn": "KNN",
        "naivebayes": "Naive Bayes",
        "decisiontree": "Decision Tree",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "ridge": "Ridge Regression",
        "extratrees": "Extra Trees",
        "gradientboosting": "Gradient Boosting",
        "catboost": "CatBoost",
        "mlp": "MLP (Réseau de Neurones)",
        "elasticnet": "ElasticNet",
        "lasso": "Lasso",
    }

    out: list[Dict[str, Any]] = []
    for name, spec in MODEL_REGISTRY._specs.items():
        tasks: list[str] = []
        if spec.supports_classification:
            tasks.append("classification")
        if spec.supports_regression:
            tasks.append("regression")
        out.append(
            {
                "key": name,
                "name": name,
                "label": labels.get(name, name),
                "aliases": list(spec.aliases),
                "tasks": tasks,
                "installed": model_is_installed(name),
            }
        )
    return out


def get_model_config(model_key: str, task: str = "classification") -> ModelSpec | None:
    """Return the ModelSpec for a given model key and task, or None if not found."""
    try:
        return MODEL_REGISTRY.get_spec(model_key, task)
    except RuntimeError:
        return None


def get_search_params(
    model_key: str,
    task: str = "classification",
    search_level: str = "balanced",
    method: str = "randomized",
) -> Dict[str, Any]:
    """Return the parameter space and search config for a model."""
    spec = get_model_config(model_key, task)
    if spec is None:
        raise ValueError(f"Unknown model: {model_key} for task={task}")
    profile = SEARCH_PROFILES.get(search_level, SEARCH_PROFILES[DEFAULT_SEARCH_LEVEL])
    if method == "grid":
        return {
            "param_space": spec.grid_for(task),
            "n_iter": None,
            "cv": profile["cv"],
            "method": "grid",
        }
    return {
        "param_space": spec.distributions_for(task),
        "n_iter": profile["n_iter"],
        "cv": profile["cv"],
        "method": "randomized",
    }
