"""
ZeroShotConfig — Phase 3
Registry of fast, reasonable starter hyperparameter configurations
per dataset scenario.  These are used by the RecommendationEngine to
pre-fill modelHyperparams so the very first training run is useful
without any HPO budget.
"""
from __future__ import annotations

from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Registry: scenario_key → {model_name: {param: value}}
# ──────────────────────────────────────────────────────────────────────────────

_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    # ── tiny / small (< 2 000 samples) ────────────────────────────────────────
    "small_balanced_binary": {
        "logisticregression": {"C": 1.0, "max_iter": 2000, "solver": "saga", "l1_ratio": 0.0},
        "randomforest": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
        "lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    },
    "small_imbalanced_binary": {
        "logisticregression": {"C": 0.5, "max_iter": 3000, "solver": "saga", "l1_ratio": 0.0, "class_weight": "balanced"},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "feature_fraction": 0.8, "bagging_fraction": 0.8},
        "randomforest": {"n_estimators": 150, "max_depth": 10, "class_weight": "balanced"},
    },
    "small_balanced_multiclass": {
        "logisticregression": {"C": 1.0, "max_iter": 3000, "solver": "saga", "l1_ratio": 0.0},
        "randomforest": {"n_estimators": 150, "max_depth": None, "max_features": "sqrt"},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 20},
    },
    "small_imbalanced_multiclass": {
        "logisticregression": {"C": 0.5, "max_iter": 4000, "solver": "saga", "l1_ratio": 0.0, "class_weight": "balanced"},
        "randomforest": {"n_estimators": 150, "max_depth": 10, "class_weight": "balanced"},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 20},
    },
    # ── medium (2 000 – 50 000 samples) ───────────────────────────────────────
    "medium_balanced_binary": {
        "logisticregression": {"C": 1.0, "max_iter": 2000, "solver": "saga", "l1_ratio": 0.0},
        "randomforest": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
        "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.9, "bagging_fraction": 0.9},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    },
    "medium_imbalanced_binary": {
        "logisticregression": {"C": 0.5, "max_iter": 3000, "solver": "saga", "l1_ratio": 0.0, "class_weight": "balanced"},
        "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.9, "bagging_fraction": 0.9},
        "randomforest": {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    },
    "medium_balanced_multiclass": {
        "randomforest": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
        "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31},
        "logisticregression": {"C": 1.0, "max_iter": 4000, "solver": "saga", "l1_ratio": 0.0},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
    "medium_imbalanced_multiclass": {
        "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31},
        "randomforest": {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
        "logisticregression": {"C": 0.5, "max_iter": 4000, "solver": "saga", "class_weight": "balanced"},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
    # ── large (> 50 000 samples) ───────────────────────────────────────────────
    "large_binary": {
        "logisticregression": {"C": 0.5, "max_iter": 1000, "solver": "saga", "l1_ratio": 0.0},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    },
    "large_multiclass": {
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31},
        "logisticregression": {"C": 0.5, "max_iter": 1000, "solver": "saga"},
    },
    # ── regression ────────────────────────────────────────────────────────────
    "regression_small": {
        "ridge": {"alpha": 1.0},
        "randomforest": {"n_estimators": 100, "max_depth": None},
        "lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15},
    },
    "regression_medium": {
        "ridge": {"alpha": 1.0},
        "randomforest": {"n_estimators": 200, "max_depth": None},
        "lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
    },
    "regression_large": {
        "ridge": {"alpha": 1.0},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31},
    },
}


def _scenario_key(
    size_cat: str,
    task_type: str,
    imbalance_ratio: float | None,
) -> str:
    is_imbalanced = (imbalance_ratio is not None) and (imbalance_ratio > 3.0)
    balance_tag = "imbalanced" if is_imbalanced else "balanced"

    if task_type == "regression":
        return f"regression_{size_cat}"

    if task_type == "binary_classification":
        if size_cat == "large":
            return "large_binary"
        return f"{size_cat}_{balance_tag}_binary"

    # multiclass
    if size_cat == "large":
        return "large_multiclass"
    return f"{size_cat}_{balance_tag}_multiclass"


def get_zero_shot_hyperparams(
    size_cat: str,
    task_type: str,
    imbalance_ratio: float | None,
    models: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Return per-model starter hyperparameters for the given scenario.
    Only returns params for models that appear in both `models` and the registry.
    Models not in the registry keep empty params (defaults will be used).
    """
    key = _scenario_key(size_cat, task_type, imbalance_ratio)
    registry = _CONFIGS.get(key, {})
    result: dict[str, dict[str, Any]] = {}
    for m in models:
        result[m] = dict(registry.get(m, {}))
    return result
