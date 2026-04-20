"""
Tree-based model specs: RandomForest, ExtraTrees, DecisionTree, GradientBoosting.
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import randint, uniform

from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.training.pipeline.models.registry import ModelSpec, log_randint


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _rf_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)


def _extratrees_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return ExtraTreesClassifier(**params) if task_type == "classification" else ExtraTreesRegressor(**params)


def _dt_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return DecisionTreeClassifier(**params) if task_type == "classification" else DecisionTreeRegressor(**params)


def _gradientboosting_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return GradientBoostingClassifier(**params) if task_type == "classification" else GradientBoostingRegressor(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        # ---------------------------------------------------------------
        # Random Forest
        # ---------------------------------------------------------------
        ModelSpec(
            name="randomforest",
            aliases=("rf",),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=True,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "max_features": "sqrt",
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "max_features": "sqrt",
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "n_estimators": [100, 300],
                    "max_depth": [5, 10, None],
                    "min_samples_leaf": [1, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "regression": {
                    "n_estimators": [100, 300],
                    "max_depth": [5, 10, None],
                    "min_samples_leaf": [2, 5],
                    "max_features": ["sqrt", "log2"],
                },
            },
            param_distributions={
                "classification": {
                    "n_estimators":      log_randint(50, 500),
                    "max_depth":         [3, 5, 8, 12, 20, None],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    "bootstrap":         [True, False],
                    "class_weight":      ["balanced", "balanced_subsample", None],
                },
                "regression": {
                    "n_estimators":      log_randint(50, 500),
                    "max_depth":         [3, 5, 8, 12, 20, None],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    "bootstrap":         [True, False],
                },
            },
            estimator_factory=_rf_factory,
        ),
        # ---------------------------------------------------------------
        # Extra Trees
        # ---------------------------------------------------------------
        ModelSpec(
            name="extratrees",
            aliases=("et",),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=True,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "max_features": "sqrt",
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "max_features": "sqrt",
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "n_estimators":      [100, 300],
                    "max_depth":         [5, 10, None],
                    "min_samples_leaf":  [1, 4],
                    "min_samples_split": [2, 5],
                    "max_features":      ["sqrt", "log2"],
                },
                "regression": {
                    "n_estimators":      [100, 300],
                    "max_depth":         [5, 10, None],
                    "min_samples_leaf":  [2, 5],
                    "min_samples_split": [2, 5],
                    "max_features":      ["sqrt", "log2"],
                },
            },
            param_distributions={
                "classification": {
                    "n_estimators":      log_randint(50, 500),
                    "max_depth":         [3, 5, 8, 12, 20, None],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    "bootstrap":         [True, False],
                    "class_weight":      ["balanced", "balanced_subsample", None],
                },
                "regression": {
                    "n_estimators":      log_randint(50, 500),
                    "max_depth":         [3, 5, 8, 12, 20, None],
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    "bootstrap":         [True, False],
                },
            },
            estimator_factory=_extratrees_factory,
        ),
        # ---------------------------------------------------------------
        # Decision Tree
        # ---------------------------------------------------------------
        ModelSpec(
            name="decisiontree",
            aliases=(),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=True,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "max_depth": None,
                    "criterion": "gini",
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
                "regression": {
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "max_depth": [3, 7, 15, None],
                    "min_samples_leaf": [1, 4, 10],
                    "criterion": ["gini", "entropy"],
                },
                "regression": {
                    "max_depth": [3, 7, 15, None],
                    "min_samples_leaf": [2, 5, 10],
                },
            },
            param_distributions={
                "classification": {
                    "criterion":         ["gini", "entropy"],
                    "max_depth":         [3, 5, 7, 10, 15, 20, None],
                    "min_samples_split": randint(2, 30),
                    "min_samples_leaf":  randint(1, 20),
                    "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                    "class_weight":      ["balanced", None],
                },
                "regression": {
                    "criterion":         ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth":         [3, 5, 7, 10, 15, 20, None],
                    "min_samples_split": randint(2, 30),
                    "min_samples_leaf":  randint(1, 20),
                    "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                },
            },
            estimator_factory=_dt_factory,
        ),
        # ---------------------------------------------------------------
        # Gradient Boosting
        # ---------------------------------------------------------------
        ModelSpec(
            name="gradientboosting",
            aliases=("gb", "gbm"),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=False,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
                "regression": {
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "n_estimators":     [100, 300],
                    "learning_rate":    [0.05, 0.1, 0.2],
                    "max_depth":        [3, 5],
                    "subsample":        [0.7, 1.0],
                    "min_samples_leaf": [1, 4],
                    "max_features":     [None, "sqrt"],
                },
                "regression": {
                    "n_estimators":     [100, 300],
                    "learning_rate":    [0.05, 0.1, 0.2],
                    "max_depth":        [3, 5],
                    "subsample":        [0.7, 1.0],
                    "min_samples_leaf": [1, 4],
                    "max_features":     [None, "sqrt"],
                },
            },
            param_distributions={
                "classification": {
                    "n_estimators":      log_randint(50, 500),
                    "learning_rate":     uniform(0.005, 0.295),
                    "max_depth":         randint(2, 8),
                    "subsample":         uniform(0.5, 0.5),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                },
                "regression": {
                    "n_estimators":      log_randint(50, 500),
                    "learning_rate":     uniform(0.005, 0.295),
                    "max_depth":         randint(2, 8),
                    "subsample":         uniform(0.5, 0.5),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf":  randint(1, 15),
                    "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                },
            },
            estimator_factory=_gradientboosting_factory,
        ),
    ]
