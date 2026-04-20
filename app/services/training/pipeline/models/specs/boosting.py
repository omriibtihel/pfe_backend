"""
Gradient-boosted tree specs: XGBoost, LightGBM, CatBoost.
All three have optional dependencies — handled with try/except at import time.
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import loguniform, randint, uniform

from app.services.training.pipeline.models.registry import ModelSpec, log_randint

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBRegressor = None   # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None  # type: ignore[assignment,misc]
    LGBMRegressor = None   # type: ignore[assignment,misc]

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    CatBoostRegressor = None   # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _xgb_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(**params)
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    return XGBRegressor(**params)


def _lgbm_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(**params)
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")
    return LGBMRegressor(**params)


def _catboost_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostClassifier(**params)
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    return CatBoostRegressor(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        # ---------------------------------------------------------------
        # XGBoost
        # ---------------------------------------------------------------
        ModelSpec(
            name="xgboost",
            aliases=(),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=False,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "n_estimators": 300,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                },
                "regression": {
                    "n_estimators": 300,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                },
            },
            param_grid={
                "classification": {
                    "n_estimators":     [200, 400],
                    "learning_rate":    [0.03, 0.08, 0.15],
                    "max_depth":        [3, 5, 8],
                    "subsample":        [0.8, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                    "reg_alpha":        [0, 0.1, 1.0],
                    "min_child_weight": [1, 3, 5],
                },
                "regression": {
                    "n_estimators":     [200, 400],
                    "learning_rate":    [0.03, 0.08, 0.15],
                    "max_depth":        [3, 5, 8],
                    "subsample":        [0.8, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                    "reg_alpha":        [0, 0.1, 1.0],
                    "min_child_weight": [1, 3, 5],
                },
            },
            param_distributions={
                "classification": {
                    "n_estimators":     log_randint(50, 500),
                    "learning_rate":    loguniform(0.005, 0.3),
                    "max_depth":        randint(3, 12),
                    "subsample":        uniform(0.5, 0.5),
                    "colsample_bytree": uniform(0.4, 0.6),
                    "reg_alpha":        loguniform(1e-3, 10),
                    "reg_lambda":       loguniform(1e-3, 10),
                    "min_child_weight": randint(1, 10),
                    "gamma":            loguniform(1e-3, 5),
                },
                "regression": {
                    "n_estimators":     log_randint(50, 500),
                    "learning_rate":    loguniform(0.005, 0.3),
                    "max_depth":        randint(3, 12),
                    "subsample":        uniform(0.5, 0.5),
                    "colsample_bytree": uniform(0.4, 0.6),
                    "reg_alpha":        loguniform(1e-3, 10),
                    "reg_lambda":       loguniform(1e-3, 10),
                    "min_child_weight": randint(1, 10),
                    "gamma":            loguniform(1e-3, 5),
                },
            },
            estimator_factory=_xgb_factory,
        ),
        # ---------------------------------------------------------------
        # LightGBM
        # ---------------------------------------------------------------
        ModelSpec(
            name="lightgbm",
            aliases=(),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=True,
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.9,
                    "bagging_freq": 5,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1,
                },
                "regression": {
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.9,
                    "bagging_freq": 5,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1,
                },
            },
            param_grid={
                "classification": {
                    "n_estimators":      [300, 500],
                    "learning_rate":     [0.02, 0.05, 0.1],
                    "num_leaves":        [31, 63, 127],
                    "min_child_samples": [10, 30],
                    "feature_fraction":  [0.8, 1.0],
                },
                "regression": {
                    "n_estimators":      [300, 500],
                    "learning_rate":     [0.02, 0.05, 0.1],
                    "num_leaves":        [31, 63, 127],
                    "min_child_samples": [10, 30],
                    "feature_fraction":  [0.8, 1.0],
                },
            },
            param_distributions={
                "classification": {
                    "n_estimators":      log_randint(50, 500),
                    "learning_rate":     loguniform(0.005, 0.3),
                    "num_leaves":        randint(15, 150),
                    "max_depth":         [-1, 5, 8, 12, 20],
                    "subsample":         uniform(0.5, 0.5),
                    "colsample_bytree":  uniform(0.4, 0.6),
                    "reg_alpha":         loguniform(1e-3, 10),
                    "reg_lambda":        loguniform(1e-3, 10),
                    "min_child_samples": randint(5, 50),
                    "min_split_gain":    loguniform(1e-4, 1.0),
                },
                "regression": {
                    "n_estimators":      log_randint(50, 500),
                    "learning_rate":     loguniform(0.005, 0.3),
                    "num_leaves":        randint(15, 150),
                    "max_depth":         [-1, 5, 8, 12, 20],
                    "subsample":         uniform(0.5, 0.5),
                    "colsample_bytree":  uniform(0.4, 0.6),
                    "reg_alpha":         loguniform(1e-3, 10),
                    "reg_lambda":        loguniform(1e-3, 10),
                    "min_child_samples": randint(5, 50),
                    "min_split_gain":    loguniform(1e-4, 1.0),
                },
            },
            estimator_factory=_lgbm_factory,
        ),
        # ---------------------------------------------------------------
        # CatBoost
        # ---------------------------------------------------------------
        ModelSpec(
            name="catboost",
            aliases=("cat", "cb"),
            supports_classification=True,
            supports_regression=True,
            supports_class_weight=False,   # CatBoost uses auto_class_weights
            supports_predict_proba=True,
            supports_sample_weight=True,
            default_params={
                "classification": {
                    "iterations": 500,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3.0,
                    "random_state": 42,
                    "verbose": 0,
                    "allow_writing_files": False,
                },
                "regression": {
                    "iterations": 500,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3.0,
                    "random_state": 42,
                    "verbose": 0,
                    "allow_writing_files": False,
                },
            },
            param_grid={
                "classification": {
                    "iterations":     [200, 400],
                    "learning_rate":  [0.03, 0.05, 0.1],
                    "depth":          [4, 6, 8],
                    "l2_leaf_reg":    [1.0, 3.0, 10.0],
                },
                "regression": {
                    "iterations":     [200, 400],
                    "learning_rate":  [0.03, 0.05, 0.1],
                    "depth":          [4, 6, 8],
                    "l2_leaf_reg":    [1.0, 3.0, 10.0],
                },
            },
            param_distributions={
                "classification": {
                    "iterations":    log_randint(100, 500),
                    "learning_rate": loguniform(0.01, 0.3),
                    "depth":         randint(4, 10),
                    "l2_leaf_reg":   loguniform(1.0, 10.0),
                    "subsample":     uniform(0.6, 0.4),
                    "colsample_bylevel": uniform(0.6, 0.4),
                },
                "regression": {
                    "iterations":    log_randint(100, 500),
                    "learning_rate": loguniform(0.01, 0.3),
                    "depth":         randint(4, 10),
                    "l2_leaf_reg":   loguniform(1.0, 10.0),
                    "subsample":     uniform(0.6, 0.4),
                    "colsample_bylevel": uniform(0.6, 0.4),
                },
            },
            estimator_factory=_catboost_factory,
        ),
    ]
