from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict
import numpy as np
from scipy.stats import loguniform, uniform, randint

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None


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
# Estimator factories
# ---------------------------------------------------------------------------

def _rf_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)


def _extratrees_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return ExtraTreesClassifier(**params) if task_type == "classification" else ExtraTreesRegressor(**params)


def _gradientboosting_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return GradientBoostingClassifier(**params) if task_type == "classification" else GradientBoostingRegressor(**params)


def _logreg_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "classification":
        raise RuntimeError("logisticregression is classification-only")
    return LogisticRegression(**params)


def _svm_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return SVC(**params) if task_type == "classification" else SVR(**params)


def _knn_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return KNeighborsClassifier(**params) if task_type == "classification" else KNeighborsRegressor(**params)


def _dt_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return DecisionTreeClassifier(**params) if task_type == "classification" else DecisionTreeRegressor(**params)


def _naivebayes_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "classification":
        raise RuntimeError("naivebayes is classification-only")
    return GaussianNB(**params)


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


def _ridge_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("ridge is regression-only")
    return Ridge(**params)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}
        self._aliases: Dict[str, str] = {}
        self._build_defaults()

    def _register(self, spec: ModelSpec) -> None:
        self._specs[spec.name] = spec
        for alias in (spec.name, *spec.aliases):
            self._aliases[alias] = spec.name

    def _build_defaults(self) -> None:
        # ---------------------------------------------------------------
        # Random Forest
        # ---------------------------------------------------------------
        self._register(
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
                        "class_weight":      ["balanced", "balanced_subsample", None],
                    },
                    "regression": {
                        "n_estimators":      log_randint(50, 500),
                        "max_depth":         [3, 5, 8, 12, 20, None],
                        "min_samples_split": randint(2, 20),
                        "min_samples_leaf":  randint(1, 15),
                        "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    },
                },
                estimator_factory=_rf_factory,
            )
        )
        # ---------------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------------
        self._register(
            ModelSpec(
                name="logisticregression",
                aliases=("logreg",),
                supports_classification=True,
                supports_regression=False,
                supports_class_weight=True,
                supports_predict_proba=True,
                supports_sample_weight=True,
                default_params={
                    "classification": {
                        "solver": "saga",
                        "C": 1.0,
                        "l1_ratio": 0,  # 0=L2, 1=L1 (replaces deprecated penalty)
                        "max_iter": 4000,
                        "class_weight": None,
                        "random_state": 42,
                    },
                },
                param_grid={
                    "classification": {
                        "C":        [0.01, 0.1, 1.0, 10.0],
                        "solver":   ["liblinear", "saga"],
                        "l1_ratio": [0, 1],   # 0=L2, 1=L1 (replaces deprecated penalty)
                    },
                },
                param_distributions={
                    "classification": {
                        "C":            loguniform(1e-3, 100),
                        "l1_ratio":     [0, 1],   # 0=L2, 1=L1 (replaces deprecated penalty)
                        "solver":       ["liblinear", "saga"],
                        "class_weight": ["balanced", None],
                    },
                },
                estimator_factory=_logreg_factory,
            )
        )
        # ---------------------------------------------------------------
        # SVM
        # ---------------------------------------------------------------
        self._register(
            ModelSpec(
                name="svm",
                aliases=(),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=True,
                supports_predict_proba=True,
                supports_sample_weight=True,
                default_params={
                    "classification": {
                        "kernel": "rbf",
                        "C": 1.0,
                        "gamma": "scale",
                        "probability": True,
                    },
                    "regression": {
                        "kernel": "rbf",
                        "C": 1.0,
                        "gamma": "scale",
                        "epsilon": 0.1,
                    },
                },
                param_grid={
                    "classification": {
                        "C": [0.1, 1.0, 10.0],
                        "gamma": ["scale", "auto"],
                    },
                    "regression": {
                        "C": [0.1, 1.0, 10.0],
                        "gamma": ["scale", "auto"],
                        "epsilon": [0.01, 0.1, 0.5],
                    },
                },
                param_distributions={
                    "classification": {
                        "C":            loguniform(1e-2, 100),
                        "kernel":       ["rbf", "linear", "poly"],
                        "gamma":        ["scale", "auto", 1e-4, 1e-3, 1e-2, 0.1],
                        "class_weight": ["balanced", None],
                    },
                    "regression": {
                        "C":       loguniform(1e-2, 100),
                        "epsilon": loguniform(1e-3, 1),
                        "kernel":  ["rbf", "linear", "poly"],
                        "gamma":   ["scale", "auto", 1e-4, 1e-3, 1e-2, 0.1],
                    },
                },
                estimator_factory=_svm_factory,
            )
        )
        # ---------------------------------------------------------------
        # KNN
        # ---------------------------------------------------------------
        self._register(
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
            )
        )
        # ---------------------------------------------------------------
        # Naive Bayes
        # ---------------------------------------------------------------
        self._register(
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
            )
        )
        # ---------------------------------------------------------------
        # Decision Tree
        # ---------------------------------------------------------------
        self._register(
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
            )
        )
        # ---------------------------------------------------------------
        # XGBoost
        # ---------------------------------------------------------------
        self._register(
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
            )
        )
        # ---------------------------------------------------------------
        # LightGBM
        # ---------------------------------------------------------------
        self._register(
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
                    },
                },
                estimator_factory=_lgbm_factory,
            )
        )
        # ---------------------------------------------------------------
        # Extra Trees
        # ---------------------------------------------------------------
        self._register(
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
                        "class_weight":      ["balanced", "balanced_subsample", None],
                    },
                    "regression": {
                        "n_estimators":      log_randint(50, 500),
                        "max_depth":         [3, 5, 8, 12, 20, None],
                        "min_samples_split": randint(2, 20),
                        "min_samples_leaf":  randint(1, 15),
                        "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
                    },
                },
                estimator_factory=_extratrees_factory,
            )
        )
        # ---------------------------------------------------------------
        # Gradient Boosting
        # ---------------------------------------------------------------
        self._register(
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
                        "learning_rate":     loguniform(0.005, 0.3),
                        "max_depth":         randint(2, 8),
                        "subsample":         uniform(0.5, 0.5),
                        "min_samples_split": randint(2, 20),
                        "min_samples_leaf":  randint(1, 15),
                        "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                    },
                    "regression": {
                        "n_estimators":      log_randint(50, 500),
                        "learning_rate":     loguniform(0.005, 0.3),
                        "max_depth":         randint(2, 8),
                        "subsample":         uniform(0.5, 0.5),
                        "min_samples_split": randint(2, 20),
                        "min_samples_leaf":  randint(1, 15),
                        "max_features":      ["sqrt", "log2", None, 0.5, 0.7],
                    },
                },
                estimator_factory=_gradientboosting_factory,
            )
        )
        # ---------------------------------------------------------------
        # Ridge (regression only)
        # ---------------------------------------------------------------
        self._register(
            ModelSpec(
                name="ridge",
                aliases=(),
                supports_classification=False,
                supports_regression=True,
                supports_class_weight=False,
                supports_predict_proba=False,
                supports_sample_weight=True,
                default_params={
                    "regression": {
                        "alpha": 1.0,
                    },
                },
                param_grid={
                    "regression": {
                        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    },
                },
                param_distributions={
                    "regression": {
                        "alpha": loguniform(1e-4, 100),
                    },
                },
                estimator_factory=_ridge_factory,
            )
        )

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


MODEL_REGISTRY = ModelRegistry()


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
        return XGBClassifier is not None and XGBRegressor is not None
    if key == "lightgbm":
        return LGBMClassifier is not None and LGBMRegressor is not None
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


# ---------------------------------------------------------------------------
# Search helpers (used by hyperparam_search.py and trainer.py)
# ---------------------------------------------------------------------------

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
