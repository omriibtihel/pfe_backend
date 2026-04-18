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
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


# ---------------------------------------------------------------------------
# MLP wrapper — accepts hidden_layer_sizes as a string like "100,50" so that
# GridSearchCV can pass string values via set_params without sklearn iterating
# over individual characters.  The conversion to tuple happens in fit().
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


def _mlp_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        return _MLPClassifier(**params)
    return _MLPRegressor(**params)


def _elasticnet_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("elasticnet is regression-only")
    return ElasticNet(**params)


def _lasso_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("lasso is regression-only")
    return Lasso(**params)


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


def _catboost_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostClassifier(**params)
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    return CatBoostRegressor(**params)


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
        # CatBoost
        # ---------------------------------------------------------------
        self._register(
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
        # ---------------------------------------------------------------
        # MLP — Multi-Layer Perceptron (classification + regression)
        # hidden_layer_sizes stored as string "100,50"; converted in fit().
        # ---------------------------------------------------------------
        self._register(
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
            )
        )
        # ---------------------------------------------------------------
        # ElasticNet (regression only)
        # ---------------------------------------------------------------
        self._register(
            ModelSpec(
                name="elasticnet",
                aliases=("elastic_net", "en"),
                supports_classification=False,
                supports_regression=True,
                supports_class_weight=False,
                supports_predict_proba=False,
                supports_sample_weight=True,
                default_params={
                    "regression": {
                        "alpha": 1.0,
                        "l1_ratio": 0.5,
                        "max_iter": 2000,
                    },
                },
                param_grid={
                    "regression": {
                        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                        "l1_ratio": [0.1, 0.5, 0.9],
                    },
                },
                param_distributions={
                    "regression": {
                        "alpha": loguniform(1e-4, 100),
                        "l1_ratio": uniform(0.0, 1.0),
                    },
                },
                estimator_factory=_elasticnet_factory,
            )
        )
        # ---------------------------------------------------------------
        # Lasso (regression only)
        # ---------------------------------------------------------------
        self._register(
            ModelSpec(
                name="lasso",
                aliases=(),
                supports_classification=False,
                supports_regression=True,
                supports_class_weight=False,
                supports_predict_proba=False,
                supports_sample_weight=True,
                default_params={
                    "regression": {
                        "alpha": 1.0,
                        "max_iter": 2000,
                    },
                },
                param_grid={
                    "regression": {
                        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    },
                },
                param_distributions={
                    "regression": {
                        "alpha": loguniform(1e-4, 100),
                        "max_iter": [1000, 2000, 5000],
                    },
                },
                estimator_factory=_lasso_factory,
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
# Adaptive parameter grids (dataset-aware: size + imbalance)
# ---------------------------------------------------------------------------
#
# Design goals:
#   - Combo count stays manageable → fast GridSearch at all sizes
#   - Tiny  (<  500): ≤ 20 combos   (each fold has few samples)
#   - Small (< 2000): ≤ 80 combos   (exhaustive grid still feasible)
#   - Medium(<50000): ≤ 100 combos  (mostly for RandomSearch; compact for Grid)
#   - Large (≥50000): ≤ 20 combos   (only a light sweep if grid is requested)
#   - Imbalanced classification: inject class_weight=["balanced", None]
#     for models that support it (doubles combos, still within budget).
#
# These override the static ModelSpec.param_grid when n_samples is known.
# ---------------------------------------------------------------------------

_ADAPTIVE_GRIDS: Dict[str, Dict[str, Dict[str, Dict[str, list]]]] = {
    # model → task → size_cat → {param: [values]}
    # Combo budget: micro ≤ 6, tiny ≤ 20, small ≤ 80, medium ≤ 100, large ≤ 20
    # micro (<100 lignes): régularisation forte obligatoire, pas de max_depth=None,
    #   n_estimators fixe (variance trop haute pour explorer), ≤ 6 combos
    "randomforest": {
        "classification": {
            # 2×2=4 — max_depth limité à 3/5 (pas de None → interdit sur <100 lignes)
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            # 2×2=4
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            # 2×3×2=12
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            # 2×3×2×2×2=48 — min_samples_split added for regularisation
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 4], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
        "regression": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            # 2×3×2×2×2=48
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [2, 5], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
    },
    "logisticregression": {
        "classification": {
            # 3×2=6 — forte régularisation (C petit), pénalité explicite
            "micro":  {"C": [0.01, 0.1, 1.0], "l1_ratio": [0, 1]},
            # 3×2=6 — l1_ratio added even for tiny (pénalité mixte L1/L2)
            "tiny":   {"C": [0.1, 1.0, 10.0], "l1_ratio": [0, 1]},
            "small":  {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "saga"]},
            "medium": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "saga"], "l1_ratio": [0, 1]},
            "large":  {"C": [0.1, 1.0, 10.0]},
        },
    },
    "svm": {
        "classification": {
            # 3 combos — C petit obligatoire sur <100 lignes, kernel fixe rbf
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
            # n_neighbors limité à sqrt(n_train) ≈ 8-9 pour 80 samples — on reste < 7
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
            # var_smoothing élevé = plus de lissage = moins d'overfitting sur <100 lignes
            "micro":  {"var_smoothing": [1e-9, 1e-7, 1e-5]},
            "tiny":   {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "small":  {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "medium": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
            "large":  {"var_smoothing": [1e-9, 1e-7, 1e-5]},
        },
    },
    "decisiontree": {
        "classification": {
            # max_depth [2,3] seulement — arbre profond sur <100 lignes = mémorisation pure
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
    # XGBoost: old static grid had 972 combos — replaced by size-aware compact grids
    "xgboost": {
        "classification": {
            # 2×2=4 — max_depth=2/3, learning_rate élevé car peu d'arbres
            # n_estimators fixe à default (300) — trop peu de data pour l'explorer
            "micro":  {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            # 2×2×2=8
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            # 2×2×3×2×2×2=96 — colsample_bytree added (anti-overfitting sur petits datasets)
            "small":  {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [3, 5, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 3]},
            # 2×2×3×2×2×2×2=96 — reg_alpha ajouté (L1, important pour features médicales)
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
            # 2×2=4 — num_leaves très bas (7/15) pour éviter surapprentissage sur <100 lignes
            "micro":  {"learning_rate": [0.05, 0.1], "num_leaves": [7, 15]},
            # 2×2×2×2=16 — reg_lambda ajouté (régularisation L2, absente avant)
            "tiny":   {"n_estimators": [200, 300], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31], "reg_lambda": [0.1, 1.0]},
            # 2×3×2×2=24 — reg_lambda ajouté
            "small":  {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63], "reg_lambda": [0.1, 1.0]},
            # 2×3×3×2×2=72 — reg_alpha + reg_lambda ajoutés
            "medium": {"n_estimators": [300, 500], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [31, 63, 127], "min_child_samples": [10, 30], "reg_alpha": [0, 0.1], "reg_lambda": [0.1, 1.0]},
            # 2×2×2×2=16
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
            # 2×3×2×2×2=48 — min_samples_split ajouté (même logique que RF)
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 4], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
        "regression": {
            "micro":  {"max_depth": [3, 5], "min_samples_leaf": [4, 8]},
            "tiny":   {"n_estimators": [100, 200], "max_depth": [5, None]},
            "small":  {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
            # 2×3×2×2×2=48
            "medium": {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [2, 5], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
            "large":  {"n_estimators": [100, 300], "max_depth": [5, 10]},
        },
    },
    "gradientboosting": {
        "classification": {
            # 2×2=4 — learning_rate modéré + forte feuille min pour régulariser
            "micro":  {"learning_rate": [0.05, 0.1], "min_samples_leaf": [8, 16]},
            # 2×2×2=8 — min_samples_leaf ajouté (anti-overfitting fort)
            "tiny":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "min_samples_leaf": [1, 4]},
            # 2×3×2×2=24
            "small":  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "min_samples_leaf": [1, 4]},
            # 2×3×2×2×2=48
            "medium": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "subsample": [0.7, 1.0], "min_samples_leaf": [1, 4]},
            # 2×2×2=8
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
    # CatBoost: taille-aware — sans ça le fallback statique donne 54 combos pour tous
    "catboost": {
        "classification": {
            # 2×2=4 — iterations bas + l2 fort pour régulariser sur très peu de données
            "micro":  {"learning_rate": [0.05, 0.1], "l2_leaf_reg": [3.0, 10.0]},
            # 2×2=4
            "tiny":   {"iterations": [200, 400], "learning_rate": [0.05, 0.1]},
            # 2×3×2=12
            "small":  {"iterations": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6]},
            # 2×3×3×2=36
            "medium": {"iterations": [200, 500], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8], "l2_leaf_reg": [1.0, 3.0, 10.0]},
            # 2×2=4
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
            # alpha élevé = forte régularisation = indispensable sur <100 lignes
            "micro":  {"alpha": [1.0, 10.0, 100.0]},
            "tiny":   {"alpha": [0.1, 1.0, 10.0]},
            "small":  {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "medium": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "large":  {"alpha": [0.1, 1.0, 10.0, 100.0]},
        },
    },
    # MLP: architecture search uniquement sur medium/large — trop coûteux sur petits datasets
    "mlp": {
        "classification": {
            # micro: seule la régularisation — architecture fixe (trop peu de données)
            "micro":  {"alpha": [0.01, 0.1, 1.0]},
            # tiny: régularisation + activation — pas d'architecture
            "tiny":   {"alpha": [0.001, 0.01, 0.1], "activation": ["relu", "tanh"]},
            # small: idem tiny (peu de données, architecture search = overfitting)
            "small":  {"alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            # medium: on peut explorer l'architecture (2×2×2=8 combos)
            "medium": {"hidden_layer_sizes": ["100", "100,50"], "alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
            # large: architecture + régularisation seulement — pas d'activation search (coût)
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
            # alpha fort + mix L1/L2 équilibré sur très peu de données
            "micro":  {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
            "tiny":   {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
            # 5×3=15
            "small":  {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.15, 0.5, 0.85]},
            # 5×5=25
            "medium": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "large":  {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]},
        },
    },
    "lasso": {
        "regression": {
            # alpha fort indispensable sur peu de données (L1 = sélection de variables)
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
    "micro":  10,   # < 100 samples: budget minimal, variance élevée → peu de candidats
    "tiny":   25,
    "small":  40,
    "medium": 60,
    "large":  80,
}

# Initial candidate count for HalvingRandomSearchCV per dataset size.
# factor=3 elimination schedule:
#   micro:  15 → 5 → 1  (21 total fits — budget minimal pour éviter le surapprentissage du search)
#   tiny:   30 → 10 → 3  (43 total fits, vs 25 for random)
#   small:  60 → 20 → 7 → 2  (89 total fits, vs 40 for random)
#   medium: 100 → 33 → 11 → 4  (148 total fits, vs 60 for random)
#   large:  50 → 17 → 6  (73 total fits, vs 80 for random — conservative for slow models)
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

    - Grid size scales with ``n_samples``:
        tiny (<500)   → ≤ 20 combos
        small (<2000) → ≤ 80 combos
        medium        → ≤ 100 combos
        large         → ≤ 20 combos
    - For imbalanced classification with supporting models,
      ``class_weight=["balanced", None]`` is injected so GridSearch
      can evaluate both class-weight strategies.
    - Falls back to the static ModelSpec grid when no adaptive grid is defined.
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

    # Inject class_weight for imbalanced classification (supporting models only)
    if (
        imbalanced
        and task_type == "classification"
        and model_key in _CLASS_WEIGHT_MODELS
        and "class_weight" not in grid
    ):
        grid["class_weight"] = ["balanced", None]

    return grid


def get_adaptive_n_iter(n_samples: int) -> int:
    """Return a sensible ``n_iter`` for RandomizedSearchCV based on dataset size."""
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
