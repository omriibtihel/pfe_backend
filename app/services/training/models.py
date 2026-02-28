from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
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


def _rf_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)


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
        self._register(
            ModelSpec(
                name="randomforest",
                aliases=("rf",),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=True,
                supports_predict_proba=True,
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
                        "n_estimators": [200, 300],
                        "max_depth": [None, 12],
                        "min_samples_leaf": [1, 2],
                    },
                    "regression": {
                        "n_estimators": [200, 300],
                        "max_depth": [None, 12],
                        "min_samples_leaf": [1, 2],
                    },
                },
                estimator_factory=_rf_factory,
            )
        )
        self._register(
            ModelSpec(
                name="logisticregression",
                aliases=("logreg",),
                supports_classification=True,
                supports_regression=False,
                supports_class_weight=True,
                supports_predict_proba=True,
                default_params={
                    "classification": {
                        "solver": "lbfgs",
                        "C": 1.0,
                        "max_iter": 4000,
                        "class_weight": None,
                        "random_state": 42,
                    },
                },
                param_grid={
                    "classification": {
                        "C": [0.5, 1.0, 2.0],
                        "solver": ["lbfgs"],
                    },
                },
                estimator_factory=_logreg_factory,
            )
        )
        self._register(
            ModelSpec(
                name="svm",
                aliases=(),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=True,
                supports_predict_proba=True,
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
                        "C": [0.5, 1.0, 2.0],
                        "gamma": ["scale", "auto"],
                    },
                    "regression": {
                        "C": [0.5, 1.0, 2.0],
                        "gamma": ["scale", "auto"],
                        "epsilon": [0.05, 0.1, 0.2],
                    },
                },
                estimator_factory=_svm_factory,
            )
        )
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
                        "n_neighbors": [5, 7, 11],
                        "weights": ["uniform", "distance"],
                    },
                    "regression": {
                        "n_neighbors": [5, 7, 11],
                        "weights": ["uniform", "distance"],
                    },
                },
                estimator_factory=_knn_factory,
            )
        )
        self._register(
            ModelSpec(
                name="naivebayes",
                aliases=("nb", "gaussiannb"),
                supports_classification=True,
                supports_regression=False,
                supports_class_weight=False,
                supports_predict_proba=True,
                default_params={
                    "classification": {
                        "var_smoothing": 1e-9,
                    },
                },
                param_grid={
                    "classification": {
                        "var_smoothing": [1e-11, 1e-9, 1e-7],
                    },
                },
                estimator_factory=_naivebayes_factory,
            )
        )
        self._register(
            ModelSpec(
                name="decisiontree",
                aliases=(),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=True,
                supports_predict_proba=True,
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
                        "max_depth": [None, 6, 12],
                        "min_samples_leaf": [1, 2, 4],
                    },
                    "regression": {
                        "max_depth": [None, 6, 12],
                        "min_samples_leaf": [1, 2, 4],
                    },
                },
                estimator_factory=_dt_factory,
            )
        )
        self._register(
            ModelSpec(
                name="xgboost",
                aliases=(),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=False,
                supports_predict_proba=True,
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
                        "eval_metric": "logloss",
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
                        "n_estimators": [250, 350],
                        "learning_rate": [0.03, 0.08],
                        "max_depth": [4, 6],
                        "subsample": [0.8, 1.0],
                    },
                    "regression": {
                        "n_estimators": [250, 350],
                        "learning_rate": [0.03, 0.08],
                        "max_depth": [4, 6],
                        "subsample": [0.8, 1.0],
                    },
                },
                estimator_factory=_xgb_factory,
            )
        )
        self._register(
            ModelSpec(
                name="lightgbm",
                aliases=(),
                supports_classification=True,
                supports_regression=True,
                supports_class_weight=True,
                supports_predict_proba=True,
                default_params={
                    "classification": {
                        "n_estimators": 500,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "feature_fraction": 1.0,
                        "bagging_fraction": 1.0,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                    "regression": {
                        "n_estimators": 500,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "feature_fraction": 1.0,
                        "bagging_fraction": 1.0,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                },
                param_grid={
                    "classification": {
                        "n_estimators": [300, 450],
                        "learning_rate": [0.03, 0.06],
                        "num_leaves": [31, 63],
                    },
                    "regression": {
                        "n_estimators": [300, 450],
                        "learning_rate": [0.03, 0.06],
                        "num_leaves": [31, 63],
                    },
                },
                estimator_factory=_lgbm_factory,
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


MODEL_REGISTRY = ModelRegistry()


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
        return {"supports_class_weight": False, "supports_predict_proba": False}
    return {
        "supports_class_weight": spec.supports_class_weight,
        "supports_predict_proba": spec.supports_predict_proba,
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
