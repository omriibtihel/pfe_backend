"""
Linear model specs: LogisticRegression, SVM, Ridge, Lasso, ElasticNet.
"""
from __future__ import annotations

from typing import Any, Dict

from scipy.stats import loguniform, uniform

from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from app.services.training.pipeline.models.registry import ModelSpec, log_randint


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _logreg_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "classification":
        raise RuntimeError("logisticregression is classification-only")
    return LogisticRegression(**params)


def _svm_factory(task_type: str, params: Dict[str, Any]) -> Any:
    return SVC(**params) if task_type == "classification" else SVR(**params)


def _ridge_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("ridge is regression-only")
    return Ridge(**params)


def _lasso_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("lasso is regression-only")
    return Lasso(**params)


def _elasticnet_factory(task_type: str, params: Dict[str, Any]) -> Any:
    if task_type != "regression":
        raise RuntimeError("elasticnet is regression-only")
    return ElasticNet(**params)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_specs() -> list[ModelSpec]:
    return [
        # ---------------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------------
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
                    "solver": "lbfgs",
                    "C": 1.0,
                    "max_iter": 10000,
                    "tol": 1e-3,
                    "class_weight": None,
                    "random_state": 42,
                },
            },
            param_grid={
                "classification": {
                    "C":        [0.01, 0.1, 1.0, 10.0],
                    "solver":   ["lbfgs", "liblinear"],
                    "max_iter": [5000, 10000],
                },
            },
            param_distributions={
                "classification": {
                    "C":            loguniform(1e-3, 100),
                    "solver":       ["lbfgs", "liblinear"],
                    "max_iter":     [5000, 10000, 20000],
                    "class_weight": ["balanced", None],
                },
            },
            estimator_factory=_logreg_factory,
        ),
        # ---------------------------------------------------------------
        # SVM
        # ---------------------------------------------------------------
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
        ),
        # ---------------------------------------------------------------
        # Ridge (regression only)
        # ---------------------------------------------------------------
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
        ),
        # ---------------------------------------------------------------
        # ElasticNet (regression only)
        # ---------------------------------------------------------------
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
                    "max_iter": 10000,
                    "tol": 1e-3,
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
        ),
        # ---------------------------------------------------------------
        # Lasso (regression only)
        # ---------------------------------------------------------------
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
                    "max_iter": 10000,
                    "tol": 1e-3,
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
        ),
    ]
