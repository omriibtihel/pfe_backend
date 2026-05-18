"""
Gradient-boosted tree specs: XGBoost, LightGBM, CatBoost.
All three have optional dependencies — handled with try/except at import time.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.preprocessing import LabelEncoder

from app.services.training.pipeline.models.registry import ModelSpec, log_randint

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBRegressor = None   # type: ignore[assignment,misc]


def _needs_label_encoding(y: np.ndarray) -> bool:
    """True when y is not already integer-encoded as [0, n_classes-1]."""
    kind = y.dtype.kind
    if kind in ("U", "O", "S", "f"):
        return True
    if kind in ("i", "u", "b"):
        unique = np.unique(y)
        n = len(unique)
        if n == 0:
            return False
        return not (int(unique.min()) == 0 and int(unique.max()) == n - 1)
    return True


if XGBClassifier is not None:

    class XGBClassifierAutoEncode(XGBClassifier):  # type: ignore[misc]
        """XGBClassifier that auto-encodes non-numeric / non-contiguous labels.

        XGBoost requires class labels to be integers in [0, n_classes-1] and
        raises ``ValueError: Invalid classes inferred from unique values of y``
        on string targets (e.g. 'B'/'M'). This subclass fits an internal
        LabelEncoder when needed and decodes predictions back so downstream
        code (Evaluator, threshold calibration, SHAP) sees the original labels.

        ``eval_set`` labels are also encoded so the early-stopping path
        (``_fit_with_early_stopping``) works on string targets.
        """

        def fit(self, X, y, sample_weight=None, **kwargs):  # type: ignore[override]
            # Clear encoder before fit so the classes_ property exposes XGBoost's
            # internal integer classes during fit (XGBoost validates y against
            # self.classes_; exposing the original string labels too early
            # causes "Invalid classes inferred from unique values of y").
            self._mv_label_encoder = None
            self._mv_decode_active = False

            y_arr = np.asarray(y)
            if _needs_label_encoding(y_arr):
                encoder = LabelEncoder()
                y_enc = encoder.fit_transform(y_arr)
            else:
                encoder = None
                y_enc = y_arr

            eval_set = kwargs.pop("eval_set", None)
            if eval_set is not None and encoder is not None:
                eval_set = [
                    (X_eval, encoder.transform(np.asarray(y_eval)))
                    for X_eval, y_eval in eval_set
                ]
            if eval_set is not None:
                kwargs["eval_set"] = eval_set

            if sample_weight is not None:
                super().fit(X, y_enc, sample_weight=sample_weight, **kwargs)
            else:
                super().fit(X, y_enc, **kwargs)

            # Activate decoding only after the underlying booster is fit.
            self._mv_label_encoder = encoder
            self._mv_decode_active = encoder is not None
            return self

        @property
        def classes_(self):  # type: ignore[override]
            if getattr(self, "_mv_decode_active", False):
                return self._mv_label_encoder.classes_
            return super().classes_

        def predict(self, X, **kwargs):  # type: ignore[override]
            enc = getattr(self, "_mv_label_encoder", None)
            if enc is not None:
                idx = np.asarray(super().predict(X, **kwargs)).astype(int)
                return enc.inverse_transform(idx)
            return super().predict(X, **kwargs)

else:
    XGBClassifierAutoEncode = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None  # type: ignore[assignment,misc]
    LGBMRegressor = None   # type: ignore[assignment,misc]


def _make_lgbm_no_fake_feature_names(base_cls: Any) -> Any:
    """Return a LightGBM subclass that hides synthetic ``Column_N`` feature names.

    LightGBM 4.x always exposes ``feature_names_in_`` as a property derived
    from the booster's ``feature_name_`` — even when ``fit`` was called with
    a numpy array.  In that case the booster auto-generates names like
    ``['Column_0', 'Column_1', ...]``.  Downstream, sklearn's
    ``_check_feature_names`` validation reads this property at predict time
    and, because the numpy ``X`` passed in has no column names, emits
    ``UserWarning: X does not have valid feature names, but LGBM... was fitted
    with feature names`` — repeated dozens of times per training run.

    The fix: track whether ``fit`` received a DataFrame.  If yes, expose
    ``feature_names_in_`` as usual (names match between fit and predict).
    If no, raise ``AttributeError`` from the property so sklearn treats the
    estimator as "no feature names tracked" and skips the validation.

    This keeps LightGBM's own ``feature_name_`` intact (still useful for
    feature-importance reporting) while making it cooperate with mixed
    DataFrame / numpy pipelines.
    """

    class _LGBMNoFakeFeatureNames(base_cls):  # type: ignore[misc, valid-type]
        def fit(self, X, y=None, **kwargs):  # type: ignore[override]
            self._mv_fit_had_columns = hasattr(X, "columns")
            return super().fit(X, y, **kwargs)

        @property
        def feature_names_in_(self) -> np.ndarray:  # type: ignore[override]
            if not getattr(self, "_mv_fit_had_columns", False):
                raise AttributeError(
                    "feature_names_in_ not set: fit was called with a numpy array."
                )
            return np.asarray(self.feature_name_)

    _LGBMNoFakeFeatureNames.__name__ = base_cls.__name__
    _LGBMNoFakeFeatureNames.__qualname__ = base_cls.__name__
    return _LGBMNoFakeFeatureNames


if LGBMClassifier is not None:
    LGBMClassifier = _make_lgbm_no_fake_feature_names(LGBMClassifier)  # type: ignore[misc]
if LGBMRegressor is not None:
    LGBMRegressor = _make_lgbm_no_fake_feature_names(LGBMRegressor)  # type: ignore[misc]

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
        if XGBClassifierAutoEncode is None:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifierAutoEncode(**params)
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
