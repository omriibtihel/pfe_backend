"""
Tests for _build_random_distributions() and the user-distribution priority
in Trainer.fit() random / halving_random paths.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.stats import loguniform, randint, uniform

from app.services.training.pipeline.trainer import _build_random_distributions


# ---------------------------------------------------------------------------
# _build_random_distributions — unit tests
# ---------------------------------------------------------------------------


class TestBuildRandomDistributions:
    """Tests for the pure helper — no sklearn needed."""

    def test_defaults_are_returned_when_no_user_input(self):
        result = _build_random_distributions("randomforest", "classification", None)
        # randomforest has default distributions — at minimum n_estimators
        assert len(result) > 0
        assert all(k.startswith("model__") for k in result)

    def test_user_distribution_overrides_default(self):
        user_dist = uniform(1, 499)  # custom distribution for n_estimators
        result = _build_random_distributions(
            "randomforest",
            "classification",
            {"n_estimators": user_dist},
        )
        assert "model__n_estimators" in result
        assert result["model__n_estimators"] is user_dist

    def test_user_list_overrides_default(self):
        user_list = [10, 20, 50]
        result = _build_random_distributions(
            "randomforest",
            "classification",
            {"n_estimators": user_list},
        )
        assert result["model__n_estimators"] is user_list

    def test_user_scipy_dist_overrides_default_for_float_param(self):
        user_dist = loguniform(1e-4, 1e2)
        result = _build_random_distributions(
            "logisticregression",
            "classification",
            {"C": user_dist},
        )
        assert "model__C" in result
        assert result["model__C"] is user_dist

    def test_default_kept_for_unspecified_params(self):
        # User overrides only n_estimators; max_depth default should still be present.
        user_dist = randint(50, 501)
        result_user = _build_random_distributions(
            "randomforest",
            "classification",
            {"n_estimators": user_dist},
        )
        result_default = _build_random_distributions("randomforest", "classification", None)

        # max_depth from defaults should survive
        if "model__max_depth" in result_default:
            assert "model__max_depth" in result_user

    def test_non_distribution_values_are_ignored(self):
        result = _build_random_distributions(
            "randomforest",
            "classification",
            {"n_estimators": 200},  # plain scalar — not a dist or list, must be skipped
        )
        result_default = _build_random_distributions("randomforest", "classification", None)
        # n_estimators should still come from defaults (scalar was ignored)
        assert result.get("model__n_estimators") is not None
        assert result["model__n_estimators"] is result_default.get("model__n_estimators")

    def test_all_keys_have_model_prefix(self):
        user_dist = {"n_estimators": randint(10, 200), "max_depth": randint(1, 20)}
        result = _build_random_distributions("randomforest", "classification", user_dist)
        assert all(k.startswith("model__") for k in result)

    def test_empty_user_dist_returns_defaults(self):
        result_empty = _build_random_distributions("randomforest", "classification", {})
        result_none = _build_random_distributions("randomforest", "classification", None)
        assert set(result_empty.keys()) == set(result_none.keys())


# ---------------------------------------------------------------------------
# Trainer.fit — integration test (RandomSearch path, sklearn mocked)
# ---------------------------------------------------------------------------


def _make_minimal_pipeline(model_type: str = "randomforest") -> Any:
    """Return a real sklearn pipeline with the given model type."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _make_xy(n: int = 120) -> tuple[Any, Any]:
    rng = np.random.default_rng(99)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(int)
    import pandas as pd
    return pd.DataFrame(X, columns=["a", "b", "c", "d"]), y


class TestTrainerRandomDistributionsMerge:
    """Integration tests that verify user distributions reach sklearn."""

    def test_user_distribution_replaces_default_in_param_distributions(self):
        """RandomizedSearchCV must receive the user's custom distribution."""
        from app.services.training.pipeline.trainer import Trainer

        X, y = _make_xy()
        pipeline = _make_minimal_pipeline()

        user_dist = randint(5, 15)  # narrow range for fast test

        captured: dict = {}

        real_rs_init = __import__("sklearn.model_selection", fromlist=["RandomizedSearchCV"]).RandomizedSearchCV.__init__

        def capture_init(self, estimator, param_distributions, **kwargs):
            captured["param_distributions"] = param_distributions
            # Call real __init__ so the object is properly initialised.
            real_rs_init(self, estimator, param_distributions, **kwargs)

        cfg = MagicMock()
        cfg.search_type = "random"
        cfg.use_grid_search = True
        cfg.k_folds = 3
        cfg.inner_cv_folds = 3
        cfg.n_iter_random_search = 2
        cfg.random_state = 42
        cfg.metrics = ["accuracy"]

        with patch(
            "sklearn.model_selection.RandomizedSearchCV.__init__",
            side_effect=capture_init,
            autospec=True,
        ):
            try:
                Trainer().fit(
                    pipeline=pipeline,
                    X_train=X,
                    y_train=y,
                    cfg=cfg,
                    model_type="randomforest",
                    task_type="classification",
                    model_param_distributions={"n_estimators": user_dist},
                    n_samples=len(y),
                )
            except Exception:
                pass  # fit() may fail after __init__ — we only care about captured args

        assert "param_distributions" in captured, (
            "RandomizedSearchCV.__init__ was not called — check test setup"
        )
        assert "model__n_estimators" in captured["param_distributions"]
        assert captured["param_distributions"]["model__n_estimators"] is user_dist
