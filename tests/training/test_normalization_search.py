"""
Tests for _coerce_search_value() and the __range__ handling in
normalize_model_hyperparams().
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from app.services.training.config.normalization import (
    _coerce_search_value,
    normalize_model_hyperparams,
)


def _dist_name(dist: object) -> str:
    """Return the scipy distribution name from a frozen distribution object."""
    return getattr(getattr(dist, "dist", None), "name", "")

# ---------------------------------------------------------------------------
# Minimal schema stubs — mirrors the shape of MODEL_HP_SCHEMA entries.
# ---------------------------------------------------------------------------

_FLOAT_SCHEMA = {"type": "float", "default": 1.0, "gt": 0.0}
_INT_SCHEMA = {"type": "int", "default": 100, "min": 1, "max": 2000}
_INT_OR_NONE_SCHEMA = {"type": "int_or_none", "default": None}
_ENUM_SCHEMA = {"type": "enum", "default": "gini", "enum": ["gini", "entropy"]}


# ---------------------------------------------------------------------------
# _coerce_search_value — unit tests
# ---------------------------------------------------------------------------


class TestCoerceSearchValueUniform:
    def test_returns_uniform_distribution(self):
        dist = _coerce_search_value(
            "C",
            {"__range__": True, "min": 0.01, "max": 100.0},
            _FLOAT_SCHEMA,
        )
        assert isinstance(dist, rv_frozen)
        assert _dist_name(dist) == "uniform"

    def test_samples_within_bounds(self):
        dist = _coerce_search_value(
            "C",
            {"__range__": True, "min": 0.5, "max": 10.0},
            _FLOAT_SCHEMA,
        )
        samples = dist.rvs(size=200, random_state=0)
        assert np.all(samples >= 0.5)
        assert np.all(samples <= 10.0)


class TestCoerceSearchValueLogUniform:
    def test_returns_loguniform_distribution(self):
        dist = _coerce_search_value(
            "C",
            {"__range__": True, "min": 1e-4, "max": 1e4, "distribution": "log_uniform"},
            _FLOAT_SCHEMA,
        )
        assert isinstance(dist, rv_frozen)
        assert _dist_name(dist) == "loguniform"

    def test_samples_within_bounds(self):
        dist = _coerce_search_value(
            "alpha",
            {"__range__": True, "min": 0.001, "max": 10.0, "distribution": "log_uniform"},
            _FLOAT_SCHEMA,
        )
        samples = dist.rvs(size=200, random_state=42)
        assert np.all(samples >= 0.001)
        assert np.all(samples <= 10.0)


class TestCoerceSearchValueRandint:
    def test_int_field_returns_randint(self):
        dist = _coerce_search_value(
            "n_estimators",
            {"__range__": True, "min": 10, "max": 500},
            _INT_SCHEMA,
        )
        assert isinstance(dist, rv_frozen)
        assert _dist_name(dist) == "randint"

    def test_int_or_none_returns_randint(self):
        dist = _coerce_search_value(
            "max_depth",
            {"__range__": True, "min": 2, "max": 20},
            _INT_OR_NONE_SCHEMA,
        )
        assert isinstance(dist, rv_frozen)
        assert _dist_name(dist) == "randint"

    def test_samples_are_integers_within_bounds(self):
        dist = _coerce_search_value(
            "n_estimators",
            {"__range__": True, "min": 50, "max": 200},
            _INT_SCHEMA,
        )
        samples = dist.rvs(size=200, random_state=7)
        assert np.all(samples == samples.astype(int))
        assert np.all(samples >= 50)
        assert np.all(samples <= 200)


class TestCoerceSearchValuePassthroughs:
    def test_list_returned_as_is(self):
        values = [1, 5, 10, 50]
        result = _coerce_search_value("n_estimators", values, _INT_SCHEMA)
        assert result == values

    def test_scalar_int_delegated(self):
        result = _coerce_search_value("n_estimators", 200, _INT_SCHEMA)
        assert result == 200

    def test_scalar_float_delegated(self):
        result = _coerce_search_value("C", 0.5, _FLOAT_SCHEMA)
        assert pytest.approx(result) == 0.5

    def test_scalar_enum_delegated(self):
        result = _coerce_search_value("criterion", "gini", _ENUM_SCHEMA)
        assert result == "gini"


class TestCoerceSearchValueInvalidRanges:
    def test_min_equal_to_max_returns_none(self):
        result = _coerce_search_value(
            "C",
            {"__range__": True, "min": 1.0, "max": 1.0},
            _FLOAT_SCHEMA,
        )
        assert result is None

    def test_min_greater_than_max_returns_none(self):
        result = _coerce_search_value(
            "C",
            {"__range__": True, "min": 10.0, "max": 1.0},
            _FLOAT_SCHEMA,
        )
        assert result is None

    def test_log_uniform_non_positive_min_returns_none(self):
        result = _coerce_search_value(
            "C",
            {"__range__": True, "min": 0.0, "max": 10.0, "distribution": "log_uniform"},
            _FLOAT_SCHEMA,
        )
        assert result is None

    def test_missing_min_key_returns_none(self):
        result = _coerce_search_value(
            "C",
            {"__range__": True, "max": 10.0},
            _FLOAT_SCHEMA,
        )
        assert result is None


# ---------------------------------------------------------------------------
# normalize_model_hyperparams — integration tests
# ---------------------------------------------------------------------------


class TestNormalizeWithRangeDict:
    def test_float_range_goes_to_param_distributions(self):
        result = normalize_model_hyperparams(
            "randomforest",
            {"n_estimators": {"__range__": True, "min": 50, "max": 500}},
            use_grid_search=True,
        )
        assert "n_estimators" in result["param_distributions"]
        dist = result["param_distributions"]["n_estimators"]
        assert isinstance(dist, rv_frozen)
        assert _dist_name(dist) == "randint"

    def test_float_uniform_range_in_distributions(self):
        result = normalize_model_hyperparams(
            "logisticregression",
            {"C": {"__range__": True, "min": 0.01, "max": 100.0}},
            use_grid_search=True,
        )
        assert "C" in result["param_distributions"]
        assert result["param_distributions"]["C"].dist.name == "uniform"

    def test_float_log_uniform_range_in_distributions(self):
        result = normalize_model_hyperparams(
            "logisticregression",
            {"C": {"__range__": True, "min": 1e-4, "max": 1e4, "distribution": "log_uniform"}},
            use_grid_search=True,
        )
        assert "C" in result["param_distributions"]
        assert result["param_distributions"]["C"].dist.name == "loguniform"

    def test_range_not_in_param_grid(self):
        result = normalize_model_hyperparams(
            "logisticregression",
            {"C": {"__range__": True, "min": 0.01, "max": 10.0}},
            use_grid_search=True,
        )
        assert "C" not in result["param_grid"]

    def test_invalid_range_produces_error(self):
        result = normalize_model_hyperparams(
            "logisticregression",
            {"C": {"__range__": True, "min": 10.0, "max": 1.0}},
            use_grid_search=True,
        )
        assert result["errors"]
        assert "C" not in result["param_distributions"]

    def test_range_without_search_produces_warning(self):
        result = normalize_model_hyperparams(
            "logisticregression",
            {"C": {"__range__": True, "min": 0.01, "max": 100.0}},
            use_grid_search=False,
        )
        assert result["warnings"]
        assert "C" not in result["param_distributions"]

    def test_list_still_goes_to_param_grid(self):
        result = normalize_model_hyperparams(
            "randomforest",
            {"n_estimators": [50, 100, 200]},
            use_grid_search=True,
        )
        assert result["param_grid"]["n_estimators"] == [50, 100, 200]
        assert "n_estimators" not in result["param_distributions"]

    def test_param_distributions_empty_for_scalar_input(self):
        result = normalize_model_hyperparams(
            "randomforest",
            {"n_estimators": 200},
            use_grid_search=True,
        )
        assert result["param_distributions"] == {}
