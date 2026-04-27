from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data.column_inference import numeric_extra_stats

BASIC_KEYS  = {"min_val", "max_val", "mean_val", "median_val", "q1_val", "q3_val", "has_negative"}
EXTRA_KEYS  = {"skewness", "outlier_count", "outlier_ratio"}
ALL_KEYS    = BASIC_KEYS | EXTRA_KEYS


class TestNumericExtraStats:
    # ── return shape ───────────────────────────────────────────────────────────

    def test_returns_all_keys_for_sufficient_data(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        assert set(numeric_extra_stats(s).keys()) == ALL_KEYS

    def test_empty_series_returns_all_none(self):
        result = numeric_extra_stats(pd.Series([], dtype=float))
        assert set(result.keys()) == ALL_KEYS
        assert all(v is None for v in result.values())

    def test_1_to_3_values_basic_stats_computed_extra_none(self):
        for n in [1, 2, 3]:
            s = pd.Series(list(range(1, n + 1)), dtype=float)
            result = numeric_extra_stats(s)
            assert set(result.keys()) == ALL_KEYS, f"Missing keys for n={n}"
            # basic stats are populated
            assert result["min_val"] is not None, f"min_val None for n={n}"
            assert result["max_val"] is not None, f"max_val None for n={n}"
            assert result["mean_val"] is not None, f"mean_val None for n={n}"
            # stats that need >= 4 values remain None
            assert result["skewness"] is None, f"skewness not None for n={n}"
            assert result["outlier_count"] is None, f"outlier_count not None for n={n}"
            assert result["outlier_ratio"] is None, f"outlier_ratio not None for n={n}"

    # ── basic numeric stats ───────────────────────────────────────────────────

    def test_min_max_correct(self):
        s = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
        result = numeric_extra_stats(s)
        assert result["min_val"] == pytest.approx(1.0)
        assert result["max_val"] == pytest.approx(5.0)

    def test_mean_correct(self):
        s = pd.Series([2.0, 4.0, 6.0, 8.0])
        result = numeric_extra_stats(s)
        assert result["mean_val"] == pytest.approx(5.0)

    def test_median_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numeric_extra_stats(s)
        assert result["median_val"] == pytest.approx(3.0)

    def test_q1_q3_ordered(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = numeric_extra_stats(s)
        assert result["q1_val"] < result["median_val"] < result["q3_val"]

    # ── outlier detection ─────────────────────────────────────────────────────

    def test_clean_series_has_zero_outliers(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numeric_extra_stats(s)
        assert result["outlier_count"] == 0
        assert result["outlier_ratio"] == pytest.approx(0.0)

    def test_extreme_outlier_detected(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = numeric_extra_stats(s)
        assert result["outlier_count"] > 0
        assert result["outlier_ratio"] > 0.0

    def test_outlier_ratio_equals_count_over_length(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numeric_extra_stats(s)
        assert result["outlier_ratio"] == pytest.approx(result["outlier_count"] / len(s))

    # ── has_negative ──────────────────────────────────────────────────────────

    def test_has_negative_true_for_negative_values(self):
        assert numeric_extra_stats(pd.Series([-1.0, 2.0, 3.0, 4.0]))["has_negative"] is True

    def test_has_negative_true_when_zero_present(self):
        # Implementation: data.min() <= 0 → True
        assert numeric_extra_stats(pd.Series([0.0, 1.0, 2.0, 3.0]))["has_negative"] is True

    def test_has_negative_false_for_all_positive(self):
        assert numeric_extra_stats(pd.Series([1.0, 2.0, 3.0, 4.0]))["has_negative"] is False

    # ── skewness ──────────────────────────────────────────────────────────────

    def test_right_skewed_positive_skewness(self):
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 10.0, 100.0, 1000.0, 5000.0])
        assert numeric_extra_stats(s)["skewness"] > 0

    def test_symmetric_skewness_near_zero(self):
        s = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert abs(numeric_extra_stats(s)["skewness"]) < 0.5

    # ── inf values ────────────────────────────────────────────────────────────

    def test_inf_does_not_raise(self):
        # Caller normally strips inf; verify function doesn't crash if one slips through
        result = numeric_extra_stats(pd.Series([1.0, 2.0, 3.0, np.inf]))
        assert isinstance(result, dict)
        assert set(result.keys()) == ALL_KEYS

    def test_all_inf_treated_as_empty_after_caller_preprocessing(self):
        # When caller does s.replace([inf,-inf], nan).dropna(), all-inf → empty
        s = pd.Series([], dtype=float)
        result = numeric_extra_stats(s)
        assert all(v is None for v in result.values())
