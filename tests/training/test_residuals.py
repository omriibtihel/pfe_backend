"""
Tests for compute_residuals.

Covers: basic structure, field types, regression-only guard (n < 5, wrong task),
histogram shape, qq_points shape, stats correctness, edge cases.
"""
import numpy as np
import pytest

from app.services.training.pipeline.residuals import compute_residuals


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_arrays(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3
    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# task_type guard
# ──────────────────────────────────────────────────────────────────────────────

def test_returns_none_for_classification():
    y_t, y_p = _make_arrays()
    assert compute_residuals(y_t, y_p, task_type="classification") is None


def test_returns_none_for_unknown_task():
    y_t, y_p = _make_arrays()
    assert compute_residuals(y_t, y_p, task_type="binary") is None


def test_returns_result_for_regression():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# n < 5 guard
# ──────────────────────────────────────────────────────────────────────────────

def test_returns_none_when_fewer_than_5_samples():
    y_t = np.array([1.0, 2.0, 3.0, 4.0])
    y_p = np.array([1.1, 2.1, 3.1, 4.1])
    assert compute_residuals(y_t, y_p, task_type="regression") is None


def test_accepts_exactly_5_samples():
    y_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_p = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# Output structure
# ──────────────────────────────────────────────────────────────────────────────

def test_top_level_keys():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert set(result.keys()) == {"stats", "histogram", "qq_points", "n_samples"}


def test_stats_keys():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    stats = result["stats"]
    assert set(stats.keys()) == {"mean", "std", "min", "max", "skewness", "correlation_with_pred"}


def test_stats_types():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    for v in result["stats"].values():
        assert isinstance(v, float)


def test_histogram_keys():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    hist = result["histogram"]
    assert "bin_edges" in hist and "counts" in hist


def test_histogram_shape():
    y_t, y_p = _make_arrays(100)
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    hist = result["histogram"]
    n_bins = len(hist["counts"])
    assert len(hist["bin_edges"]) == n_bins + 1, "bin_edges should be n_bins + 1"


def test_histogram_counts_are_ints():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    for c in result["histogram"]["counts"]:
        assert isinstance(c, int)


def test_histogram_bin_edges_are_floats():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    for e in result["histogram"]["bin_edges"]:
        assert isinstance(e, float)


def test_qq_points_is_list_of_pairs():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    for pt in result["qq_points"]:
        assert len(pt) == 2
        assert isinstance(pt[0], float) and isinstance(pt[1], float)


def test_qq_points_at_most_50():
    y_t, y_p = _make_arrays(200)
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert len(result["qq_points"]) <= 50


def test_n_samples_correct():
    n = 70
    y_t, y_p = _make_arrays(n)
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert result["n_samples"] == n


# ──────────────────────────────────────────────────────────────────────────────
# Stats correctness
# ──────────────────────────────────────────────────────────────────────────────

def test_residual_mean_near_zero_for_unbiased_model():
    """When predictions are unbiased (residuals symmetric), mean should be near 0."""
    rng = np.random.default_rng(42)
    y_t = rng.standard_normal(200)
    y_p = y_t + rng.standard_normal(200) * 0.1  # small unbiased noise
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert abs(result["stats"]["mean"]) < 0.1


def test_residual_mean_nonzero_for_biased_model():
    y_t = np.ones(50)
    y_p = np.zeros(50)  # always predicts 0 → residuals all +1
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert abs(result["stats"]["mean"] - 1.0) < 1e-6


def test_correlation_with_pred_near_zero_for_good_model():
    """Residuals uncorrelated with predictions → no systematic pattern."""
    rng = np.random.default_rng(7)
    y_t = rng.standard_normal(300)
    y_p = y_t + rng.standard_normal(300) * 0.05
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert abs(result["stats"]["correlation_with_pred"]) < 0.3


def test_std_positive():
    y_t, y_p = _make_arrays(50)
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert result["stats"]["std"] > 0.0


def test_min_le_max():
    y_t, y_p = _make_arrays()
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    assert result["stats"]["min"] <= result["stats"]["max"]


# ──────────────────────────────────────────────────────────────────────────────
# Constant predictions edge case
# ──────────────────────────────────────────────────────────────────────────────

def test_constant_predictions_no_crash():
    """Constant predictions → correlation is undefined → should be replaced with 0."""
    y_t = np.random.randn(30)
    y_p = np.full(30, 5.0)  # constant prediction → NaN pearson correlation
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    corr = result["stats"]["correlation_with_pred"]
    assert np.isfinite(corr), "correlation should be finite (NaN replaced with 0)"


# ──────────────────────────────────────────────────────────────────────────────
# JSON-safe output
# ──────────────────────────────────────────────────────────────────────────────

def test_all_values_json_serializable():
    import json
    y_t, y_p = _make_arrays(80)
    result = compute_residuals(y_t, y_p, task_type="regression")
    assert result is not None
    json.dumps(result)  # should not raise
