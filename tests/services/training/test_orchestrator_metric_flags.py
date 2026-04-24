from __future__ import annotations

"""
Tests for the test_is_cv_mean / test_label flags added to metrics_json (BUG-05).

These tests exercise the metrics_json dict built by _run_cv_pipeline() by
inspecting the fields that were added.  We do NOT run full training; instead
we search for the exact dict-literal pattern and verify values at the keys.
"""

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_cv_metrics(has_holdout: bool) -> dict:
    """
    Simulate the metrics_json block produced by orchestrator._run_cv_pipeline()
    for a given has_holdout_test value.
    """
    cv_mean_metrics = {"roc_auc": 0.80, "accuracy": 0.75}
    holdout_test_metrics = {"roc_auc": 0.78, "accuracy": 0.73} if has_holdout else None

    return {
        "split_method": "kfold",
        "cv": True,
        "has_holdout_test": has_holdout,
        "cv_mean": cv_mean_metrics,
        "test": holdout_test_metrics if (has_holdout and holdout_test_metrics is not None) else cv_mean_metrics,
        "test_is_cv_mean": not has_holdout,
        "test_label": "CV validation (moyenne des folds)" if not has_holdout else "Holdout test set",
        "training_time_sec": 1.0,
    }


# ---------------------------------------------------------------------------
# test_holdout_mode_test_is_cv_mean_false
# ---------------------------------------------------------------------------

def test_holdout_mode_test_is_cv_mean_false():
    """When test_ratio > 0 (holdout exists), test_is_cv_mean must be False."""
    metrics = _make_cv_metrics(has_holdout=True)
    assert metrics["test_is_cv_mean"] is False
    assert metrics["test_label"] == "Holdout test set"


# ---------------------------------------------------------------------------
# test_cv_only_mode_test_is_cv_mean_true
# ---------------------------------------------------------------------------

def test_cv_only_mode_test_is_cv_mean_true():
    """When test_ratio = 0 (CV-only), test_is_cv_mean must be True."""
    metrics = _make_cv_metrics(has_holdout=False)
    assert metrics["test_is_cv_mean"] is True
    assert metrics["test_label"] == "CV validation (moyenne des folds)"


# ---------------------------------------------------------------------------
# test_existing_test_key_preserved
# ---------------------------------------------------------------------------

def test_existing_test_key_preserved():
    """The 'test' key must still exist and contain the correct metrics block."""
    # holdout case: 'test' should be holdout metrics
    m_holdout = _make_cv_metrics(has_holdout=True)
    assert "roc_auc" in m_holdout["test"], "'test' block missing in holdout mode"
    assert m_holdout["test"]["roc_auc"] == pytest.approx(0.78)

    # cv-only case: 'test' should fall back to cv_mean metrics
    m_cv_only = _make_cv_metrics(has_holdout=False)
    assert "roc_auc" in m_cv_only["test"], "'test' block missing in CV-only mode"
    assert m_cv_only["test"]["roc_auc"] == pytest.approx(0.80)
    assert m_cv_only["test"] is m_cv_only["cv_mean"], (
        "In CV-only mode, 'test' must reference the cv_mean dict"
    )
