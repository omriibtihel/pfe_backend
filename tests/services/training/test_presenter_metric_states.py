from __future__ import annotations

"""
Tests for get_primary_metric() 3-state logic (BUG-04).
"""

import logging
import pytest
from unittest.mock import patch

import app.models.dataset          # noqa: F401 — mapper pre-load
import app.models.dataset_version  # noqa: F401
import app.models.training         # noqa: F401

from app.services.training.presenter import get_primary_metric, MetricNotApplicable
from app.schemas.training.results import PrimaryMetric


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mget(d: dict):
    """Return a mget-style callable that reads from a flat dict."""
    def _fn(k):
        v = d.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return _fn


# ---------------------------------------------------------------------------
# test_success_state_returns_valid_float
# ---------------------------------------------------------------------------

def test_success_state_returns_valid_float():
    metrics_all = {
        "test": {"roc_auc": 0.85, "accuracy": 0.90},
    }
    mget = _mget({"roc_auc": 0.85, "accuracy": 0.90})
    result = get_primary_metric("classification", metrics_all, mget)

    assert result.status == "success"
    assert isinstance(result.value, float)
    assert result.value > 0
    assert result.name in ("roc_auc", "accuracy")


# ---------------------------------------------------------------------------
# test_not_applicable_state
# ---------------------------------------------------------------------------

def test_not_applicable_state():
    """When metrics_json has no valid numeric metric, status must be 'not_applicable'."""
    metrics_all: dict = {}
    mget = _mget({})
    result = get_primary_metric("classification", metrics_all, mget)

    assert result.status == "not_applicable"
    assert result.value is None
    assert result.name == "unknown"
    assert result.displayName == "—"


# ---------------------------------------------------------------------------
# test_error_state_on_parse_exception
# ---------------------------------------------------------------------------

def test_error_state_on_parse_exception():
    """When the function itself raises unexpectedly, status must be 'error'."""
    # Pass a mget that raises to trigger the outer except
    def _bad_mget(k):
        raise RuntimeError("deliberate failure for testing")

    result = get_primary_metric("classification", {}, _bad_mget)

    assert result.status == "error"
    assert result.value is None
    assert result.name == "error"


# ---------------------------------------------------------------------------
# test_error_state_does_not_swallow_log
# ---------------------------------------------------------------------------

def test_error_state_does_not_swallow_log(caplog):
    """When status='error', the exception must be logged at ERROR level."""
    def _bad_mget(k):
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR, logger="app.services.training.presenter"):
        result = get_primary_metric("classification", {}, _bad_mget)

    assert result.status == "error"
    assert any("get_primary_metric" in r.message for r in caplog.records), (
        f"Expected ERROR log from get_primary_metric, got: {[r.message for r in caplog.records]}"
    )
