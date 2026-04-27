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

from pydantic import ValidationError

from app.services.training.presenter import _build_cv_info, get_primary_metric, MetricNotApplicable
from app.schemas.training.results import PrimaryMetric, ModelResultResponse, EvaluationSource


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

# ---------------------------------------------------------------------------
# test_cv_test_score_zero_not_discarded (BUG: falsy 0.0 was treated as "no score")
# ---------------------------------------------------------------------------

def test_cv_test_score_zero_rmse_not_discarded():
    """RMSE=0.0 on a perfect synthetic model must survive as the resolved test_score.

    Previously, _build_cv_info returned 0.0 as the default sentinel AND as a
    legitimate score, and the callers used `if is_cv and cv_test_score` which
    silently discarded 0.0 (falsy) and fell through to the primary_score fallback.
    """
    metrics_all = {
        "cv": True,
        "has_holdout_test": False,
        "cv_summary": {
            "mean": {"rmse": 0.0},
        },
    }
    _cv_info, cv_test_score, is_cv, _has_holdout = _build_cv_info(
        metrics_all, primary_name="rmse"
    )

    # _build_cv_info must return the explicit 0.0, not None
    assert is_cv is True
    assert cv_test_score == 0.0

    # Simulate the resolution logic as written in model_to_list_result /
    # model_to_detail_result.  Use a non-zero fallback so a regression is obvious.
    raw_test = None
    primary_value = 0.87  # wrong fallback that would mask the perfect score
    test_score = (
        cv_test_score if is_cv and cv_test_score is not None
        else (float(raw_test) if raw_test is not None else primary_value)
    )

    assert test_score == 0.0, (
        f"Expected 0.0 (perfect RMSE) but got {test_score}; "
        "the falsy check would have silently used the fallback value instead."
    )


def test_cv_test_score_none_when_not_cv():
    """When is_cv is False, _build_cv_info returns None (not 0.0) for the score override."""
    metrics_all = {}  # no "cv" key → not a CV run
    _cv_info, cv_test_score, is_cv, _has_holdout = _build_cv_info(
        metrics_all, primary_name="rmse"
    )

    assert is_cv is False
    assert cv_test_score is None


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


# ---------------------------------------------------------------------------
# test_task_type_literal_rejects_unknown (schema contract)
# ---------------------------------------------------------------------------

def _minimal_model_result_payload(task_type: str) -> dict:
    """Minimal valid payload for ModelResultResponse with the given task_type."""
    return {
        "id": "1",
        "modelType": "LogisticRegression",
        "taskType": task_type,
        "primaryMetric": {
            "name": "roc_auc",
            "value": 0.85,
            "displayName": "ROC-AUC",
            "status": "success",
            "direction": "higher_is_better",
        },
        "metrics": {},
        "trainingTime": 1.0,
        "isSaved": False,
        "isActive": False,
    }


def test_task_type_literal_rejects_unknown():
    """ModelResultResponse must raise ValidationError for an unrecognised taskType.

    If a new task type is added backend-side, this test will fail and force an
    explicit update to the Literal in results.py — preventing a silent mismatch
    with the frontend union type.
    """
    with pytest.raises(ValidationError):
        ModelResultResponse(**_minimal_model_result_payload("segmentation"))


def test_task_type_literal_accepts_classification():
    ModelResultResponse(**_minimal_model_result_payload("classification"))


def test_task_type_literal_accepts_regression():
    ModelResultResponse(**_minimal_model_result_payload("regression"))
