from __future__ import annotations

"""
Tests for the falsy-default pattern fix (Chain 3 — Phase 2).

Verifies that None-check replaces the `or DEFAULT` anti-pattern so that
falsy-but-valid values (random_state=0, testRatio=0) are preserved.
"""

import pytest

from app.services.training.config.schema.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Minimal valid payload factory
# ---------------------------------------------------------------------------

def _payload(**overrides) -> dict:
    """Build the smallest valid payload, then apply overrides."""
    base = {
        "targetColumn": "target",
        "taskType": "classification",
        "models": ["logisticregression"],
        "metrics": ["accuracy"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# test_random_state_zero_is_preserved
# ---------------------------------------------------------------------------

def test_random_state_zero_is_preserved():
    """random_state=0 is a valid reproducibility seed — must NOT be silently replaced by 42."""
    cfg = TrainingConfig.from_front(_payload(randomState=0))
    assert cfg.random_state == 0


def test_random_state_negative_is_preserved():
    """Negative random_state values are valid (sklearn accepts them)."""
    cfg = TrainingConfig.from_front(_payload(randomState=-1))
    assert cfg.random_state == -1


# ---------------------------------------------------------------------------
# test_absent_key_uses_default
# ---------------------------------------------------------------------------

def test_absent_key_uses_default():
    """When randomState is absent from payload, the default 42 must be used."""
    cfg = TrainingConfig.from_front(_payload())
    assert cfg.random_state == 42


# ---------------------------------------------------------------------------
# test_test_ratio_zero_is_preserved
# ---------------------------------------------------------------------------

def test_test_ratio_zero_is_preserved():
    """testRatio=0 is valid in CV-only mode (no holdout test set)."""
    cfg = TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=5, testRatio=0))
    assert cfg.test_ratio == 0.0


# ---------------------------------------------------------------------------
# test_k_folds_zero_raises_if_invalid
# ---------------------------------------------------------------------------

def test_k_folds_zero_raises_if_invalid():
    """kFolds=0 is structurally invalid for CV; must raise ValueError not silently become 5."""
    with pytest.raises(ValueError, match="kFolds"):
        TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=0))


def test_k_folds_one_raises():
    """kFolds=1 is also too small for CV (need at least 2)."""
    with pytest.raises(ValueError, match="kFolds"):
        TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=1))


def test_k_folds_not_checked_for_holdout():
    """kFolds is irrelevant in holdout mode — kFolds=0 must NOT raise."""
    cfg = TrainingConfig.from_front(_payload(splitMethod="holdout", kFolds=0))
    assert cfg.k_folds == 0  # stored as-is; validate() only checks CV methods


# ---------------------------------------------------------------------------
# test_resolved_params_in_artifacts
# ---------------------------------------------------------------------------

def test_resolved_params_in_artifacts():
    """resolved_params must capture the actual resolved value — random_state=0 when 0 was passed."""
    cfg = TrainingConfig.from_front(_payload(randomState=0))
    assert "randomState" in cfg.resolved_params
    assert cfg.resolved_params["randomState"] == 0


def test_resolved_params_default_captured():
    """When a param is absent, resolved_params must record the default that was used."""
    cfg = TrainingConfig.from_front(_payload())
    assert cfg.resolved_params["randomState"] == 42
    assert cfg.resolved_params["kFolds"] == 5


def test_resolved_params_in_as_dict():
    """as_dict() must include resolved_params so it flows into session artifacts."""
    cfg = TrainingConfig.from_front(_payload(randomState=0))
    d = cfg.as_dict()
    assert "resolved_params" in d
    assert d["resolved_params"]["randomState"] == 0


# ---------------------------------------------------------------------------
# test_variance_threshold_zero_preserved (preprocessing.py fix)
# ---------------------------------------------------------------------------

def test_variance_threshold_zero_preserved():
    """varianceThreshold=0 disables variance filtering — must not be silently replaced by 0.01."""
    cfg = TrainingConfig.from_front(_payload(preprocessing={
        "advancedParams": {"varianceThreshold": 0}
    }))
    assert cfg.preprocessing.variance_threshold == 0.0
