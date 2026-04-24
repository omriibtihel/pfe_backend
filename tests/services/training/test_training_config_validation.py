from __future__ import annotations

"""
Tests for the TrainingConfig.validate() method (Chain 3 — Phase 4).

validate() is called automatically at the end of from_front(), so each
test exercises the public from_front() entry point.
"""

import pytest

from app.services.training.config.schema.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(**overrides) -> dict:
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
# test_valid_config_no_error
# ---------------------------------------------------------------------------

def test_valid_config_no_error():
    """A fully valid payload must not raise."""
    cfg = TrainingConfig.from_front(_payload())
    assert cfg.target_column == "target"


# ---------------------------------------------------------------------------
# test_k_folds_below_2_raises
# ---------------------------------------------------------------------------

def test_k_folds_below_2_raises():
    """kFolds < 2 with a CV split method must raise ValueError."""
    with pytest.raises(ValueError, match="kFolds"):
        TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=1))


def test_k_folds_exactly_2_is_valid():
    """kFolds=2 is the minimum valid value for CV."""
    cfg = TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=2, testRatio=0))
    assert cfg.k_folds == 2


# ---------------------------------------------------------------------------
# test_test_ratio_above_95_raises
# ---------------------------------------------------------------------------

def test_test_ratio_above_95_raises():
    """testRatio > 95% leaves less than 5% for training — must raise ValueError."""
    with pytest.raises(ValueError, match="testRatio"):
        TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=5, testRatio=96))


def test_test_ratio_exactly_95_is_valid():
    """testRatio=95 is the maximum allowed."""
    cfg = TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=5, testRatio=95))
    assert abs(cfg.test_ratio - 0.95) < 0.001


def test_test_ratio_zero_is_valid():
    """testRatio=0 (CV-only mode) is explicitly valid."""
    cfg = TrainingConfig.from_front(_payload(splitMethod="kfold", kFolds=5, testRatio=0))
    assert cfg.test_ratio == 0.0


# ---------------------------------------------------------------------------
# test_ratios_not_summing_100_raises
# ---------------------------------------------------------------------------

def test_ratios_not_summing_100_raises():
    """trainRatio + valRatio + testRatio must sum to 100 in holdout mode."""
    with pytest.raises(ValueError, match="sum"):
        # 70 + 20 + 20 = 110
        TrainingConfig.from_front(_payload(trainRatio=70, valRatio=20, testRatio=20))


def test_ratios_summing_100_is_valid():
    """70 + 15 + 15 = 100 — must pass."""
    cfg = TrainingConfig.from_front(_payload(trainRatio=70, valRatio=15, testRatio=15))
    assert cfg.train_ratio == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# test_random_state_negative_is_valid
# ---------------------------------------------------------------------------

def test_random_state_negative_is_valid():
    """Negative random_state values are accepted (sklearn treats -1 as valid)."""
    cfg = TrainingConfig.from_front(_payload(randomState=-1))
    assert cfg.random_state == -1


# ---------------------------------------------------------------------------
# test_n_repeats_below_1_raises
# ---------------------------------------------------------------------------

def test_n_repeats_below_1_raises():
    """nRepeats < 1 with repeated CV must raise ValueError."""
    with pytest.raises(ValueError, match="nRepeats"):
        TrainingConfig.from_front(_payload(
            splitMethod="repeated_stratified_kfold",
            kFolds=5,
            testRatio=0,
            nRepeats=0,
        ))


def test_n_repeats_one_is_valid():
    """nRepeats=1 is the minimum valid value for repeated CV."""
    cfg = TrainingConfig.from_front(_payload(
        splitMethod="repeated_stratified_kfold",
        kFolds=5,
        testRatio=0,
        nRepeats=1,
    ))
    assert cfg.n_repeats == 1
