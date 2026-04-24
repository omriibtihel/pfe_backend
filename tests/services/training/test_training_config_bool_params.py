from __future__ import annotations

"""
Tests for boolean parameter integrity (Chain 3 — Phase 3).

All boolean params go through _to_bool() which correctly distinguishes
None (→ default) from False (→ False).  These tests are regression guards
ensuring that params with DEFAULT=True never silently override an explicit False.
"""

import pytest

from app.services.training.config.schema.training_config import TrainingConfig


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
# test_bool_false_is_preserved
# ---------------------------------------------------------------------------

def test_shuffle_false_preserved_when_default_is_true():
    """shuffle has DEFAULT=True. Passing shuffle=False must NOT be overridden."""
    cfg = TrainingConfig.from_front(_payload(shuffle=False))
    assert cfg.shuffle is False


def test_shuffle_true_when_absent():
    """When shuffle is absent, DEFAULT=True must be used."""
    cfg = TrainingConfig.from_front(_payload())
    assert cfg.shuffle is True


def test_shuffle_true_when_explicitly_true():
    """Explicit True must be preserved."""
    cfg = TrainingConfig.from_front(_payload(shuffle=True))
    assert cfg.shuffle is True


def test_debug_false_is_default():
    """debug has DEFAULT=False. Absent → False."""
    cfg = TrainingConfig.from_front(_payload())
    assert cfg.debug is False


def test_debug_false_preserved():
    """Explicit debug=False is preserved (same as default but confirmed)."""
    cfg = TrainingConfig.from_front(_payload(debug=False))
    assert cfg.debug is False


def test_debug_true_is_preserved():
    """debug=True is correctly set."""
    cfg = TrainingConfig.from_front(_payload(debug=True))
    assert cfg.debug is True


def test_shuffle_string_false_is_preserved():
    """The frontend might send shuffle='false' (string). Must be treated as False."""
    cfg = TrainingConfig.from_front(_payload(shuffle="false"))
    assert cfg.shuffle is False


def test_shuffle_string_true_preserved():
    """shuffle='true' (string) must be treated as True."""
    cfg = TrainingConfig.from_front(_payload(shuffle="true"))
    assert cfg.shuffle is True
