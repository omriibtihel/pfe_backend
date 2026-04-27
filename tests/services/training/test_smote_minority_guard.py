"""Tests for the SMOTE minority-count guard shared by the K-Fold and LOO paths.

Both pipelines call ``_smote_minority_guard`` before ``apply_prefit``. The
guard downgrades a SMOTE / SMOTETomek decision to "no resampling" when the
minority class on a fold has too few samples for ``k_neighbors``-based
interpolation. This avoids the imblearn exception that would otherwise
fail the fold and hides any silent degeneracy.

These tests pin the exact contract:
  - SMOTE/SMOTETomek + min_count <= k → fallback decision + structured warning.
  - SMOTE/SMOTETomek + min_count > k → original decision, no warning.
  - Non-SMOTE strategies → passthrough, no warning.
  - Edge cases (empty array, multi-class, k_neighbors=None default).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.preparation_ml.balancing import BalancingDecision
from app.services.training.orchestrator import (
    _default_decision_for_non_classification,
    _smote_minority_guard,
)


def _make_decision(strategy: str, k_neighbors: int | None = 5) -> BalancingDecision:
    """Build a minimal BalancingDecision for testing the guard."""
    return BalancingDecision(
        strategy=strategy,
        apply_threshold=False,
        threshold_strategy="maximize_f1",
        rationale="test",
        refit_metric="f1",
        smote_k_neighbors=k_neighbors,
        class_weight_mode="balanced",
        audit_flags=[strategy],
        optimal_threshold=0.5,
    )


# ---------------------------------------------------------------------------
# Trigger conditions
# ---------------------------------------------------------------------------

def test_smote_skipped_when_one_minority_sample():
    """1 minority sample with k=5 → guard fires, fallback to non-resampling."""
    # 19 zeros, 1 one — exactly the "one fold has only 1 minority sample"
    # scenario from the audit task.
    y = np.array([0] * 19 + [1] * 1)
    decision = _make_decision("smote", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is not None
    assert "smote_skipped_minority_too_small" in warning
    assert "min_count=1" in warning
    assert "k_neighbors=5" in warning
    assert new_decision.strategy == "none"
    # Returned decision is the canonical fallback so the fold runs unaffected.
    fallback = _default_decision_for_non_classification()
    assert new_decision.strategy == fallback.strategy
    assert new_decision.apply_threshold == fallback.apply_threshold


def test_smote_skipped_at_boundary_min_eq_k():
    """min_count == k_neighbors is still insufficient — SMOTE needs k+1 minorities."""
    y = np.array([0] * 15 + [1] * 5)  # min_count = 5
    decision = _make_decision("smote", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is not None
    assert new_decision.strategy == "none"


def test_smote_passes_when_minority_strictly_greater_than_k():
    """min_count > k_neighbors → original decision passes through unchanged."""
    y = np.array([0] * 20 + [1] * 6)  # min_count = 6, k = 5
    decision = _make_decision("smote", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is None
    assert new_decision is decision  # identity preserved


def test_smote_tomek_also_guarded():
    """SMOTETomek wraps SMOTE internally, so the same guard applies."""
    y = np.array([0] * 10 + [1] * 2)
    decision = _make_decision("smote_tomek", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is not None
    assert "strategy=smote_tomek" in warning
    assert new_decision.strategy == "none"


# ---------------------------------------------------------------------------
# Pass-through cases (no guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["none", "class_weight", "sample_weight", "random_undersampling"])
def test_non_smote_strategies_passthrough(strategy: str):
    """The guard is SMOTE-specific. Other strategies must not be downgraded."""
    y = np.array([0] * 19 + [1] * 1)  # would trigger if SMOTE
    decision = _make_decision(strategy, k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is None
    assert new_decision is decision


def test_empty_y_passthrough():
    """Empty y_train_fold should not crash; just return the decision."""
    decision = _make_decision("smote", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, np.array([]))

    assert warning is None
    assert new_decision is decision


# ---------------------------------------------------------------------------
# Edge cases on k_neighbors
# ---------------------------------------------------------------------------

def test_k_neighbors_none_defaults_to_five():
    """smote_k_neighbors=None must use the imblearn default (5) for the comparison."""
    y = np.array([0] * 20 + [1] * 5)  # min_count = 5, default k = 5 → fallback
    decision = _make_decision("smote", k_neighbors=None)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is not None
    assert "k_neighbors=5" in warning
    assert new_decision.strategy == "none"


def test_k_neighbors_smaller_allows_smote():
    """A smaller k_neighbors should make tiny minorities viable."""
    y = np.array([0] * 20 + [1] * 3)  # min_count = 3, k = 2 → ok
    decision = _make_decision("smote", k_neighbors=2)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is None
    assert new_decision is decision


# ---------------------------------------------------------------------------
# Multi-class: guard reads the smallest class
# ---------------------------------------------------------------------------

def test_multiclass_min_class_drives_guard():
    """In multi-class settings the guard must look at the rarest class."""
    # 30 of class 0, 10 of class 1, 2 of class 2 → min_count = 2
    y = np.array([0] * 30 + [1] * 10 + [2] * 2)
    decision = _make_decision("smote", k_neighbors=5)

    new_decision, warning = _smote_minority_guard(decision, y)

    assert warning is not None
    assert "min_count=2" in warning
    assert new_decision.strategy == "none"
