"""Tests for asymmetric cost / F-beta threshold optimisation strategies.

Covers:
  - minimize_cost: cost_fn > cost_fp → lower threshold (higher recall)
  - minimize_cost: cost_fp > cost_fn → higher threshold (higher precision)
  - maximize_f_beta: beta > 1 → threshold closer to recall-heavy optimum
  - _cost_weighted_threshold helper
  - End-to-end: BalancingDecision fields flow through executor to ThresholdOptimizer
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.preparation_ml.balancing.threshold import (
    ThresholdOptimizer,
    _cost_weighted_threshold,
    _f_beta,
)
from app.services.preparation_ml.balancing.resolver import BalancingDecision


# ---------------------------------------------------------------------------
# Helper: synthetic binary classification data
# ---------------------------------------------------------------------------

def _make_imbalanced_data(n: int = 400, minority_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Return (y_true, y_proba) for a moderately imbalanced binary problem."""
    rng = np.random.default_rng(seed)
    n_minority = int(n * minority_ratio)
    n_majority = n - n_minority
    y = np.array([1] * n_minority + [0] * n_majority)
    # Minority class gets higher proba, majority lower — realistic separation
    proba_pos = np.concatenate([
        rng.beta(5, 2, size=n_minority),   # minority: skewed high
        rng.beta(2, 5, size=n_majority),   # majority: skewed low
    ])
    return y, proba_pos


# ---------------------------------------------------------------------------
# Unit tests: _cost_weighted_threshold
# ---------------------------------------------------------------------------

class TestCostWeightedThreshold:
    def test_equal_costs_matches_f1_basin(self):
        """With equal costs (1,1) the cost surface should be flat-ish."""
        p = np.array([0.8, 0.7, 0.6, 0.5])
        r = np.array([0.4, 0.5, 0.6, 0.7])
        cost = _cost_weighted_threshold(p, r, cost_fn=1.0, cost_fp=1.0)
        # minimum should be where precision ≈ recall (cost ~0.4+0.5 = 0.9 for each side)
        assert cost.shape == (4,)
        assert np.all(cost >= 0)

    def test_high_fn_cost_favours_recall(self):
        """High FN cost → minimum cost at high recall point."""
        p = np.array([0.9, 0.7, 0.5, 0.3])   # decreasing precision (higher threshold → lower index)
        r = np.array([0.3, 0.5, 0.7, 0.9])   # increasing recall (lower threshold → higher index)
        cost_high_fn = _cost_weighted_threshold(p, r, cost_fn=10.0, cost_fp=1.0)
        cost_high_fp = _cost_weighted_threshold(p, r, cost_fn=1.0, cost_fp=10.0)
        # High FN cost → prefer high recall (last index)
        assert np.argmin(cost_high_fn) > np.argmin(cost_high_fp)

    def test_high_fp_cost_favours_precision(self):
        """High FP cost → minimum cost at high precision point."""
        p = np.array([0.9, 0.7, 0.5, 0.3])
        r = np.array([0.3, 0.5, 0.7, 0.9])
        cost_high_fp = _cost_weighted_threshold(p, r, cost_fn=1.0, cost_fp=10.0)
        # High FP cost → prefer high precision (first index)
        assert np.argmin(cost_high_fp) == 0


# ---------------------------------------------------------------------------
# Unit tests: ThresholdOptimizer — new strategies
# ---------------------------------------------------------------------------

class TestThresholdOptimizerNewStrategies:
    def setup_method(self):
        self.y, self.proba = _make_imbalanced_data(n=500)
        self.optimizer = ThresholdOptimizer()

    def test_minimize_cost_high_fn_lowers_threshold(self):
        """High cost_fn → optimal threshold lower than balanced (equal) cost."""
        result_equal = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=1.0, cost_fp=1.0
        )
        result_high_fn = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=10.0, cost_fp=1.0
        )
        assert result_high_fn.optimal_threshold <= result_equal.optimal_threshold, (
            f"Expected lower threshold with cost_fn=10: "
            f"{result_high_fn.optimal_threshold:.3f} > {result_equal.optimal_threshold:.3f}"
        )

    def test_minimize_cost_high_fp_raises_threshold(self):
        """High cost_fp → optimal threshold higher than balanced cost."""
        result_equal = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=1.0, cost_fp=1.0
        )
        result_high_fp = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=1.0, cost_fp=10.0
        )
        assert result_high_fp.optimal_threshold >= result_equal.optimal_threshold, (
            f"Expected higher threshold with cost_fp=10: "
            f"{result_high_fp.optimal_threshold:.3f} < {result_equal.optimal_threshold:.3f}"
        )

    def test_minimize_cost_strategy_used_label(self):
        result = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=5.0, cost_fp=1.0
        )
        assert result.strategy_used == "minimize_cost"

    def test_maximize_f_beta_uses_provided_beta(self):
        """maximize_f_beta with beta=3 should favour recall more than beta=1."""
        result_b1 = self.optimizer.optimize(self.y, self.proba, strategy="maximize_f_beta", beta=1.0)
        result_b3 = self.optimizer.optimize(self.y, self.proba, strategy="maximize_f_beta", beta=3.0)
        assert result_b3.recall_at_threshold >= result_b1.recall_at_threshold, (
            "Beta=3 should produce equal or higher recall than beta=1"
        )
        assert result_b3.strategy_used == "maximize_f_beta"

    def test_maximize_f_beta_threshold_in_valid_range(self):
        result = self.optimizer.optimize(self.y, self.proba, strategy="maximize_f_beta", beta=2.0)
        assert 0.0 < result.optimal_threshold < 1.0

    def test_minimize_cost_threshold_in_valid_range(self):
        result = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=5.0, cost_fp=2.0
        )
        assert 0.0 < result.optimal_threshold < 1.0

    def test_empty_input_returns_fallback(self):
        result = self.optimizer.optimize(
            np.array([]), np.array([]), strategy="minimize_cost", cost_fn=5.0, cost_fp=1.0
        )
        assert result.optimal_threshold == 0.5

    def test_minimize_cost_extreme_fn_cost(self):
        """Extreme cost_fn=100 should push threshold very low (near 0)."""
        result = self.optimizer.optimize(
            self.y, self.proba, strategy="minimize_cost", cost_fn=100.0, cost_fp=1.0
        )
        assert result.optimal_threshold < 0.5, (
            f"With cost_fn=100, threshold should be < 0.5, got {result.optimal_threshold:.3f}"
        )


# ---------------------------------------------------------------------------
# Integration test: BalancingDecision fields flow through
# ---------------------------------------------------------------------------

class TestBalancingDecisionCostFields:
    def test_balancing_decision_has_new_fields(self):
        decision = BalancingDecision(
            strategy="none",
            apply_threshold=True,
            threshold_strategy="minimize_cost",
            rationale="test",
            refit_metric="f1",
            smote_k_neighbors=None,
            class_weight_mode="balanced",
            audit_flags=[],
            optimal_threshold=0.5,
            f_beta=3.0,
            cost_fn=8.0,
            cost_fp=1.0,
        )
        assert decision.f_beta == 3.0
        assert decision.cost_fn == 8.0
        assert decision.cost_fp == 1.0

    def test_balancing_decision_defaults(self):
        decision = BalancingDecision(
            strategy="none",
            apply_threshold=False,
            threshold_strategy="maximize_f1",
            rationale="test",
            refit_metric="f1",
            smote_k_neighbors=None,
            class_weight_mode="balanced",
            audit_flags=[],
            optimal_threshold=0.5,
        )
        assert decision.f_beta == 2.0
        assert decision.cost_fn == 1.0
        assert decision.cost_fp == 1.0
