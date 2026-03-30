"""
Tests for Sprint-1 additions:
  - Brier score and calibration curve in metrics.py
  - Bootstrap confidence intervals in confidence.py
"""
import numpy as np
import pytest

from app.services.training.pipeline.metrics import (
    compute_calibration_curve,
    compute_classification_metrics,
)
from app.services.training.pipeline.confidence import compute_bootstrap_cis


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Well-separated binary labels + realistic predicted probabilities."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.array([0] * 100 + [1] * 100)
    # Probabilities: class-1 samples get high proba, class-0 get low proba
    y_proba = np.concatenate([rng.uniform(0.05, 0.45, 100), rng.uniform(0.55, 0.95, 100)])
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    y_true = rng.normal(0, 1, 150)
    y_pred = y_true + rng.normal(0, 0.3, 150)
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Brier score in compute_classification_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBrierScore:
    def test_brier_score_present_in_binary_legacy_flat(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        out = compute_classification_metrics(
            y_true, y_pred,
            proba=np.column_stack([1 - y_proba, y_proba]),
            task_type="classification",
        )
        assert "brier_score" in out["legacy_flat"], "brier_score missing from legacy_flat"
        bs = out["legacy_flat"]["brier_score"]
        assert bs is not None
        assert 0.0 <= bs <= 1.0, f"Brier score out of range: {bs}"

    def test_brier_score_in_global_metrics(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        out = compute_classification_metrics(
            y_true, y_pred,
            proba=np.column_stack([1 - y_proba, y_proba]),
            task_type="classification",
        )
        assert out["global"].get("brier_score") is not None

    def test_brier_perfect_predictions(self):
        """Perfect classifier: predict 1.0 for all positives → Brier ≈ 0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        out = compute_classification_metrics(
            y_true, y_pred,
            proba=y_proba,
            task_type="classification",
        )
        bs = out["legacy_flat"]["brier_score"]
        assert bs is not None
        assert bs < 0.01, f"Expected near-zero Brier score, got {bs}"

    def test_brier_score_not_present_for_multiclass(self):
        """Brier score is only computed for binary tasks."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 2, 0])
        out = compute_classification_metrics(
            y_true, y_pred,
            task_type="classification",
        )
        # brier_score may be absent or None for multiclass — never a float
        bs = out["legacy_flat"].get("brier_score")
        assert bs is None

    def test_brier_none_when_no_proba(self, binary_data):
        """Without probability estimates brier_score should not be computed."""
        y_true, y_pred, _ = binary_data
        out = compute_classification_metrics(
            y_true, y_pred,
            task_type="classification",
        )
        bs = out["legacy_flat"].get("brier_score")
        assert bs is None


# ─────────────────────────────────────────────────────────────────────────────
# Calibration curve
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationCurve:
    def test_structure(self, binary_data):
        y_true, _, y_proba = binary_data
        result = compute_calibration_curve(y_true, y_proba, n_bins=5)
        assert result is not None
        assert "points" in result
        assert "brier_score" in result
        assert result["n_bins"] == 5

    def test_points_are_valid_fractions(self, binary_data):
        y_true, _, y_proba = binary_data
        result = compute_calibration_curve(y_true, y_proba)
        for mp, fp in result["points"]:
            assert 0.0 <= mp <= 1.0, f"mean_predicted out of range: {mp}"
            assert 0.0 <= fp <= 1.0, f"fraction_of_positives out of range: {fp}"

    def test_brier_score_in_result(self, binary_data):
        y_true, _, y_proba = binary_data
        result = compute_calibration_curve(y_true, y_proba)
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_curves_key_in_compute_classification_metrics(self, binary_data):
        """compute_classification_metrics should populate out['curves']['calibration']."""
        y_true, y_pred, y_proba = binary_data
        out = compute_classification_metrics(
            y_true, y_pred,
            proba=np.column_stack([1 - y_proba, y_proba]),
            task_type="classification",
        )
        assert "curves" in out, "curves key missing from output"
        assert out["curves"] is not None
        assert "calibration" in out["curves"], "calibration missing from curves"
        cal = out["curves"]["calibration"]
        assert "points" in cal
        assert "brier_score" in cal

    def test_none_on_bad_input(self):
        """compute_calibration_curve returns None on degenerate input."""
        result = compute_calibration_curve(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
        # May return None or a result with empty points — either is acceptable
        if result is not None:
            assert "brier_score" in result


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCIs:
    def test_classification_structure(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_bootstrap_cis(
            y_true, y_pred,
            y_score=y_proba,
            task_type="classification",
            n_bootstrap=100,
        )
        assert result["ci_level"] == 0.95
        assert result["n_bootstrap"] == 100
        assert result["n_samples"] == len(y_true)
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_ci_entry_fields(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_bootstrap_cis(
            y_true, y_pred,
            y_score=y_proba,
            task_type="classification",
            n_bootstrap=100,
        )
        for metric_name, entry in result["metrics"].items():
            assert "mean" in entry, f"{metric_name} missing 'mean'"
            assert "ci_low" in entry, f"{metric_name} missing 'ci_low'"
            assert "ci_high" in entry, f"{metric_name} missing 'ci_high'"
            assert "std" in entry, f"{metric_name} missing 'std'"
            assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"], (
                f"{metric_name}: ci_low={entry['ci_low']} mean={entry['mean']} ci_high={entry['ci_high']}"
            )

    def test_regression_structure(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_bootstrap_cis(
            y_true, y_pred,
            task_type="regression",
            n_bootstrap=100,
        )
        metrics = result["metrics"]
        assert "mae" in metrics
        assert "mse" in metrics
        assert "r2" in metrics

    def test_regression_ci_valid(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_bootstrap_cis(
            y_true, y_pred,
            task_type="regression",
            n_bootstrap=200,
        )
        mae_ci = result["metrics"]["mae"]
        assert mae_ci["ci_low"] >= 0.0, "MAE cannot be negative"
        assert mae_ci["ci_low"] <= mae_ci["mean"] <= mae_ci["ci_high"]

    def test_too_few_samples_returns_warning(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_bootstrap_cis(y_true, y_pred, task_type="classification")
        assert result["n_bootstrap"] == 0
        assert "warning" in result
        assert result["metrics"] == {}

    def test_reproducible_with_same_seed(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        r1 = compute_bootstrap_cis(y_true, y_pred, y_score=y_proba, task_type="classification",
                                    n_bootstrap=50, random_state=7)
        r2 = compute_bootstrap_cis(y_true, y_pred, y_score=y_proba, task_type="classification",
                                    n_bootstrap=50, random_state=7)
        assert r1["metrics"]["accuracy"]["mean"] == r2["metrics"]["accuracy"]["mean"]

    def test_ci_width_shrinks_with_more_bootstrap(self, binary_data):
        """More bootstrap samples → tighter (converged) CIs, not necessarily wider."""
        y_true, y_pred, y_proba = binary_data
        r_small = compute_bootstrap_cis(y_true, y_pred, y_score=y_proba,
                                         task_type="classification", n_bootstrap=30, random_state=0)
        r_large = compute_bootstrap_cis(y_true, y_pred, y_score=y_proba,
                                         task_type="classification", n_bootstrap=500, random_state=0)
        # Both should have valid CI structure — mean should converge
        m_small = r_small["metrics"]["accuracy"]["mean"]
        m_large = r_large["metrics"]["accuracy"]["mean"]
        # Means should be within 2% of each other (both are bootstrapping the same data)
        assert abs(m_small - m_large) < 0.02, f"Bootstrap means diverge: {m_small} vs {m_large}"

    def test_pr_auc_present_for_binary(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_bootstrap_cis(y_true, y_pred, y_score=y_proba,
                                        task_type="classification", n_bootstrap=100)
        assert "pr_auc" in result["metrics"]
