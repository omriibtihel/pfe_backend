"""
Tests for Sprint-2 learning curves module.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from app.services.training.pipeline.learning_curves import compute_learning_curve


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Structure tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningCurveStructure:
    def test_returns_expected_keys(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification", n_points=5, cv_splits=3,
        )
        assert result is not None
        required = {"train_sizes", "train_mean", "train_std", "val_mean", "val_std", "scoring", "n_samples"}
        assert required.issubset(set(result.keys())), f"Missing keys: {required - set(result.keys())}"

    def test_array_lengths_match(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification", n_points=6, cv_splits=3,
        )
        assert result is not None
        n = len(result["train_sizes"])
        assert len(result["train_mean"]) == n
        assert len(result["train_std"]) == n
        assert len(result["val_mean"]) == n
        assert len(result["val_std"]) == n

    def test_n_points_respected(self, classification_data):
        X, y = classification_data
        for n_pts in [4, 6]:
            result = compute_learning_curve(
                LogisticRegression(max_iter=300),
                X, y, task_type="classification", n_points=n_pts, cv_splits=3,
            )
            assert result is not None
            # Can be <= n_pts due to deduplication of sizes
            assert len(result["train_sizes"]) <= n_pts

    def test_train_sizes_increasing(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        sizes = result["train_sizes"]
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)), "train_sizes should be increasing"

    def test_scores_in_valid_range_classification(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        for v in result["train_mean"] + result["val_mean"]:
            assert 0.0 <= v <= 1.0, f"F1 score out of [0,1]: {v}"

    def test_stds_non_negative(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        for v in result["train_std"] + result["val_std"]:
            assert v >= 0.0, f"std cannot be negative: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Regression task
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningCurveRegression:
    def test_regression_returns_result(self, regression_data):
        X, y = regression_data
        result = compute_learning_curve(
            Ridge(),
            X, y, task_type="regression",
        )
        assert result is not None
        assert result["scoring"] == "r2"

    def test_regression_r2_range(self, regression_data):
        X, y = regression_data
        result = compute_learning_curve(
            Ridge(),
            X, y, task_type="regression",
        )
        assert result is not None
        # R2 can be negative (bad fit) but should be < 1 in practice
        for v in result["train_mean"] + result["val_mean"]:
            assert v <= 1.0, f"R2 cannot exceed 1: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningCurveEdgeCases:
    def test_too_small_dataset_returns_none(self):
        """Dataset with fewer samples than minimum threshold → None."""
        X = np.random.default_rng(0).standard_normal((10, 3))
        y = np.array([0, 1] * 5)
        result = compute_learning_curve(
            LogisticRegression(), X, y, task_type="classification",
        )
        assert result is None

    def test_n_samples_matches_input(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        assert result["n_samples"] == len(y)

    def test_scoring_field_is_string(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        assert isinstance(result["scoring"], str)

    def test_multiclass_uses_weighted_f1(self):
        """With 3 classes, scoring should be f1_weighted."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((150, 4))
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        result = compute_learning_curve(
            LogisticRegression(max_iter=500),
            X, y, task_type="classification", n_points=4,
        )
        assert result is not None
        assert result["scoring"] == "f1_weighted"

    def test_binary_uses_f1(self, classification_data):
        X, y = classification_data
        result = compute_learning_curve(
            LogisticRegression(max_iter=300),
            X, y, task_type="classification",
        )
        assert result is not None
        assert result["scoring"] == "f1"
