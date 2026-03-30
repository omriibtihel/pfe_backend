"""
Tests for compute_permutation_importance.

Covers: basic structure, field types, sorting, n < 20 guard,
feature_names parameter, regression mode, exception safety.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.training.pipeline.importance import compute_permutation_importance


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_clf_pipeline(n_features: int = 5):
    X, y = make_classification(n_samples=100, n_features=n_features, random_state=0)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=10, random_state=0))])
    pipe.fit(df, y)
    return pipe, df, y, cols


def _make_reg_pipeline(n_features: int = 4):
    X, y = make_regression(n_samples=100, n_features=n_features, random_state=0)
    cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=10, random_state=0))])
    pipe.fit(df, y)
    return pipe, df, y, cols


# ──────────────────────────────────────────────────────────────────────────────
# Basic structure tests
# ──────────────────────────────────────────────────────────────────────────────

def test_returns_list():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert isinstance(result, list), "Should return a list"


def test_non_empty_result():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None and len(result) > 0


def test_item_has_required_fields():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    for item in result:
        assert "feature" in item
        assert "mean" in item
        assert "std" in item


def test_field_types():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    for item in result:
        assert isinstance(item["feature"], str)
        assert isinstance(item["mean"], float)
        assert isinstance(item["std"], float)


def test_sorted_descending_by_mean():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    means = [d["mean"] for d in result]
    assert means == sorted(means, reverse=True), "Items should be sorted by mean descending"


def test_std_non_negative():
    pipe, df, y, cols = _make_clf_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    for item in result:
        assert item["std"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# n < 20 guard
# ──────────────────────────────────────────────────────────────────────────────

def test_returns_none_when_too_few_samples():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    pipe = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipe.fit(df, y)
    # Use only 19 samples
    result = compute_permutation_importance(pipe, df.iloc[:19], y[:19], task_type="classification")
    assert result is None


def test_accepts_exactly_20_samples():
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    pipe = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipe.fit(df, y)
    result = compute_permutation_importance(pipe, df.iloc[:20], y[:20], task_type="classification")
    assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# feature_names parameter
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_feature_names():
    pipe, df, y, _ = _make_clf_pipeline(n_features=5)
    custom_names = [f"custom_{i}" for i in range(5)]
    result = compute_permutation_importance(
        pipe, df, y, task_type="classification", feature_names=custom_names
    )
    assert result is not None
    returned_names = {item["feature"] for item in result}
    assert returned_names.issubset(set(custom_names))


def test_default_feature_names_from_dataframe_columns():
    pipe, df, y, cols = _make_clf_pipeline(n_features=4)
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    returned_names = {item["feature"] for item in result}
    assert returned_names.issubset(set(cols))


# ──────────────────────────────────────────────────────────────────────────────
# Regression mode
# ──────────────────────────────────────────────────────────────────────────────

def test_regression_mode():
    pipe, df, y, _ = _make_reg_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="regression")
    assert isinstance(result, list) and len(result) > 0


def test_regression_sorted_descending():
    pipe, df, y, _ = _make_reg_pipeline()
    result = compute_permutation_importance(pipe, df, y, task_type="regression")
    assert result is not None
    means = [d["mean"] for d in result]
    assert means == sorted(means, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Max features cap
# ──────────────────────────────────────────────────────────────────────────────

def test_at_most_30_features_returned():
    # Create a dataset with 40 features
    X, y = make_classification(n_samples=100, n_features=40, n_informative=10, random_state=0)
    cols = [f"x{i}" for i in range(40)]
    df = pd.DataFrame(X, columns=cols)
    pipe = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipe.fit(df, y)
    result = compute_permutation_importance(pipe, df, y, task_type="classification")
    assert result is not None
    assert len(result) <= 30


# ──────────────────────────────────────────────────────────────────────────────
# Exception safety
# ──────────────────────────────────────────────────────────────────────────────

def test_returns_none_on_exception():
    """Passing an unfitted object should not crash — returns None."""
    class BrokenPipe:
        def predict(self, X):
            raise RuntimeError("broken")

    df = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    y = np.random.randint(0, 2, 50)
    result = compute_permutation_importance(BrokenPipe(), df, y, task_type="classification")
    assert result is None
