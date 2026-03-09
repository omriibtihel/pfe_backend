"""Tests for DatasetProfiler (Phase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.training.dataset_profiler import DatasetProfiler, DatasetProfile


@pytest.fixture()
def profiler() -> DatasetProfiler:
    return DatasetProfiler()


def _make_binary_df(n: int = 500, ir: float = 1.0) -> pd.DataFrame:
    """Create a balanced or imbalanced binary classification DataFrame."""
    minority = max(1, int(n / (1 + ir)))
    majority = n - minority
    y = np.array([0] * majority + [1] * minority)
    rng = np.random.RandomState(42)
    X = rng.randn(n, 4)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


def _make_regression_df(n: int = 800) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(n, 5), columns=list("abcde"))
    df["target"] = rng.randn(n) * 10 + 50
    return df


def _make_multiclass_df(n: int = 600, n_classes: int = 4) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    X = rng.randn(n, 3)
    df = pd.DataFrame(X, columns=["x", "y", "z"])
    df["label"] = (rng.randint(0, n_classes, size=n)).astype(str)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Basic shape tests
# ──────────────────────────────────────────────────────────────────────────────

def test_binary_classification_detected(profiler):
    df = _make_binary_df(n=500, ir=1.0)
    prof = profiler.profile(df, "target")
    assert prof.task_type == "binary_classification"
    assert prof.n_classes == 2
    assert prof.n_samples == 500
    assert prof.n_features == 4


def test_regression_detected(profiler):
    df = _make_regression_df(800)
    prof = profiler.profile(df, "target")
    assert prof.task_type == "regression"
    assert prof.n_classes is None
    assert prof.imbalance_ratio is None


def test_multiclass_detected(profiler):
    df = _make_multiclass_df(n=600, n_classes=4)
    prof = profiler.profile(df, "label")
    assert prof.task_type == "multiclass_classification"
    assert prof.n_classes == 4


# ──────────────────────────────────────────────────────────────────────────────
# Size categories
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n, expected_cat", [
    (100, "tiny"),
    (1_000, "small"),
    (10_000, "medium"),
    (100_000, "large"),
])
def test_size_categories(profiler, n, expected_cat):
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
    df["t"] = np.tile([0, 1], n // 2 + 1)[:n]
    prof = profiler.profile(df, "t")
    assert prof.dataset_size_category == expected_cat


# ──────────────────────────────────────────────────────────────────────────────
# Imbalance
# ──────────────────────────────────────────────────────────────────────────────

def test_balanced_dataset(profiler):
    df = _make_binary_df(n=500, ir=1.0)
    prof = profiler.profile(df, "target")
    assert prof.imbalance_ratio is not None
    assert prof.imbalance_ratio <= 2.0
    assert prof.recommended_resampling is None


def test_severe_imbalance(profiler):
    df = _make_binary_df(n=500, ir=20.0)
    prof = profiler.profile(df, "target")
    assert prof.imbalance_ratio is not None
    assert prof.imbalance_ratio > 10.0
    assert prof.recommended_resampling in {"smote_tomek", "smote", "class_weight"}


# ──────────────────────────────────────────────────────────────────────────────
# Missing values
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_values_detected(profiler):
    df = _make_binary_df(n=200, ir=1.0)
    df.loc[0:19, "a"] = np.nan   # 10% of col a
    prof = profiler.profile(df, "target")
    assert prof.has_missing_values is True
    assert prof.missing_ratio > 0.0


def test_no_missing_values(profiler):
    df = _make_binary_df(n=200, ir=1.0)
    prof = profiler.profile(df, "target")
    assert prof.has_missing_values is False
    assert prof.missing_ratio == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CV strategy
# ──────────────────────────────────────────────────────────────────────────────

def test_cv_strategy_small_classification(profiler):
    df = _make_binary_df(n=300, ir=1.0)
    prof = profiler.profile(df, "target")
    assert prof.recommended_cv_strategy == "stratified_kfold"


def test_cv_strategy_large_dataset(profiler):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(60_000, 3), columns=["a", "b", "c"])
    df["t"] = np.tile([0, 1], 30_000)
    prof = profiler.profile(df, "t")
    assert prof.recommended_cv_strategy == "holdout"


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_target_column_raises(profiler):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column"):
        profiler.profile(df, "nonexistent")


# ──────────────────────────────────────────────────────────────────────────────
# Metric selection
# ──────────────────────────────────────────────────────────────────────────────

def test_metric_regression(profiler):
    df = _make_regression_df(n=800)
    prof = profiler.profile(df, "target")
    assert prof.recommended_metric == "rmse"


def test_metric_balanced_binary(profiler):
    df = _make_binary_df(n=500, ir=1.0)
    prof = profiler.profile(df, "target")
    assert prof.recommended_metric == "roc_auc"


def test_metric_imbalanced_binary(profiler):
    df = _make_binary_df(n=500, ir=15.0)
    prof = profiler.profile(df, "target")
    assert prof.recommended_metric in {"f1", "pr_auc"}
