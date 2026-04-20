"""
Tests for global and local SHAP computation.

Covers: output structure, field types, sorting, explainer routing,
too-small dataset guard, graceful failure, local SHAP per-row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.services.shap import compute_global_shap, compute_local_shap
from app.services.shap._utils import extract_model_from_pipeline, get_explainer_type


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_clf_pipe(n_samples=100, n_features=5, estimator=None):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                n_informative=4, n_redundant=1, random_state=0)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipe.fit(df, y)
    return pipe, df, y, cols


def _make_reg_pipe(n_samples=100, n_features=4):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=0)
    cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    pipe.fit(df, y)
    return pipe, df, y, cols


# ──────────────────────────────────────────────────────────────────────────────
# _utils tests
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_model_from_sklearn_pipeline():
    pipe, df, y, _ = _make_clf_pipe()
    model = extract_model_from_pipeline(pipe)
    assert isinstance(model, RandomForestClassifier)


def test_get_explainer_type_tree():
    assert get_explainer_type(RandomForestClassifier()) == "tree"
    assert get_explainer_type(DecisionTreeClassifier()) == "tree"
    assert get_explainer_type(GradientBoostingClassifier()) == "tree"


def test_get_explainer_type_linear():
    assert get_explainer_type(LogisticRegression()) == "linear"
    assert get_explainer_type(Ridge()) == "linear"


def test_get_explainer_type_kernel():
    from sklearn.svm import SVC
    assert get_explainer_type(SVC()) == "kernel"
    from sklearn.neighbors import KNeighborsClassifier
    assert get_explainer_type(KNeighborsClassifier()) == "kernel"


# ──────────────────────────────────────────────────────────────────────────────
# Global SHAP — structure
# ──────────────────────────────────────────────────────────────────────────────

def test_global_shap_returns_dict():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert isinstance(result, dict)


def test_global_shap_top_level_keys():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    expected_keys = {"summary", "expected_value", "explainer_type", "n_samples", "background_data"}
    assert expected_keys.issubset(result.keys())


def test_global_shap_summary_non_empty():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    assert len(result["summary"]) > 0


def test_global_shap_summary_item_fields():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    for item in result["summary"]:
        assert "feature" in item
        assert "mean_abs_shap" in item
        assert "mean_shap" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["mean_abs_shap"], float)
        assert isinstance(item["mean_shap"], float)


def test_global_shap_summary_sorted_descending():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    vals = [d["mean_abs_shap"] for d in result["summary"]]
    assert vals == sorted(vals, reverse=True)


def test_global_shap_mean_abs_non_negative():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    for item in result["summary"]:
        assert item["mean_abs_shap"] >= 0.0


def test_global_shap_explainer_type_tree_for_rf():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    assert result["explainer_type"] == "tree"


def test_global_shap_explainer_type_linear_for_logreg():
    pipe, df, y, _ = _make_clf_pipe(estimator=LogisticRegression(max_iter=200))
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    assert result["explainer_type"] == "linear"


def test_global_shap_n_samples_positive():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    assert result["n_samples"] > 0


def test_global_shap_at_most_30_features():
    X, y = make_classification(n_samples=100, n_features=40, n_informative=10, random_state=0)
    cols = [f"x{i}" for i in range(40)]
    df = pd.DataFrame(X, columns=cols)
    pipe = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipe.fit(df, y)
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    assert len(result["summary"]) <= 30


# ──────────────────────────────────────────────────────────────────────────────
# Global SHAP — n < 10 guard
# ──────────────────────────────────────────────────────────────────────────────

def test_global_shap_returns_none_when_too_few_samples():
    pipe, df, y, _ = _make_clf_pipe(n_samples=50)
    result = compute_global_shap(pipe, df.iloc[:9], task_type="classification")
    assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Global SHAP — regression
# ──────────────────────────────────────────────────────────────────────────────

def test_global_shap_regression():
    pipe, df, y, _ = _make_reg_pipe()
    result = compute_global_shap(pipe, df, task_type="regression")
    assert result is not None
    assert result["explainer_type"] == "linear"
    assert len(result["summary"]) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Global SHAP — JSON-safe
# ──────────────────────────────────────────────────────────────────────────────

def test_global_shap_json_serializable():
    import json
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_global_shap(pipe, df, task_type="classification")
    assert result is not None
    json.dumps(result)


# ──────────────────────────────────────────────────────────────────────────────
# Local SHAP — structure
# ──────────────────────────────────────────────────────────────────────────────

def test_local_shap_returns_list():
    pipe, df, y, _ = _make_clf_pipe()
    row = df.iloc[[0]]
    result = compute_local_shap(pipe, row, task_type="classification")
    assert isinstance(result, list)


def test_local_shap_non_empty():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_local_shap(pipe, df.iloc[[0]], task_type="classification")
    assert result is not None and len(result) > 0


def test_local_shap_item_fields():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_local_shap(pipe, df.iloc[[0]], task_type="classification")
    assert result is not None
    for item in result:
        assert "feature" in item
        assert "shap_value" in item
        assert "data" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["shap_value"], float)


def test_local_shap_sorted_by_abs_descending():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_local_shap(pipe, df.iloc[[0]], task_type="classification")
    assert result is not None
    abs_vals = [abs(item["shap_value"]) for item in result]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_local_shap_regression():
    pipe, df, y, _ = _make_reg_pipe()
    result = compute_local_shap(pipe, df.iloc[[0]], task_type="regression")
    assert result is not None and len(result) > 0


def test_local_shap_json_serializable():
    import json
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_local_shap(pipe, df.iloc[[0]], task_type="classification")
    assert result is not None
    json.dumps(result)


def test_local_shap_returns_none_for_empty_row():
    pipe, df, y, _ = _make_clf_pipe()
    result = compute_local_shap(pipe, df.iloc[0:0], task_type="classification")
    assert result is None
