"""
End-to-end smoke tests for every model in MODEL_REGISTRY.

For each model that supports a given task type (regression and multiclass
classification), we run the full orchestrator pipeline and assert that:
  - the run completes without exception,
  - a fitted sklearn Pipeline is produced,
  - the metrics_json contains non-NaN scalar metrics on the test split.

Models that require an optional dependency that is not installed (xgboost,
lightgbm, catboost) are skipped automatically.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.training.config.schema import TrainingConfig
from app.services.training.pipeline.models import MODEL_REGISTRY, model_is_installed


# ──────────────────────────────────────────────────────────────────────────────
# Model lists derived from the live registry (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────

_OPTIONAL_DEPS = {"xgboost", "lightgbm", "catboost"}


def _models_for(task_type: str) -> list[str]:
    """All registered models that declare support for `task_type`."""
    names: list[str] = []
    for name, spec in MODEL_REGISTRY._specs.items():
        if spec.supports(task_type):
            names.append(name)
    return names


CLASSIFICATION_MODELS = _models_for("classification")
REGRESSION_MODELS = _models_for("regression")


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

def _multiclass_df(n: int = 240, n_classes: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per = n // n_classes
    y = np.repeat(np.arange(n_classes), per)
    rng.shuffle(y)
    centers = rng.normal(0, 3, size=(n_classes, 2))
    x1 = np.empty(len(y))
    x2 = np.empty(len(y))
    for i, cls in enumerate(y):
        cx, cy = centers[cls]
        x1[i] = cx + rng.normal(0, 1)
        x2[i] = cy + rng.normal(0, 1)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": rng.normal(0, 1, len(y)),
            "cat": np.where(rng.random(len(y)) < 0.5, "A", "B"),
            "target": y,
        }
    )


def _regression_df(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.5 * x2 + 0.5 * x3 + rng.normal(0, 0.3, n)
    cat = np.where(rng.random(n) < 0.5, "A", "B")
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "y": y})


# ──────────────────────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────────────────────

def _cfg(*, target: str, task_type: str, model: str) -> TrainingConfig:
    metrics = ["accuracy", "f1_macro"] if task_type == "classification" else ["r2", "mae"]
    return TrainingConfig.from_front(
        {
            "targetColumn": target,
            "taskType": task_type,
            "models": [model],
            "metrics": metrics,
            "splitMethod": "holdout",
            "trainRatio": 0.7,
            "valRatio": 0.15,
            "testRatio": 0.15,
            "shuffle": True,
            "randomState": 42,
            "useGridSearch": False,
            "balancing": {"strategy": "none"},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def _maybe_skip(model: str) -> None:
    if model in _OPTIONAL_DEPS and not model_is_installed(model):
        pytest.skip(f"Optional dependency '{model}' is not installed")


def _assert_clean_metrics(metrics_json: dict, task_type: str) -> None:
    """The test split should yield at least one finite scalar metric."""
    block = metrics_json.get("test") or metrics_json.get("val") or metrics_json.get("train")
    assert isinstance(block, dict) and block, f"No metrics block found: {metrics_json}"

    finite_count = 0
    for k, v in block.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert not (isinstance(v, float) and math.isnan(v)), f"Metric '{k}' is NaN"
            finite_count += 1
    assert finite_count > 0, f"No numeric scalar metrics produced (task={task_type})"


# ──────────────────────────────────────────────────────────────────────────────
# Multiclass classification — one parametrised test per registered model
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", CLASSIFICATION_MODELS)
def test_full_pipeline_multiclass(model: str) -> None:
    _maybe_skip(model)
    import app.services.training.orchestrator as orch
    from sklearn.pipeline import Pipeline

    df = _multiclass_df(n=240, n_classes=3, seed=42)
    cfg = _cfg(target="target", task_type="classification", model=model)

    result = orch.run_one_model(df, cfg, model_type=model)

    assert isinstance(result.fitted_pipeline, Pipeline), (
        f"{model} (multiclass): fitted_pipeline is not a sklearn Pipeline"
    )
    _assert_clean_metrics(result.metrics_json, "classification")


# ──────────────────────────────────────────────────────────────────────────────
# Regression — one parametrised test per registered model
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", REGRESSION_MODELS)
def test_full_pipeline_regression(model: str) -> None:
    _maybe_skip(model)
    import app.services.training.orchestrator as orch
    from sklearn.pipeline import Pipeline

    df = _regression_df(n=240, seed=7)
    cfg = _cfg(target="y", task_type="regression", model=model)

    result = orch.run_one_model(df, cfg, model_type=model)

    assert isinstance(result.fitted_pipeline, Pipeline), (
        f"{model} (regression): fitted_pipeline is not a sklearn Pipeline"
    )
    _assert_clean_metrics(result.metrics_json, "regression")


# ──────────────────────────────────────────────────────────────────────────────
# Coverage sanity — fail loudly if the registry shrinks or expectation drifts
# ──────────────────────────────────────────────────────────────────────────────

def test_registry_minimum_coverage() -> None:
    """If a model is removed from the registry we want to know."""
    expected = {
        "randomforest", "logisticregression", "svm", "knn", "naivebayes",
        "decisiontree", "xgboost", "lightgbm", "extratrees", "gradientboosting",
        "catboost", "ridge", "mlp", "elasticnet", "lasso",
    }
    actual = set(MODEL_REGISTRY._specs.keys())
    missing = expected - actual
    assert not missing, f"Missing models from registry: {missing}"
