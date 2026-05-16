"""
Tests for end-to-end training reproducibility.

Guarantees that two runs with identical config (same random_state) produce
bit-identical metrics, and that changing the seed actually changes results.
This catches regressions where a stochastic component (splitter, SMOTE,
GridSearch CV, RandomForest, train_test_split, learning-curve splitter, ...)
silently bypasses cfg.random_state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.training.config.schema import TrainingConfig


def _classification_df(n: int = 200, n_minority: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_majority = n - n_minority
    y = np.array([0] * n_majority + [1] * n_minority)
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "age": rng.normal(52, 10, n),
            "bmi": rng.normal(25, 5, n),
            "glucose": rng.normal(110, 20, n),
            "sex": np.where(rng.random(n) < 0.5, "F", "M"),
            "label": y,
        }
    )


def _regression_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _cfg(
    *,
    target: str,
    task_type: str,
    random_state: int,
    split_method: str = "holdout",
    balancing: str = "none",
) -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": target,
            "taskType": task_type,
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1"] if task_type == "classification" else ["r2", "mae"],
            "splitMethod": split_method,
            "kFolds": 3,
            "shuffle": True,
            "trainRatio": 0.7,
            "valRatio": 0.15,
            "testRatio": 0.15,
            "randomState": random_state,
            "useGridSearch": False,
            "balancing": {"strategy": balancing},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def _scalar_metrics(metrics_json: dict) -> dict[str, float]:
    """Extract only numeric scalar metrics for deterministic comparison."""
    out: dict[str, float] = {}
    for section in ("train", "val", "test"):
        block = metrics_json.get(section)
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"{section}.{k}"] = float(v)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Holdout — same seed → identical metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_classification_same_seed_is_bit_identical():
    import app.services.training.orchestrator as orch

    df = _classification_df(n=200, n_minority=50, seed=1)

    r1 = orch.run_one_model(df.copy(), _cfg(target="label", task_type="classification", random_state=123), model_type="randomforest")
    r2 = orch.run_one_model(df.copy(), _cfg(target="label", task_type="classification", random_state=123), model_type="randomforest")

    m1 = _scalar_metrics(r1.metrics_json)
    m2 = _scalar_metrics(r2.metrics_json)

    assert m1 and m2, "Expected non-empty metrics blocks"
    assert m1.keys() == m2.keys()
    for k in m1:
        assert m1[k] == pytest.approx(m2[k], abs=1e-12, rel=1e-9), (
            f"Metric '{k}' diverges between two same-seed runs: {m1[k]} vs {m2[k]}"
        )


def test_holdout_regression_same_seed_is_bit_identical():
    import app.services.training.orchestrator as orch

    df = _regression_df(n=200, seed=3)

    r1 = orch.run_one_model(df.copy(), _cfg(target="y", task_type="regression", random_state=7), model_type="randomforest")
    r2 = orch.run_one_model(df.copy(), _cfg(target="y", task_type="regression", random_state=7), model_type="randomforest")

    m1 = _scalar_metrics(r1.metrics_json)
    m2 = _scalar_metrics(r2.metrics_json)

    assert m1.keys() == m2.keys()
    for k in m1:
        assert m1[k] == pytest.approx(m2[k], abs=1e-12, rel=1e-9), (
            f"Metric '{k}' diverges between two same-seed runs: {m1[k]} vs {m2[k]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Different seed → different metrics (sanity: seed actually has an effect)
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_different_seed_changes_metrics():
    import app.services.training.orchestrator as orch

    df = _classification_df(n=200, n_minority=50, seed=1)

    r1 = orch.run_one_model(df.copy(), _cfg(target="label", task_type="classification", random_state=123), model_type="randomforest")
    r2 = orch.run_one_model(df.copy(), _cfg(target="label", task_type="classification", random_state=999), model_type="randomforest")

    m1 = _scalar_metrics(r1.metrics_json)
    m2 = _scalar_metrics(r2.metrics_json)

    assert any(m1[k] != m2[k] for k in m1.keys() & m2.keys()), (
        "Changing random_state should affect at least one metric. "
        "If this fails, a stochastic component is ignoring cfg.random_state."
    )


# ──────────────────────────────────────────────────────────────────────────────
# SMOTE path — same seed → identical metrics (covers resampler seeding)
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_with_smote_same_seed_is_bit_identical():
    import app.services.training.orchestrator as orch

    df = _classification_df(n=300, n_minority=40, seed=5)

    r1 = orch.run_one_model(
        df.copy(),
        _cfg(target="label", task_type="classification", random_state=42, balancing="smote"),
        model_type="randomforest",
    )
    r2 = orch.run_one_model(
        df.copy(),
        _cfg(target="label", task_type="classification", random_state=42, balancing="smote"),
        model_type="randomforest",
    )

    m1 = _scalar_metrics(r1.metrics_json)
    m2 = _scalar_metrics(r2.metrics_json)

    assert m1.keys() == m2.keys()
    for k in m1:
        assert m1[k] == pytest.approx(m2[k], abs=1e-12, rel=1e-9), (
            f"SMOTE path: metric '{k}' diverges between two same-seed runs: {m1[k]} vs {m2[k]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# K-Fold CV — same seed → identical fold-level results
# ──────────────────────────────────────────────────────────────────────────────

def test_kfold_same_seed_is_bit_identical():
    import app.services.training.orchestrator as orch

    df = _classification_df(n=200, n_minority=50, seed=11)

    cfg1 = _cfg(target="label", task_type="classification", random_state=2026, split_method="stratified_kfold")
    cfg2 = _cfg(target="label", task_type="classification", random_state=2026, split_method="stratified_kfold")

    r1 = orch.run_one_model(df.copy(), cfg1, model_type="randomforest")
    r2 = orch.run_one_model(df.copy(), cfg2, model_type="randomforest")

    fr1 = r1.metrics_json.get("fold_results", [])
    fr2 = r2.metrics_json.get("fold_results", [])

    assert len(fr1) == len(fr2) > 0

    for i, (a, b) in enumerate(zip(fr1, fr2)):
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        ma = a.get("val_metrics", {})
        mb = b.get("val_metrics", {})
        for k in ma:
            if isinstance(ma[k], (int, float)) and isinstance(mb.get(k), (int, float)):
                assert ma[k] == pytest.approx(mb[k], abs=1e-12, rel=1e-9), (
                    f"Fold {i}: val metric '{k}' diverges: {ma[k]} vs {mb[k]}"
                )
