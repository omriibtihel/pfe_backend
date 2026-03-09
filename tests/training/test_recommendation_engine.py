"""Tests for RecommendationEngine, MetricSelector, ImbalanceHandler, ConfigBuilder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.training.dataset_profiler import DatasetProfiler
from app.services.training.recommendation_engine import RecommendationEngine
from app.services.training.metric_selector import select_metrics
from app.services.training.imbalance_handler import recommend_imbalance_strategy
from app.services.training.config_builder import TrainingConfigBuilder
from app.services.training.config import TrainingConfig


profiler = DatasetProfiler()
engine = RecommendationEngine()
builder = TrainingConfigBuilder()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_profile(n: int = 500, ir: float = 1.0, task: str = "binary"):
    rng = np.random.RandomState(42)
    minority = max(5, int(n / (1 + ir)))
    majority = n - minority
    if task == "binary":
        y_vals = [0] * majority + [1] * minority
        df = pd.DataFrame(rng.randn(n, 4), columns=list("abcd"))
        df["t"] = y_vals
        return profiler.profile(df, "t")
    elif task == "regression":
        df = pd.DataFrame(rng.randn(n, 4), columns=list("abcd"))
        df["t"] = rng.randn(n) * 5 + 10
        return profiler.profile(df, "t")
    else:  # multiclass
        df = pd.DataFrame(rng.randn(n, 4), columns=list("abcd"))
        df["t"] = (rng.randint(0, 4, n)).astype(str)
        return profiler.profile(df, "t")


# ──────────────────────────────────────────────────────────────────────────────
# RecommendationEngine
# ──────────────────────────────────────────────────────────────────────────────

def test_recommendation_returns_models():
    prof = _make_profile(n=600, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    assert isinstance(rec.recommended_models, list)
    assert len(rec.recommended_models) >= 1
    assert len(rec.recommended_models) <= 4


def test_recommendation_config_payload_valid():
    prof = _make_profile(n=600, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    payload = rec.training_config_payload
    assert "taskType" in payload
    assert "models" in payload
    assert "balancing" in payload
    assert payload["configMode"] == "intelligent"


def test_recommendation_has_reasoning():
    prof = _make_profile(n=600, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    assert isinstance(rec.reasoning, dict)
    assert len(rec.reasoning) > 0


def test_recommendation_regression_metric():
    prof = _make_profile(n=800, task="regression")
    rec = engine.recommend(prof)
    assert rec.recommended_metric == "rmse"
    assert "ridge" in rec.recommended_models or "lightgbm" in rec.recommended_models


def test_recommendation_imbalanced_suggests_resampling():
    prof = _make_profile(n=500, ir=15.0, task="binary")
    rec = engine.recommend(prof)
    assert rec.recommended_resampling is not None
    assert rec.recommended_resampling != "none"


def test_recommendation_balanced_no_resampling():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    assert rec.recommended_resampling is None or rec.recommended_resampling == "none"


def test_recommendation_large_dataset_uses_holdout():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(60_000, 3), columns=["a", "b", "c"])
    df["t"] = np.tile([0, 1], 30_000)
    prof = profiler.profile(df, "t")
    rec = engine.recommend(prof)
    assert rec.recommended_cv_strategy == "holdout"


def test_recommendation_small_uses_cv():
    prof = _make_profile(n=200, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    assert rec.recommended_cv_strategy in {"stratified_kfold", "kfold"}


# ──────────────────────────────────────────────────────────────────────────────
# MetricSelector
# ──────────────────────────────────────────────────────────────────────────────

def test_metric_selector_regression():
    prof = _make_profile(n=500, task="regression")
    sel = select_metrics(prof)
    assert sel.primary == "rmse"
    assert "mae" in sel.secondary


def test_metric_selector_balanced_binary():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    sel = select_metrics(prof)
    assert sel.primary == "roc_auc"


def test_metric_selector_imbalanced_binary():
    prof = _make_profile(n=500, ir=20.0, task="binary")
    sel = select_metrics(prof)
    assert sel.primary in {"f1", "pr_auc"}


def test_metric_selector_multiclass():
    prof = _make_profile(n=500, task="multiclass")
    sel = select_metrics(prof)
    assert "f1" in sel.primary


# ──────────────────────────────────────────────────────────────────────────────
# ImbalanceHandler
# ──────────────────────────────────────────────────────────────────────────────

def test_imbalance_handler_regression():
    prof = _make_profile(n=500, task="regression")
    rec = recommend_imbalance_strategy(prof)
    assert rec.strategy is None
    assert rec.severity == "none"


def test_imbalance_handler_balanced():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = recommend_imbalance_strategy(prof)
    assert rec.strategy is None


def test_imbalance_handler_mild():
    prof = _make_profile(n=500, ir=2.5, task="binary")
    rec = recommend_imbalance_strategy(prof)
    assert rec.strategy == "class_weight"
    assert rec.severity == "mild"


def test_imbalance_handler_severe_smote():
    prof = _make_profile(n=500, ir=20.0, task="binary")
    rec = recommend_imbalance_strategy(prof, minority_count=25)
    assert rec.strategy in {"smote_tomek", "smote", "class_weight"}
    assert rec.apply_threshold is True


def test_imbalance_handler_smote_infeasible():
    prof = _make_profile(n=100, ir=15.0, task="binary")
    rec = recommend_imbalance_strategy(prof, minority_count=3)
    # SMOTE needs ≥ 6 minority samples
    assert rec.strategy == "class_weight"
    assert len(rec.feasibility_notes) > 0


# ──────────────────────────────────────────────────────────────────────────────
# TrainingConfigBuilder
# ──────────────────────────────────────────────────────────────────────────────

def test_builder_intelligent_no_overrides():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    payload = builder.build_intelligent(rec)
    assert payload["configMode"] == "intelligent"
    assert "models" in payload
    assert "balancing" in payload


def test_builder_intelligent_with_overrides():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    payload = builder.build_intelligent(rec, user_overrides={"models": ["logisticregression"]})
    assert payload["models"] == ["logisticregression"]
    assert "models" in payload.get("userOverrides", [])
    assert payload["configMode"] == "intelligent"


def test_builder_override_non_overridable_key_ignored():
    """taskType is not in _OVERRIDABLE_KEYS and should be ignored in user_overrides."""
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    original_task = rec.training_config_payload.get("taskType")
    payload = builder.build_intelligent(rec, user_overrides={"taskType": "regression"})
    # taskType override should be silently ignored
    assert payload.get("taskType") == original_task


def test_builder_manual_mode():
    user_payload = {
        "taskType": "classification",
        "models": ["randomforest"],
        "metrics": ["f1"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "balancing": {"strategy": "none"},
    }
    payload = builder.build_manual(user_payload)
    assert payload["configMode"] == "manual"
    assert payload["models"] == ["randomforest"]


def test_builder_resolve_intelligent():
    prof = _make_profile(n=500, ir=1.0, task="binary")
    rec = engine.recommend(prof)
    user_meta = {"targetColumn": "t", "datasetVersionId": 42, "taskType": "classification"}
    payload = builder.resolve("intelligent", recommendation=rec, user_payload=user_meta)
    assert payload["targetColumn"] == "t"
    assert payload["datasetVersionId"] == 42


def test_builder_resolve_manual_requires_payload():
    with pytest.raises(ValueError, match="user_payload"):
        builder.resolve("manual", user_payload=None)


def test_builder_resolve_intelligent_requires_recommendation():
    with pytest.raises(ValueError, match="recommendation"):
        builder.resolve("intelligent", recommendation=None)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end: recommendation → TrainingConfig.from_front()
# ──────────────────────────────────────────────────────────────────────────────

def test_recommendation_to_training_config():
    """The recommended payload must be parseable by TrainingConfig.from_front()."""
    prof = _make_profile(n=500, ir=2.0, task="binary")
    rec = engine.recommend(prof)
    payload = builder.resolve(
        "intelligent",
        recommendation=rec,
        user_payload={"targetColumn": "my_target", "datasetVersionId": 1, "taskType": "classification"},
    )
    cfg = TrainingConfig.from_front(payload)
    assert cfg.task_type == "classification"
    assert len(cfg.models) > 0
    assert cfg.target_column == "my_target"


def test_manual_mode_respects_user_choices():
    """Manual mode must not inject system recommendations."""
    user_payload = {
        "targetColumn": "label",
        "taskType": "classification",
        "models": ["svm"],
        "metrics": ["accuracy"],
        "splitMethod": "holdout",
        "trainRatio": 80,
        "valRatio": 10,
        "testRatio": 10,
        "balancing": {"strategy": "none"},
    }
    payload = builder.build_manual(user_payload)
    cfg = TrainingConfig.from_front(payload)
    assert "svm" in cfg.models
    assert cfg.split_method == "holdout"
    assert str(cfg.balancing.strategy) == "none"
