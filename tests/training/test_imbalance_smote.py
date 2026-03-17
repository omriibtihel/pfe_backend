from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.services.balancing.profiler import ImbalanceLevel, profile_binary_dataset
from app.services.balancing.resolver import resolve
from app.services.training.config.schema import TrainingConfig
from app.services.training.orchestrator import run_one_model


def test_profile_binary_dataset_exposes_all_expected_strategies():
    y_train = np.array([0] * 90 + [1] * 10, dtype=int)
    profile = profile_binary_dataset(y=y_train, X_shape=(len(y_train), 4))

    strategy_ids = [s.id for s in profile.available_strategies]
    assert strategy_ids == [
        "none",
        "class_weight",
        "smote",
        "smote_tomek",
        "random_undersampling",
        "threshold_optimization",
    ]
    assert profile.imbalance_level in {
        ImbalanceLevel.MILD,
        ImbalanceLevel.MODERATE,
        ImbalanceLevel.SEVERE,
        ImbalanceLevel.CRITICAL,
    }
    assert profile.needs_balancing is True
    assert profile.default_recommendation in {"class_weight", "smote", "smote_tomek", "threshold_optimization"}


def test_resolve_class_weight_falls_back_to_sample_weight_when_model_has_no_class_weight():
    y_train = np.array([0] * 120 + [1] * 40, dtype=int)
    profile = profile_binary_dataset(y=y_train, X_shape=(len(y_train), 3))
    cfg = SimpleNamespace(
        strategy="class_weight",
        apply_threshold=False,
        threshold_strategy="maximize_f1",
        min_recall_constraint=None,
    )

    decision = resolve(
        profile=profile,
        config=cfg,
        model_supports_class_weight=False,
        model_supports_predict_proba=True,
        model_supports_sample_weight=True,
    )

    assert decision.strategy == "sample_weight"
    assert "fallback_sample_weight:model_no_class_weight" in decision.audit_flags
    assert decision.rationale


def test_resolve_class_weight_falls_back_to_none_when_model_supports_neither_weighting():
    y_train = np.array([0] * 120 + [1] * 40, dtype=int)
    profile = profile_binary_dataset(y=y_train, X_shape=(len(y_train), 3))
    cfg = SimpleNamespace(
        strategy="class_weight",
        apply_threshold=False,
        threshold_strategy="maximize_f1",
        min_recall_constraint=None,
    )

    decision = resolve(
        profile=profile,
        config=cfg,
        model_supports_class_weight=False,
        model_supports_predict_proba=True,
        model_supports_sample_weight=False,
    )

    assert decision.strategy == "none"
    assert "fallback_none:model_no_class_weight_or_sample_weight" in decision.audit_flags
    assert decision.rationale


def test_knn_class_weight_request_does_not_fail_kfold_training():
    rng = np.random.default_rng(17)
    majority = pd.DataFrame(
        {
            "age": rng.normal(42.0, 4.0, size=120),
            "biomarker": rng.normal(-0.8, 0.3, size=120),
            "Outcome": np.zeros(120, dtype=int),
        }
    )
    minority = pd.DataFrame(
        {
            "age": rng.normal(63.0, 4.0, size=40),
            "biomarker": rng.normal(1.1, 0.3, size=40),
            "Outcome": np.ones(40, dtype=int),
        }
    )
    df = pd.concat([majority, minority], ignore_index=True).sample(frac=1.0, random_state=17).reset_index(drop=True)
    cfg = TrainingConfig.from_front(
        {
            "targetColumn": "Outcome",
            "taskType": "classification",
            "models": ["knn"],
            "metrics": ["accuracy", "f1"],
            "splitMethod": "stratified_kfold",
            "trainRatio": 80,
            "valRatio": 0,
            "testRatio": 0,
            "kFolds": 5,
            "useGridSearch": False,
            "balancing": {
                "strategy": "class_weight",
                "apply_threshold": False,
                "threshold_strategy": "maximize_f1",
            },
            "preprocessing": {
                "defaults": {
                    "numericImputation": "median",
                    "numericScaling": "standard",
                    "categoricalImputation": "none",
                    "categoricalEncoding": "none",
                },
                "columns": {},
            },
        }
    )

    result = run_one_model(df, cfg, model_type="knn")

    assert result.metrics_json["cv_summary"]["n_folds_ok"] == 5
    assert all(fold["status"] == "ok" for fold in result.metrics_json["fold_results"])
    assert result.fitted_pipeline.named_steps["model"].__class__.__name__ == "KNeighborsClassifier"
