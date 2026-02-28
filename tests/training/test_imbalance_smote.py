from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.training.balancing.profiler import ImbalanceLevel, profile_binary_dataset
from app.services.training.balancing.resolver import resolve


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
    )

    assert decision.strategy == "sample_weight"
    assert "fallback_sample_weight:model_no_class_weight" in decision.audit_flags
    assert decision.rationale
