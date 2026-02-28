from __future__ import annotations

import numpy as np
import pandas as pd

import app.services.training.orchestrator as orchestrator
from app.services.training.config import TrainingConfig


def _make_imbalanced_binary_df(n_majority: int = 180, n_minority: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y = np.array([0] * n_majority + [1] * n_minority, dtype=int)
    rng.shuffle(y)

    age = rng.normal(loc=52.0, scale=11.0, size=len(y))
    blood_pressure = 118.0 + 0.45 * age + 6.0 * y + rng.normal(0.0, 5.0, size=len(y))
    sex = np.where(rng.random(len(y)) < (0.45 + 0.15 * y), "F", "M")
    smoker = np.where(rng.random(len(y)) < (0.20 + 0.35 * y), "yes", "no")

    return pd.DataFrame(
        {
            "age": age,
            "blood_pressure": blood_pressure,
            "sex": sex,
            "smoker": smoker,
            "Outcome": y,
        }
    )


def _cfg(*, strategy: str = "none", models: list[str] | None = None, apply_threshold: bool = False) -> TrainingConfig:
    selected_models = models or ["randomforest"]
    return TrainingConfig.from_front(
        {
            "targetColumn": "Outcome",
            "taskType": "classification",
            "models": selected_models,
            "metrics": ["accuracy", "f1"],
            "splitMethod": "holdout",
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": 5,
            "useGridSearch": False,
            "balancing": {
                "strategy": strategy,
                "apply_threshold": apply_threshold,
                "threshold_strategy": "maximize_f1",
                "user_confirmed": True,
            },
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def test_artifacts_include_preprocessing_model_and_balancing_audit():
    df = _make_imbalanced_binary_df()
    cfg = _cfg(strategy="none")

    res = orchestrator.run_one_model(df, cfg, model_type="randomforest")
    artifacts = res.artifacts_json

    preprocessing = artifacts.get("preprocessing", {})
    applied = preprocessing.get("applied", {})
    assert applied.get("class_name") == "ColumnTransformer"
    assert isinstance(applied.get("params"), dict)
    assert isinstance(applied.get("sample_input_shape"), list)
    assert isinstance(applied.get("sample_transform_shape"), list)
    assert isinstance(applied.get("n_features_out"), int)
    assert applied.get("n_features_out", 0) > 0
    assert preprocessing.get("defaults", {}).get("numericImputation") == "median"
    assert preprocessing.get("defaults", {}).get("categoricalEncoding") == "onehot"
    assert isinstance(preprocessing.get("effectiveByColumn"), dict)
    assert isinstance(preprocessing.get("droppedColumns"), list)
    assert isinstance(preprocessing.get("columnTypes"), dict)

    model = artifacts.get("model", {})
    assert model.get("class_name") == "RandomForestClassifier"
    assert isinstance(model.get("params"), dict)
    assert model.get("params", {}).get("n_estimators") == 200

    feature_importance = artifacts.get("feature_importance")
    assert isinstance(feature_importance, list)
    assert len(feature_importance) > 0
    assert isinstance(feature_importance[0], dict)
    assert isinstance(feature_importance[0].get("feature"), str)
    assert isinstance(feature_importance[0].get("importance"), float)
    assert feature_importance[0]["importance"] >= feature_importance[-1]["importance"]

    training_schema = artifacts.get("training_schema", {})
    assert training_schema.get("target") == "Outcome"
    assert isinstance(training_schema.get("feature_names"), list)
    assert isinstance(training_schema.get("dtypes"), dict)
    assert isinstance(training_schema.get("preprocessing_config"), dict)

    balancing = artifacts.get("balancing", {})
    assert balancing.get("strategy_applied") == "none"
    assert balancing.get("rationale")
    assert isinstance(balancing.get("audit_flags"), list)
    assert isinstance(balancing.get("imbalance_ratio"), float)
    assert isinstance(balancing.get("warnings"), list)


def test_feature_importance_works_for_logistic_regression_via_coefficients():
    df = _make_imbalanced_binary_df()
    cfg = _cfg(strategy="none", models=["logisticregression"])

    res = orchestrator.run_one_model(df, cfg, model_type="logisticregression")
    feature_importance = res.artifacts_json.get("feature_importance")

    assert isinstance(feature_importance, list)
    assert len(feature_importance) > 0
    assert isinstance(feature_importance[0].get("feature"), str)
    assert isinstance(feature_importance[0].get("importance"), float)
