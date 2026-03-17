from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from app.services.training.config.schema import TrainingConfig
from app.services.preprocessing.preprocessing import build_preprocessor
from app.services.training.config.validation import validate_training_config_payload


def _base_payload(model: str = "randomforest") -> dict:
    return {
        "targetColumn": "Outcome",
        "taskType": "classification",
        "models": [model],
        "metrics": ["accuracy", "f1"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "kFolds": 5,
        "useGridSearch": False,
        "useSmote": False,
        "preprocessing": {
            "defaults": {
                "numericImputation": "none",
                "numericScaling": "none",
                "categoricalImputation": "none",
                "categoricalEncoding": "none",
            },
            "columns": {},
        },
    }


def test_validate_returns_error_when_categorical_columns_and_encoding_none():
    df = pd.DataFrame(
        {
            "age": [50, 42, 61, 45, 53, 58, 47, 66, 55, 44],
            "city": ["A", "B", "A", "C", "A", "B", "A", "C", "B", "A"],
            "Outcome": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )

    out = validate_training_config_payload(_base_payload("randomforest"), df)
    assert any("categoricalEncoding='none'" in msg for msg in out["errors"])
    assert isinstance(out.get("effective_preprocessing_by_column"), dict)


def test_validate_returns_error_when_missing_values_and_imputation_none_for_sklearn():
    df = pd.DataFrame(
        {
            "age": [51, 49, np.nan, 53, 44, 61, np.nan, 58, 47, 55, 46, 63],
            "Outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    out = validate_training_config_payload(_base_payload("svm"), df)
    assert any("NaN detectes sur colonnes actives avec imputation='none'" in msg for msg in out["errors"])


def test_preprocessor_builder_uses_passthrough_group_for_numeric_column_when_all_methods_are_none():
    df = pd.DataFrame(
        {
            "age": [50, 42, 61, 45, 53, 58],
            "city": ["A", "B", "A", "C", "A", "B"],
            "Outcome": [0, 1, 0, 1, 0, 0],
        }
    )
    cfg = TrainingConfig.from_front(_base_payload("randomforest"))
    spec = build_preprocessor(df.drop(columns=["Outcome"]), cfg.preprocessing)

    assert isinstance(spec.preprocessor, ColumnTransformer)
    passthrough_groups = [t for t in spec.preprocessor.transformers if t[0] == "passthrough_cols"]
    assert len(passthrough_groups) == 1
    assert "age" in passthrough_groups[0][2]
    assert spec.selected_methods["defaults"]["numericImputation"] == "none"
    assert spec.selected_methods["defaults"]["numericScaling"] == "none"
    assert spec.selected_methods["defaults"]["categoricalImputation"] == "none"
    assert spec.selected_methods["defaults"]["categoricalEncoding"] == "none"


def test_preprocessor_builder_groups_identical_numeric_configs():
    n_rows = 25
    n_cols = 100
    data = {f"num_{i}": np.arange(n_rows, dtype=float) + i for i in range(n_cols)}
    data["Outcome"] = np.where(np.arange(n_rows) % 2 == 0, 0, 1)
    df = pd.DataFrame(data)

    payload = _base_payload("randomforest")
    payload["preprocessing"] = {
        "defaults": {
            "numericImputation": "median",
            "numericScaling": "standard",
            "categoricalImputation": "none",
            "categoricalEncoding": "none",
        },
        "columns": {},
    }
    cfg = TrainingConfig.from_front(payload)
    spec = build_preprocessor(df.drop(columns=["Outcome"]), cfg.preprocessing)

    assert isinstance(spec.preprocessor, ColumnTransformer)
    numeric_transformers = [t for t in spec.preprocessor.transformers if str(t[0]).startswith("num")]
    assert len(numeric_transformers) == 1
    assert len(numeric_transformers[0][2]) == n_cols
    scaler = numeric_transformers[0][1].named_steps.get("scaler")
    assert scaler is not None
    assert scaler.with_mean is True


def test_onehot_emits_all_observed_categories():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "C", "C", "D"],
            "Outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    payload = _base_payload("randomforest")
    payload["preprocessing"] = {
        "defaults": {
            "numericImputation": "none",
            "numericScaling": "none",
            "categoricalImputation": "none",
            "categoricalEncoding": "onehot",
        },
        "columns": {},
    }

    cfg = TrainingConfig.from_front(payload)
    X = df.drop(columns=["Outcome"])
    spec = build_preprocessor(X, cfg.preprocessing)
    fitted = spec.preprocessor.fit(X)

    feature_names = [str(v) for v in fitted.get_feature_names_out()]
    city_features = [name for name in feature_names if "city" in name]

    assert len(city_features) == 4
    assert all("infrequent" not in name for name in city_features)
