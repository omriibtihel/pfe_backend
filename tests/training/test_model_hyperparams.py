from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.training.config import TrainingConfig, get_training_capabilities
from app.services.training.orchestrator import run_one_model
from app.services.training.validation import validate_training_config_payload


def _make_df(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    age = rng.normal(52.0, 10.0, size=rows)
    biomarker = rng.normal(0.0, 1.0, size=rows) + 0.04 * age
    noise = rng.normal(0.0, 0.7, size=rows)
    outcome = (age * 0.03 + biomarker + noise > np.median(age) * 0.03).astype(int)
    return pd.DataFrame(
        {
            "age": age,
            "biomarker": biomarker,
            "Outcome": outcome,
        }
    )


def _payload(
    *,
    model: str,
    use_grid_search: bool = False,
    model_hyperparams: dict[str, dict] | None = None,
) -> dict:
    return {
        "targetColumn": "Outcome",
        "taskType": "classification",
        "models": [model],
        "metrics": ["accuracy", "f1"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "kFolds": 3,
        "useGridSearch": use_grid_search,
        "useSmote": False,
        "preprocessing": {
            "defaults": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "none",
                "categoricalEncoding": "none",
            },
            "columns": {},
        },
        "modelHyperparams": model_hyperparams or {},
    }


def test_randomforest_hyperparams_are_applied_in_estimator():
    df = _make_df()
    payload = _payload(
        model="randomforest",
        model_hyperparams={
            "randomforest": {
                "n_estimators": 123,
                "max_depth": 9,
                "max_features": "log2",
            }
        },
    )
    cfg = TrainingConfig.from_front(payload)
    res = run_one_model(df, cfg, model_type="randomforest")
    params = res.fitted_pipeline.named_steps["model"].get_params()

    assert int(params.get("n_estimators")) == 123
    assert int(params.get("max_depth")) == 9
    assert str(params.get("max_features")) == "log2"


def test_naivebayes_hyperparams_are_applied_in_estimator():
    df = _make_df()
    payload = _payload(
        model="naivebayes",
        model_hyperparams={
            "naivebayes": {
                "var_smoothing": 1e-7,
            }
        },
    )
    cfg = TrainingConfig.from_front(payload)
    res = run_one_model(df, cfg, model_type="naivebayes")
    params = res.fitted_pipeline.named_steps["model"].get_params()

    assert abs(float(params.get("var_smoothing")) - 1e-7) < 1e-15


def test_naivebayes_runs_with_onehot_preprocessing():
    df = _make_df()
    df["sex"] = np.where(np.arange(len(df)) % 2 == 0, "F", "M")
    payload = _payload(
        model="naivebayes",
        model_hyperparams={"naivebayes": {"var_smoothing": 1e-9}},
    )
    payload["preprocessing"]["defaults"]["categoricalImputation"] = "most_frequent"
    payload["preprocessing"]["defaults"]["categoricalEncoding"] = "onehot"
    cfg = TrainingConfig.from_front(payload)
    res = run_one_model(df, cfg, model_type="naivebayes")

    assert res.fitted_pipeline.named_steps["model"].__class__.__name__ == "GaussianNB"


def test_validation_svm_linear_removes_gamma_and_adds_warning_detail():
    df = _make_df()
    payload = _payload(
        model="svm",
        model_hyperparams={
            "svm": {
                "kernel": "linear",
                "gamma": "scale",
                "C": 2.0,
            }
        },
    )

    out = validate_training_config_payload(payload, df)
    svm_hp = ((out.get("normalized_config") or {}).get("modelHyperparams") or {}).get("svm") or {}

    assert "gamma" not in svm_hp
    assert any(
        d.get("code") == "HP_SVM_GAMMA_IGNORED" and d.get("severity") == "warning" and d.get("model") == "svm"
        for d in (out.get("error_details") or [])
    )


def test_gridsearch_lists_build_param_grid_and_save_best_params():
    df = _make_df()
    payload = _payload(
        model="decisiontree",
        use_grid_search=True,
        model_hyperparams={
            "decisiontree": {
                "max_depth": [3, 5],
                "criterion": ["gini", "entropy"],
            }
        },
    )
    cfg = TrainingConfig.from_front(payload)
    res = run_one_model(df, cfg, model_type="decisiontree")

    hyperparams_artifacts = res.artifacts_json.get("hyperparams", {})
    grid_search = res.artifacts_json.get("grid_search", {})

    assert grid_search.get("enabled") is True
    assert hyperparams_artifacts.get("param_grid", {}).get("max_depth") == [3, 5]
    assert hyperparams_artifacts.get("param_grid", {}).get("criterion") == ["gini", "entropy"]
    assert isinstance(hyperparams_artifacts.get("best"), dict)
    assert len(hyperparams_artifacts.get("best", {})) > 0


def test_validation_unknown_hyperparam_key_is_warned_and_dropped():
    df = _make_df()
    payload = _payload(
        model="randomforest",
        model_hyperparams={
            "randomforest": {
                "n_estimators": 120,
                "foo_bar": 99,
            }
        },
    )

    out = validate_training_config_payload(payload, df)
    rf_hp = ((out.get("normalized_config") or {}).get("modelHyperparams") or {}).get("randomforest") or {}

    assert rf_hp.get("n_estimators") == 120
    assert "foo_bar" not in rf_hp
    assert any(
        d.get("code") == "HP_UNKNOWN" and d.get("severity") == "warning" and d.get("model") == "randomforest"
        for d in (out.get("error_details") or [])
    )


def test_capabilities_expose_model_hyperparams_schema_and_model_installation():
    caps = get_training_capabilities()
    schema = caps.get("modelHyperparamsSchema", {})
    models = caps.get("availableModels", [])

    assert isinstance(schema, dict)
    assert schema.get("svm", {}).get("gamma", {}).get("default") == "scale"
    assert schema.get("randomforest", {}).get("n_estimators", {}).get("min") == 10
    assert schema.get("logisticregression", {}).get("max_iter", {}).get("default") == 4000
    assert schema.get("naivebayes", {}).get("var_smoothing", {}).get("default") == 1e-9
    assert isinstance(models, list) and len(models) > 0
    assert all("key" in m and "label" in m and "installed" in m for m in models)
    assert any(m.get("key") == "naivebayes" and m.get("installed") is True for m in models)
