from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.training import TrainedModel
from app.services.training.config.schema import TrainingConfig
from app.services.training.orchestrator import run_one_model
from app.services.training.output.persistence import load_pipeline, save_pipeline
from app.services.training.output.predictor import predict_with_trained_model
from app.services.training.pipeline.splitters import make_holdout_split


def _make_dataset(n_rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    y = np.array([0] * int(n_rows * 0.78) + [1] * (n_rows - int(n_rows * 0.78)), dtype=int)
    rng.shuffle(y)

    age = rng.normal(52.0, 10.0, size=n_rows)
    cholesterol = 170.0 + 0.55 * age + 10.0 * y + rng.normal(0.0, 8.0, size=n_rows)
    # Missingness to test train-only fitted imputer stats.
    missing_mask = rng.random(n_rows) < 0.2
    cholesterol[missing_mask] = np.nan

    symptom = np.where(rng.random(n_rows) < (0.3 + 0.35 * y), "yes", "no")
    region = np.where(rng.random(n_rows) < 0.5, "north", "south")

    return pd.DataFrame(
        {
            "age": age,
            "cholesterol": cholesterol,
            "symptom": symptom,
            "region": region,
            "Outcome": y,
        }
    )


def _cfg(
    *,
    model: str = "randomforest",
    use_smote: bool = False,
    use_grid_search: bool = False,
    k_folds: int = 3,
) -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": "Outcome",
            "taskType": "classification",
            "models": [model],
            "metrics": ["accuracy", "f1"],
            "splitMethod": "holdout",
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": k_folds,
            "useGridSearch": use_grid_search,
            "useSmote": use_smote,
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def test_preprocessing_is_fitted_on_train_only_and_reused_for_test():
    df = _make_dataset()
    cfg = _cfg(model="randomforest", use_smote=False, use_grid_search=False)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].values
    split = make_holdout_split(
        X,
        y,
        task_type=cfg.task_type,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_state=42,
    )
    expected_median = float(np.nanmedian(split.X_train["cholesterol"].to_numpy(dtype=float)))

    res = run_one_model(df, cfg, model_type="randomforest")
    prep = res.fitted_pipeline.named_steps["prep"]
    num_cols = list(prep.transformers_[0][2])
    chol_idx = num_cols.index("cholesterol")
    imputer = prep.named_transformers_["num"].named_steps["imputer"]

    assert np.isclose(float(imputer.statistics_[chol_idx]), expected_median, equal_nan=True)

    stats_before = np.asarray(imputer.statistics_, dtype=float).copy()
    _ = res.fitted_pipeline.predict(split.X_test)
    stats_after = np.asarray(prep.named_transformers_["num"].named_steps["imputer"].statistics_, dtype=float).copy()
    assert np.allclose(stats_before, stats_after, equal_nan=True)


def test_pipeline_save_reload_keeps_same_predictions(tmp_path):
    df = _make_dataset()
    cfg = _cfg(model="randomforest", use_smote=False, use_grid_search=False)
    res = run_one_model(df, cfg, model_type="randomforest")

    pkl_path = save_pipeline(res.fitted_pipeline, tmp_path, "randomforest")
    reloaded = load_pipeline(pkl_path)

    X_sample = df.drop(columns=["Outcome"]).head(30).copy()
    pred_before = res.fitted_pipeline.predict(X_sample)
    pred_after = reloaded.predict(X_sample)
    assert np.array_equal(pred_before, pred_after)


def test_prediction_alignment_handles_missing_and_extra_columns(tmp_path):
    df = _make_dataset()
    cfg = _cfg(model="randomforest", use_smote=False, use_grid_search=False)
    res = run_one_model(df, cfg, model_type="randomforest")

    pkl_path = save_pipeline(res.fitted_pipeline, tmp_path, "randomforest")
    tm = TrainedModel(
        id=999,
        session_id=1,
        project_id=1,
        model_type="randomforest",
        task_type="classification",
        metrics_json={},
        artifacts_json={
            "model_pkl": str(pkl_path),
            "dataset_version_id": 123,
            "training_schema": res.artifacts_json.get("training_schema", {}),
        },
    )

    base = df.drop(columns=["Outcome"]).head(15).copy()
    misaligned = base.drop(columns=["symptom"]).copy()
    misaligned["unexpected_feature"] = np.arange(len(misaligned))

    out = predict_with_trained_model(tm, misaligned)
    assert out["model_id"] == 999
    assert out["dataset_version_id"] == 123
    assert out["n_rows"] == len(misaligned)
    assert isinstance(out["preview"], list)
    assert len(out["preview"]) > 0


def test_gridsearch_enabled_exposes_best_params_and_summary():
    df = _make_dataset()
    cfg = _cfg(model="logisticregression", use_smote=False, use_grid_search=True, k_folds=3)
    res = run_one_model(df, cfg, model_type="logisticregression")

    gs = res.artifacts_json.get("grid_search", {})
    assert gs.get("enabled") is True
    assert isinstance(gs.get("best_params"), dict)
    assert len(gs.get("best_params", {})) > 0
    assert isinstance(gs.get("cv_results_summary"), list)
    assert gs.get("n_candidates", 0) > 0
    assert isinstance(gs.get("best_score"), float)
    assert isinstance(gs.get("refit_metric"), str)
