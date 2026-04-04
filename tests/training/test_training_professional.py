from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

import app.services.training.pipeline.trainer as trainer_module
from app.models.training import TrainedModel
from app.services.training.config.schema import TrainingConfig
from app.services.training.orchestrator import run_one_model
from app.services.training.output.persistence import load_pipeline, save_pipeline
from app.services.training.output.predictor import predict_with_trained_model
from app.services.training.pipeline.splitters import make_holdout_split
from app.services.training.utils import sanitize_json_payload


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


def _make_smote_starved_dataset(n_rows: int = 300, n_minority: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(1337)
    y = np.array([0] * (n_rows - n_minority) + [1] * n_minority, dtype=int)
    rng.shuffle(y)

    age = rng.normal(50.0, 11.0, size=n_rows)
    cholesterol = 165.0 + 0.4 * age + 14.0 * y + rng.normal(0.0, 7.0, size=n_rows)
    symptom = np.where(rng.random(n_rows) < (0.12 + 0.55 * y), "yes", "no")
    scanner = np.where(rng.random(n_rows) < 0.5, "A", "B")

    return pd.DataFrame(
        {
            "age": age,
            "cholesterol": cholesterol,
            "symptom": symptom,
            "scanner": scanner,
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


def _force_single_process_grid_search(monkeypatch) -> None:
    def _grid_search_single(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return SklearnGridSearchCV(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "GridSearchCV", _grid_search_single)


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


def test_gridsearch_enabled_exposes_best_params_and_summary(monkeypatch):
    _force_single_process_grid_search(monkeypatch)
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


def test_sanitize_json_payload_replaces_nested_nan_values():
    payload = {
        "score": float("nan"),
        "curve": np.array([0.1, np.nan, np.inf]),
        "nested": {"train": [1.0, float("-inf"), np.float64(2.5)]},
    }

    sanitized = sanitize_json_payload(payload)

    assert sanitized["score"] is None
    assert sanitized["curve"] == [0.1, None, None]
    assert sanitized["nested"]["train"] == [1.0, None, 2.5]
    json.dumps(sanitized, allow_nan=False)


def test_gridsearch_smote_payload_is_json_safe_when_inner_folds_are_tight(monkeypatch):
    _force_single_process_grid_search(monkeypatch)
    df = _make_smote_starved_dataset()
    cfg = _cfg(model="logisticregression", use_smote=True, use_grid_search=True, k_folds=5)

    res = run_one_model(df, cfg, model_type="logisticregression")

    gs = res.artifacts_json.get("grid_search", {})
    rows = gs.get("cv_results_summary", [])

    assert gs.get("enabled") is True
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert any(row.get("mean_test_score") is not None for row in rows)

    json.dumps(res.metrics_json, allow_nan=False)
    json.dumps(res.artifacts_json, allow_nan=False)
