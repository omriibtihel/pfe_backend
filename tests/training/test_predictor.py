"""
Tests for app/services/training/predictor.py

Covers:
- validate_feature_schema: missing columns, extra columns, reordering
- _get_optimal_threshold: extracts calibrated threshold, fallbacks
- predict_with_trained_model: threshold applied, all rows returned, input_data echoed
- predict_rows_json: manual input mode
- predict_to_csv: correct CSV structure
"""
from __future__ import annotations

import csv
import io
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.training.output.predictor import (
    _get_optimal_threshold,
    predict_rows_json,
    predict_to_csv,
    predict_with_trained_model,
    validate_feature_schema,
)
from app.services.training.config.schema import TrainingConfig
from app.services.training.orchestrator import run_one_model


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_binary_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_pos = n // 3
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "age": rng.uniform(20, 80, n),
            "bmi": rng.uniform(18, 40, n),
            "bp": rng.uniform(60, 140, n),
            "smoker": rng.choice(["yes", "no"], n),
            "target": y,
        }
    )


def _base_cfg(split_method: str = "holdout") -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": "target",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1"],
            "splitMethod": split_method,
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": 3,
            "useGridSearch": False,
            "balancing": {"strategy": "none", "apply_threshold": False},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def _make_mock_trained_model(
    fitted_pipeline,
    artifacts_json: dict,
    task_type: str = "classification",
    model_type: str = "randomforest",
    model_id: int = 1,
    session_id: int = 1,
):
    """Return a mock TrainedModel ORM object with the given pipeline stored on disk."""
    import joblib
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    joblib.dump(fitted_pipeline, tmp.name)
    tmp.close()

    m = MagicMock()
    m.id = model_id
    m.session_id = session_id
    m.model_type = model_type
    m.task_type = task_type
    m.artifacts_json = {**artifacts_json, "model_pkl": tmp.name}
    return m, tmp.name


def _train_and_get_pipeline(df: pd.DataFrame, cfg: TrainingConfig):
    """Train a model and return (pipeline, artifacts_json, training_schema)."""
    result = run_one_model(df, cfg, "randomforest")
    return result.fitted_pipeline, result.artifacts_json


# ──────────────────────────────────────────────────────────────────────────────
# validate_feature_schema
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateFeatureSchema:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"age": [1, 2], "bmi": [20, 22]})
        schema = {"feature_names": ["age", "bmi", "bp"]}
        with pytest.raises(RuntimeError, match="bp"):
            validate_feature_schema(df, schema)

    def test_extra_columns_dropped_silently(self):
        df = pd.DataFrame({"age": [1, 2], "bmi": [20, 22], "extra": [99, 99]})
        schema = {"feature_names": ["age", "bmi"]}
        result = validate_feature_schema(df, schema)
        assert list(result.columns) == ["age", "bmi"]

    def test_reorders_columns_to_match_schema(self):
        df = pd.DataFrame({"bmi": [20, 22], "age": [30, 40]})
        schema = {"feature_names": ["age", "bmi"]}
        result = validate_feature_schema(df, schema)
        assert list(result.columns) == ["age", "bmi"]

    def test_no_schema_passes_through(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = validate_feature_schema(df, {})
        assert list(result.columns) == ["a", "b"]

    def test_error_message_lists_all_missing(self):
        df = pd.DataFrame({"age": [1]})
        schema = {"feature_names": ["age", "bmi", "bp", "smoker"]}
        with pytest.raises(RuntimeError) as exc_info:
            validate_feature_schema(df, schema)
        msg = str(exc_info.value)
        assert "bmi" in msg
        assert "bp" in msg
        assert "smoker" in msg


# ──────────────────────────────────────────────────────────────────────────────
# _get_optimal_threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestGetOptimalThreshold:
    def test_extracts_threshold_from_artifacts(self):
        artifacts = {"thresholding": {"optimal_threshold": 0.35}}
        assert _get_optimal_threshold(artifacts, "classification") == pytest.approx(0.35)

    def test_fallback_when_absent(self):
        assert _get_optimal_threshold({}, "classification") == pytest.approx(0.5)

    def test_fallback_when_none(self):
        artifacts = {"thresholding": {"optimal_threshold": None}}
        assert _get_optimal_threshold(artifacts, "classification") == pytest.approx(0.5)

    def test_returns_05_for_regression(self):
        artifacts = {"thresholding": {"optimal_threshold": 0.3}}
        assert _get_optimal_threshold(artifacts, "regression") == pytest.approx(0.5)

    def test_fallback_when_thresholding_not_dict(self):
        artifacts = {"thresholding": "disabled"}
        assert _get_optimal_threshold(artifacts, "classification") == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# predict_with_trained_model
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictWithTrainedModel:
    def setup_method(self):
        self.df = _make_binary_df(n=120)
        self.cfg = _base_cfg()
        self.pipeline, self.artifacts = _train_and_get_pipeline(self.df, self.cfg)

    def teardown_method(self):
        import os
        if hasattr(self, "_tmp_path") and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def _get_mock_model(self, **artifact_overrides):
        arts = {**self.artifacts, **artifact_overrides}
        m, path = _make_mock_trained_model(self.pipeline, arts)
        self._tmp_path = path
        return m

    def test_returns_all_rows(self):
        X = self.df.drop(columns=["target"])
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == len(X)
        assert len(result["rows"]) == len(X)

    def test_row_has_required_keys(self):
        X = self.df.drop(columns=["target"]).head(5)
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        for row in result["rows"]:
            assert "row_index" in row
            assert "prediction" in row
            assert "score" in row
            assert "input_data" in row

    def test_input_data_echoed_in_rows(self):
        X = self.df.drop(columns=["target"]).head(3)
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        for i, row in enumerate(result["rows"]):
            assert row["row_index"] == i
            assert "age" in row["input_data"]
            assert "bmi" in row["input_data"]

    def test_threshold_recorded_in_result(self):
        X = self.df.drop(columns=["target"]).head(5)
        arts = {**self.artifacts, "thresholding": {"optimal_threshold": 0.3}}
        m = self._get_mock_model(**{"thresholding": {"optimal_threshold": 0.3}})
        m.artifacts_json["thresholding"] = {"optimal_threshold": 0.3}
        result = predict_with_trained_model(m, X)
        assert result["threshold_used"] == pytest.approx(0.3)

    def test_applies_threshold_shifts_predictions(self):
        """
        Using threshold 0.01 forces almost all predictions to the positive class.
        Using threshold 0.99 forces almost all to negative.
        The two results must differ on a meaningful dataset.
        """
        X = self.df.drop(columns=["target"])

        arts_low = {**self.artifacts, "thresholding": {"optimal_threshold": 0.01}}
        m_low, path_low = _make_mock_trained_model(self.pipeline, arts_low)
        result_low = predict_with_trained_model(m_low, X)

        arts_high = {**self.artifacts, "thresholding": {"optimal_threshold": 0.99}}
        m_high, path_high = _make_mock_trained_model(self.pipeline, arts_high)
        result_high = predict_with_trained_model(m_high, X)

        import os
        os.unlink(path_low)
        os.unlink(path_high)

        preds_low = [r["prediction"] for r in result_low["rows"]]
        preds_high = [r["prediction"] for r in result_high["rows"]]
        assert preds_low != preds_high, (
            "Threshold 0.01 vs 0.99 should produce different predictions on binary classification"
        )

    def test_missing_required_column_raises(self):
        # A column entirely absent from the input file is a hard error:
        # silently filling it with NaN can mask upload mistakes and yield
        # predictions driven only by the imputer's training median.
        X_bad = self.df[["age"]].copy()  # missing bmi, bp, smoker
        m = self._get_mock_model()
        with pytest.raises(RuntimeError, match="missing"):
            predict_with_trained_model(m, X_bad)

    def test_missing_values_inside_present_columns_are_imputed(self):
        # A column that is *present* but contains NaN cells must NOT raise:
        # the imputer saved in the pipeline handles NaNs as it did at training.
        X = self.df.drop(columns=["target"]).head(5).copy()
        X.loc[X.index[0], "age"] = np.nan
        X.loc[X.index[1], "bmi"] = np.nan
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == len(X)
        assert len(result["rows"]) == len(X)

    def test_extra_column_is_ignored(self):
        X = self.df.drop(columns=["target"]).head(5).copy()
        X["unrelated_extra"] = 999
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == len(X)

    def test_summary_class_distribution_present(self):
        X = self.df.drop(columns=["target"])
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        assert "class_distribution" in result["summary"]
        dist = result["summary"]["class_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(X)

    def test_feature_names_expected_in_result(self):
        X = self.df.drop(columns=["target"]).head(5)
        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        assert isinstance(result["feature_names_expected"], list)
        assert len(result["feature_names_expected"]) > 0

    def test_missing_pkl_raises_runtime_error(self):
        m = MagicMock()
        m.id = 1
        m.session_id = 1
        m.model_type = "randomforest"
        m.task_type = "classification"
        m.artifacts_json = {"model_pkl": "/nonexistent/path/model.pkl"}
        X = self.df.drop(columns=["target"]).head(3)
        with pytest.raises(RuntimeError, match="not found"):
            predict_with_trained_model(m, X)


# ──────────────────────────────────────────────────────────────────────────────
# predict_rows_json
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictRowsJson:
    def setup_method(self):
        self.df = _make_binary_df(n=120)
        self.cfg = _base_cfg()
        self.pipeline, self.artifacts = _train_and_get_pipeline(self.df, self.cfg)

    def teardown_method(self):
        import os
        if hasattr(self, "_tmp_path") and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def test_single_row_manual_input(self):
        row = self.df.drop(columns=["target"]).iloc[0].to_dict()
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        result = predict_rows_json(m, [row])
        assert result["n_rows"] == 1
        assert len(result["rows"]) == 1

    def test_multiple_rows_manual_input(self):
        rows = self.df.drop(columns=["target"]).head(10).to_dict(orient="records")
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        result = predict_rows_json(m, rows)
        assert result["n_rows"] == 10

    def test_empty_rows_raises(self):
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        with pytest.raises(RuntimeError, match="No rows"):
            predict_rows_json(m, [])


# ──────────────────────────────────────────────────────────────────────────────
# predict_to_csv
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictToCsv:
    def setup_method(self):
        self.df = _make_binary_df(n=60)
        self.cfg = _base_cfg()
        self.pipeline, self.artifacts = _train_and_get_pipeline(self.df, self.cfg)

    def teardown_method(self):
        import os
        if hasattr(self, "_tmp_path") and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def test_csv_has_correct_row_count(self):
        X = self.df.drop(columns=["target"])
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        csv_str = predict_to_csv(m, X)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == len(X)

    def test_csv_has_required_columns(self):
        X = self.df.drop(columns=["target"]).head(5)
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        csv_str = predict_to_csv(m, X)
        reader = csv.DictReader(io.StringIO(csv_str))
        fieldnames = reader.fieldnames or []
        assert "row_index" in fieldnames
        assert "prediction" in fieldnames
        assert "score" in fieldnames

    def test_csv_includes_input_feature_columns(self):
        X = self.df.drop(columns=["target"]).head(5)
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        csv_str = predict_to_csv(m, X)
        reader = csv.DictReader(io.StringIO(csv_str))
        fieldnames = reader.fieldnames or []
        for col in ["age", "bmi", "bp"]:
            assert col in fieldnames

    def test_csv_row_indices_are_sequential(self):
        X = self.df.drop(columns=["target"]).head(5)
        m, path = _make_mock_trained_model(self.pipeline, self.artifacts)
        self._tmp_path = path
        csv_str = predict_to_csv(m, X)
        reader = csv.DictReader(io.StringIO(csv_str))
        indices = [int(r["row_index"]) for r in reader]
        assert indices == list(range(5))


# ──────────────────────────────────────────────────────────────────────────────
# ThresholdedClassifier — class-label preservation across save/load
# ──────────────────────────────────────────────────────────────────────────────

class TestThresholdedClassifierLabelPreservation:
    """The wrapper must return the *original* class labels — never coerced to 0/1."""

    def _fit_simple_lr(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        return clf

    def _make_separable(self, n: int = 200, labels=(0, 1)):
        rng = np.random.default_rng(42)
        X_pos = rng.normal(loc=2.0, scale=1.0, size=(n // 2, 3))
        X_neg = rng.normal(loc=-2.0, scale=1.0, size=(n // 2, 3))
        X = np.vstack([X_pos, X_neg])
        y = np.array([labels[1]] * (n // 2) + [labels[0]] * (n // 2))
        idx = rng.permutation(n)
        return X[idx], y[idx]

    def test_text_labels_preserved_through_save_load(self, tmp_path):
        from app.services.training.output.persistence import save_pipeline, load_pipeline
        X, y = self._make_separable(labels=("sain", "malade"))
        clf = self._fit_simple_lr(X, y)

        # 'malade' is the positive class; classes_ is sorted alphabetically
        # by sklearn → ["malade", "sain"] so positive_class_index = 0.
        pos_label = "malade"
        pos_idx = int(list(clf.classes_).index(pos_label))

        pkl = save_pipeline(
            clf, tmp_path, "lr",
            threshold=0.35,
            positive_class_index=pos_idx,
            positive_label=pos_label,
        )
        loaded = load_pipeline(pkl)
        preds = loaded.predict(X)

        assert set(np.unique(preds)).issubset({"sain", "malade"})
        # And both classes appear (separable data + custom threshold)
        assert "malade" in preds
        # Predictions are NOT 0/1
        assert not set(preds).issubset({0, 1, "0", "1"})

    def test_numeric_non_0_1_labels_preserved(self, tmp_path):
        from app.services.training.output.persistence import save_pipeline, load_pipeline
        X, y = self._make_separable(labels=(1, 2))
        clf = self._fit_simple_lr(X, y)

        pos_label = 2
        pos_idx = int(list(clf.classes_).index(pos_label))
        pkl = save_pipeline(
            clf, tmp_path, "lr",
            threshold=0.4,
            positive_class_index=pos_idx,
            positive_label=pos_label,
        )
        loaded = load_pipeline(pkl)
        preds = loaded.predict(X)

        assert set(np.unique(preds)).issubset({1, 2})
        assert 0 not in set(np.unique(preds))

    def test_threshold_shift_changes_predictions_with_text_labels(self, tmp_path):
        from app.services.training.output.persistence import save_pipeline, load_pipeline
        X, y = self._make_separable(labels=("sain", "malade"))
        clf = self._fit_simple_lr(X, y)
        pos_label = "malade"
        pos_idx = int(list(clf.classes_).index(pos_label))

        pkl_low = save_pipeline(
            clf, tmp_path, "lr_low",
            threshold=0.05,
            positive_class_index=pos_idx,
            positive_label=pos_label,
        )
        pkl_high = save_pipeline(
            clf, tmp_path, "lr_high",
            threshold=0.95,
            positive_class_index=pos_idx,
            positive_label=pos_label,
        )
        preds_low = load_pipeline(pkl_low).predict(X)
        preds_high = load_pipeline(pkl_high).predict(X)
        # Low threshold floods 'malade' class; high threshold floods 'sain'.
        assert (preds_low == "malade").sum() > (preds_high == "malade").sum()
        # Both still produce only the original labels.
        assert set(preds_low).issubset({"sain", "malade"})
        assert set(preds_high).issubset({"sain", "malade"})

    def test_default_threshold_unchanged_returns_estimator_predict(self, tmp_path):
        # threshold=0.5 (or None) must NOT wrap the pipeline.
        from app.services.training.output.persistence import (
            ThresholdedClassifier, save_pipeline, load_pipeline,
        )
        X, y = self._make_separable(labels=(0, 1))
        clf = self._fit_simple_lr(X, y)
        pkl = save_pipeline(clf, tmp_path, "lr_default", threshold=0.5)
        loaded = load_pipeline(pkl)
        assert not isinstance(loaded, ThresholdedClassifier)

    def test_score_is_probability_of_positive_class(self, tmp_path):
        # The score returned alongside predictions must be the prob of the
        # *positive* class column — even when sklearn orders classes_
        # such that the positive class is column 0 (e.g. "malade" < "sain").
        from app.services.training.output.persistence import save_pipeline, load_pipeline
        from app.services.training.output.predictor import _run_inference

        X, y = self._make_separable(labels=("sain", "malade"))
        clf = self._fit_simple_lr(X, y)
        pos_label = "malade"
        pos_idx = int(list(clf.classes_).index(pos_label))

        pkl = save_pipeline(
            clf, tmp_path, "lr",
            threshold=0.5,  # no wrapping; predictor extracts column directly
            positive_class_index=pos_idx,
            positive_label=pos_label,
        )
        loaded = load_pipeline(pkl)

        df_X = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        y_pred, y_score = _run_inference(
            loaded, df_X, "classification",
            threshold=0.5,
            positive_class_index=pos_idx,
        )

        proba_full = loaded.predict_proba(df_X)
        np.testing.assert_allclose(y_score, proba_full[:, pos_idx])

    def test_fallback_to_estimator_predict_when_no_predict_proba(self, tmp_path):
        # A regressor-like estimator without predict_proba should not crash
        # when accidentally wrapped: the wrapper falls back to estimator.predict.
        from app.services.training.output.persistence import ThresholdedClassifier

        class _NoProba:
            classes_ = np.array(["a", "b"])

            def predict(self, X):
                return np.array(["a"] * len(X))

        wrapper = ThresholdedClassifier(_NoProba(), threshold=0.3, positive_label="b")
        out = wrapper.predict(np.zeros((3, 2)))
        assert list(out) == ["a", "a", "a"]
