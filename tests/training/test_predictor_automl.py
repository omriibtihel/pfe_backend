"""
Tests for AutoML-specific prediction logic in predictor.py

Covers:
- _fix_string_dtypes: pandas 3.0 StringDtype → object conversion
- _apply_automl_features: medical interaction feature reconstruction
- predict_with_trained_model (AutoML path):
    - Old model (no automl_feature_pairs key) → raises RuntimeError
    - New model with empty feature pairs → works
    - New model with feature pairs → features reconstructed in output
    - StringDtype columns handled before inference
- _run_inference: FLAML classes_ fallback via pipeline.model.estimator.classes_
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.training.output.predictor import (
    _apply_automl_features,
    _fix_string_dtypes,
    _run_inference,
    predict_with_trained_model,
)
from app.services.training.config.schema import TrainingConfig
from app.services.training.orchestrator import run_one_model


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y = np.array([1] * (n // 3) + [0] * (n - n // 3))
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


def _base_cfg() -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": "target",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy"],
            "splitMethod": "holdout",
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


def _make_mock_model(pipeline, artifacts: dict, task_type: str = "classification"):
    """Serialize pipeline to a temp file and return a mock TrainedModel."""
    import joblib

    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    joblib.dump(pipeline, tmp.name)
    tmp.close()

    m = MagicMock()
    m.id = 1
    m.session_id = 1
    m.model_type = "randomforest"
    m.task_type = task_type
    m.artifacts_json = {**artifacts, "model_pkl": tmp.name}
    return m, tmp.name


# ──────────────────────────────────────────────────────────────────────────────
# _fix_string_dtypes
# ──────────────────────────────────────────────────────────────────────────────

class TestFixStringDtypes:
    def test_converts_string_dtype_to_object(self):
        df = pd.DataFrame({"a": pd.array(["x", "y"], dtype="string"), "b": [1, 2]})
        result = _fix_string_dtypes(df)
        assert result["a"].dtype == object

    def test_no_op_when_already_object(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        result = _fix_string_dtypes(df)
        assert result["a"].dtype == object
        assert list(result.columns) == ["a", "b"]

    def test_columns_index_is_object_dtype(self):
        df = pd.DataFrame({"age": [1.0], "bmi": [2.0]})
        result = _fix_string_dtypes(df)
        assert result.columns.dtype == object

    def test_mixed_dtypes_only_string_converted(self):
        df = pd.DataFrame({
            "a": pd.array(["x", "y"], dtype="string"),
            "b": [1, 2],
            "c": [1.5, 2.5],
        })
        result = _fix_string_dtypes(df)
        assert result["a"].dtype == object
        assert result["b"].dtype != object  # int unchanged
        assert result["c"].dtype != object  # float unchanged


# ──────────────────────────────────────────────────────────────────────────────
# _apply_automl_features
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyAutomlFeatures:
    def test_creates_interaction_feature(self):
        df = pd.DataFrame({"age": [30.0, 40.0], "bmi": [25.0, 30.0]})
        pairs = [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}]
        result = _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert "_med_agexbmi" in result.columns
        assert result["_med_agexbmi"].iloc[0] == pytest.approx(30.0 * 25.0)
        assert result["_med_agexbmi"].iloc[1] == pytest.approx(40.0 * 30.0)

    def test_multiple_pairs(self):
        df = pd.DataFrame({"a": [2.0], "b": [3.0], "c": [4.0]})
        pairs = [
            {"col1": "a", "col2": "b", "name": "_med_axb"},
            {"col1": "b", "col2": "c", "name": "_med_bxc"},
        ]
        result = _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert result["_med_axb"].iloc[0] == pytest.approx(6.0)
        assert result["_med_bxc"].iloc[0] == pytest.approx(12.0)

    def test_original_columns_preserved(self):
        df = pd.DataFrame({"age": [30.0], "bmi": [25.0]})
        pairs = [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}]
        result = _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert "age" in result.columns
        assert "bmi" in result.columns

    def test_missing_column_skipped_gracefully(self):
        df = pd.DataFrame({"age": [30.0]})
        pairs = [{"col1": "age", "col2": "missing_col", "name": "_med_feature"}]
        result = _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert "_med_feature" not in result.columns

    def test_empty_pairs_returns_original(self):
        df = pd.DataFrame({"age": [30.0], "bmi": [25.0]})
        result = _apply_automl_features(df, {"automl_feature_pairs": []})
        assert list(result.columns) == ["age", "bmi"]

    def test_absent_key_returns_original(self):
        df = pd.DataFrame({"age": [30.0]})
        result = _apply_automl_features(df, {})
        assert list(result.columns) == ["age"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"age": [30.0], "bmi": [25.0]})
        original_cols = list(df.columns)
        pairs = [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}]
        _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert list(df.columns) == original_cols

    def test_columns_index_object_after_apply(self):
        df = pd.DataFrame({"age": [30.0], "bmi": [25.0]})
        pairs = [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}]
        result = _apply_automl_features(df, {"automl_feature_pairs": pairs})
        assert result.columns.dtype == object


# ──────────────────────────────────────────────────────────────────────────────
# _run_inference — FLAML classes_ fallback
# ──────────────────────────────────────────────────────────────────────────────

class TestRunInferenceFlamlFallback:
    def _make_flaml_mock(self, classes, proba):
        """Build a mock pipeline that mimics a FLAML AutoML object (no named_steps)."""
        mock = MagicMock()
        mock.predict_proba.return_value = proba
        # Explicitly set direct classes_ to None (not exposed on FLAML root object)
        mock.classes_ = None
        # Empty named_steps — no "model" key
        mock.named_steps = {}
        # FLAML path: pipeline.model.estimator.classes_
        mock.model.estimator.classes_ = classes
        return mock

    def test_uses_flaml_classes_for_threshold(self):
        """Threshold is applied when classes_ retrieved via FLAML fallback path."""
        classes = np.array([0, 1])
        proba = np.array([[0.4, 0.6], [0.7, 0.3], [0.45, 0.55]])
        mock = self._make_flaml_mock(classes, proba)

        X = pd.DataFrame({"a": [1, 2, 3]})
        # Use threshold != 0.5 so the threshold branch is actually exercised
        y_pred, y_score = _run_inference(mock, X, "classification", threshold=0.4)

        assert y_score is not None
        # 0.6 >= 0.4 → 1; 0.3 < 0.4 → 0; 0.55 >= 0.4 → 1
        assert list(y_pred) == [1, 0, 1]

    def test_threshold_shifts_predictions(self):
        """Changing threshold from 0.4 to 0.6 flips borderline prediction."""
        classes = np.array([0, 1])
        proba = np.array([[0.5, 0.5]])  # borderline case
        mock_low = self._make_flaml_mock(classes, proba)
        mock_high = self._make_flaml_mock(classes, proba)

        X = pd.DataFrame({"a": [1]})
        y_low, _ = _run_inference(mock_low, X, "classification", threshold=0.4)
        y_high, _ = _run_inference(mock_high, X, "classification", threshold=0.6)

        assert y_low[0] == 1   # 0.5 >= 0.4
        assert y_high[0] == 0  # 0.5 < 0.6

    def test_score_returned_for_flaml_pipeline(self):
        classes = np.array([0, 1])
        proba = np.array([[0.3, 0.7]])
        mock = self._make_flaml_mock(classes, proba)

        X = pd.DataFrame({"a": [1]})
        _, y_score = _run_inference(mock, X, "classification", threshold=0.5)

        assert y_score is not None
        assert y_score[0] == pytest.approx(0.7)


# ──────────────────────────────────────────────────────────────────────────────
# predict_with_trained_model — AutoML path
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictWithTrainedModelAutoML:
    def setup_method(self):
        self.df = _make_df(n=120)
        self.cfg = _base_cfg()
        result = run_one_model(self.df, self.cfg, "randomforest")
        self.pipeline = result.fitted_pipeline
        self.artifacts = result.artifacts_json
        self._tmp_paths: list[str] = []

    def teardown_method(self):
        for p in self._tmp_paths:
            if os.path.exists(p):
                os.unlink(p)

    def _automl_model(self, extra_artifacts: dict):
        base = {**self.artifacts, "automl": {"time_budget_s": 60}, **extra_artifacts}
        m, path = _make_mock_model(self.pipeline, base)
        self._tmp_paths.append(path)
        return m

    def test_old_automl_model_without_feature_pairs_key_works(self):
        """Old AutoML model missing automl_feature_pairs key still predicts (no interaction features)."""
        # No "automl_feature_pairs" key at all (old model) — should proceed gracefully
        m = self._automl_model({})
        m.artifacts_json.pop("automl_feature_pairs", None)

        X = self.df.drop(columns=["target"]).head(5)
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == 5

    def test_automl_model_empty_pairs_works(self):
        """automl_feature_pairs=[] (no interaction features) — prediction succeeds."""
        m = self._automl_model({"automl_feature_pairs": []})
        X = self.df.drop(columns=["target"]).head(10)
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == 10
        assert len(result["rows"]) == 10

    def test_interaction_feature_reconstructed_in_output(self):
        """
        When automl_feature_pairs lists (age, bmi), the predictor reconstructs
        _med_agexbmi before inference, and the column appears in row input_data.
        """
        # Train a pipeline that includes the interaction feature
        df_with_feat = self.df.copy()
        df_with_feat["_med_agexbmi"] = df_with_feat["age"] * df_with_feat["bmi"]

        cfg2 = TrainingConfig.from_front({
            "targetColumn": "target",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy"],
            "splitMethod": "holdout",
            "trainRatio": 70, "valRatio": 15, "testRatio": 15,
            "kFolds": 3, "useGridSearch": False,
            "balancing": {"strategy": "none", "apply_threshold": False},
            "preprocessing": {
                "numericImputation": "median", "numericScaling": "standard",
                "categoricalImputation": "most_frequent", "categoricalEncoding": "onehot",
            },
        })
        result2 = run_one_model(df_with_feat, cfg2, "randomforest")
        pipeline2 = result2.fitted_pipeline
        artifacts2 = result2.artifacts_json

        arts = {
            **artifacts2,
            "automl": {"time_budget_s": 60},
            "automl_feature_pairs": [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}],
        }
        m, path = _make_mock_model(pipeline2, arts)
        self._tmp_paths.append(path)

        # Input without the interaction feature (predictor must rebuild it)
        X_raw = self.df.drop(columns=["target"]).head(5)
        pred_result = predict_with_trained_model(m, X_raw)

        assert pred_result["n_rows"] == 5
        # The interaction column must appear in input_data (was added by _apply_automl_features)
        first_row_input = pred_result["rows"][0]["input_data"]
        assert "_med_agexbmi" in first_row_input

    def test_interaction_feature_value_correct(self):
        """Reconstructed interaction feature equals age * bmi."""
        m = self._automl_model({
            "automl_feature_pairs": [{"col1": "age", "col2": "bmi", "name": "_med_agexbmi"}],
        })
        X = self.df.drop(columns=["target"]).head(3)
        result = predict_with_trained_model(m, X)

        for i, row in enumerate(result["rows"]):
            expected = float(X.iloc[i]["age"]) * float(X.iloc[i]["bmi"])
            actual = row["input_data"]["_med_agexbmi"]
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_string_dtype_columns_handled(self):
        """pandas 3.0 StringDtype columns are coerced before inference (no crash)."""
        m = self._automl_model({"automl_feature_pairs": []})
        X = self.df.drop(columns=["target"]).head(5).copy()
        X["smoker"] = X["smoker"].astype("string")  # simulate pandas 3.0 dtype
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == 5

    def test_non_automl_model_skips_automl_path(self):
        """When artifacts['automl'] is absent, string coercion and feature_pairs checks are skipped."""
        # Regular model (no "automl" key)
        m, path = _make_mock_model(self.pipeline, self.artifacts)
        self._tmp_paths.append(path)
        X = self.df.drop(columns=["target"]).head(5)
        result = predict_with_trained_model(m, X)
        assert result["n_rows"] == 5
