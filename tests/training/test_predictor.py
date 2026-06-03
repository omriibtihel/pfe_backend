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

    def test_whitespace_cells_in_numeric_columns_are_imputed(self):
        # When a CSV/Excel upload contains a literal space " " or empty string
        # in a numeric column, pandas reads it as object dtype.  Without
        # normalisation the column reaches SimpleImputer(strategy="median")
        # which crashes with:
        #   "could not convert string to float: ' '"
        # _normalize_input_dtypes must turn these blanks into NaN so the
        # pipeline imputer fills them with the training median.
        X = self.df.drop(columns=["target"]).head(5).copy()
        # Force the numeric column to object dtype with whitespace/empty cells.
        X["age"] = X["age"].astype(object)
        X.loc[X.index[0], "age"] = " "
        X.loc[X.index[2], "age"] = ""
        X.loc[X.index[3], "age"] = "  \t "

        m = self._get_mock_model()
        result = predict_with_trained_model(m, X)
        # Did not crash; all rows produced a prediction.
        assert result["n_rows"] == len(X)
        assert len(result["rows"]) == len(X)
        for row in result["rows"]:
            assert row["prediction"] is not None

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


# ──────────────────────────────────────────────────────────────────────────────
# Train/Predict preprocessing parity
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainPredictPreprocessingParity:
    """The fitted pipeline saved at training time must apply the *exact* same
    imputation/scaling/encoding transformations at prediction time.  No
    re-fitting, no parameter drift: the imputer median, the scaler mean/scale,
    and the encoder category mapping are all frozen at training time and
    reused verbatim on new data.

    This is what makes the model trustworthy:
    - The user picks (e.g.) "robust" scaling at training time.
    - The pipeline learns the train-set median and IQR.
    - At prediction time, those exact stats are used to scale new rows.
    - If a row contains a NaN, the imputer fills it with the train median —
      never the prediction-set median (which would leak distribution).
    """

    def _build_df(self, n: int = 80, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "age":    rng.uniform(20, 80, n),
                "bmi":    rng.uniform(18, 40, n),
                "region": rng.choice(["north", "south", "east", "west"], n),
                "target": rng.integers(0, 2, n),
            }
        )

    def _cfg_with_user_preprocessing(self, **overrides) -> TrainingConfig:
        # User chooses *atypical* methods so we know the test fails loud if
        # train and predict somehow diverge.
        prep = {
            "numericImputation":     "median",
            "numericScaling":        "robust",     # <-- not the default
            "categoricalImputation": "most_frequent",
            "categoricalEncoding":   "onehot",
        }
        prep.update(overrides.get("preprocessing", {}))
        return TrainingConfig.from_front(
            {
                "targetColumn": "target",
                "taskType": "classification",
                "models": ["logisticregression"],
                "metrics": ["accuracy"],
                "splitMethod": "holdout",
                "trainRatio": 80, "valRatio": 0, "testRatio": 20,
                "useGridSearch": False,
                "balancing": {"strategy": "none", "apply_threshold": False},
                "preprocessing": prep,
            }
        )

    def test_predict_uses_frozen_imputer_median(self, tmp_path):
        # Train with one dataset, predict on a *different* one whose median is
        # very different.  The imputer must use the TRAIN median, not the
        # predict-set median.
        df_train = self._build_df(seed=0)
        cfg = self._cfg_with_user_preprocessing()

        from app.services.training.orchestrator import run_one_model
        result = run_one_model(df_train, cfg, "logisticregression")
        pipeline = result.fitted_pipeline

        # Sanity: the prep step is a ColumnTransformer with a SimpleImputer.
        prep = pipeline.named_steps["prep"]
        from sklearn.compose import ColumnTransformer
        assert isinstance(prep, ColumnTransformer)

        # Find the numeric SimpleImputer and capture its frozen median.
        train_imputer = None
        for _name, sub_pipe, _cols in prep.transformers_:
            if hasattr(sub_pipe, "named_steps") and "imputer" in sub_pipe.named_steps:
                imp = sub_pipe.named_steps["imputer"]
                if hasattr(imp, "statistics_") and imp.strategy == "median":
                    train_imputer = imp
                    break
        assert train_imputer is not None
        train_medians = train_imputer.statistics_.copy()

        # Build a prediction dataframe with a totally different scale for "age".
        df_pred = self._build_df(n=10, seed=99).drop(columns=["target"])
        df_pred["age"] = df_pred["age"] + 1000  # shift far away from train median
        df_pred.loc[df_pred.index[0], "age"] = np.nan  # NaN to be imputed

        # Predict — must not fail, must use train median for imputation.
        from app.services.training.output.predictor import predict_with_trained_model
        m, path = _make_mock_trained_model(pipeline, result.artifacts_json)
        predict_with_trained_model(m, df_pred)

        # After predict, the imputer statistics_ must be UNCHANGED.
        np.testing.assert_array_equal(train_imputer.statistics_, train_medians)

    def test_predict_uses_frozen_scaler_stats(self, tmp_path):
        df_train = self._build_df(seed=1)
        cfg = self._cfg_with_user_preprocessing()  # numericScaling="robust"

        from app.services.training.orchestrator import run_one_model
        result = run_one_model(df_train, cfg, "logisticregression")
        pipeline = result.fitted_pipeline

        prep = pipeline.named_steps["prep"]
        train_scaler = None
        for _name, sub_pipe, _cols in prep.transformers_:
            if hasattr(sub_pipe, "named_steps") and "scaler" in sub_pipe.named_steps:
                train_scaler = sub_pipe.named_steps["scaler"]
                break
        assert train_scaler is not None
        # RobustScaler exposes center_ (median) and scale_ (IQR) after fit.
        train_center = train_scaler.center_.copy()
        train_scale = train_scaler.scale_.copy()

        # Predict on a dataset with much wider numeric range.
        df_pred = self._build_df(n=20, seed=42).drop(columns=["target"])
        df_pred["age"] = df_pred["age"] * 5  # huge shift in scale

        from app.services.training.output.predictor import predict_with_trained_model
        m, _path = _make_mock_trained_model(pipeline, result.artifacts_json)
        predict_with_trained_model(m, df_pred)

        # Scaler stats must NOT have moved — proves predict uses transform(),
        # not fit_transform().
        np.testing.assert_array_equal(train_scaler.center_, train_center)
        np.testing.assert_array_equal(train_scaler.scale_, train_scale)

    def test_predict_uses_frozen_onehot_categories(self, tmp_path):
        # If a category appears at predict time but was NOT in train,
        # OneHotEncoder(handle_unknown="ignore") must NOT learn it — it must
        # be silently dropped, exactly as if the column were missing.
        df_train = self._build_df(seed=2)
        cfg = self._cfg_with_user_preprocessing()

        from app.services.training.orchestrator import run_one_model
        result = run_one_model(df_train, cfg, "logisticregression")
        pipeline = result.fitted_pipeline

        prep = pipeline.named_steps["prep"]
        train_encoder = None
        for _name, sub_pipe, _cols in prep.transformers_:
            if hasattr(sub_pipe, "named_steps") and "encoder" in sub_pipe.named_steps:
                train_encoder = sub_pipe.named_steps["encoder"]
                break
        assert train_encoder is not None
        train_categories = [arr.copy() for arr in train_encoder.categories_]

        # Inject an unseen category "moon" into region.
        df_pred = self._build_df(n=8, seed=7).drop(columns=["target"])
        df_pred["region"] = ["moon"] * len(df_pred)

        from app.services.training.output.predictor import predict_with_trained_model
        m, _path = _make_mock_trained_model(pipeline, result.artifacts_json)
        # Must NOT raise (handle_unknown="ignore"); must NOT learn "moon".
        predict_with_trained_model(m, df_pred)

        for before, after in zip(train_categories, train_encoder.categories_):
            np.testing.assert_array_equal(before, after)
        # And "moon" is nowhere in the encoder's categories.
        all_cats = [str(c) for arr in train_encoder.categories_ for c in arr]
        assert "moon" not in all_cats

    def test_transform_output_identical_when_input_identical(self, tmp_path):
        # The strongest parity check: feeding the *training rows* back through
        # the loaded pipeline must produce the exact same preprocessed array
        # whether we call prep.transform(X) directly or via the full pipeline.
        df_train = self._build_df(seed=3)
        cfg = self._cfg_with_user_preprocessing()

        from app.services.training.orchestrator import run_one_model
        result = run_one_model(df_train, cfg, "logisticregression")
        pipeline = result.fitted_pipeline

        # Save & reload (matches what happens in production).
        from app.services.training.output.persistence import save_pipeline, load_pipeline
        pkl = save_pipeline(pipeline, tmp_path, "lr", threshold=0.5)
        reloaded = load_pipeline(pkl)

        X_sample = df_train.drop(columns=["target"]).head(5)

        # Apply each step manually up to (but excluding) the model.
        x = X_sample
        for name, step in reloaded.named_steps.items():
            if name == "model":
                break
            x = step.transform(x)

        # Same input passed through pipeline.predict_proba — internally calls
        # the same .transform() chain, so the underlying preprocessed array is
        # identical.  We assert that the model's decision_function on x equals
        # the one obtained through the full pipeline.
        manual_scores = reloaded.named_steps["model"].decision_function(x)
        pipeline_scores = reloaded.decision_function(X_sample)
        np.testing.assert_allclose(manual_scores, pipeline_scores, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# Drift detection robustness (regression: "Cannot cast object dtype to float64")
# ──────────────────────────────────────────────────────────────────────────────
#
# _normalize_input_dtypes turns a column whose values don't parse as numeric into
# a pd.Categorical. When the training stats marked that column "numeric",
# _detect_drift used to call series.astype(float) on the Categorical, which raises
# ValueError("Cannot cast object dtype to float64") and failed the whole
# prediction even though pipeline.predict() had already succeeded.

from app.services.training.output.predictor import _detect_drift, _normalize_input_dtypes


def test_detect_drift_does_not_crash_on_non_numeric_values_in_numeric_column():
    schema = {"column_stats": {"age": {"type": "numeric", "mean": 50.0, "std": 10.0}}}
    # Mixed/non-numeric content for a column the model expects to be numeric.
    raw = pd.DataFrame({"age": pd.Categorical(["n/a", "unknown", "??", "n/a"])})
    # Must not raise, and produces no mean-shift signal (nothing numeric remains).
    assert _detect_drift(raw, schema) == []


def test_detect_drift_coerces_numeric_strings_and_flags_shift():
    schema = {"column_stats": {"age": {"type": "numeric", "mean": 50.0, "std": 5.0}}}
    # Values arrive as strings (pd.Categorical of numeric strings) far from train mean.
    raw = pd.DataFrame({"age": pd.Categorical(["90", "92", "95", "absent"])})
    warnings = _detect_drift(raw, schema)
    assert any(w["type"] == "mean_shift" and w["severity"] == "critical" for w in warnings)


def test_detect_drift_after_normalize_input_dtypes_pipeline():
    """Mirror the real predict flow: normalize input dtypes, then detect drift."""
    schema = {"column_stats": {"score": {"type": "numeric", "mean": 0.0, "std": 1.0}}}
    raw = pd.DataFrame({"score": pd.Series(["bad", "missing", "n/a"], dtype="string")})
    normalized = _normalize_input_dtypes(raw, pipeline=None)
    # Should not raise regardless of how normalization typed the column.
    assert _detect_drift(normalized, schema) == []
