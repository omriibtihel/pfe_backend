from __future__ import annotations

from types import SimpleNamespace

from app.services.reporting.feature_glossary import FeatureGlossary
from app.services.reporting.model_context import build_reporting_model_context


def _model():
    return SimpleNamespace(
        metrics_json={
            "holdout_test_metrics": {
                "legacy_flat": {
                    "accuracy": 0.775,
                    "balanced_accuracy": 0.761,
                    "roc_auc": 0.84,
                    "precision_pos": 0.667,
                    "recall_pos": 0.716,
                    "f1_pos": 0.69,
                },
                "meta": {"positive_label": 1, "threshold_used": 0.5},
                "confusion_matrix": {"labels": [0, 1], "matrix": [[121, 29], [23, 58]]},
            },
        },
        artifacts_json={
            "split_info": {
                "method": "stratified_kfold",
                "k_folds": 5,
                "n_samples": 768,
                "test_rows": 231,
            },
            "training_schema": {
                "target": "Outcome",
                "feature_names": [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "BMI",
                    "DiabetesPedigreeFunction",
                ],
                "column_stats": {
                    "Glucose": {"mean": 121.59, "min": 57, "max": 198},
                    "BMI": {"mean": 32.62, "min": 18.2, "max": 67.1},
                },
            },
            "feature_importance": [
                {"feature": "num__Glucose", "importance": 0.2704},
                {"feature": "num__BMI", "importance": 0.1555},
            ],
        },
    )


def test_model_context_translates_positive_class_label():
    """The generic class-label logic translates raw "1" / "0" into a human-
    readable positive/negative indication, independent of dataset name."""
    ctx = build_reporting_model_context(
        _model(),
        raw_prediction=1,
        lang="fr",
        glossary=FeatureGlossary(),
    )
    # Generic output: "resultat positif suggere" — no dataset-specific disease name.
    assert "positif" in str(ctx.display_prediction)
    assert ctx.class_context.raw_label == "1"
    assert ctx.class_context.target_name == "Outcome"
    assert ctx.class_context.positive_class == "1"
    assert ctx.class_context.label_meaning == "classe positive"


def test_model_context_translates_negative_class_label():
    ctx = build_reporting_model_context(
        _model(),
        raw_prediction=0,
        lang="fr",
        glossary=FeatureGlossary(),
    )
    assert "negatif" in str(ctx.display_prediction)
    assert ctx.class_context.label_meaning == "classe negative"


def test_model_context_extracts_quality_and_dataset_summary():
    ctx = build_reporting_model_context(_model(), raw_prediction=1, lang="fr", glossary=FeatureGlossary())
    values = {m.label: m.value for m in ctx.model_quality}
    assert "Exactitude globale" in values
    assert values["Exactitude globale"] == "77.5 %"
    assert any("768" in item for item in ctx.dataset_summary)
    assert any("231" in item for item in ctx.dataset_summary)


def test_model_context_extracts_feature_metadata_aliases():
    ctx = build_reporting_model_context(_model(), raw_prediction=1, lang="fr", glossary=FeatureGlossary())
    glucose = ctx.feature_metadata["Glucose"]
    assert "moyenne entrainement" in glucose["training_reference"]
    assert "121.59" in glucose["training_reference"]
    assert "rang global 1/2" in glucose["global_importance"]
    assert ctx.feature_metadata["num__Glucose"]["global_importance"] == glucose["global_importance"]
    assert ctx.feature_metadata["glucose"]["training_reference"] == glucose["training_reference"]
