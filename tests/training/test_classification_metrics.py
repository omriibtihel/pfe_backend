from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from app.services.training.presenter import model_to_front_result as _model_to_front_result
from app.services.training.pipeline.metrics import compute_classification_metrics


def test_imbalanced_binary_keeps_recall_pos_separate_from_weighted_recall() -> None:
    y_true = np.array([0] * 95 + [1] * 5, dtype=int)
    y_pred = np.zeros_like(y_true)

    out = compute_classification_metrics(y_true, y_pred, labels=[0, 1])
    legacy = out["legacy_flat"]
    binary = out["binary"]

    assert out["meta"]["classification_type"] == "binary"
    assert legacy["recall_weighted"] == pytest.approx(legacy["accuracy"], abs=1e-12)
    assert binary["recall_pos"] < legacy["recall_weighted"]
    assert legacy["recall"] == pytest.approx(binary["recall_pos"])


def test_multiclass_per_class_and_averages_are_consistent() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2], dtype=int)

    out = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])

    assert out["meta"]["classification_type"] == "multiclass"

    recalls = np.array(
        [
            out["per_class"]["0"]["recall"],
            out["per_class"]["1"]["recall"],
            out["per_class"]["2"]["recall"],
        ],
        dtype=float,
    )
    supports = np.array(
        [
            out["per_class"]["0"]["support"],
            out["per_class"]["1"]["support"],
            out["per_class"]["2"]["support"],
        ],
        dtype=float,
    )

    expected_macro_recall = float(recalls.mean())
    expected_weighted_recall = float(np.average(recalls, weights=supports))

    assert out["averaged"]["macro"]["recall"] == pytest.approx(expected_macro_recall)
    assert out["averaged"]["weighted"]["recall"] == pytest.approx(expected_weighted_recall)
    assert out["legacy_flat"]["recall"] == pytest.approx(out["averaged"]["macro"]["recall"])


def test_binary_roc_auc_uses_continuous_scores_not_hard_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_pred = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.10, 0.20, 0.80, 0.90], dtype=float)

    out = compute_classification_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        labels=[0, 1],
        positive_label=1,
    )

    auc_from_scores = float(roc_auc_score(y_true, y_score))
    auc_from_predictions = float(roc_auc_score(y_true, y_pred))

    assert out["global"]["roc_auc"] == pytest.approx(auc_from_scores)
    assert out["global"]["roc_auc"] != pytest.approx(auc_from_predictions)


def test_binary_auc_pr_auc_use_positive_label_column_with_string_labels() -> None:
    y_true = np.array(["No", "No", "Yes", "Yes"], dtype=object)
    y_pred = np.array(["No", "No", "Yes", "Yes"], dtype=object)
    y_proba = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.10, 0.90],
        ],
        dtype=float,
    )

    out = compute_classification_metrics(
        y_true,
        y_pred,
        proba=y_proba,
        labels=["No", "Yes"],
        positive_label="No",
    )

    y_true_bin = np.asarray(y_true == "No", dtype=int)
    expected_auc = float(roc_auc_score(y_true_bin, y_proba[:, 0]))
    expected_pr_auc = float(average_precision_score(y_true_bin, y_proba[:, 0]))

    assert out["binary"]["positive_label"] == "No"
    assert out["binary"]["f1_pos"] == pytest.approx(1.0)
    assert out["global"]["roc_auc"] == pytest.approx(expected_auc)
    assert out["global"]["pr_auc"] == pytest.approx(expected_pr_auc)
    assert out["meta"]["auc_score_source"] == "proba"
    assert out["meta"]["auc_pos_index"] == 0
    assert out["meta"]["threshold_used"] == pytest.approx(0.5)


def test_binary_auc_not_computable_when_single_class_in_split() -> None:
    y_true = np.array(["No", "No", "No"], dtype=object)
    y_pred = np.array(["No", "No", "No"], dtype=object)
    y_proba = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.85, 0.15],
        ],
        dtype=float,
    )

    out = compute_classification_metrics(
        y_true,
        y_pred,
        proba=y_proba,
        labels=["No", "Yes"],
        positive_label="Yes",
    )

    assert out["global"]["roc_auc"] is None
    assert out["global"]["pr_auc"] is None
    assert any("single class" in str(msg).lower() for msg in out["warnings"])


def test_multiclass_requested_f1_uses_weighted_average_for_legacy_flat() -> None:
    y_true = np.array([0] * 50 + [1] * 10 + [2] * 5, dtype=int)
    y_pred = np.array([0] * 45 + [1] * 5 + [1] * 8 + [0] * 2 + [2] * 3 + [1] * 2, dtype=int)

    out = compute_classification_metrics(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        requested_metrics=["accuracy", "f1"],
    )

    assert out["meta"]["averaging_defaults"]["legacy_primary_average"] == "weighted"
    assert out["legacy_flat"]["f1"] == pytest.approx(out["averaged"]["weighted"]["f1"])
    assert out["legacy_flat"]["f1"] != pytest.approx(out["averaged"]["macro"]["f1"])


def test_front_result_keeps_unavailable_metrics_as_none() -> None:
    metrics = compute_classification_metrics(
        np.array([0, 0, 0], dtype=int),
        np.array([0, 0, 0], dtype=int),
        proba=np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.85, 0.15],
            ],
            dtype=float,
        ),
        labels=[0, 1],
        positive_label=1,
    )
    model = SimpleNamespace(
        id=1,
        model_type="logisticregression",
        task_type="classification",
        metrics_json={"test": metrics, "train": metrics, "training_time_sec": 0.1},
        artifacts_json={},
        is_saved=False,
    )

    front = _model_to_front_result(model)

    assert front["metrics"]["rocAuc"] is None
