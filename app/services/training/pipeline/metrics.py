from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

from app.services.training.utils import to_python_scalar


def _to_json_label(value: Any) -> Any:
    value = to_python_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _array_shape(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    try:
        shape = np.asarray(value).shape
        return [int(s) for s in shape]
    except Exception:
        return None


def _stable_unique(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for raw_value in values:
        value = to_python_scalar(raw_value)
        if isinstance(value, float) and np.isnan(value):
            key = ("float", "nan")
        else:
            key = (type(value).__name__, str(value))
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _matching_label(label: Any, labels: Sequence[Any]) -> Any:
    label_str = str(to_python_scalar(label))
    for candidate in labels:
        if str(to_python_scalar(candidate)) == label_str:
            return candidate
    return None


def _is_zero_one_label_set(labels: Sequence[Any]) -> bool:
    try:
        normalized = {float(to_python_scalar(v)) for v in labels}
    except Exception:
        return False
    return normalized == {0.0, 1.0}


def _detect_classification_type(y_true: np.ndarray) -> str:
    target_kind = type_of_target(y_true)
    if target_kind in {"multilabel-indicator", "multiclass-multioutput"}:
        return "multilabel"
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        return "multilabel"
    n_classes = int(np.unique(y_true).size)
    return "binary" if n_classes == 2 else "multiclass"


def _resolve_primary_average(requested_metrics: Optional[Sequence[str]], classification_type: str) -> str:
    if classification_type == "binary":
        return "binary_pos"

    requested = [str(m or "").strip().lower() for m in (requested_metrics or [])]
    for metric_name in requested:
        if metric_name in {
            "weighted",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "average_weighted",
        }:
            return "weighted"
        if metric_name in {
            "micro",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "average_micro",
        }:
            return "micro"
        if metric_name in {
            "macro",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "average_macro",
        }:
            return "macro"
        # Align legacy "precision/recall/f1" display with trainer scoring in multiclass.
        if classification_type == "multiclass" and metric_name in {"precision", "recall", "f1"}:
            return "weighted"
    return "macro"


def _resolve_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[Any]]) -> list[Any]:
    if labels is not None and len(labels) > 0:
        return [to_python_scalar(v) for v in labels]

    values: list[Any] = []
    values.extend(np.asarray(y_true).ravel().tolist())
    values.extend(np.asarray(y_pred).ravel().tolist())
    return _stable_unique(values)


def _pick_positive_label(
    y_true: np.ndarray,
    labels: Sequence[Any],
    positive_label: Any,
    warnings: list[str],
) -> Any:
    if positive_label is not None:
        matched = _matching_label(positive_label, labels)
        if matched is not None:
            return matched
        warnings.append(
            f"positive_label '{positive_label}' is not present in labels; selecting automatically."
        )

    if _is_zero_one_label_set(labels):
        for label in labels:
            try:
                if float(to_python_scalar(label)) == 1.0:
                    return label
            except Exception:
                continue
    else:
        warnings.append(
            "positive_label is missing for non {0,1} binary labels; auto-selection may be unstable. "
            "Set positiveLabel explicitly from the frontend."
        )

    counts: dict[str, tuple[Any, int]] = {}
    y_flat = np.asarray(y_true).ravel()
    for label in labels:
        count = int(np.sum(y_flat == label))
        counts[str(to_python_scalar(label))] = (label, count)
    rare_key = min(counts.keys(), key=lambda k: counts[k][1])
    chosen = counts[rare_key][0]
    warnings.append(
        f"positive_label was auto-selected as rarest class: '{_to_json_label(chosen)}'."
    )
    return chosen


def _safe_average_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str,
) -> dict[str, Any]:
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        f2 = float(fbeta_score(y_true, y_pred, beta=2.0, average=average, zero_division=0))
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f2": f2,
            "support": None,
        }
    except Exception:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "support": None,
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_binary_score(
    *,
    y_score: Any,
    proba: Any,
    score_labels: Optional[Sequence[Any]],
    positive_label: Any,
    estimator: Any = None,
    warnings: Optional[list[str]] = None,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[int], Optional[float]]:
    score_labels_list = [to_python_scalar(v) for v in (score_labels or [])]
    if not score_labels_list and estimator is not None:
        score_labels_list = [to_python_scalar(v) for v in (get_class_labels(estimator) or [])]
    pos_index = None
    if len(score_labels_list) > 0:
        matched = _matching_label(positive_label, score_labels_list)
        if matched in score_labels_list:
            pos_index = int(score_labels_list.index(matched))
        elif warnings is not None:
            warnings.append(
                f"positive_label '{positive_label}' was not found in model classes for score extraction; "
                "falling back to default score orientation."
            )

    if proba is not None:
        arr = np.asarray(proba)
        if arr.ndim == 1:
            return arr, "proba", 0, 0.5
        if arr.ndim == 2:
            idx = pos_index
            if idx is None:
                idx = 1 if arr.shape[1] > 1 else 0
            if idx is not None and idx < arr.shape[1]:
                return arr[:, idx], "proba", int(idx), 0.5

    if y_score is not None:
        arr = np.asarray(y_score)
        if arr.ndim == 1:
            score_vec = arr.astype(float, copy=False)
            idx = pos_index
            if idx is None and len(score_labels_list) == 2:
                idx = 1
            if len(score_labels_list) == 2 and idx == 0:
                # Binary decision_function is usually oriented toward classes_[1].
                score_vec = -score_vec
            return score_vec, "decision", int(idx) if idx is not None else None, 0.0
        if arr.ndim == 2:
            idx = pos_index
            if idx is None:
                idx = 1 if arr.shape[1] > 1 else 0
            if idx is not None and idx < arr.shape[1]:
                return arr[:, idx], "decision", int(idx), 0.0

    return None, None, None, None


def _compute_multiclass_auc(
    *,
    y_true: np.ndarray,
    labels: Sequence[Any],
    y_score: Any,
    proba: Any,
    warnings: list[str],
    score_labels: Optional[Sequence[Any]] = None,
) -> tuple[Optional[float], Optional[float]]:
    if int(np.unique(y_true).size) < 2:
        warnings.append(
            "ROC-AUC/PR-AUC unavailable for this split: y_true contains a single class."
        )
        return None, None

    score_matrix = None
    source_name = None
    if proba is not None:
        arr = np.asarray(proba)
        if arr.ndim == 2:
            score_matrix = arr
            source_name = "predict_proba"
    if score_matrix is None and y_score is not None:
        arr = np.asarray(y_score)
        if arr.ndim == 2:
            score_matrix = arr
            source_name = "decision_function"

    if score_matrix is None:
        warnings.append("ROC-AUC/PR-AUC unavailable for multiclass: no probability/score matrix.")
        return None, None

    # Reorder score_matrix columns to match `labels` when the model's class order
    # (score_labels) differs from resolved_labels — prevents column misalignment in AUC.
    if score_labels is not None and list(score_labels) != list(labels):
        str_score = [str(s) for s in score_labels]
        str_labels = [str(lbl) for lbl in labels]
        if set(str_score) == set(str_labels):
            try:
                col_order = [str_score.index(lbl) for lbl in str_labels]
                score_matrix = score_matrix[:, col_order]
            except (ValueError, IndexError):
                warnings.append(
                    "ROC-AUC/PR-AUC: could not reorder score columns to match labels."
                )
                return None, None

    if score_matrix.shape[1] != len(labels):
        warnings.append(
            "ROC-AUC/PR-AUC unavailable for multiclass: score columns do not match labels."
        )
        return None, None

    roc_auc = None
    try:
        roc_auc = float(
            roc_auc_score(
                y_true,
                score_matrix,
                labels=np.asarray(labels),
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception as exc:
        warnings.append(f"ROC-AUC unavailable for multiclass: {exc}")

    pr_auc = None
    try:
        y_true_bin = label_binarize(y_true, classes=np.asarray(labels))
        if y_true_bin.ndim == 2 and y_true_bin.shape[1] == score_matrix.shape[1]:
            pr_auc = float(average_precision_score(y_true_bin, score_matrix, average="macro"))
        else:
            warnings.append("PR-AUC unavailable for multiclass: incompatible one-vs-rest target shape.")
    except Exception as exc:
        warnings.append(f"PR-AUC unavailable for multiclass ({source_name}): {exc}")

    return roc_auc, pr_auc


def _compute_multilabel_auc(
    *,
    y_true: np.ndarray,
    y_score: Any,
    proba: Any,
    warnings: list[str],
) -> tuple[Optional[float], Optional[float]]:
    score_matrix = None
    if proba is not None:
        arr = np.asarray(proba)
        if arr.ndim == 2:
            score_matrix = arr
    if score_matrix is None and y_score is not None:
        arr = np.asarray(y_score)
        if arr.ndim == 2:
            score_matrix = arr

    if score_matrix is None:
        warnings.append("ROC-AUC/PR-AUC unavailable for multilabel: no score matrix.")
        return None, None
    if score_matrix.shape != y_true.shape:
        warnings.append("ROC-AUC/PR-AUC unavailable for multilabel: score matrix shape mismatch.")
        return None, None

    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(y_true, score_matrix, average="macro"))
    except Exception as exc:
        warnings.append(f"ROC-AUC unavailable for multilabel: {exc}")

    pr_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, score_matrix, average="macro"))
    except Exception as exc:
        warnings.append(f"PR-AUC unavailable for multilabel: {exc}")

    return roc_auc, pr_auc


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 100) -> list[list[float]]:
    """Return [[x, y], ...] downsampled to at most max_points points."""
    n = len(x)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return [[round(float(x[i]), 6), round(float(y[i]), 6)] for i in idx]


def compute_roc_pr_curves(
    y_true_bin: np.ndarray,
    score_vector: np.ndarray,
    *,
    max_points: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Compute downsampled ROC and Precision-Recall curves for binary classification.

    Returns a dict with keys:
      "roc":  [[fpr, tpr], ...]  (max_points entries)
      "pr":   [[recall, precision], ...]  (max_points entries)
    or None if computation fails.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, score_vector)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true_bin, score_vector)
        # precision_recall_curve returns arrays in decreasing threshold order; flip so recall is ascending
        precision_arr = precision_arr[::-1]
        recall_arr = recall_arr[::-1]
        return {
            "roc": _downsample_curve(fpr, tpr, max_points),
            "pr": _downsample_curve(recall_arr, precision_arr, max_points),
        }
    except Exception:
        return None


def compute_calibration_curve(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Compute a calibration (reliability) curve and Brier score for binary classification.

    Returns a dict with:
      "points":       [[mean_predicted_prob, fraction_of_positives], ...]
      "brier_score":  float  (lower is better; perfect calibration → 0)
      "n_bins":       int
    or None if computation fails.
    """
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_bin, y_proba, n_bins=n_bins, strategy="uniform"
        )
        brier = float(brier_score_loss(y_true_bin, y_proba))
        points = [
            [float(mp), float(fp)]
            for mp, fp in zip(mean_predicted_value.tolist(), fraction_of_positives.tolist())
        ]
        return {"points": points, "brier_score": brier, "n_bins": n_bins}
    except Exception:
        return None


def compute_classification_metrics(
    y_true,
    y_pred,
    *,
    y_score=None,
    proba=None,
    labels: Optional[Sequence[Any]] = None,
    estimator: Any = None,
    positive_label: Any = None,
    requested_metrics: Optional[Sequence[str]] = None,
    task_type: str = "classification",
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    warnings: list[str] = []
    estimator_classes = get_class_labels(estimator) if estimator is not None else None
    model_classes = [to_python_scalar(v) for v in (estimator_classes or labels or [])]

    if str(task_type or "").strip().lower() != "classification":
        return {}

    classification_type = _detect_classification_type(y_true_arr)
    resolved_labels = _resolve_labels(y_true_arr, y_pred_arr, labels)
    if classification_type != "multilabel":
        classification_type = "binary" if len(resolved_labels) == 2 else "multiclass"
    averaging_primary = _resolve_primary_average(requested_metrics, classification_type)

    global_metrics: Dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": None,
        "roc_auc": None,
        "pr_auc": None,
        "specificity": None,
    }

    if classification_type == "multilabel":
        n_outputs = int(y_true_arr.shape[1]) if y_true_arr.ndim == 2 else 0
        resolved_labels = (
            [to_python_scalar(v) for v in labels]
            if labels is not None and len(labels) == n_outputs
            else list(range(n_outputs))
        )

        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            average=None,
            zero_division=0,
        )
        per_class = {}
        for idx in range(len(per_precision)):
            key = str(_to_json_label(resolved_labels[idx] if idx < len(resolved_labels) else idx))
            per_class[key] = {
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }

        averaged = {
            "macro": _safe_average_block(y_true_arr, y_pred_arr, average="macro"),
            "weighted": _safe_average_block(y_true_arr, y_pred_arr, average="weighted"),
            "micro": _safe_average_block(y_true_arr, y_pred_arr, average="micro"),
        }

        cm = multilabel_confusion_matrix(y_true_arr, y_pred_arr).astype(int).tolist()
        warnings.append("Multilabel confusion_matrix is returned as one 2x2 matrix per label.")

        roc_auc, pr_auc = _compute_multilabel_auc(
            y_true=y_true_arr,
            y_score=y_score,
            proba=proba,
            warnings=warnings,
        )
        global_metrics["roc_auc"] = roc_auc
        global_metrics["pr_auc"] = pr_auc

        legacy_src = averaged.get(averaging_primary) or averaged.get("macro", {})
        legacy_flat: Dict[str, Any] = {
            "accuracy": global_metrics["accuracy"],
            "precision": legacy_src.get("precision"),
            "recall": legacy_src.get("recall"),
            "f1": legacy_src.get("f1"),
            "roc_auc": global_metrics.get("roc_auc"),
            "pr_auc": global_metrics.get("pr_auc"),
            "precision_macro": averaged["macro"].get("precision"),
            "recall_macro": averaged["macro"].get("recall"),
            "f1_macro": averaged["macro"].get("f1"),
            "precision_weighted": averaged["weighted"].get("precision"),
            "recall_weighted": averaged["weighted"].get("recall"),
            "f1_weighted": averaged["weighted"].get("f1"),
            "precision_micro": averaged["micro"].get("precision"),
            "recall_micro": averaged["micro"].get("recall"),
            "f1_micro": averaged["micro"].get("f1"),
            "balanced_accuracy": None,
            "specificity": None,
        }

        out: Dict[str, Any] = {
            "global": global_metrics,
            "averaged": averaged,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": [_to_json_label(v) for v in resolved_labels],
                "matrix": cm,
            },
            "warnings": warnings,
            "meta": {
                "classification_type": classification_type,
                "labels": [_to_json_label(v) for v in resolved_labels],
                "positive_label": None,
                "resolved_positive_label": None,
                "model_classes": [_to_json_label(v) for v in model_classes] if model_classes else None,
                "auc_score_source": None,
                "auc_pos_index": None,
                "threshold_used": None,
                "averaging_defaults": {
                    "legacy_primary_average": averaging_primary,
                    "available_averages": ["macro", "weighted", "micro"],
                },
                "score_shapes": {
                    "proba": _array_shape(proba),
                    "y_score": _array_shape(y_score),
                },
            },
            "legacy_flat": legacy_flat,
        }
        out.update(legacy_flat)
        return out

    if not resolved_labels:
        warnings.append("No labels resolved from y_true/y_pred.")

    true_unique = _stable_unique(np.asarray(y_true_arr).ravel().tolist())
    missing_in_true = [label for label in resolved_labels if _matching_label(label, true_unique) is None]
    if missing_in_true:
        warnings.append(
            "Some labels are absent in y_true for this split: "
            + ", ".join(str(_to_json_label(v)) for v in missing_in_true)
        )

    score_labels = list(model_classes) if model_classes else list(resolved_labels)
    auc_score_source: Optional[str] = None
    auc_pos_index: Optional[int] = None
    threshold_used: Optional[float] = None
    binary_block: Dict[str, Any] = {}
    if classification_type == "binary":
        pos_label = _pick_positive_label(y_true_arr, resolved_labels, positive_label, warnings)
        neg_candidates = [label for label in resolved_labels if _matching_label(label, [pos_label]) is None]
        neg_label = neg_candidates[0] if neg_candidates else None
        if neg_label is not None:
            resolved_labels = [neg_label, pos_label]

        try:
            precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
                y_true_arr,
                y_pred_arr,
                average="binary",
                pos_label=pos_label,
                zero_division=0,
            )
            f2_pos = float(fbeta_score(y_true_arr, y_pred_arr, beta=2.0, pos_label=pos_label, average="binary", zero_division=0))
            binary_block = {
                "positive_label": _to_json_label(pos_label),
                "precision_pos": float(precision_pos),
                "recall_pos": float(recall_pos),
                "f1_pos": float(f1_pos),
                "f2_pos": f2_pos,
            }
        except Exception as exc:
            warnings.append(f"Binary positive-class metrics unavailable: {exc}")
            binary_block = {
                "positive_label": _to_json_label(pos_label),
                "precision_pos": None,
                "recall_pos": None,
                "f1_pos": None,
                "f2_pos": None,
            }
    else:
        pos_label = None

    try:
        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=np.asarray(resolved_labels),
            average=None,
            zero_division=0,
        )
        per_class = {
            str(_to_json_label(label)): {
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }
            for idx, label in enumerate(resolved_labels)
        }
    except Exception as exc:
        warnings.append(f"Per-class metrics unavailable: {exc}")
        per_class = {}

    averaged = {
        "macro": _safe_average_block(y_true_arr, y_pred_arr, average="macro"),
        "weighted": _safe_average_block(y_true_arr, y_pred_arr, average="weighted"),
        "micro": _safe_average_block(y_true_arr, y_pred_arr, average="micro"),
    }

    cm_matrix = []
    try:
        cm_matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=np.asarray(resolved_labels)).astype(int).tolist()
    except Exception as exc:
        warnings.append(f"Confusion matrix unavailable: {exc}")

    if classification_type in {"binary", "multiclass"}:
        try:
            global_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        except Exception as exc:
            warnings.append(f"balanced_accuracy unavailable: {exc}")

    if classification_type == "binary":
        if len(cm_matrix) == 2 and len(cm_matrix[0]) == 2:
            tn = float(cm_matrix[0][0])
            fp = float(cm_matrix[0][1])
            fn = float(cm_matrix[1][0])
            denom_spec = tn + fp
            global_metrics["specificity"] = float(tn / denom_spec) if denom_spec > 0 else None
            # NPV — "among patients declared healthy, how many truly are?" Critical for medical.
            denom_npv = tn + fn
            global_metrics["npv"] = float(tn / denom_npv) if denom_npv > 0 else None

        score_vector, auc_score_source, auc_pos_index, threshold_used = _extract_binary_score(
            y_score=y_score,
            proba=proba,
            score_labels=score_labels,
            positive_label=pos_label,
            estimator=estimator,
            warnings=warnings,
        )
        if score_vector is None:
            warnings.append("ROC-AUC/PR-AUC unavailable for binary classification: no continuous score.")
        else:
            y_true_bin = np.asarray(y_true_arr == pos_label, dtype=int)
            if int(np.unique(y_true_bin).size) < 2:
                warnings.append(
                    "ROC-AUC/PR-AUC unavailable for this split: y_true contains a single class."
                )
            else:
                try:
                    global_metrics["roc_auc"] = float(roc_auc_score(y_true_bin, score_vector))
                except Exception as exc:
                    warnings.append(f"ROC-AUC unavailable for binary classification: {exc}")
                try:
                    global_metrics["pr_auc"] = float(average_precision_score(y_true_bin, score_vector))
                except Exception as exc:
                    warnings.append(f"PR-AUC unavailable for binary classification: {exc}")
                try:
                    global_metrics["brier_score"] = float(brier_score_loss(y_true_bin, score_vector))
                except Exception as exc:
                    warnings.append(f"Brier score unavailable: {exc}")
        try:
            global_metrics["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))
        except Exception as exc:
            warnings.append(f"MCC unavailable: {exc}")

    if classification_type == "multiclass":
        roc_auc, pr_auc = _compute_multiclass_auc(
            y_true=y_true_arr,
            labels=resolved_labels,
            y_score=y_score,
            proba=proba,
            warnings=warnings,
            score_labels=score_labels,
        )
        global_metrics["roc_auc"] = roc_auc
        global_metrics["pr_auc"] = pr_auc
        try:
            global_metrics["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))
        except Exception as exc:
            warnings.append(f"MCC unavailable: {exc}")

    if classification_type == "binary":
        legacy_flat: Dict[str, Any] = {
            "accuracy": global_metrics["accuracy"],
            "precision": binary_block.get("precision_pos"),
            "recall": binary_block.get("recall_pos"),
            "f1": binary_block.get("f1_pos"),
            "roc_auc": global_metrics.get("roc_auc"),
            "pr_auc": global_metrics.get("pr_auc"),
            "brier_score": global_metrics.get("brier_score"),
            "precision_pos": binary_block.get("precision_pos"),
            "recall_pos": binary_block.get("recall_pos"),
            "f1_pos": binary_block.get("f1_pos"),
            "precision_macro": averaged["macro"].get("precision"),
            "recall_macro": averaged["macro"].get("recall"),
            "f1_macro": averaged["macro"].get("f1"),
            "precision_weighted": averaged["weighted"].get("precision"),
            "recall_weighted": averaged["weighted"].get("recall"),
            "f1_weighted": averaged["weighted"].get("f1"),
            "precision_micro": averaged["micro"].get("precision"),
            "recall_micro": averaged["micro"].get("recall"),
            "f1_micro": averaged["micro"].get("f1"),
            "balanced_accuracy": global_metrics.get("balanced_accuracy"),
            "specificity": global_metrics.get("specificity"),
            "npv": global_metrics.get("npv"),
            "mcc": global_metrics.get("mcc"),
        }
    else:
        source_average = averaged.get(averaging_primary) or averaged.get("macro", {})
        legacy_flat = {
            "accuracy": global_metrics["accuracy"],
            "precision": source_average.get("precision"),
            "recall": source_average.get("recall"),
            "f1": source_average.get("f1"),
            "roc_auc": global_metrics.get("roc_auc"),
            "pr_auc": global_metrics.get("pr_auc"),
            "precision_macro": averaged["macro"].get("precision"),
            "recall_macro": averaged["macro"].get("recall"),
            "f1_macro": averaged["macro"].get("f1"),
            "precision_weighted": averaged["weighted"].get("precision"),
            "recall_weighted": averaged["weighted"].get("recall"),
            "f1_weighted": averaged["weighted"].get("f1"),
            "precision_micro": averaged["micro"].get("precision"),
            "recall_micro": averaged["micro"].get("recall"),
            "f1_micro": averaged["micro"].get("f1"),
            "balanced_accuracy": global_metrics.get("balanced_accuracy"),
            "specificity": None,
            "npv": None,
            "mcc": global_metrics.get("mcc"),
        }

    out = {
        "global": global_metrics,
        "averaged": averaged,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [_to_json_label(v) for v in resolved_labels],
            "matrix": cm_matrix,
        },
        "warnings": warnings,
        "meta": {
            "classification_type": classification_type,
            "labels": [_to_json_label(v) for v in resolved_labels],
            "positive_label": _to_json_label(pos_label) if pos_label is not None else None,
            "resolved_positive_label": _to_json_label(pos_label) if pos_label is not None else None,
            "model_classes": [_to_json_label(v) for v in model_classes] if model_classes else None,
            "auc_score_source": auc_score_source,
            "auc_pos_index": int(auc_pos_index) if auc_pos_index is not None else None,
            "threshold_used": threshold_used,
            "averaging_defaults": {
                "legacy_primary_average": averaging_primary,
                "available_averages": ["macro", "weighted", "micro"],
            },
            "score_shapes": {
                "proba": _array_shape(proba),
                "y_score": _array_shape(y_score),
            },
        },
        "legacy_flat": legacy_flat,
    }
    if classification_type == "binary":
        out["binary"] = binary_block
        # Compute ROC/PR + calibration curves (stored so orchestrator can copy into artifacts_json)
        if pos_label is not None and score_vector is not None:
            _y_true_bin = np.asarray(y_true_arr == pos_label, dtype=int)
            if int(np.unique(_y_true_bin).size) == 2:
                curves = compute_roc_pr_curves(_y_true_bin, score_vector) or {}
                # Use proba for calibration when available (more accurate); fall back to score_vector
                _proba_for_cal = None
                if proba is not None:
                    try:
                        _proba_for_cal = _extract_binary_score(
                            y_score=None,
                            proba=proba,
                            score_labels=score_labels,
                            positive_label=pos_label,
                            estimator=None,
                            warnings=[],
                        )[0]
                    except Exception:
                        pass
                cal_proba = _proba_for_cal if _proba_for_cal is not None else score_vector
                cal_result = compute_calibration_curve(_y_true_bin, cal_proba)
                if cal_result is not None:
                    curves["calibration"] = cal_result
                out["curves"] = curves if curves else None

    out.update(legacy_flat)
    return out


def is_binary(y: np.ndarray) -> bool:
    try:
        return len(np.unique(y)) == 2
    except Exception:
        return False


def get_proba_or_score(model, X) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y_proba = None
    y_score = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
    except Exception:
        y_proba = None
    try:
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
    except Exception:
        y_score = None
    return y_proba, y_score


def get_class_labels(model) -> Optional[list[Any]]:
    def _classes_from_estimator(estimator: Any) -> Any:
        classes_local = getattr(estimator, "classes_", None)
        if classes_local is not None:
            return classes_local
        named_steps = getattr(estimator, "named_steps", None)
        if isinstance(named_steps, dict):
            inner = named_steps.get("model")
            classes_local = getattr(inner, "classes_", None)
            if classes_local is not None:
                return classes_local
        return None

    candidate = model
    for _ in range(3):
        classes = _classes_from_estimator(candidate)
        if classes is not None:
            return [to_python_scalar(v) for v in np.asarray(classes).ravel().tolist()]
        next_estimator = getattr(candidate, "best_estimator_", None)
        if next_estimator is None:
            break
        candidate = next_estimator
    return None


def classification_metrics(
    y_true,
    y_pred,
    *,
    y_proba=None,
    y_score=None,
    labels: Optional[Sequence[Any]] = None,
    estimator: Any = None,
    positive_label: Any = None,
    requested_metrics: Optional[Sequence[str]] = None,
    task_type: str = "classification",
) -> Dict[str, Any]:
    return compute_classification_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        proba=y_proba,
        labels=labels,
        estimator=estimator,
        positive_label=positive_label,
        requested_metrics=requested_metrics,
        task_type=task_type,
    )


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def safe_confusion_matrix(y_true, y_pred) -> list[list[int]]:
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.astype(int).tolist()
    except Exception:
        return []
