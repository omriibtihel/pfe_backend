"""
ROC, Precision-Recall, and calibration curves for binary classification,
plus One-vs-Rest variants for multiclass.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


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


def compute_multiclass_ovr_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_labels: Sequence[Any],
    *,
    max_points: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Compute One-vs-Rest ROC and Precision-Recall curves for multiclass classification.

    Returns a dict shaped as::

        {
          "multiclassRoc": {
            "perClass": [{"label": str, "points": [[fpr,tpr], ...], "auc": float|None}, ...],
            "macroAuc": float|None,
          },
          "multiclassPr": {
            "perClass": [{"label": str, "points": [[recall,prec], ...], "auc": float|None}, ...],
            "macroAuc": float|None,   # mean Average Precision across classes
          },
        }

    A class is silently skipped when ``y_true`` does not contain both
    "is_class_k" and "is_not_class_k" samples (curve would be ill-defined).
    Returns ``None`` if no class survives the filter or if shapes mismatch.
    """
    try:
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] != len(class_labels):
            return None

        per_class_roc: list[Dict[str, Any]] = []
        per_class_pr: list[Dict[str, Any]] = []
        aucs_roc: list[float] = []
        aucs_pr: list[float] = []

        for k, label in enumerate(class_labels):
            y_bin = np.asarray(y_true == label, dtype=int)
            if int(np.unique(y_bin).size) < 2:
                continue
            scores_k = proba[:, k]

            fpr, tpr, _ = roc_curve(y_bin, scores_k)
            prec, rec, _ = precision_recall_curve(y_bin, scores_k)
            prec, rec = prec[::-1], rec[::-1]

            try:
                auc_roc: Optional[float] = float(roc_auc_score(y_bin, scores_k))
            except Exception:
                auc_roc = None
            try:
                ap: Optional[float] = float(average_precision_score(y_bin, scores_k))
            except Exception:
                ap = None

            per_class_roc.append({
                "label": str(label),
                "points": _downsample_curve(fpr, tpr, max_points),
                "auc": auc_roc,
            })
            per_class_pr.append({
                "label": str(label),
                "points": _downsample_curve(rec, prec, max_points),
                "auc": ap,
            })
            if auc_roc is not None:
                aucs_roc.append(auc_roc)
            if ap is not None:
                aucs_pr.append(ap)

        if not per_class_roc:
            return None

        return {
            "multiclassRoc": {
                "perClass": per_class_roc,
                "macroAuc": float(np.mean(aucs_roc)) if aucs_roc else None,
            },
            "multiclassPr": {
                "perClass": per_class_pr,
                "macroAuc": float(np.mean(aucs_pr)) if aucs_pr else None,
            },
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
