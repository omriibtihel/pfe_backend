"""
ROC, Precision-Recall, and calibration curves for binary classification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve


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
