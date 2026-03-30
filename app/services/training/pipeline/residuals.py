"""
Residual analysis for regression models.

Provides diagnostic statistics that help users understand whether a regression
model's errors are random (good) or systematic (bad):

  - stats: mean/std/skewness/correlation of residuals with predictions
  - histogram: distribution of residuals (should be approximately Gaussian)
  - qq_points: Q-Q plot data to visually assess normality

All computation is on the TEST set (no training data leakage).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


_QQ_POINTS = 50
_HIST_BINS = 20


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str = "regression",
) -> Optional[Dict[str, Any]]:
    """
    Compute residual diagnostics for a regression model.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted values (test set).
    task_type:
        Only processes "regression"; returns None for any other task.

    Returns
    -------
    dict with shape::

        {
            "stats": {
                "mean":                float,   # should be ≈ 0 for unbiased model
                "std":                 float,
                "min":                 float,
                "max":                 float,
                "skewness":            float,   # 0 = symmetric
                "correlation_with_pred": float, # 0 = no pattern with predictions
            },
            "histogram": {
                "bin_edges": [float, ...],   # length n_bins + 1
                "counts":    [int,   ...],   # length n_bins
            },
            "qq_points": [[theoretical_z, residual], ...],  # for Q-Q plot
            "n_samples": int,
        }

    or None if task_type != "regression" or computation fails.
    """
    if str(task_type).strip().lower() != "regression":
        return None

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) < 5:
        return None

    try:
        residuals = y_t - y_p

        # Stats
        r_mean = float(np.mean(residuals))
        r_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        r_min = float(np.min(residuals))
        r_max = float(np.max(residuals))
        r_skew = float(stats.skew(residuals))
        corr_with_pred = float(np.corrcoef(y_p, residuals)[0, 1]) if len(residuals) > 1 else 0.0
        # Replace nan correlation (constant predictions) with 0
        if not np.isfinite(corr_with_pred):
            corr_with_pred = 0.0

        # Histogram
        counts, bin_edges = np.histogram(residuals, bins=_HIST_BINS)

        # Q-Q plot: theoretical standard-normal quantiles vs sorted residuals
        n = len(residuals)
        qq_result = stats.probplot(residuals, dist="norm")
        theoretical_q: np.ndarray = qq_result[0][0]
        sample_q: np.ndarray = qq_result[0][1]

        # Downsample to _QQ_POINTS evenly-spaced indices
        if len(theoretical_q) > _QQ_POINTS:
            idx = np.linspace(0, len(theoretical_q) - 1, _QQ_POINTS).astype(int)
            theoretical_q = theoretical_q[idx]
            sample_q = sample_q[idx]

        qq_points: List[List[float]] = [
            [float(t), float(s)]
            for t, s in zip(theoretical_q.tolist(), sample_q.tolist())
        ]

        return {
            "stats": {
                "mean": r_mean,
                "std": r_std,
                "min": r_min,
                "max": r_max,
                "skewness": r_skew,
                "correlation_with_pred": corr_with_pred,
            },
            "histogram": {
                "bin_edges": [float(e) for e in bin_edges.tolist()],
                "counts": [int(c) for c in counts.tolist()],
            },
            "qq_points": qq_points,
            "n_samples": int(n),
        }
    except Exception:
        return None
