"""
Regression metrics and utility helpers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    warn: List[str] = []

    # ── Detect NaN / Inf in predictions ──────────────────────────────────────
    n_nan = int(np.sum(np.isnan(y_pred_arr)))
    n_inf = int(np.sum(np.isinf(y_pred_arr)))
    if n_nan > 0:
        warn.append(
            f"y_pred contains {n_nan} NaN value(s) — the model may have diverged "
            "(check for extreme feature values, log(0), or unconstrained MLP)."
        )
    if n_inf > 0:
        warn.append(
            f"y_pred contains {n_inf} Inf value(s) — numerical instability detected."
        )

    # Filter out invalid samples for metric computation
    valid = np.isfinite(y_pred_arr) & np.isfinite(y_true_arr)
    n_invalid = int((~valid).sum())
    if n_invalid > 0 and n_invalid < len(y_pred_arr):
        warn.append(f"{n_invalid} sample(s) with NaN/Inf excluded from metrics.")
        y_true_c = y_true_arr[valid]
        y_pred_c = y_pred_arr[valid]
    elif n_invalid == len(y_pred_arr):
        result: Dict[str, Any] = {"mae": None, "mse": None, "rmse": None, "r2": None, "mape": None}
        warn.append("All predictions are invalid (NaN/Inf) — no metrics can be computed.")
        result["warnings"] = warn
        return result
    else:
        y_true_c = y_true_arr
        y_pred_c = y_pred_arr

    mse = float(mean_squared_error(y_true_c, y_pred_c))
    r2_val = float(r2_score(y_true_c, y_pred_c))

    # ── MAPE: skip zero-target samples to avoid division by zero ─────────────
    mape: Optional[float] = None
    nonzero = y_true_c != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true_c[nonzero] - y_pred_c[nonzero]) / y_true_c[nonzero])))
        n_zero_targets = int((~nonzero).sum())
        if n_zero_targets > 0:
            warn.append(
                f"{n_zero_targets} sample(s) with y_true=0 excluded from MAPE "
                "(division by zero)."
            )
    else:
        warn.append("MAPE not computed: all true values are 0.")

    # ── Warn when R² is negative (model worse than constant baseline) ─────────
    if r2_val < 0:
        warn.append(
            f"R²={r2_val:.4f} < 0: the model predicts worse than a constant "
            "mean baseline. Possible causes: severe underfitting, data leakage "
            "in the inverse direction, or a train/test mismatch."
        )

    result = {
        "mae":  float(mean_absolute_error(y_true_c, y_pred_c)),
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "r2":   r2_val,
        "mape": mape,
    }
    if warn:
        result["warnings"] = warn
    return result


def safe_confusion_matrix(y_true: Any, y_pred: Any) -> list[list[int]]:
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.astype(int).tolist()
    except Exception:
        return []
