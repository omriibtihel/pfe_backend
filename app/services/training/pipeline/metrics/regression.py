"""
Regression metrics and utility helpers.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def safe_confusion_matrix(y_true: Any, y_pred: Any) -> list[list[int]]:
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.astype(int).tolist()
    except Exception:
        return []
