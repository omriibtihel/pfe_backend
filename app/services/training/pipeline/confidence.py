"""
Bootstrap confidence intervals for training evaluation metrics.

Computes non-parametric percentile confidence intervals via Efron-style
bootstrap resampling.  Designed for small-to-medium eval sets where
asymptotic Gaussian intervals are unreliable.

Exported function:
    compute_bootstrap_cis(y_true, y_pred, *, y_score, task_type, ...)
        → dict ready to be stored in metrics_json["confidence_intervals"]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ci_entry(values: np.ndarray, ci_level: float) -> Dict[str, float]:
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(values, alpha * 100))
    hi = float(np.percentile(values, (1 - alpha) * 100))
    return {
        "mean": float(np.mean(values)),
        "ci_low": lo,
        "ci_high": hi,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _bootstrap_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    *,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = len(y_true)
    labels = list(np.unique(y_true))
    is_binary = len(labels) == 2

    acc_s: List[float] = []
    f1_s: List[float] = []
    prec_s: List[float] = []
    rec_s: List[float] = []
    roc_s: List[float] = []
    pr_s: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]

        # Skip bootstrap samples where not all classes appear (degenerate)
        if len(np.unique(yt)) < 2:
            continue

        acc_s.append(float(accuracy_score(yt, yp)))

        avg = "binary" if is_binary else "macro"
        try:
            f1_s.append(float(f1_score(yt, yp, average=avg, zero_division=0)))
            prec_s.append(float(precision_score(yt, yp, average=avg, zero_division=0)))
            rec_s.append(float(recall_score(yt, yp, average=avg, zero_division=0)))
        except Exception:
            pass

        if y_score is not None:
            ys = y_score[idx]
            try:
                if is_binary:
                    roc_s.append(float(roc_auc_score(yt, ys)))
                    pr_s.append(float(average_precision_score(yt, ys)))
                else:
                    roc_s.append(float(roc_auc_score(yt, ys, multi_class="ovr", average="macro")))
            except Exception:
                pass

    metrics: Dict[str, Any] = {}
    if acc_s:
        metrics["accuracy"] = _ci_entry(np.array(acc_s), ci_level)
    if f1_s:
        metrics["f1"] = _ci_entry(np.array(f1_s), ci_level)
    if prec_s:
        metrics["precision"] = _ci_entry(np.array(prec_s), ci_level)
    if rec_s:
        metrics["recall"] = _ci_entry(np.array(rec_s), ci_level)
    if roc_s:
        metrics["roc_auc"] = _ci_entry(np.array(roc_s), ci_level)
    if pr_s:
        metrics["pr_auc"] = _ci_entry(np.array(pr_s), ci_level)

    return metrics


def _bootstrap_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = len(y_true)

    mae_s: List[float] = []
    mse_s: List[float] = []
    r2_s: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mae_s.append(float(mean_absolute_error(yt, yp)))
        mse_s.append(float(mean_squared_error(yt, yp)))
        try:
            r2_s.append(float(r2_score(yt, yp)))
        except Exception:
            pass

    metrics: Dict[str, Any] = {}
    if mae_s:
        metrics["mae"] = _ci_entry(np.array(mae_s), ci_level)
    if mse_s:
        metrics["mse"] = _ci_entry(np.array(mse_s), ci_level)
    if r2_s:
        metrics["r2"] = _ci_entry(np.array(r2_s), ci_level)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_bootstrap_cis(
    y_true,
    y_pred,
    *,
    y_score=None,
    task_type: str = "classification",
    n_bootstrap: int = 300,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute non-parametric bootstrap confidence intervals for evaluation metrics.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted values.
    y_score:
        Continuous score (proba column for positive class, or decision_function).
        Used for ROC-AUC / PR-AUC CIs on classification tasks.
    task_type:
        "classification" | "regression"
    n_bootstrap:
        Number of bootstrap resamples (default 300 — fast but reliable for n ≥ 50).
    ci_level:
        Confidence level, e.g. 0.95 → 95 % CI (default).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    dict with shape::

        {
            "ci_level": 0.95,
            "n_bootstrap": 300,
            "n_samples": int,
            "metrics": {
                "accuracy": {"mean": float, "ci_low": float, "ci_high": float, "std": float},
                "f1": {...},
                ...
            }
        }
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score) if y_score is not None else None
    n = len(y_true_arr)

    if n < 10:
        return {
            "ci_level": ci_level,
            "n_bootstrap": 0,
            "n_samples": n,
            "metrics": {},
            "warning": "Too few samples for bootstrap CIs (minimum 10).",
        }

    rng = np.random.default_rng(random_state)

    task = str(task_type).strip().lower()
    if task == "regression":
        metrics = _bootstrap_regression(
            y_true_arr, y_pred_arr,
            n_bootstrap=n_bootstrap, ci_level=ci_level, rng=rng,
        )
    else:
        metrics = _bootstrap_classification(
            y_true_arr, y_pred_arr, y_score_arr,
            n_bootstrap=n_bootstrap, ci_level=ci_level, rng=rng,
        )

    return {
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
        "metrics": metrics,
    }
