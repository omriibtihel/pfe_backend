from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


@dataclass(frozen=True)
class ThresholdResult:
    optimal_threshold: float
    strategy_used: str
    f1_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    default_threshold_f1: float
    improvement_delta: float


def _f_beta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = float(beta) ** 2
    denom = (b2 * precision) + recall
    out = np.zeros_like(precision, dtype=float)
    valid = denom > 0
    out[valid] = (1.0 + b2) * precision[valid] * recall[valid] / denom[valid]
    return out


def _as_binary_minority(y_true: np.ndarray) -> np.ndarray:
    values, counts = np.unique(y_true, return_counts=True)
    if len(values) != 2:
        return np.asarray([], dtype=int)
    minority_label = values[int(np.argmin(counts))]
    return (y_true == minority_label).astype(int)


class ThresholdOptimizer:
    def optimize(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        strategy: str,
        min_recall: float = 0.70,
        beta: float = 2.0,
    ) -> ThresholdResult:
        y_arr = np.asarray(y_true).reshape(-1)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
            proba_arr = proba_arr[:, 1]
        proba_arr = proba_arr.reshape(-1)

        if y_arr.size == 0 or proba_arr.size == 0 or y_arr.size != proba_arr.size:
            return ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="maximize_f1",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )

        y_bin = _as_binary_minority(y_arr)
        if y_bin.size == 0:
            return ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="maximize_f1",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )

        y_default = (proba_arr >= 0.5).astype(int)
        default_f1 = float(f1_score(y_bin, y_default, zero_division=0))

        precision, recall, thresholds = precision_recall_curve(y_bin, proba_arr)
        if thresholds.size == 0:
            return ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="maximize_f1",
                f1_at_threshold=default_f1,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=default_f1,
                improvement_delta=0.0,
            )

        precision = np.nan_to_num(np.asarray(precision[:-1], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        recall = np.nan_to_num(np.asarray(recall[:-1], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        thresholds = np.asarray(thresholds, dtype=float)
        f1_scores = _f_beta(precision, recall, beta=1.0)
        f2_scores = _f_beta(precision, recall, beta=float(beta))

        selected_idx = int(np.nanargmax(f1_scores))
        strategy_used = "maximize_f1"
        strategy_norm = str(strategy or "maximize_f1").strip().lower()

        if strategy_norm == "maximize_f2":
            selected_idx = int(np.nanargmax(f2_scores))
            strategy_used = "maximize_f2"
        elif strategy_norm == "min_recall":
            target_recall = float(min_recall)
            if not (0.0 < target_recall < 1.0):
                target_recall = 0.70
            valid = np.where(recall >= target_recall)[0]
            if valid.size == 0:
                selected_idx = int(np.nanargmax(f1_scores))
                strategy_used = "maximize_f1"
            else:
                best_valid = valid[int(np.nanargmax(precision[valid]))]
                selected_idx = int(best_valid)
                strategy_used = "min_recall"
        elif strategy_norm == "precision_recall_balance":
            distance = np.abs(precision - recall)
            selected_idx = int(np.nanargmin(distance))
            strategy_used = "precision_recall_balance"
        else:
            selected_idx = int(np.nanargmax(f1_scores))
            strategy_used = "maximize_f1"

        optimal_threshold = float(thresholds[selected_idx])
        f1_at_threshold = float(f1_scores[selected_idx])
        precision_at_threshold = float(precision[selected_idx])
        recall_at_threshold = float(recall[selected_idx])

        return ThresholdResult(
            optimal_threshold=optimal_threshold,
            strategy_used=strategy_used,
            f1_at_threshold=f1_at_threshold,
            precision_at_threshold=precision_at_threshold,
            recall_at_threshold=recall_at_threshold,
            default_threshold_f1=default_f1,
            improvement_delta=float(f1_at_threshold - default_f1),
        )
