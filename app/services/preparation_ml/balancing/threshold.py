from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve


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


def _cost_weighted_threshold(
    precisions: np.ndarray,
    recalls: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> np.ndarray:
    """
    Coût normalisé par seuil = cost_fp*(1-precision) + cost_fn*(1-recall).

    Un coût_fn élevé (FN coûteux) → seuil bas → sensibilité favorisée.
    Un coût_fp élevé (FP coûteux) → seuil haut → spécificité favorisée.
    """
    return cost_fp * (1.0 - precisions) + cost_fn * (1.0 - recalls)


def _as_binary_minority(y_true: np.ndarray) -> np.ndarray:
    values, counts = np.unique(y_true, return_counts=True)
    if len(values) != 2:
        return np.asarray([], dtype=int)
    minority_label = values[int(np.argmin(counts))]
    return (y_true == minority_label).astype(int)


def _as_binary_for_positive(y_true: np.ndarray, positive_label: Any) -> np.ndarray:
    """Map positive_label → 1, everything else → 0.

    When positive_label is None, falls back to the minority class so existing
    behaviour is preserved for callers that don't supply a positive label.
    """
    if positive_label is None:
        return _as_binary_minority(y_true)
    return (y_true == positive_label).astype(int)


class ThresholdOptimizer:
    def optimize(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        strategy: str,
        min_recall: float = 0.70,
        beta: float = 2.0,
        cost_fn: float = 1.0,
        cost_fp: float = 1.0,
        classes: np.ndarray | None = None,
        positive_label: Any = None,
    ) -> ThresholdResult:
        y_arr = np.asarray(y_true).reshape(-1)
        proba_arr = np.asarray(y_proba)

        # Determine which probability column corresponds to the positive class.
        # Priority: use positive_label (from resolved_positive_label at training time).
        # Fallback: use the minority class column (original behavior).
        # Both the proba column and y_bin must reference the same class so that
        # F1 optimisation is coherent — misaligning them would invert the threshold.
        selected_col = 1  # default
        effective_positive_label: Any = positive_label

        if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
            if classes is not None and len(classes) == 2:
                if positive_label is not None:
                    # Align with the explicitly provided positive class.
                    pos_str = str(positive_label)
                    for i, cls in enumerate(classes):
                        if str(cls) == pos_str:
                            selected_col = i
                            break
                elif y_arr.size > 0:
                    # Fall back: minority class (original behavior for callers that
                    # don't supply a positive_label, e.g. {0,1} standard encoding).
                    values, counts = np.unique(y_arr, return_counts=True)
                    if len(values) == 2:
                        minority_label = values[int(np.argmin(counts))]
                        try:
                            selected_col = list(classes).index(minority_label)
                        except ValueError:
                            selected_col = 1
                        effective_positive_label = classes[selected_col]
                else:
                    effective_positive_label = classes[selected_col]
            proba_arr = proba_arr[:, selected_col]
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

        # y_bin must be 1 wherever y_true == positive class — same class as proba_arr.
        y_bin = _as_binary_for_positive(y_arr, effective_positive_label)
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
        elif strategy_norm == "maximize_f_beta":
            fb_scores = _f_beta(precision, recall, beta=float(beta))
            selected_idx = int(np.nanargmax(fb_scores))
            strategy_used = "maximize_f_beta"
        elif strategy_norm == "minimize_cost":
            cost_scores = _cost_weighted_threshold(precision, recall, cost_fn=float(cost_fn), cost_fp=float(cost_fp))
            selected_idx = int(np.nanargmin(cost_scores))
            strategy_used = "minimize_cost"
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
        elif strategy_norm == "youden":
            # Youden Index J = sensitivity + specificity - 1 = TPR - FPR
            # Standard in medical diagnosis — maximises the distance from random guessing
            try:
                fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_bin, proba_arr)
                j_scores = tpr_arr - fpr_arr
                best_roc_idx = int(np.nanargmax(j_scores))
                optimal_threshold = float(roc_thresholds[best_roc_idx])
                # Recompute F1/precision/recall at this threshold for reporting
                y_pred_youden = (proba_arr >= optimal_threshold).astype(int)
                f1_youden = float(f1_score(y_bin, y_pred_youden, zero_division=0))
                prec_youden = float(precision[selected_idx]) if len(precision) > 0 else 0.0
                rec_youden = float(recall[selected_idx]) if len(recall) > 0 else 0.0
                # Align precision/recall to the found threshold via PR curve
                thresh_diffs = np.abs(thresholds - optimal_threshold)
                pr_idx = int(np.argmin(thresh_diffs))
                prec_youden = float(precision[pr_idx])
                rec_youden = float(recall[pr_idx])
                return ThresholdResult(
                    optimal_threshold=optimal_threshold,
                    strategy_used="youden",
                    f1_at_threshold=f1_youden,
                    precision_at_threshold=prec_youden,
                    recall_at_threshold=rec_youden,
                    default_threshold_f1=default_f1,
                    improvement_delta=float(f1_youden - default_f1),
                )
            except Exception:
                selected_idx = int(np.nanargmax(f1_scores))
                strategy_used = "maximize_f1"
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
