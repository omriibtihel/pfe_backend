from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from .resolver import BalancingDecision
from .threshold import ThresholdOptimizer, ThresholdResult

logger = logging.getLogger(__name__)


class BalancingExecutor:
    def __init__(self) -> None:
        self.last_threshold_result: ThresholdResult | None = None
        self.postfit_warnings: list[str] = []

    def apply_prefit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        decision: BalancingDecision,
    ) -> tuple[Any, np.ndarray, dict[str, Any]]:
        strategy = str(decision.strategy or "none").strip().lower()
        y_arr = np.asarray(y_train)

        if strategy in {"none", "threshold_only", "threshold_optimization"}:
            return X_train, y_arr, {}

        if strategy == "class_weight":
            return X_train, y_arr, {"class_weight": str(decision.class_weight_mode or "balanced")}

        if strategy == "sample_weight":
            weights = compute_sample_weight(class_weight=str(decision.class_weight_mode or "balanced"), y=y_arr)
            return X_train, y_arr, {"sample_weight": np.asarray(weights)}

        if strategy == "smote":
            try:
                from imblearn.over_sampling import SMOTE
            except ImportError as exc:
                raise RuntimeError("imbalanced-learn is required. pip install imbalanced-learn") from exc
            sampler = SMOTE(random_state=decision.random_state, k_neighbors=int(decision.smote_k_neighbors or 5))
            X_res, y_res = sampler.fit_resample(X_train, y_arr)
            return X_res, np.asarray(y_res), {}

        if strategy == "smote_tomek":
            try:
                from imblearn.combine import SMOTETomek
                from imblearn.over_sampling import SMOTE
            except ImportError as exc:
                raise RuntimeError("imbalanced-learn is required. pip install imbalanced-learn") from exc
            smote = SMOTE(random_state=decision.random_state, k_neighbors=int(decision.smote_k_neighbors or 5))
            sampler = SMOTETomek(random_state=decision.random_state, smote=smote)
            X_res, y_res = sampler.fit_resample(X_train, y_arr)
            return X_res, np.asarray(y_res), {}

        if strategy == "random_undersampling":
            try:
                from imblearn.under_sampling import RandomUnderSampler
            except ImportError as exc:
                raise RuntimeError("imbalanced-learn is required. pip install imbalanced-learn") from exc
            sampler = RandomUnderSampler(random_state=decision.random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_arr)
            return X_res, np.asarray(y_res), {}

        raise RuntimeError(f"Unknown balancing strategy '{strategy}'")

    def apply_postfit(
        self,
        model: Any,
        X_val: Any,
        y_val: np.ndarray | None,
        decision: BalancingDecision,
    ) -> float:
        self.postfit_warnings = []
        self.last_threshold_result = None

        if not bool(decision.apply_threshold):
            self.last_threshold_result = ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="disabled",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )
            return 0.5

        y_arr = np.asarray(y_val) if y_val is not None else np.asarray([])
        if y_arr.size == 0 or X_val is None:
            self.postfit_warnings.append("threshold_optimization_skipped_no_validation_data")
            self.last_threshold_result = ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="maximize_f1",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )
            return 0.5

        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            warning = "threshold_optimization_skipped_predict_proba_not_available"
            self.postfit_warnings.append(warning)
            logger.warning(warning)
            self.last_threshold_result = ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="predict_proba_unavailable",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )
            return 0.5

        try:
            y_proba = predict_proba(X_val)
            optimizer = ThresholdOptimizer()
            result = optimizer.optimize(
                y_true=y_arr,
                y_proba=np.asarray(y_proba),
                strategy=str(decision.threshold_strategy or "maximize_f1"),
                min_recall=(
                    float(decision.min_recall_constraint)
                    if decision.min_recall_constraint is not None
                    else 0.70
                ),
                beta=2.0,
            )
            self.last_threshold_result = result
            return float(result.optimal_threshold)
        except Exception as exc:
            warning = f"threshold_optimization_failed_{exc.__class__.__name__.lower()}"
            self.postfit_warnings.append(warning)
            logger.warning("Threshold optimization failed: %s", exc)
            self.last_threshold_result = ThresholdResult(
                optimal_threshold=0.5,
                strategy_used="maximize_f1",
                f1_at_threshold=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                default_threshold_f1=0.0,
                improvement_delta=0.0,
            )
            return 0.5
