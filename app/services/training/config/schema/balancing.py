"""
BalancingConfig frozen dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.training.config.normalization import (
    _as_dict,
    _norm_choice,
    _to_bool,
    _to_optional_float,
)
from app.services.training.config.schema.types import (
    BALANCING_STRATEGIES,
    THRESHOLD_STRATEGIES,
    BalancingStrategy,
    ThresholdStrategy,
)


@dataclass(frozen=True)
class BalancingConfig:
    strategy: BalancingStrategy = "none"
    apply_threshold: bool = False
    threshold_strategy: ThresholdStrategy = "maximize_f1"
    min_recall_constraint: float | None = None
    f_beta: float = 2.0
    cost_fn: float = 1.0
    cost_fp: float = 1.0

    @staticmethod
    def from_front(raw: Any, *, legacy_use_smote: bool = False) -> "BalancingConfig":
        data = _as_dict(raw)
        legacy_default_strategy = "smote" if legacy_use_smote else "none"

        strategy = _norm_choice(
            data.get("strategy"),
            BALANCING_STRATEGIES,
            legacy_default_strategy,
        )
        apply_threshold = _to_bool(
            data.get("apply_threshold", data.get("applyThreshold")),
            default=False,
        )
        threshold_strategy = _norm_choice(
            data.get("threshold_strategy", data.get("thresholdStrategy")),
            THRESHOLD_STRATEGIES,
            "maximize_f1",
        )
        min_recall_constraint = _to_optional_float(
            data.get("min_recall_constraint", data.get("minRecallConstraint"))
        )
        if min_recall_constraint is not None and not (0.0 < min_recall_constraint < 1.0):
            min_recall_constraint = None

        f_beta_raw = _to_optional_float(data.get("f_beta", data.get("fBeta")))
        f_beta = max(0.1, min(10.0, float(f_beta_raw))) if f_beta_raw is not None else 2.0
        cost_fn_raw = _to_optional_float(data.get("cost_fn", data.get("costFn")))
        cost_fn = max(0.0, min(100.0, float(cost_fn_raw))) if cost_fn_raw is not None else 1.0
        cost_fp_raw = _to_optional_float(data.get("cost_fp", data.get("costFp")))
        cost_fp = max(0.0, min(100.0, float(cost_fp_raw))) if cost_fp_raw is not None else 1.0

        return BalancingConfig(
            strategy=strategy,  # type: ignore[arg-type]
            apply_threshold=apply_threshold,
            threshold_strategy=threshold_strategy,  # type: ignore[arg-type]
            min_recall_constraint=min_recall_constraint,
            f_beta=f_beta,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "apply_threshold": bool(self.apply_threshold),
            "threshold_strategy": str(self.threshold_strategy),
            "min_recall_constraint": self.min_recall_constraint,
            "f_beta": self.f_beta,
            "cost_fn": self.cost_fn,
            "cost_fp": self.cost_fp,
        }
