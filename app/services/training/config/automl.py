from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class AutoMLConfig:
    """Configuration for an AutoML run (FLAML-based, full-pipeline)."""

    dataset_version_id: int
    target_column: str
    task_type: Literal["classification", "regression"]
    time_budget: int = 60       # seconds (10–3600)
    metric: Optional[str] = None  # None = auto-select per task type
    test_ratio: float = 0.2
    positive_label: Optional[str] = None
    f_beta: float = 1.0   # F-beta for threshold optimisation (1.0=F1, 2.0=F2/recall-heavy)
    random_state: int = 42

    @classmethod
    def from_front(cls, payload: dict) -> "AutoMLConfig":
        raw_metric = payload.get("metric") or None
        # Normalize empty string to None
        if isinstance(raw_metric, str) and not raw_metric.strip():
            raw_metric = None
        return cls(
            dataset_version_id=int(payload["datasetVersionId"]),
            target_column=str(payload["targetColumn"]).strip(),
            task_type=payload.get("taskType", "classification"),
            time_budget=int(payload.get("timeBudget", 60)),
            metric=raw_metric,
            test_ratio=float(payload.get("testRatio", 0.2)),
            positive_label=payload.get("positiveLabel") or None,
            f_beta=float(payload.get("fBeta", payload.get("f_beta", 1.0))),
            random_state=int(payload.get("randomState", payload.get("random_state", 42))),
        )
