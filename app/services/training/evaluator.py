from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    classification_metrics,
    get_class_labels,
    get_proba_or_score,
    regression_metrics,
    safe_confusion_matrix,
)


@dataclass(frozen=True)
class DatasetEval:
    metrics: Dict[str, Any]
    predictions: np.ndarray
    confusion_matrix: Optional[list[Any]] = None


class Evaluator:
    def __init__(
        self,
        task_type: str,
        *,
        requested_metrics: Optional[Sequence[str]] = None,
        positive_label: Any = None,
    ):
        self.task_type = task_type
        self.requested_metrics = [str(m).strip().lower() for m in (requested_metrics or [])]
        self.positive_label = positive_label

    def evaluate(self, pipeline: Any, X: pd.DataFrame, y: np.ndarray) -> DatasetEval:
        y_pred = pipeline.predict(X)
        y_proba, y_score = get_proba_or_score(pipeline, X)
        labels = get_class_labels(pipeline)

        if self.task_type == "classification":
            metrics = classification_metrics(
                y,
                y_pred,
                y_proba=y_proba,
                y_score=y_score,
                labels=labels,
                estimator=pipeline,
                positive_label=self.positive_label,
                requested_metrics=self.requested_metrics,
                task_type=self.task_type,
            )
            cm_payload = metrics.get("confusion_matrix") if isinstance(metrics, dict) else {}
            cm = cm_payload.get("matrix") if isinstance(cm_payload, dict) else safe_confusion_matrix(y, y_pred)
            return DatasetEval(metrics=metrics, predictions=y_pred, confusion_matrix=cm)

        metrics = regression_metrics(y, y_pred)
        return DatasetEval(metrics=metrics, predictions=y_pred, confusion_matrix=None)
