"""
Binary / label-set utilities shared across metric modules.

Single source of truth for:
  - _is_zero_one_label_set   (was duplicated in orchestrator.py and metrics.py)
  - _detect_classification_type
  - is_binary
  - get_proba_or_score
  - get_class_labels
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from sklearn.utils.multiclass import type_of_target

from app.services.training.utils import to_python_scalar


def _is_zero_one_label_set(labels: Sequence[Any]) -> bool:
    try:
        normalized = {float(to_python_scalar(v)) for v in labels}
    except Exception:
        return False
    return normalized == {0.0, 1.0}


def _detect_classification_type(y_true: np.ndarray) -> str:
    target_kind = type_of_target(y_true)
    if target_kind in {"multilabel-indicator", "multiclass-multioutput"}:
        return "multilabel"
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        return "multilabel"
    n_classes = int(np.unique(y_true).size)
    return "binary" if n_classes == 2 else "multiclass"


def is_binary(y: np.ndarray) -> bool:
    try:
        return len(np.unique(y)) == 2
    except Exception:
        return False


def get_proba_or_score(model: Any, X: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y_proba = None
    y_score = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
    except Exception:
        y_proba = None
    try:
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
    except Exception:
        y_score = None
    return y_proba, y_score


def get_class_labels(model: Any) -> Optional[list[Any]]:
    def _classes_from_estimator(estimator: Any) -> Any:
        classes_local = getattr(estimator, "classes_", None)
        if classes_local is not None:
            return classes_local
        named_steps = getattr(estimator, "named_steps", None)
        if isinstance(named_steps, dict):
            inner = named_steps.get("model")
            classes_local = getattr(inner, "classes_", None)
            if classes_local is not None:
                return classes_local
        return None

    candidate = model
    for _ in range(3):
        classes = _classes_from_estimator(candidate)
        if classes is not None:
            return [to_python_scalar(v) for v in np.asarray(classes).ravel().tolist()]
        next_estimator = getattr(candidate, "best_estimator_", None)
        if next_estimator is None:
            break
        candidate = next_estimator
    return None
