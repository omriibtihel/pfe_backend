from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.models.training import TrainedModel
from app.services.training.utils import sanitize_json_payload

logger = logging.getLogger(__name__)


class ThresholdedClassifier:
    """Wraps a fitted sklearn-compatible binary classifier and overrides the
    decision boundary used by ``predict()``.  ``predict_proba`` is unchanged.

    Clinical rationale: displaying evaluation metrics at a custom threshold
    (e.g. 0.35 to maximise sensitivity for a rare disease) while the saved
    pipeline still predicts at sklearn's default 0.5 would produce results
    that differ from the metrics shown to clinicians.  Embedding the threshold
    here ensures that inference, evaluation, and reported metrics all refer to
    the exact same decision boundary.

    The wrapper returns the *original* class labels (e.g. "malade"/"sain",
    1/2, …) — never coerced to 0/1 — by reading ``estimator.classes_`` and
    using the recorded ``positive_label`` or ``positive_class_index`` to pick
    the positive column from ``predict_proba``. If anything is incompatible
    (no ``predict_proba``, multiclass, missing classes), the wrapper falls
    back transparently to ``estimator.predict(X)``.
    """

    def __init__(
        self,
        estimator: Any,
        threshold: float = 0.5,
        *,
        positive_class_index: Optional[int] = None,
        positive_label: Any = None,
    ) -> None:
        self.estimator = estimator
        self.threshold = threshold
        self.positive_class_index = positive_class_index
        self.positive_label = positive_label

    def _resolve_classes(self) -> Optional[np.ndarray]:
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            named = getattr(self.estimator, "named_steps", None)
            model = None
            if named is not None:
                try:
                    model = named.get("model")
                except Exception:
                    model = None
            if model is not None:
                classes = getattr(model, "classes_", None)
        if classes is None:
            try:
                classes = self.estimator.model.estimator.classes_
            except AttributeError:
                pass
        return np.asarray(classes) if classes is not None else None

    def _resolve_positive_index(self, classes: np.ndarray) -> int:
        # Prefer label match (robust to reordering of classes_)
        if self.positive_label is not None:
            try:
                return int(list(classes).index(self.positive_label))
            except ValueError:
                pass
            try:
                return int([str(c) for c in classes].index(str(self.positive_label)))
            except ValueError:
                pass
        if (
            self.positive_class_index is not None
            and 0 <= int(self.positive_class_index) < len(classes)
        ):
            return int(self.positive_class_index)
        # Fallback: sklearn binary convention (positive = column 1)
        return len(classes) - 1

    def predict(self, X: Any) -> np.ndarray:
        try:
            proba = self.estimator.predict_proba(X)
        except Exception as exc:
            logger.warning(
                "ThresholdedClassifier: predict_proba failed (%s); "
                "falling back to estimator.predict",
                exc,
            )
            return np.asarray(self.estimator.predict(X))

        if proba is None or proba.ndim != 2 or proba.shape[1] != 2:
            return np.asarray(self.estimator.predict(X))

        classes = self._resolve_classes()
        if classes is None or len(classes) != 2:
            return np.asarray(self.estimator.predict(X))

        pos_idx = self._resolve_positive_index(classes)
        neg_idx = 1 - pos_idx if pos_idx in (0, 1) else 0
        pos_cls = classes[pos_idx]
        neg_cls = classes[neg_idx]
        score = proba[:, pos_idx]
        return np.asarray(np.where(score >= self.threshold, pos_cls, neg_cls))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def __getattr__(self, name: str) -> Any:
        # Transparent delegation for classes_, feature_names_in_, etc.
        # Guard against unpickling / recursion before __init__ has run.
        if name == "estimator":
            raise AttributeError(name)
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


def save_pipeline(
    pipeline: Any,
    out_dir: Path,
    model_type: str,
    *,
    threshold: Optional[float] = None,
    positive_class_index: Optional[int] = None,
    positive_label: Any = None,
) -> Path:
    """Serialise *pipeline* to ``out_dir/<model_type>.pkl``.

    When *threshold* is provided and differs from 0.5 (binary classification),
    the pipeline is wrapped in a :class:`ThresholdedClassifier` before saving
    so that ``predict()`` at inference time uses the same decision boundary
    that was optimised during training.

    ``positive_class_index`` / ``positive_label`` (only one is needed) tell
    the wrapper which column of ``predict_proba`` carries the positive class,
    so the threshold is applied to the correct probability and ``predict()``
    returns the original class labels (e.g. "malade"/"sain", 1/2, …) rather
    than 0/1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model_type}.pkl"

    if threshold is not None and abs(threshold - 0.5) > 1e-6:
        pipeline_to_save = ThresholdedClassifier(
            pipeline,
            threshold,
            positive_class_index=positive_class_index,
            positive_label=positive_label,
        )
        logger.info(
            "Saving pipeline with embedded threshold %.4f (pos_idx=%s, pos_label=%s)",
            threshold,
            positive_class_index,
            positive_label,
        )
    else:
        pipeline_to_save = pipeline
        logger.debug("Saving pipeline with default threshold 0.5")

    joblib.dump(pipeline_to_save, p)
    return p


def load_pipeline(path: Path | str) -> Any:
    return joblib.load(Path(path))


def persist_trained_model(
    db: Session,
    *,
    session_id: int,
    project_id: int,
    model_type: str,
    task_type: str,
    metrics_json: Dict[str, Any],
    artifacts_json: Dict[str, Any],
) -> TrainedModel:
    obj = TrainedModel(
        session_id=session_id,
        project_id=project_id,
        model_type=model_type,
        task_type=task_type,
        metrics_json=sanitize_json_payload(metrics_json),
        artifacts_json=sanitize_json_payload(artifacts_json),
    )
    db.add(obj)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(obj)
    return obj
