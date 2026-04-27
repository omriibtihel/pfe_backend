"""
MetaLearner — Phase 11
Records each training outcome so the system can improve its
recommendations over time.

Storage: one JSON-lines file per project under
  <PROJECTS_PATH>/<project_id>/meta_learning.jsonl

Format is append-only; no external DB required.  Records are loaded
on demand and kept in memory only for the duration of a request.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mse", "log_loss", "brier_score"}


def _is_better_score(metric_name: str, challenger: float, champion: float) -> bool:
    if (metric_name or "").lower() in _LOWER_IS_BETTER_METRICS:
        return challenger < champion
    return challenger > champion


# ──────────────────────────────────────────────────────────────────────────────
# Record dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRecord:
    # Dataset characteristics
    dataset_meta_features: dict[str, Any]
    n_samples: int
    n_features: int
    task_type: str                     # "binary_classification" | "multiclass_classification" | "regression"
    dataset_size_category: str

    # Training outcome
    best_algorithm: str
    best_hyperparameters: dict[str, Any]
    best_score: float
    metric_used: str

    # Config details
    resampling_method: str | None
    split_method: str
    search_type: str
    config_mode: str                   # "manual" | "automl"
    user_overrides: dict[str, Any] | None

    # Timing
    training_time_seconds: float
    session_id: int
    project_id: int
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# MetaLearner
# ──────────────────────────────────────────────────────────────────────────────

class MetaLearner:
    """
    Persist and retrieve training records for meta-learning.

    Usage::

        ml = MetaLearner(projects_path="/path/to/projects")
        ml.record(project_id=1, record=TrainingRecord(...))
        history = ml.load_records(project_id=1)
    """

    def __init__(self, projects_path: str | os.PathLike) -> None:
        self._root = Path(projects_path)

    def _record_path(self, project_id: int) -> Path:
        return self._root / str(project_id) / "meta_learning.jsonl"

    def record(self, project_id: int, record: TrainingRecord) -> None:
        """Append a training record to the project's meta-learning log."""
        path = self._record_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        except OSError as exc:
            logger.warning("MetaLearner: could not write record: %s", exc)

    def load_records(self, project_id: int) -> list[TrainingRecord]:
        """Load all records for a project."""
        path = self._record_path(project_id)
        if not path.exists():
            return []
        records: list[TrainingRecord] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        records.append(TrainingRecord(**data))
                    except Exception as e:
                        logger.debug("MetaLearner: skipping malformed record: %s", e)
        except OSError as exc:
            logger.warning("MetaLearner: could not read records: %s", exc)
        return records

    def best_algorithm_for_profile(
        self,
        project_id: int,
        task_type: str,
        size_category: str,
        metric: str,
    ) -> str | None:
        """
        Return the best-performing algorithm seen for similar profiles.
        Returns None if no matching history found.
        """
        import math

        records = self.load_records(project_id)
        candidates = [
            r for r in records
            if r.task_type == task_type
            and r.dataset_size_category == size_category
            and r.metric_used == metric
            and isinstance(r.best_score, (int, float))
            and math.isfinite(r.best_score)
        ]
        if not candidates:
            return None

        best = candidates[0]
        for r in candidates[1:]:
            if _is_better_score(metric, r.best_score, best.best_score):
                best = r

        logger.info(
            "MetaLearner warm-start: suggest '%s' (score=%.4f) from %d matching records.",
            best.best_algorithm, best.best_score, len(candidates),
        )
        return best.best_algorithm


def build_training_record(
    *,
    session_id: int,
    project_id: int,
    profile_meta: dict[str, Any],
    best_model_type: str,
    best_hyperparams: dict[str, Any],
    best_score: float,
    metric_used: str,
    resampling_method: str | None,
    split_method: str,
    search_type: str,
    config_mode: str,
    user_overrides: dict[str, Any] | None,
    training_time_seconds: float,
) -> TrainingRecord:
    """Helper to construct a TrainingRecord from orchestrator outputs."""
    return TrainingRecord(
        dataset_meta_features=profile_meta,
        n_samples=int(profile_meta.get("n_samples", 0)),
        n_features=int(profile_meta.get("n_features", 0)),
        task_type=str(profile_meta.get("task_type", "unknown")),
        dataset_size_category=str(profile_meta.get("dataset_size_category", "unknown")),
        best_algorithm=best_model_type,
        best_hyperparameters=best_hyperparams,
        best_score=float(best_score),
        metric_used=metric_used,
        resampling_method=resampling_method,
        split_method=split_method,
        search_type=search_type,
        config_mode=config_mode,
        user_overrides=user_overrides,
        training_time_seconds=float(training_time_seconds),
        session_id=session_id,
        project_id=project_id,
    )
