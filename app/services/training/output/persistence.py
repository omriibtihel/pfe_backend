from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from sqlalchemy.orm import Session

from app.models.training import TrainedModel
from app.services.training.utils import sanitize_json_payload


def save_pipeline(pipeline: Any, out_dir: Path, model_type: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model_type}.pkl"
    joblib.dump(pipeline, p)
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
