from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core.config import PROJECTS_PATH
from app.models.training import TrainingSession

from app.services.training.config import TrainingConfig
from app.services.training.dataset_loader import resolve_dataset_path, load_dataframe
from app.services.training.orchestrator import run_one_model
from app.services.training.persistence import save_pipeline, persist_trained_model

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _log_event(event: str, **payload: Any) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


def _update_session(db: Session, s: TrainingSession, **fields):
    for k, v in fields.items():
        setattr(s, k, v)
    db.add(s)
    db.commit()
    db.refresh(s)


def _append_session_message(db: Session, s: TrainingSession, msg: str):
    current = (s.error_message or "").strip()
    msg = msg.strip()
    if not current:
        new_msg = msg
    else:
        new_msg = current if msg in current else (current + "\n" + msg)
    _update_session(db, s, error_message=new_msg)


def run_training_session(session_id: int) -> None:
    """
    Worker appelé par ta route (background task).
    Version PRO (refactor) : split->preprocess->smote(train only)->fit->eval->persist.
    Pour l’instant: orchestrator = HOLDOUT (kfold arrive juste après).
    """
    db = SessionLocal()
    try:
        s: TrainingSession | None = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not s:
            return

        _log_event("training.session.start", session_id=session_id)
        _update_session(db, s, status="running", progress=3, started_at=_now())

        cfg_raw: Dict[str, Any] = s.config_json or {}
        cfg = TrainingConfig.from_front(cfg_raw)

        if not cfg.target_column:
            raise RuntimeError("targetColumn is required")
        if not cfg.models:
            raise RuntimeError("No model selected")

        dataset_path, dv_id = resolve_dataset_path(db, s.project_id, s.dataset_version_id)
        df = load_dataframe(dataset_path)

        out_dir = PROJECTS_PATH / str(s.project_id) / "training_models" / str(session_id)

        total = max(1, len(cfg.models))
        success_count = 0

        _update_session(db, s, dataset_version_id=dv_id, progress=10)

        for i, model_type in enumerate(cfg.models, start=1):
            _update_session(db, s, progress=min(95, int(10 + (i - 1) * (80 / total))))
            _log_event("training.session.model.start", session_id=session_id, model_type=model_type, index=i, total=total)

            try:
                res = run_one_model(df, cfg, model_type=model_type)

                # save artifact
                pkl_path = save_pipeline(res.fitted_pipeline, out_dir, model_type)
                res.artifacts_json["model_pkl"] = str(pkl_path)
                res.artifacts_json["dataset_version_id"] = dv_id

                # persist trained model
                persist_trained_model(
                    db,
                    session_id=session_id,
                    project_id=s.project_id,
                    model_type=str(model_type),
                    task_type=str(res.task_type),
                    metrics_json=res.metrics_json,
                    artifacts_json=res.artifacts_json,
                )

                success_count += 1
                _log_event("training.session.model.success", session_id=session_id, model_type=model_type)

            except Exception as e:
                _append_session_message(db, s, f"[{model_type}] {str(e)}")
                _log_event(
                    "training.session.model.error",
                    session_id=session_id,
                    model_type=model_type,
                    reason=str(e),
                )

        final_status = "succeeded" if success_count > 0 else "failed"
        _update_session(db, s, status=final_status, progress=100, finished_at=_now())
        _log_event("training.session.end", session_id=session_id, status=final_status, success_count=success_count)

    except Exception as e:
        s2 = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if s2:
            _update_session(
                db,
                s2,
                status="failed",
                progress=min(int(getattr(s2, "progress", 0) or 0), 99),
                error_message=str(e),
                finished_at=_now(),
            )
        _log_event("training.session.fatal_error", session_id=session_id, reason=str(e))
    finally:
        db.close()
