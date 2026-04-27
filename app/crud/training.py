# app/crud/training.py
"""
Data-access layer for training entities (TrainingSession, TrainedModel, DatasetVersion).
No HTTP imports, no business logic — only DB queries and mutations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.dataset_version import DatasetVersion
from app.models.project import Project
from app.models.training import TrainedModel, TrainingSession


# ──────────────────────────────────────────────────────────────────────────────
# TrainingSession
# ──────────────────────────────────────────────────────────────────────────────

def get_session(
    db: Session, session_id: int, project_id: int
) -> Optional[TrainingSession]:
    return (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )


def list_sessions_for_project(
    db: Session, project_id: int
) -> list[TrainingSession]:
    return (
        db.query(TrainingSession)
        .filter(TrainingSession.project_id == project_id)
        .order_by(TrainingSession.id.desc())
        .all()
    )


def create_session(
    db: Session,
    *,
    project_id: int,
    version_id: int,
    config_json: dict[str, Any],
) -> TrainingSession:
    s = TrainingSession(
        project_id=project_id,
        dataset_version_id=version_id,
        status="queued",
        progress=0,
        config_json=config_json,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def rename_session(db: Session, session: TrainingSession, name: str) -> None:
    config = dict(session.config_json) if isinstance(session.config_json, dict) else {}
    config["name"] = name
    session.config_json = config
    db.commit()


def delete_session_and_files(db: Session, session: TrainingSession) -> None:
    """Delete all associated .pkl files, then remove the session from the DB."""
    for m in session.models:
        try:
            arts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
            pkl = arts.get("model_pkl")
            if pkl:
                p = Path(pkl)
                if p.exists():
                    p.unlink()
        except Exception:
            pass
    db.delete(session)
    db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# TrainedModel
# ──────────────────────────────────────────────────────────────────────────────

def get_model(
    db: Session,
    model_id: int,
    *,
    session_id: int,
    project_id: int,
) -> Optional[TrainedModel]:
    return (
        db.query(TrainedModel)
        .filter(
            TrainedModel.id == model_id,
            TrainedModel.session_id == session_id,
            TrainedModel.project_id == project_id,
        )
        .first()
    )


def list_models_for_session(
    db: Session, session_id: int
) -> list[TrainedModel]:
    return (
        db.query(TrainedModel)
        .filter(TrainedModel.session_id == session_id)
        .order_by(TrainedModel.id.asc())
        .all()
    )


def save_model(
    db: Session,
    model: TrainedModel,
    project: Project,
) -> int | None:
    """Mark model as saved, set it as project's active model. Returns previous active_model_id."""
    previous_active_model_id = project.active_model_id

    model.is_saved = True
    db.add(model)

    project.active_model_id = model.id
    db.add(project)
    db.commit()

    return previous_active_model_id


def unsave_model(
    db: Session,
    model: TrainedModel,
    project: Project,
) -> None:
    model.is_saved = False
    db.add(model)

    if project.active_model_id is not None and int(project.active_model_id) == int(model.id):
        project.active_model_id = None
        db.add(project)

    db.commit()


def delete_model_and_file(
    db: Session,
    model: TrainedModel,
    project: Project,
) -> None:
    """Unset active model if needed, delete pkl from disk, delete DB record."""
    if project.active_model_id is not None and int(project.active_model_id) == int(model.id):
        project.active_model_id = None
        db.add(project)

    artifacts = model.artifacts_json if isinstance(model.artifacts_json, dict) else {}
    pkl_path_str = artifacts.get("model_pkl")
    if pkl_path_str:
        try:
            pkl_path = Path(str(pkl_path_str))
            if pkl_path.exists():
                pkl_path.unlink()
        except Exception:
            pass

    db.delete(model)
    db.commit()


def get_saved_models_with_context(
    db: Session,
    project_id: int,
    active_id: int | None,
) -> tuple[list[TrainedModel], dict[int, TrainingSession], dict[int, str]]:
    """
    Returns (models, sessions_by_id, version_name_by_id).

    Includes both explicitly saved models (is_saved=True) and the active model,
    plus a backward-compat pass for pre-migration rows saved only via artifacts_json.
    """
    query = db.query(TrainedModel).filter(TrainedModel.project_id == project_id)
    if active_id is not None:
        query = query.filter(
            or_(TrainedModel.is_saved == True, TrainedModel.id == active_id)  # noqa: E712
        )
    else:
        query = query.filter(TrainedModel.is_saved == True)  # noqa: E712

    models: list[TrainedModel] = query.order_by(
        TrainedModel.created_at.desc(), TrainedModel.id.desc()
    ).all()

    # Backward-compat: include models saved only in artifacts_json (pre-migration rows).
    legacy_ids = {int(m.id) for m in models}
    legacy_candidates = (
        db.query(TrainedModel)
        .filter(TrainedModel.project_id == project_id, TrainedModel.is_saved == False)  # noqa: E712
        .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
        .all()
    )
    for lm in legacy_candidates:
        arts = lm.artifacts_json if isinstance(lm.artifacts_json, dict) else {}
        if bool(arts.get("saved", False)) and int(lm.id) not in legacy_ids:
            models = list(models) + [lm]

    if not models:
        return [], {}, {}

    session_ids = list({int(m.session_id) for m in models})
    sessions = (
        db.query(TrainingSession).filter(TrainingSession.id.in_(session_ids)).all()
        if session_ids else []
    )
    sessions_by_id = {int(s.id): s for s in sessions}

    version_ids = list(
        {int(s.dataset_version_id) for s in sessions if getattr(s, "dataset_version_id", None) is not None}
    )
    versions = (
        db.query(DatasetVersion).filter(DatasetVersion.id.in_(version_ids)).all()
        if version_ids else []
    )
    version_name_by_id = {int(v.id): str(v.name) for v in versions}

    return models, sessions_by_id, version_name_by_id


# ──────────────────────────────────────────────────────────────────────────────
# DatasetVersion
# ──────────────────────────────────────────────────────────────────────────────

def get_version(
    db: Session, version_id: int, project_id: int
) -> Optional[DatasetVersion]:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
