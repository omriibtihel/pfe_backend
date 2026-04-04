"""
Couche d'accès aux données pour les entités imaging (ImagingSession, ImagingTrainedModel).
Aucune logique métier ici — uniquement des requêtes et mutations DB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.imaging import ImagingSession, ImagingTrainedModel


# ──────────────────────────────────────────────────────────────────────────────
# ImagingSession
# ──────────────────────────────────────────────────────────────────────────────

def create_session(
    db: Session,
    *,
    project_id: int,
    config_json: Dict[str, Any],
) -> ImagingSession:
    s = ImagingSession(
        project_id=project_id,
        status="queued",
        progress=0,
        config_json=config_json,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(
    db: Session,
    session_id: int,
    project_id: int,
) -> Optional[ImagingSession]:
    return (
        db.query(ImagingSession)
        .filter(
            ImagingSession.id == session_id,
            ImagingSession.project_id == project_id,
        )
        .first()
    )


def list_sessions(
    db: Session,
    project_id: int,
) -> List[ImagingSession]:
    return (
        db.query(ImagingSession)
        .filter(ImagingSession.project_id == project_id)
        .order_by(ImagingSession.id.desc())
        .all()
    )


def delete_session_and_files(
    db: Session,
    session: ImagingSession,
) -> None:
    """Supprime les fichiers .pt associés, puis la session en DB."""
    for m in session.models:
        try:
            arts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
            pt = arts.get("model_pt")
            if pt:
                p = Path(pt)
                if p.exists():
                    p.unlink()
        except Exception:
            pass
    db.delete(session)
    db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# ImagingTrainedModel
# ──────────────────────────────────────────────────────────────────────────────

def get_model(
    db: Session,
    model_id: int,
    *,
    session_id: int,
    project_id: int,
) -> Optional[ImagingTrainedModel]:
    return (
        db.query(ImagingTrainedModel)
        .filter(
            ImagingTrainedModel.id == model_id,
            ImagingTrainedModel.session_id == session_id,
            ImagingTrainedModel.project_id == project_id,
        )
        .first()
    )


def list_saved_models(
    db: Session,
    project_id: int,
) -> List[ImagingTrainedModel]:
    """Retourne tous les modèles sauvegardés (is_saved=True) d'un projet."""
    return (
        db.query(ImagingTrainedModel)
        .filter(
            ImagingTrainedModel.project_id == project_id,
            ImagingTrainedModel.is_saved == True,  # noqa: E712
        )
        .order_by(ImagingTrainedModel.id.desc())
        .all()
    )


def get_model_by_id(
    db: Session,
    model_id: int,
    project_id: int,
) -> Optional[ImagingTrainedModel]:
    """Récupère un modèle par son id sans filtrer par session."""
    return (
        db.query(ImagingTrainedModel)
        .filter(
            ImagingTrainedModel.id == model_id,
            ImagingTrainedModel.project_id == project_id,
        )
        .first()
    )


def list_models_for_session(
    db: Session,
    session_id: int,
) -> List[ImagingTrainedModel]:
    return (
        db.query(ImagingTrainedModel)
        .filter(ImagingTrainedModel.session_id == session_id)
        .order_by(ImagingTrainedModel.id.asc())
        .all()
    )


def save_model(
    db: Session,
    model: ImagingTrainedModel,
) -> None:
    model.is_saved = True
    artifacts = dict(model.artifacts_json) if isinstance(model.artifacts_json, dict) else {}
    artifacts["saved"] = True
    model.artifacts_json = artifacts
    db.add(model)
    db.commit()


def unsave_model(
    db: Session,
    model: ImagingTrainedModel,
) -> None:
    model.is_saved = False
    artifacts = dict(model.artifacts_json) if isinstance(model.artifacts_json, dict) else {}
    artifacts["saved"] = False
    model.artifacts_json = artifacts
    db.add(model)
    db.commit()
