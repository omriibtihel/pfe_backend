"""
Persistance des modèles imagerie :
- Sauvegarde des poids PyTorch (.pt)
- Insertion en base de ImagingTrainedModel
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from sqlalchemy.orm import Session

from app.models.imaging import ImagingTrainedModel
from app.services.training.utils import sanitize_json_payload

logger = logging.getLogger(__name__)


def save_imaging_weights(
    model: nn.Module,
    out_dir: Path,
    model_name: str,
) -> Path:
    """
    Sauvegarde les poids du modèle dans out_dir/{model_name}.pt.

    Retourne le chemin absolu du fichier sauvegardé.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), path)
    logger.info("Poids sauvegardés : %s", path)
    return path


def persist_imaging_model(
    db: Session,
    *,
    session_id: int,
    project_id: int,
    model_name: str,
    task_type: str,
    metrics_json: Dict[str, Any],
    artifacts_json: Dict[str, Any],
) -> ImagingTrainedModel:
    """Insère un ImagingTrainedModel en base de données."""
    obj = ImagingTrainedModel(
        session_id=session_id,
        project_id=project_id,
        model_name=model_name,
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
    logger.info(
        "ImagingTrainedModel persisted: id=%d session=%d model=%s",
        obj.id, session_id, model_name,
    )
    return obj
