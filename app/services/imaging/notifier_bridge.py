"""
Pont vers le training_notifier existant pour les événements SSE du pipeline imagerie.

Les événements imagerie utilisent des types distincts (préfixe "imaging.")
mais transitent par le même singleton TrainingNotifier.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.training.notifier import training_notifier

logger = logging.getLogger(__name__)


class ImagingEventType:
    STARTED        = "imaging.started"
    EPOCH_COMPLETE = "imaging.epoch.complete"
    MODEL_COMPLETE = "imaging.model.complete"
    FINAL_COMPLETE = "imaging.final.complete"
    ERROR          = "imaging.error"
    PROGRESS       = "imaging.progress"


def emit_started(session_id: int) -> None:
    training_notifier.emit_sync(
        session_id,
        ImagingEventType.STARTED,
        {"sessionId": session_id},
    )


def emit_epoch(
    session_id: int,
    *,
    epoch: int,
    total_epochs: int,
    model_name: str,
    model_index: int,
    total_models: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
) -> None:
    """
    Calcule la progression globale et émet un événement EPOCH_COMPLETE.

    Progression :
    - 10 % de démarrage
    - 80 % distribués sur tous les modèles × toutes les epochs
    - 10 % réservés pour la fin
    """
    model_fraction = (model_index - 1) / max(total_models, 1)
    epoch_fraction = epoch / max(total_epochs, 1) / max(total_models, 1)
    progress = int(10 + 80 * (model_fraction + epoch_fraction))

    training_notifier.emit_sync(
        session_id,
        ImagingEventType.EPOCH_COMPLETE,
        {
            "modelName":   model_name,
            "modelIndex":  model_index,
            "totalModels": total_models,
            "epoch":       epoch,
            "totalEpochs": total_epochs,
            "trainLoss":   round(train_loss, 6),
            "valLoss":     round(val_loss, 6),
            "valAcc":      round(val_acc, 6),
            "progress":    min(progress, 89),
        },
    )


def emit_model_complete(
    session_id: int,
    *,
    model_name: str,
    model_index: int,
    total_models: int,
    metrics: Dict[str, Any],
) -> None:
    progress = int(10 + 80 * model_index / max(total_models, 1))
    training_notifier.emit_sync(
        session_id,
        ImagingEventType.MODEL_COMPLETE,
        {
            "modelName":   model_name,
            "modelIndex":  model_index,
            "totalModels": total_models,
            "metrics":     metrics,
            "progress":    min(progress, 95),
        },
    )


def emit_final_complete(session_id: int, *, results: Optional[list] = None) -> None:
    training_notifier.emit_sync(
        session_id,
        ImagingEventType.FINAL_COMPLETE,
        {"sessionId": session_id, "results": results or [], "progress": 100},
    )


def emit_error(session_id: int, *, message: str) -> None:
    training_notifier.emit_sync(
        session_id,
        ImagingEventType.ERROR,
        {"sessionId": session_id, "message": message},
    )
