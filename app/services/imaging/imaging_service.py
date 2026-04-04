"""
run_imaging_session — background task du pipeline imagerie.

Mirrors run_training_session() dans training_service.py.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core.config import PROJECTS_PATH
from app.models.imaging import ImagingSession

from app.services.imaging.config.schema import ImagingConfig
from app.services.imaging.data.dataset import ImageFolderDataset, make_splits
from app.services.imaging.data.transforms import build_train_transforms, build_val_transforms
from app.services.imaging.pipeline.models import build_model, get_device
from app.services.imaging.pipeline.trainer import ImagingTrainer
from app.services.imaging.pipeline.evaluator import ImagingEvaluator
from app.services.imaging.output.persistence import save_imaging_weights, persist_imaging_model
from app.services.imaging.notifier_bridge import (
    emit_started, emit_epoch, emit_model_complete,
    emit_final_complete, emit_error,
)

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _update_session(db: Session, s: ImagingSession, **fields):
    for k, v in fields.items():
        setattr(s, k, v)
    db.add(s)
    db.commit()
    db.refresh(s)


def run_imaging_session(session_id: int) -> None:
    """
    Background task — appelé par la route POST /imaging/sessions.

    Flow :
      1. Charge la session depuis la DB
      2. Émet STARTED
      3. Désérialise ImagingConfig depuis config_json
      4. Construit le dataset et les splits
      5. Pour chaque modèle :
           a. Entraîne avec ImagingTrainer (callbacks SSE par epoch)
           b. Évalue sur le test split
           c. Sauvegarde les poids .pt
           d. Persiste ImagingTrainedModel en DB
           e. Émet MODEL_COMPLETE
      6. Met à jour le statut de la session
      7. Émet FINAL_COMPLETE
    """
    t_start = time.monotonic()
    db = SessionLocal()

    try:
        s: ImagingSession | None = (
            db.query(ImagingSession)
            .filter(ImagingSession.id == session_id)
            .first()
        )
        if not s:
            logger.error("ImagingSession introuvable : id=%d", session_id)
            return

        _update_session(db, s, status="running", progress=3, started_at=_now())
        emit_started(session_id)

        cfg_raw: Dict[str, Any] = s.config_json or {}
        cfg = ImagingConfig.from_front(cfg_raw)

        # DataLoader workers with multiprocessing spawn on Windows cause
        # spurious KeyboardInterrupt tracebacks during worker cleanup.
        # Force num_workers=0 (main-thread loading) on Windows.
        if sys.platform == "win32":
            cfg.num_workers = 0

        _evaluator = ImagingEvaluator()

        # ── Résoudre le dossier images ────────────────────────────────────────
        images_dir = cfg.images_dir
        if not images_dir:
            images_dir = str(PROJECTS_PATH / str(s.project_id) / "images")
            cfg.images_dir = images_dir

        # ── Auto-détecter classes ─────────────────────────────────────────────
        discovered = ImageFolderDataset.discover_classes(images_dir)
        if not discovered:
            msg = f"Aucune classe trouvée dans '{images_dir}'."
            _update_session(db, s, status="failed", error_message=msg, finished_at=_now())
            emit_error(session_id, message=msg)
            return

        if not cfg.class_names:
            cfg.class_names = discovered
        cfg.num_classes = len(cfg.class_names)

        # ── Dataset complet (sans transform, pour splitter) ───────────────────
        full_dataset = ImageFolderDataset(images_dir, class_names=cfg.class_names)
        train_ds_base, val_ds_base, test_ds_base = make_splits(
            full_dataset,
            val_split=cfg.val_split,
            test_split=cfg.test_split,
        )

        device = get_device()
        logger.info(
            "run_imaging_session: session=%d models=%s device=%s classes=%s n=%d",
            session_id, cfg.model_names, device, cfg.class_names, len(full_dataset),
        )

        total_models = len(cfg.model_names)
        out_dir = PROJECTS_PATH / str(s.project_id) / "imaging_models" / str(session_id)
        results_summary = []

        for model_index, model_name in enumerate(cfg.model_names, start=1):
            _update_session(db, s,
                current_model=f"{model_name} ({model_index}/{total_models})",
                progress=int(5 + 85 * (model_index - 1) / total_models),
            )

            try:
                # Transformer les splits avec les bons pipelines
                train_transform = build_train_transforms(cfg.augmentation, cfg.image_size)
                val_transform = build_val_transforms(cfg.augmentation, cfg.image_size)

                train_ds_base.transform = train_transform
                val_ds_base.transform = val_transform
                test_ds_base.transform = val_transform

                train_loader = DataLoader(
                    train_ds_base,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                val_loader = DataLoader(
                    val_ds_base,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                )
                test_loader = DataLoader(
                    test_ds_base,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                )

                model = build_model(model_name, cfg.num_classes, cfg.pretrained)

                def _epoch_cb(epoch, total_epochs, train_loss, val_loss, val_acc):
                    emit_epoch(
                        session_id,
                        epoch=epoch,
                        total_epochs=total_epochs,
                        model_name=model_name,
                        model_index=model_index,
                        total_models=total_models,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )

                trainer = ImagingTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=cfg,
                    model_name=model_name,
                    device=device,
                    epoch_callback=_epoch_cb,
                )

                training_result = trainer.train()

                # ── Évaluation sur test ───────────────────────────────────────
                test_metrics = _evaluator.evaluate_classification(
                    model, test_loader, cfg.class_names, device,
                )

                # ── Sauvegarder les poids ─────────────────────────────────────
                pt_path = save_imaging_weights(model, out_dir, model_name)

                # ── Construire metrics_json et artifacts_json ─────────────────
                metrics_json = {
                    **test_metrics,
                    "best_epoch":     training_result.best_epoch,
                    "best_val_loss":  training_result.best_val_loss,
                    "best_val_acc":   training_result.best_val_acc,
                    "epoch_curves":   training_result.epoch_curves(),
                    "training_time_s": training_result.training_time_s,
                }

                artifacts_json = {
                    "model_pt":    str(pt_path),
                    "class_names": cfg.class_names,
                    "num_classes": cfg.num_classes,
                    "image_size":  cfg.image_size,
                    "pretrained":  cfg.pretrained,
                    "model_name":  model_name,
                }

                # ── Persister en DB ───────────────────────────────────────────
                persist_imaging_model(
                    db,
                    session_id=session_id,
                    project_id=s.project_id,
                    model_name=model_name,
                    task_type=cfg.task_type,
                    metrics_json=metrics_json,
                    artifacts_json=artifacts_json,
                )

                results_summary.append({
                    "modelName": model_name,
                    "accuracy": test_metrics.get("accuracy"),
                    "f1Macro": test_metrics.get("f1_macro"),
                })

                emit_model_complete(
                    session_id,
                    model_name=model_name,
                    model_index=model_index,
                    total_models=total_models,
                    metrics={"accuracy": test_metrics.get("accuracy"), "f1Macro": test_metrics.get("f1_macro")},
                )

                logger.info(
                    "run_imaging_session: model=%s done accuracy=%.4f",
                    model_name, test_metrics.get("accuracy", 0),
                )

            except Exception as exc:
                logger.exception("run_imaging_session: erreur sur modèle %s", model_name)
                _update_session(db, s,
                    error_message=f"{model_name}: {exc}",
                )
                # Continuer avec les autres modèles

        # ── Finalisation ──────────────────────────────────────────────────────
        elapsed = round(time.monotonic() - t_start, 2)
        _update_session(db, s,
            status="succeeded",
            progress=100,
            current_model=None,
            finished_at=_now(),
        )
        emit_final_complete(session_id, results=results_summary)
        logger.info(
            "run_imaging_session: session=%d terminée en %.1f s",
            session_id, elapsed,
        )

    except Exception as exc:
        logger.exception("run_imaging_session: erreur fatale session=%d", session_id)
        try:
            _update_session(db, s, status="failed", error_message=str(exc), finished_at=_now())
        except Exception:
            pass
        emit_error(session_id, message=str(exc))
    finally:
        db.close()
