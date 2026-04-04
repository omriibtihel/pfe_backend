"""
ImagingTrainer — boucle d'entraînement par epoch pour un seul modèle.

Émet des événements SSE via epoch_callback après chaque epoch.
Sauvegarde le meilleur checkpoint (val_loss minimal) dans un fichier temporaire.
"""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.services.imaging.config.schema import ImagingConfig
from app.services.imaging.pipeline.models import freeze_backbone, unfreeze_all

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    duration_s: float


@dataclass
class TrainingResult:
    model_name: str
    task_type: str
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    epoch_results: List[EpochResult] = field(default_factory=list)
    best_checkpoint_path: Optional[str] = None
    training_time_s: float = 0.0

    def epoch_curves(self) -> List[Dict[str, Any]]:
        return [
            {
                "epoch": r.epoch,
                "trainLoss": round(r.train_loss, 6),
                "valLoss": round(r.val_loss, 6),
                "valAcc": round(r.val_acc, 6),
            }
            for r in self.epoch_results
        ]


class ImagingTrainer:
    """
    Boucle d'entraînement pour un modèle PyTorch.

    Parameters
    ----------
    model:
        Modèle nn.Module déjà construit (tête remplacée).
    train_loader / val_loader:
        DataLoaders pour l'entraînement et la validation.
    cfg:
        ImagingConfig (epochs, lr, weight_decay, freeze_backbone, etc.).
    model_name:
        Nom du modèle (pour freeze_backbone / logs).
    device:
        Périphérique de calcul.
    epoch_callback:
        Optionnel — appelé après chaque epoch avec
        (epoch, total_epochs, train_loss, val_loss, val_acc).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ImagingConfig,
        model_name: str,
        device: torch.device,
        epoch_callback: Optional[Callable] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.model_name = model_name
        self.device = device
        self.epoch_callback = epoch_callback

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.learning_rate * 1e-2,
        )

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="imaging_ckpt_"))

    # ── Public ─────────────────────────────────────────────────────────────────

    def train(self) -> TrainingResult:
        """Lance la boucle d'entraînement complète."""
        t_start = time.monotonic()

        if self.cfg.freeze_backbone:
            freeze_backbone(self.model, self.model_name)
            # Recréer l'optimiseur pour ne voir que les params entraînables
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )

        best_val_loss = float("inf")
        best_epoch = 0
        best_val_acc = 0.0
        best_ckpt = self._tmp_dir / f"{self.model_name}_best.pt"
        epoch_results: List[EpochResult] = []

        _skip_scheduler_step = False

        for epoch in range(1, self.cfg.epochs + 1):
            # Dégeler le backbone après N epochs
            if self.cfg.freeze_backbone and epoch == self.cfg.unfreeze_after_epoch + 1:
                unfreeze_all(self.model)
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.cfg.learning_rate * 0.1,  # LR réduit pour le fine-tuning global
                    weight_decay=self.cfg.weight_decay,
                )
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, self.cfg.epochs - epoch),
                    eta_min=self.cfg.learning_rate * 1e-3,
                )
                # Skip scheduler.step() on the epoch the scheduler is recreated;
                # optimizer.step() inside _train_one_epoch will prime it first.
                _skip_scheduler_step = True
                logger.info("ImagingTrainer[%s]: backbone dégel à epoch %d", self.model_name, epoch)

            t_epoch = time.monotonic()
            train_loss = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            if _skip_scheduler_step:
                _skip_scheduler_step = False
            else:
                self.scheduler.step()
            duration = time.monotonic() - t_epoch

            er = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                duration_s=round(duration, 2),
            )
            epoch_results.append(er)

            # Sauvegarder le meilleur checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_ckpt)
                logger.debug(
                    "ImagingTrainer[%s]: nouveau meilleur checkpoint epoch=%d val_loss=%.4f",
                    self.model_name, epoch, val_loss,
                )

            logger.info(
                "ImagingTrainer[%s] epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                self.model_name, epoch, self.cfg.epochs,
                train_loss, val_loss, val_acc,
            )

            if self.epoch_callback:
                try:
                    self.epoch_callback(
                        epoch=epoch,
                        total_epochs=self.cfg.epochs,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )
                except Exception:
                    logger.warning("epoch_callback a échoué silencieusement.", exc_info=True)

        # Recharger le meilleur checkpoint dans le modèle
        if best_ckpt.exists():
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.device, weights_only=True)
            )

        return TrainingResult(
            model_name=self.model_name,
            task_type=self.cfg.task_type,
            best_epoch=best_epoch,
            best_val_loss=round(best_val_loss, 6),
            best_val_acc=round(best_val_acc, 6),
            epoch_results=epoch_results,
            best_checkpoint_path=str(best_ckpt),
            training_time_s=round(time.monotonic() - t_start, 2),
        )

    # ── Boucles internes ───────────────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        return total_loss / max(n, 1)

    def _validate(self) -> tuple[float, float]:
        """Retourne (val_loss, val_accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                n += images.size(0)
        return total_loss / max(n, 1), correct / max(n, 1)
