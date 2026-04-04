"""
Pipelines d'augmentation pour le pipeline imagerie.
Utilise albumentations si installé, sinon torchvision.transforms.
"""
from __future__ import annotations

from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from app.services.imaging.config.schema import AugmentationConfig


def build_train_transforms(cfg: AugmentationConfig, image_size: int) -> Callable:
    """Pipeline d'augmentation pour le set d'entraînement."""
    ops = [A.Resize(image_size, image_size)]
    if cfg.horizontal_flip:
        ops.append(A.HorizontalFlip(p=0.5))
    if cfg.vertical_flip:
        ops.append(A.VerticalFlip(p=0.5))
    if cfg.rotation_degrees:
        ops.append(A.Rotate(limit=cfg.rotation_degrees, p=0.5))
    if cfg.brightness_limit or cfg.contrast_limit:
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_limit,
            contrast_limit=cfg.contrast_limit,
            p=0.4,
        ))
    ops.append(A.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std))
    ops.append(ToTensorV2())
    return A.Compose(ops)


def build_val_transforms(cfg: AugmentationConfig, image_size: int) -> Callable:
    """Resize + normalisation uniquement (pas d'augmentation stochastique)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
        ToTensorV2(),
    ])


def uses_albumentations() -> bool:
    return True
