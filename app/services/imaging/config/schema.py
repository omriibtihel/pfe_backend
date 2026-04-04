"""
ImagingConfig — configuration interne du pipeline imagerie.
Mirrors TrainingConfig.from_front() from app/services/training/config/schema.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ImagingConfig:
    task_type: str                      # "image_classification"
    model_names: List[str]              # ["resnet50", "efficientnet_b0"]
    image_size: int = 224
    pretrained: bool = True
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    freeze_backbone: bool = True
    unfreeze_after_epoch: int = 5
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 0                # 0 = main thread (safe on Windows)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_classes: int = 2                # auto-détecté depuis le dossier images
    class_names: List[str] = field(default_factory=list)
    images_dir: str = ""                # /storage/projects/{id}/images/

    @classmethod
    def from_front(cls, raw: Dict[str, Any]) -> "ImagingConfig":
        """Désérialise depuis le payload camelCase du frontend."""
        aug_raw = raw.get("augmentation", {})
        aug = AugmentationConfig(
            horizontal_flip=bool(aug_raw.get("horizontalFlip", True)),
            vertical_flip=bool(aug_raw.get("verticalFlip", False)),
            rotation_degrees=int(aug_raw.get("rotationDegrees", 15)),
            brightness_limit=float(aug_raw.get("brightnessLimit", 0.2)),
            contrast_limit=float(aug_raw.get("contrastLimit", 0.2)),
            normalize_mean=aug_raw.get("normalizeMean", [0.485, 0.456, 0.406]),
            normalize_std=aug_raw.get("normalizeStd", [0.229, 0.224, 0.225]),
        )
        return cls(
            task_type=raw.get("taskType", "image_classification"),
            model_names=raw.get("models", []),
            image_size=int(raw.get("imageSize", 224)),
            pretrained=bool(raw.get("pretrained", True)),
            epochs=int(raw.get("epochs", 20)),
            batch_size=int(raw.get("batchSize", 32)),
            learning_rate=float(raw.get("learningRate", 1e-4)),
            weight_decay=float(raw.get("weightDecay", 1e-5)),
            freeze_backbone=bool(raw.get("freezeBackbone", True)),
            unfreeze_after_epoch=int(raw.get("unfreezeAfterEpoch", 5)),
            val_split=float(raw.get("valSplit", 0.2)),
            test_split=float(raw.get("testSplit", 0.1)),
            num_workers=int(raw.get("numWorkers", 0)),
            augmentation=aug,
            num_classes=int(raw.get("numClasses", 2)),
            class_names=raw.get("classNames", []),
            images_dir=raw.get("imagesDir", ""),
        )
