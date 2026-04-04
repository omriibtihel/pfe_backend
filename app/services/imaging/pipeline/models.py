"""
Registry de modèles de deep learning pour la classification d'images.

Supporte : ResNet18/50/101, EfficientNet-B0/B3, VGG16, DenseNet121.
Toutes les têtes de classification sont remplacées par nn.Linear(in_features, num_classes).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Réseau convolutif léger entraîné de zéro (sans poids ImageNet).
    4 blocs Conv → BN → ReLU → MaxPool, tête FC adaptative.
    Adapté aux petits datasets médicaux quand le transfer learning n'est pas souhaité.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1 — 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloc 2 — 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloc 3 — 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloc 4 — 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# (factory_fn, head_attr, in_features)
# simple_cnn : factory_fn=None — géré directement dans build_model
_REGISTRY: Dict[str, Tuple[Any, str, int]] = {
    "resnet18":        (tv_models.resnet18,        "fc",          512),
    "resnet50":        (tv_models.resnet50,         "fc",          2048),
    "resnet101":       (tv_models.resnet101,        "fc",          2048),
    "efficientnet_b0": (tv_models.efficientnet_b0,  "classifier",  1280),
    "efficientnet_b3": (tv_models.efficientnet_b3,  "classifier",  1536),
    "vgg16":           (tv_models.vgg16,            "classifier",  4096),
    "densenet121":     (tv_models.densenet121,      "classifier",  1024),
    "simple_cnn":      (None,                       "simple_cnn",  256),
}

SUPPORTED_MODELS = list(_REGISTRY.keys())


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Construit et retourne un modèle prêt à l'entraînement.

    La dernière couche de classification est remplacée par
    nn.Linear(in_features, num_classes).

    Parameters
    ----------
    model_name:
        Nom du modèle (voir SUPPORTED_MODELS).
    num_classes:
        Nombre de classes de sortie.
    pretrained:
        Si True, charge les poids ImageNet (IMAGENET1K_V1).
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Modèle inconnu : '{model_name}'. "
            f"Modèles supportés : {SUPPORTED_MODELS}"
        )

    factory_fn, head_attr, in_features = _REGISTRY[model_name]

    # SimpleCNN is built from scratch — no pretrained weights
    if head_attr == "simple_cnn":
        logger.debug("build_model: SimpleCNN, num_classes=%d", num_classes)
        return SimpleCNN(num_classes=num_classes)

    weights = "IMAGENET1K_V1" if pretrained else None
    model: nn.Module = factory_fn(weights=weights)

    new_head = nn.Linear(in_features, num_classes)

    if head_attr == "fc":
        model.fc = new_head
    elif head_attr == "classifier":
        # EfficientNet / DenseNet ont un Sequential ; VGG16 a Sequential[-1] = Linear
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = new_head
        else:
            model.classifier = new_head

    logger.debug(
        "build_model: %s, num_classes=%d, pretrained=%s",
        model_name, num_classes, pretrained,
    )
    return model


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Gèle tous les paramètres sauf la tête de classification.
    Utilisé pour le fine-tuning par transfer learning (phase 1).
    SimpleCNN n'a pas de backbone pré-entraîné — freeze ignoré.
    """
    _fn, head_attr, _in = _REGISTRY.get(model_name, (None, "fc", None))

    # SimpleCNN is trained from scratch — nothing to freeze
    if head_attr == "simple_cnn":
        logger.debug("freeze_backbone: %s — SimpleCNN, freeze ignoré", model_name)
        return

    for param in model.parameters():
        param.requires_grad = False

    if head_attr and hasattr(model, head_attr):
        head = getattr(model, head_attr)
        for param in head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug("freeze_backbone: %s — %d paramètres entraînables (tête seule)", model_name, trainable)


def unfreeze_all(model: nn.Module) -> None:
    """Dégèle tous les paramètres (fine-tuning complet)."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug("unfreeze_all: %d paramètres entraînables", trainable)


def get_device() -> torch.device:
    """Retourne CUDA si disponible, sinon CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
