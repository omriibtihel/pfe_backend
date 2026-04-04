"""
Inférence sur une image unique avec un modèle ImagingTrainedModel sauvegardé.

Le modèle chargé est mis en cache en mémoire (clé = chemin .pt) pour éviter
de recharger les poids à chaque requête.
"""
from __future__ import annotations

import io
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.imaging.config.schema import AugmentationConfig
from app.services.imaging.data.transforms import build_val_transforms, uses_albumentations
from app.services.imaging.pipeline.models import build_model

logger = logging.getLogger(__name__)

# ── Cache LRU simple (max 3 modèles en mémoire) ────────────────────────────────
_MAX_CACHE = 3
# {model_pt_path: (model, device)}
_MODEL_CACHE: OrderedDict[str, Tuple[Any, Any]] = OrderedDict()


def _get_or_load_model(
    model_pt_path: str,
    model_name: str,
    num_classes: int,
):
    """Retourne le modèle depuis le cache ou le charge depuis le disque."""
    if model_pt_path in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(model_pt_path)  # LRU refresh
        model, device = _MODEL_CACHE[model_pt_path]
        logger.debug("predictor: cache hit — %s", model_pt_path)
        return model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("predictor: chargement du modèle %s depuis %s", model_name, model_pt_path)

    model = build_model(model_name, num_classes, pretrained=False)
    state_dict = torch.load(model_pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if len(_MODEL_CACHE) >= _MAX_CACHE:
        evicted_key, _ = _MODEL_CACHE.popitem(last=False)
        logger.debug("predictor: éviction cache — %s", evicted_key)

    _MODEL_CACHE[model_pt_path] = (model, device)
    return model, device


def predict_image(
    model_pt_path: str,
    model_name: str,
    class_names: List[str],
    image_size: int,
    image_bytes: bytes,
) -> Dict[str, Any]:
    """
    Prétraite l'image et retourne les probabilités par classe.

    Returns
    -------
    {
        "predicted_class": str,
        "confidence": float,
        "probabilities": {class: float, ...}   # triées par proba décroissante
    }
    """
    num_classes = len(class_names)
    model, device = _get_or_load_model(model_pt_path, model_name, num_classes)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    aug_cfg = AugmentationConfig()
    transform = build_val_transforms(aug_cfg, image_size)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if uses_albumentations():
        tensor = transform(image=np.array(pil_image))["image"]
    else:
        tensor = transform(pil_image)

    tensor = tensor.unsqueeze(0).to(device)

    # ── Inférence ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()

    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    probabilities = {class_names[i]: round(float(probs[i]), 6) for i in range(len(class_names))}

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": round(float(probs[pred_idx]), 6),
        "probabilities": dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)),
    }
