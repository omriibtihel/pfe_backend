"""
ImageFolderDataset — torch.utils.data.Dataset pour la classification d'images.

Structure attendue du dossier :
    images_dir/
        class_a/   *.jpg  *.jpeg  *.png  *.bmp  *.tiff  *.tif
        class_b/   ...
        ...

Les sous-dossiers sont triés alphabétiquement pour garantir un encodage
label ↔ index déterministe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ImageFolderDataset:
    """
    Dataset PyTorch-compatible (implémente __len__ et __getitem__).

    Parameters
    ----------
    images_dir:
        Chemin absolu vers le dossier racine contenant les sous-dossiers de classes.
    class_names:
        Liste des noms de classes à inclure. Si None, toutes les classes
        détectées dans images_dir sont utilisées (tri alphabétique).
    transform:
        Pipeline de transformation appliqué à chaque image (PIL → Tensor).
    indices:
        Si fourni, seuls les échantillons aux index donnés sont inclus.
        Utilisé pour créer train/val/test splits sans duplication des données.
    """

    def __init__(
        self,
        images_dir: str,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Découvrir les classes
        if class_names:
            self.class_names = sorted(class_names)
        else:
            self.class_names = self._discover_classes()

        if not self.class_names:
            raise ValueError(
                f"Aucune sous-classe détectée dans '{images_dir}'. "
                "Vérifiez que le dossier contient des sous-dossiers de classes."
            )

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Construire la liste de tous les (chemin, label) pairs
        self._all_samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.images_dir / cls
            if not cls_dir.is_dir():
                logger.warning("Classe '%s' introuvable dans '%s' — ignorée.", cls, images_dir)
                continue
            label = self.class_to_idx[cls]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    self._all_samples.append((p, label))

        if not self._all_samples:
            raise ValueError(
                f"Aucune image trouvée dans '{images_dir}' pour les classes {self.class_names}."
            )

        # Appliquer les indices si fournis
        if indices is not None:
            self._samples = [self._all_samples[i] for i in indices]
        else:
            self._samples = list(self._all_samples)

        logger.debug(
            "ImageFolderDataset: %d images, %d classes, transform=%s",
            len(self._samples),
            len(self.class_names),
            type(self.transform).__name__ if self.transform else "None",
        )

    # ── Méthodes statiques ────────────────────────────────────────────────────

    @staticmethod
    def discover_classes(images_dir: str) -> List[str]:
        """Retourne la liste triée des classes (sous-dossiers) dans images_dir."""
        p = Path(images_dir)
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    @staticmethod
    def count_per_class(images_dir: str) -> dict:
        """Retourne {class_name: nb_images} pour chaque sous-dossier."""
        result = {}
        p = Path(images_dir)
        if not p.exists():
            return result
        for d in sorted(p.iterdir()):
            if d.is_dir():
                count = sum(
                    1 for f in d.iterdir()
                    if f.suffix.lower() in _SUPPORTED_EXTENSIONS
                )
                result[d.name] = count
        return result

    # ── Interface Dataset ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Impossible de lire '%s': %s — image noire utilisée.", path, exc)
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform is not None:
            # albumentations Compose expose une liste `.transforms` et attend un ndarray
            if hasattr(self.transform, "transforms"):
                try:
                    tensor = self.transform(image=np.array(img))["image"]
                except (TypeError, KeyError):
                    tensor = self.transform(img)
            else:
                tensor = self.transform(img)
        else:
            tensor = TF.to_tensor(img)

        return tensor, label

    # ── Helpers privés ────────────────────────────────────────────────────────

    def _discover_classes(self) -> List[str]:
        if not self.images_dir.exists():
            return []
        return sorted([d.name for d in self.images_dir.iterdir() if d.is_dir()])


def make_splits(
    dataset: ImageFolderDataset,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple["ImageFolderDataset", "ImageFolderDataset", "ImageFolderDataset"]:
    """
    Découpe le dataset en train / val / test de façon stratifiée (par classe).

    Retourne trois ImageFolderDataset partageant les mêmes données source
    mais avec des indices disjoints.
    """
    rng = np.random.default_rng(seed)
    all_samples = dataset._all_samples

    # Grouper par classe
    class_indices: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(all_samples):
        class_indices.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []

    for label, idxs in class_indices.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n = len(arr)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        test_idx.extend(arr[:n_test].tolist())
        val_idx.extend(arr[n_test: n_test + n_val].tolist())
        train_idx.extend(arr[n_test + n_val:].tolist())

    def _subset(indices):
        s = ImageFolderDataset.__new__(ImageFolderDataset)
        s.images_dir = dataset.images_dir
        s.class_names = dataset.class_names
        s.class_to_idx = dataset.class_to_idx
        s._all_samples = dataset._all_samples
        s._samples = [dataset._all_samples[i] for i in indices]
        s.transform = dataset.transform
        return s

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)
