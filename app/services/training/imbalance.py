from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np

IMBALANCE_THRESHOLD = 0.20  # minority ratio below this means imbalanced binary data

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


def is_binary(y: np.ndarray) -> bool:
    try:
        return len(np.unique(y)) == 2
    except Exception:
        return False


def class_counts(y: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def minority_ratio(y: np.ndarray) -> Optional[float]:
    try:
        _, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        if total <= 0:
            return None
        return float(counts.min() / total)
    except Exception:
        return None


def should_use_smote(y_train: np.ndarray) -> bool:
    if not is_binary(y_train):
        return False
    mr = minority_ratio(y_train)
    return mr is not None and mr < IMBALANCE_THRESHOLD


def build_smote_for_train(y_train: np.ndarray, random_state: int = 42) -> Tuple[Optional[Any], dict]:
    """
    Return (smote_obj, metadata).
    SMOTE is valid only if:
    - imblearn is installed
    - classification is binary
    - minority class has enough samples
    """
    if SMOTE is None:
        return None, {"enabled": False, "reason": "imblearn_not_installed"}

    if not is_binary(y_train):
        return None, {"enabled": False, "reason": "not_binary"}

    mr = minority_ratio(y_train)

    counts = list(class_counts(y_train).values())
    minority = min(counts) if counts else 0
    if minority < 6:
        # SMOTE needs at least k_neighbors + 1 minority points (k=5 by default).
        return None, {
            "enabled": False,
            "reason": "minority_too_small",
            "minority_count": int(minority),
            "minority_ratio": mr,
        }

    k = max(1, min(5, int(minority) - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote, {"enabled": True, "k_neighbors": k, "minority_ratio": mr}
