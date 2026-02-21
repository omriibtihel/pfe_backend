from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.training.splitters import make_holdout_split


def _binary_dataset(n_majority: int, n_minority: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_majority + [1] * n_minority, dtype=int)
    rng.shuffle(y)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=len(y)),
            "f2": rng.normal(size=len(y)),
        }
    )
    return X, y


def test_holdout_binary_split_keeps_both_classes_in_train_val_test() -> None:
    X, y = _binary_dataset(n_majority=380, n_minority=20, seed=2026)  # 95/5

    split = make_holdout_split(
        X,
        y,
        task_type="classification",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert len(np.unique(split.y_train)) == 2
    assert split.y_val is not None
    assert len(np.unique(split.y_val)) == 2
    assert len(np.unique(split.y_test)) == 2


def test_holdout_binary_can_fallback_to_no_validation_when_needed() -> None:
    X, y = _binary_dataset(n_majority=58, n_minority=2, seed=7)

    split = make_holdout_split(
        X,
        y,
        task_type="classification",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert split.y_val is None
    assert len(np.unique(split.y_train)) == 2
    assert len(np.unique(split.y_test)) == 2
    assert any("falling back to train/test split" in msg for msg in split.warnings)
