from __future__ import annotations

import numpy as np

import app.services.training.imbalance as imbalance


def test_build_smote_for_train_does_not_skip_balanced_binary_when_called(monkeypatch):
    class _FakeSmote:
        def __init__(self, random_state: int, k_neighbors: int):
            self.random_state = random_state
            self.k_neighbors = k_neighbors

    monkeypatch.setattr(imbalance, "SMOTE", _FakeSmote)
    y_train = np.array([0] * 40 + [1] * 40, dtype=int)

    smote_obj, meta = imbalance.build_smote_for_train(y_train, random_state=123)

    assert smote_obj is not None
    assert meta.get("enabled") is True
    assert meta.get("k_neighbors") == 5
    assert meta.get("minority_ratio") == 0.5
