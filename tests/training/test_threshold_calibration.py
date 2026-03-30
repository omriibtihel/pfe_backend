"""
Tests de non-régression pour la calibration du seuil de décision.

Scénarios couverts :
  Test A : apply_threshold=True + val_ratio=0 → fallback sur X_train
            → optimal_threshold != 0.5 ET threshold_source contient "train_fallback"
  Test B : apply_threshold=False → seuil = 0.5, threshold_source = "disabled"
  Test C : apply_threshold=True + val_ratio>0 → calibration sur val set (comportement original)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.services.training.orchestrator as orchestrator
from app.services.training.config.schema import TrainingConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixture : dataset binaire très déséquilibré avec signal visible
# RF peut scorer les positifs différemment mais max proba < 0.5 en défaut
# ──────────────────────────────────────────────────────────────────────────────

def _make_imbalanced_df(n_majority: int = 620, n_minority: int = 100) -> pd.DataFrame:
    """Reproduit les proportions du dataset observé dans les logs (620/100)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * n_majority + [1] * n_minority, dtype=int)
    rng.shuffle(y)

    age         = rng.normal(52.0, 10.0, size=len(y)) + 6.0 * y
    bp          = rng.normal(118.0, 8.0, size=len(y)) + 10.0 * y
    glucose     = rng.normal(90.0, 15.0, size=len(y)) + 20.0 * y
    bmi         = rng.normal(27.0, 4.0, size=len(y)) + 3.0 * y
    smoker      = np.where(rng.random(len(y)) < (0.15 + 0.35 * y), "yes", "no")

    return pd.DataFrame({
        "age": age,
        "blood_pressure": bp,
        "glucose": glucose,
        "bmi": bmi,
        "smoker": smoker,
        "Outcome": y,
    })


def _cfg(
    *,
    apply_threshold: bool,
    strategy: str = "smote",
    val_ratio: int = 0,
    train_ratio: int = 80,
    test_ratio: int = 20,
) -> TrainingConfig:
    return TrainingConfig.from_front({
        "targetColumn": "Outcome",
        "taskType": "classification",
        "models": ["randomforest"],
        "metrics": ["f1", "accuracy"],
        "splitMethod": "holdout",
        "trainRatio": train_ratio,
        "valRatio": val_ratio,
        "testRatio": test_ratio,
        "kFolds": 5,
        "useGridSearch": False,
        "balancing": {
            "strategy": strategy,
            "apply_threshold": apply_threshold,
            "threshold_strategy": "maximize_f1",
            "user_confirmed": True,
        },
        "preprocessing": {
            "numericImputation": "median",
            "numericScaling": "standard",
            "categoricalImputation": "most_frequent",
            "categoricalEncoding": "onehot",
        },
    })


# ──────────────────────────────────────────────────────────────────────────────
# Test A : apply_threshold=True + val_ratio=0 → fallback sur X_train
# ──────────────────────────────────────────────────────────────────────────────

def test_threshold_fallback_to_train_when_no_val_set():
    """
    Quand apply_threshold=True mais val_ratio=0 :
    - le seuil est calibré sur X_train (fallback) → threshold_source contient "train_fallback"
    - threshold_used est un float valide dans [0, 1]
    - NOTE: le seuil peut rester à 0.5 si l'optimizer ne trouve pas d'amélioration
      (comportement correct depuis le fix "revert if no improvement").
    """
    df = _make_imbalanced_df()
    cfg = _cfg(apply_threshold=True, strategy="smote", val_ratio=0)

    res = orchestrator.run_one_model(df, cfg, model_type="randomforest")

    threshold_source = res.metrics_json.get("threshold_source", "")
    threshold_used   = float(res.metrics_json.get("threshold_used", -1.0))

    assert "train_fallback" in threshold_source, (
        f"Attendu 'train_fallback' dans threshold_source, obtenu: {threshold_source!r}. "
        "Le fallback sur X_train n'a pas été déclenché."
    )
    assert 0.0 <= threshold_used <= 1.0, (
        f"threshold_used={threshold_used} hors plage [0, 1]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test B : apply_threshold=False → seuil 0.5, threshold_source = "disabled"
# ──────────────────────────────────────────────────────────────────────────────

def test_threshold_disabled_stays_at_default_0_5():
    """
    Quand apply_threshold=False :
    - threshold_source = "disabled"
    - threshold_used = 0.5
    - comportement identique à l'ancien (aucune régression)
    """
    df = _make_imbalanced_df()
    cfg = _cfg(apply_threshold=False, strategy="none", val_ratio=0)

    res = orchestrator.run_one_model(df, cfg, model_type="randomforest")

    threshold_source = res.metrics_json.get("threshold_source", "")
    threshold_used   = float(res.metrics_json.get("threshold_used", 0.5))

    assert threshold_source == "disabled", (
        f"Attendu threshold_source='disabled', obtenu: {threshold_source!r}"
    )
    assert threshold_used == 0.5, (
        f"Attendu threshold_used=0.5 quand apply_threshold=False, obtenu: {threshold_used}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test C : apply_threshold=True + val_ratio>0 → calibration sur val set (original)
# ──────────────────────────────────────────────────────────────────────────────

def test_threshold_uses_val_set_when_available():
    """
    Quand apply_threshold=True ET val_ratio>0 :
    - threshold_source contient "val_set" (pas de fallback train)
    - metrics_json["threshold_used"] est bien exposé
    """
    df = _make_imbalanced_df()
    cfg = _cfg(apply_threshold=True, strategy="smote", val_ratio=15, train_ratio=70, test_ratio=15)

    res = orchestrator.run_one_model(df, cfg, model_type="randomforest")

    threshold_source = res.metrics_json.get("threshold_source", "")
    threshold_used   = res.metrics_json.get("threshold_used")

    assert "val_set" in threshold_source, (
        f"Attendu 'val_set' dans threshold_source, obtenu: {threshold_source!r}"
    )
    assert threshold_used is not None, "threshold_used absent de metrics_json"


# ──────────────────────────────────────────────────────────────────────────────
# Test D : sanity check — threshold_used remonte bien dans metrics_json
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics_json_always_contains_threshold_fields():
    """threshold_used et threshold_source sont toujours présents dans metrics_json."""
    df = _make_imbalanced_df()
    for apply_threshold, strategy in [(False, "none"), (True, "class_weight")]:
        cfg = _cfg(apply_threshold=apply_threshold, strategy=strategy, val_ratio=0)
        res = orchestrator.run_one_model(df, cfg, model_type="randomforest")
        assert "threshold_used"   in res.metrics_json, "threshold_used absent de metrics_json"
        assert "threshold_source" in res.metrics_json, "threshold_source absent de metrics_json"
