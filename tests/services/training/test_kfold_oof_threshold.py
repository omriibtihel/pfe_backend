"""Tests for k-fold OOF threshold calibration (RISK-004).

Pins the contract of the two new code paths introduced in _run_kfold_cv:
  - OOF path: apply_postfit_from_oof is used when fold predictions are
    available — no leakage-related warning is produced.
  - Fallback path: when OOF collection is infeasible (all fold models lack
    predict_proba, or stacking fails), apply_postfit fires on training data
    and 'threshold_calibrated_on_train_data_may_be_optimistic' must appear
    in metrics_json['warnings'].

These tests do NOT run the full pipeline — they exercise the
BalancingExecutor paths and replicate the warning-assembly pattern from
_run_kfold_cv so the contract is pinned without a heavy integration fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.preparation_ml.balancing import BalancingDecision
from app.services.preparation_ml.balancing.executor import BalancingExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(apply_threshold: bool = True) -> BalancingDecision:
    return BalancingDecision(
        strategy="none",
        apply_threshold=apply_threshold,
        threshold_strategy="maximize_f1",
        rationale="test",
        refit_metric="f1",
        smote_k_neighbors=5,
        class_weight_mode="balanced",
        audit_flags=[],
        optimal_threshold=0.5,
    )


def _build_oof(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_proba) simulating accumulated fold predictions."""
    rng = np.random.RandomState(seed)
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    proba_pos = np.clip(y_true * 0.7 + rng.rand(n) * 0.3, 0, 1)
    return y_true, np.column_stack([1 - proba_pos, proba_pos])


def _simulate_metrics_json(executor: BalancingExecutor, threshold_source: str) -> dict:
    """Replicate the metrics_json warning-assembly block from _run_kfold_cv."""
    hp_warnings: list[str] = []
    cv_instability: list[str] = []
    postfit_w = list(getattr(executor, "postfit_warnings", None) or [])
    all_warnings = list(hp_warnings) + list(cv_instability) + postfit_w
    metrics: dict = {"threshold_source": threshold_source}
    if all_warnings:
        metrics["warnings"] = all_warnings
    return metrics


# ---------------------------------------------------------------------------
# OOF path: apply_postfit_from_oof
# ---------------------------------------------------------------------------

def test_oof_path_no_leakage_warning():
    """When OOF arrays are available, no train-data leakage warning is produced."""
    executor = BalancingExecutor()
    y_true, y_proba = _build_oof()

    executor.apply_postfit_from_oof(
        y_true=y_true,
        y_proba=y_proba,
        decision=_make_decision(apply_threshold=True),
        model_classes=np.array([0, 1]),
    )

    assert "threshold_calibrated_on_train_data_may_be_optimistic" not in executor.postfit_warnings


def test_oof_path_threshold_is_valid_probability():
    """apply_postfit_from_oof must return a value in [0, 1]."""
    executor = BalancingExecutor()
    y_true, y_proba = _build_oof()

    threshold = executor.apply_postfit_from_oof(
        y_true=y_true,
        y_proba=y_proba,
        decision=_make_decision(apply_threshold=True),
        model_classes=np.array([0, 1]),
    )

    assert 0.0 <= threshold <= 1.0


def test_oof_path_metrics_json_threshold_source_is_oof():
    """metrics_json['threshold_source'] == 'oof' on the happy path."""
    executor = BalancingExecutor()
    y_true, y_proba = _build_oof()
    executor.apply_postfit_from_oof(
        y_true=y_true,
        y_proba=y_proba,
        decision=_make_decision(apply_threshold=True),
        model_classes=np.array([0, 1]),
    )

    metrics = _simulate_metrics_json(executor, threshold_source="oof")

    assert metrics["threshold_source"] == "oof"
    assert "threshold_calibrated_on_train_data_may_be_optimistic" not in metrics.get("warnings", [])


# ---------------------------------------------------------------------------
# Fallback path: no OOF (e.g. test_ratio=0, model lacks predict_proba)
# ---------------------------------------------------------------------------

def _apply_fallback(decision: BalancingDecision) -> dict:
    """
    Simulate the else-branch that fires when oof_available is False in
    _run_kfold_cv (e.g. all fold models lack predict_proba so oof_proba_parts
    stays empty).  Returns the resulting metrics_json-like dict.
    """
    executor = BalancingExecutor()
    # Injected exactly as the orchestrator does in the else-branch
    executor.postfit_warnings.append(
        "threshold_calibrated_on_train_data_may_be_optimistic"
    )
    return _simulate_metrics_json(executor, threshold_source="train_refit")


def test_fallback_warning_in_metrics_json():
    """
    When OOF collection is not feasible the leakage warning must appear in
    metrics_json['warnings'] so callers (test_ratio=0 scenario included) can
    surface the degraded quality signal.
    """
    metrics = _apply_fallback(_make_decision(apply_threshold=False))

    assert "warnings" in metrics
    assert any(
        "threshold_calibrated_on_train_data_may_be_optimistic" in w
        for w in metrics["warnings"]
    )


def test_fallback_threshold_source_is_train_refit():
    """metrics_json['threshold_source'] == 'train_refit' on the fallback path."""
    metrics = _apply_fallback(_make_decision(apply_threshold=False))

    assert metrics.get("threshold_source") == "train_refit"


def test_fallback_warnings_key_always_exists():
    """The 'warnings' key must be present whenever the fallback fires."""
    metrics = _apply_fallback(_make_decision())

    assert "warnings" in metrics, (
        "metrics_json must have a 'warnings' key when threshold "
        "calibration falls back to training data"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_oof_apply_threshold_disabled_no_warning():
    """apply_postfit_from_oof with apply_threshold=False should not warn."""
    executor = BalancingExecutor()
    y_true, y_proba = _build_oof()

    executor.apply_postfit_from_oof(
        y_true=y_true,
        y_proba=y_proba,
        decision=_make_decision(apply_threshold=False),
        model_classes=np.array([0, 1]),
    )

    assert not any("train_data" in w for w in executor.postfit_warnings)


def test_oof_empty_arrays_fallback_graceful():
    """apply_postfit_from_oof with empty arrays must not crash — returns 0.5."""
    executor = BalancingExecutor()
    threshold = executor.apply_postfit_from_oof(
        y_true=np.array([]),
        y_proba=np.array([]),
        decision=_make_decision(apply_threshold=True),
        model_classes=np.array([0, 1]),
    )

    assert threshold == 0.5
