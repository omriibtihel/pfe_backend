"""
ImbalanceHandler — Phase 5
Thin wrapper that derives an actionable imbalance recommendation from a
DatasetProfile.  For binary classification it also delegates to the
existing balancing/profiler internals for detailed strategy feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.data.profiler import DatasetProfile


@dataclass(frozen=True)
class ImbalanceRecommendation:
    strategy: str | None           # None → no action needed (balanced or regression)
    apply_threshold: bool
    reasoning: str
    severity: str                  # "none" | "mild" | "moderate" | "severe" | "critical"
    feasibility_notes: list[str]


def recommend_imbalance_strategy(
    profile: DatasetProfile,
    *,
    minority_count: int | None = None,
) -> ImbalanceRecommendation:
    """
    Derive the recommended balancing strategy from a DatasetProfile.

    Parameters
    ----------
    profile:
        The DatasetProfile produced by DatasetProfiler.
    minority_count:
        Absolute count of the minority class.  If supplied, SMOTE
        feasibility (≥ 6 samples) is checked more precisely.
    """
    task = profile.task_type
    ir = profile.imbalance_ratio

    if task == "regression":
        return ImbalanceRecommendation(
            strategy=None,
            apply_threshold=False,
            reasoning="Regression task: no class balancing needed.",
            severity="none",
            feasibility_notes=[],
        )

    if ir is None or ir <= 1.5:
        return ImbalanceRecommendation(
            strategy=None,
            apply_threshold=False,
            reasoning="Dataset is balanced (IR ≤ 1.5): no resampling required.",
            severity="none",
            feasibility_notes=[],
        )

    severity = _severity(ir)
    notes: list[str] = []

    # SMOTE feasibility
    smote_feasible = (
        task == "binary_classification"
        and (minority_count is None or minority_count >= 6)
    )
    if not smote_feasible and minority_count is not None and minority_count < 6:
        notes.append(f"SMOTE requires ≥ 6 minority samples (got {minority_count}); falling back to class_weight.")

    if ir <= 3.0:
        strategy = "class_weight"
        reasoning = f"Mild imbalance (IR={ir:.1f}): class_weight adjusts the loss without synthetic data."
        apply_threshold = False
    elif ir <= 10.0:
        if smote_feasible:
            strategy = "smote"
            reasoning = f"Moderate imbalance (IR={ir:.1f}): SMOTE synthesises minority samples before training."
        else:
            strategy = "class_weight"
            reasoning = f"Moderate imbalance (IR={ir:.1f}): SMOTE infeasible, using class_weight."
        apply_threshold = False
    else:
        # Severe / critical
        if smote_feasible:
            strategy = "smote_tomek"
            reasoning = (
                f"Severe imbalance (IR={ir:.1f}): SMOTE + Tomek links synthesis and boundary cleanup. "
                "Threshold optimisation is also advised."
            )
        else:
            strategy = "class_weight"
            reasoning = f"Severe imbalance (IR={ir:.1f}): SMOTE infeasible, using class_weight + threshold optimisation."
        apply_threshold = True

    return ImbalanceRecommendation(
        strategy=strategy,
        apply_threshold=apply_threshold,
        reasoning=reasoning,
        severity=severity,
        feasibility_notes=notes,
    )


def _severity(ir: float) -> str:
    if ir <= 1.5:
        return "none"
    if ir <= 3.0:
        return "mild"
    if ir <= 10.0:
        return "moderate"
    if ir <= 50.0:
        return "severe"
    return "critical"
