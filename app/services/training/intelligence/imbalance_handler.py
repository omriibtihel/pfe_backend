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
    n_samples: int | None = None,
) -> ImbalanceRecommendation:
    """
    Derive the recommended balancing strategy from a DatasetProfile.

    Parameters
    ----------
    profile:
        The DatasetProfile produced by DatasetProfiler.
    minority_count:
        Absolute count of the minority class.  If supplied, SMOTE
        feasibility (≥ 6 samples, synthetic ratio) is checked precisely.
    n_samples:
        Total number of samples.  Falls back to profile.n_samples when absent.
        Used to detect tiny-dataset situations where SMOTE is unreliable.
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

    _n = n_samples if n_samples is not None else profile.n_samples
    is_tiny = _n < 200

    # SMOTE feasibility — mirrors the logic in balancing/profiler._smote_feasibility
    smote_min_ok = minority_count is None or minority_count >= 6
    smote_not_tiny_starved = not (is_tiny and minority_count is not None and minority_count < 20)

    # Synthetic ratio check: if IR > 10 and minority is very small, SMOTE would generate
    # > 10× synthetic samples for the minority class → unreliable
    synthetic_ratio_ok = True
    if minority_count is not None and minority_count > 0:
        majority_estimate = _n - minority_count
        synthetic_to_generate = max(0, majority_estimate - minority_count)
        if synthetic_to_generate / minority_count > 10:
            synthetic_ratio_ok = False
            notes.append(
                f"SMOTE synthetic ratio too high ({synthetic_to_generate / minority_count:.1f}×): "
                "falling back to class_weight."
            )

    smote_feasible = (
        task == "binary_classification"
        and smote_min_ok
        and smote_not_tiny_starved
        and synthetic_ratio_ok
    )

    if not smote_min_ok and minority_count is not None:
        notes.append(f"SMOTE requires ≥ 6 minority samples (got {minority_count}); falling back to class_weight.")
    if not smote_not_tiny_starved and minority_count is not None:
        notes.append(
            f"Tiny dataset (n={_n}) with only {minority_count} minority samples: "
            "SMOTE bypassed to avoid synthetic-data dominance."
        )

    if ir <= 3.0:
        # Mild imbalance: class_weight is safe and sufficient on all dataset sizes
        strategy = "class_weight"
        reasoning = (
            f"Mild imbalance (IR={ir:.1f}): class_weight adjusts the loss function "
            "without generating synthetic data — safe at all dataset sizes."
        )
        apply_threshold = False

    elif ir <= 10.0:
        # Moderate imbalance
        if smote_feasible and not is_tiny:
            strategy = "smote"
            reasoning = (
                f"Moderate imbalance (IR={ir:.1f}): SMOTE synthesises minority samples "
                "on the training split only, improving minority-class coverage."
            )
            apply_threshold = False
        elif smote_feasible and is_tiny:
            # Technically feasible but risky on tiny data
            strategy = "class_weight"
            reasoning = (
                f"Moderate imbalance (IR={ir:.1f}): SMOTE is technically feasible but bypassed "
                f"on tiny dataset (n={_n}) — class_weight avoids synthetic-data dominance."
            )
            apply_threshold = True  # add threshold for extra robustness on tiny data
        else:
            strategy = "class_weight"
            reasoning = (
                f"Moderate imbalance (IR={ir:.1f}): SMOTE infeasible — "
                "class_weight reweights the loss function."
            )
            apply_threshold = False

    else:
        # Severe / critical imbalance
        if smote_feasible and not is_tiny:
            strategy = "smote_tomek"
            reasoning = (
                f"Severe imbalance (IR={ir:.1f}): SMOTE + Tomek combines oversampling with "
                "boundary cleanup, giving the best minority-class coverage. "
                "Threshold optimisation is also advised."
            )
            apply_threshold = True
        elif smote_feasible and is_tiny:
            strategy = "class_weight"
            reasoning = (
                f"Severe imbalance (IR={ir:.1f}): SMOTE bypassed on tiny dataset (n={_n}) — "
                "class_weight + threshold optimisation is the safest approach."
            )
            apply_threshold = True
        else:
            strategy = "class_weight"
            reasoning = (
                f"Severe imbalance (IR={ir:.1f}): SMOTE infeasible — "
                "class_weight + threshold optimisation is applied."
            )
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
