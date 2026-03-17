"""
MetricSelector — Phase 4
Selects primary and secondary metrics from a DatasetProfile.
Medical context: always include recall, precision, and confusion matrix
when they are relevant.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.data.profiler import DatasetProfile


@dataclass(frozen=True)
class MetricSelection:
    primary: str
    secondary: list[str]
    reasoning: str


# Maps task_type + imbalance_level → (primary, secondaries, reason)
def select_metrics(profile: DatasetProfile) -> MetricSelection:
    task = profile.task_type
    ir = profile.imbalance_ratio or 1.0

    if task == "regression":
        return MetricSelection(
            primary="rmse",
            secondary=["mae", "r2"],
            reasoning="RMSE is the standard primary metric for regression; MAE and R² provide complementary views.",
        )

    if task == "binary_classification":
        if ir <= 1.5:
            return MetricSelection(
                primary="roc_auc",
                secondary=["f1", "precision", "recall", "accuracy"],
                reasoning="Dataset is balanced; ROC-AUC gives a robust discrimination measure.",
            )
        if ir <= 5.0:
            return MetricSelection(
                primary="f1",
                secondary=["roc_auc", "precision", "recall", "pr_auc"],
                reasoning="Mild imbalance: F1 balances precision and recall; PR-AUC tracks minority performance.",
            )
        # Severe / critical imbalance
        return MetricSelection(
            primary="pr_auc",
            secondary=["f1", "recall", "precision", "roc_auc"],
            reasoning=(
                f"Imbalance ratio={ir:.1f}: PR-AUC is more informative than ROC-AUC under severe imbalance. "
                "Monitor recall explicitly for the minority class."
            ),
        )

    # multiclass
    if ir <= 2.0:
        return MetricSelection(
            primary="f1_macro",
            secondary=["accuracy", "f1_weighted", "precision_macro", "recall_macro"],
            reasoning="Multiclass balanced: F1-macro weights all classes equally.",
        )
    return MetricSelection(
        primary="f1_weighted",
        secondary=["f1_macro", "accuracy", "precision_weighted", "recall_weighted"],
        reasoning="Multiclass imbalanced: F1-weighted accounts for class frequencies.",
    )
