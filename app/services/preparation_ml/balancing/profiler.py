from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DatasetScale(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ImbalanceLevel(str, Enum):
    BALANCED = "balanced"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class StrategyImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class BinaryClassProfile:
    label: Any
    count: int
    ratio: float
    role: str


@dataclass(frozen=True)
class AvailableStrategy:
    id: str
    label: str
    description: str
    impact: str
    recommended: bool
    feasible: bool
    infeasible_reason: str | None = None


@dataclass(frozen=True)
class DataProfile:
    n_samples: int
    n_features: int
    scale: str
    majority: BinaryClassProfile
    minority: BinaryClassProfile
    imbalance_ratio: float
    minority_ratio: float
    imbalance_level: ImbalanceLevel
    needs_balancing: bool
    available_strategies: list[AvailableStrategy]
    default_recommendation: str
    summary_message: str
    warnings: list[str]
    metric_advice: list[str]


def is_binary(y: np.ndarray) -> bool:
    try:
        return len(np.unique(np.asarray(y))) == 2
    except Exception:
        return False


def class_counts(y: np.ndarray) -> dict[str, int]:
    try:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}
    except Exception:
        return {}


def minority_ratio(y: np.ndarray | None) -> float | None:
    if y is None:
        return None
    try:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        total = int(counts.sum())
        if total <= 0:
            return None
        return float(int(counts.min()) / total)
    except Exception:
        return None


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _infer_scale(n_samples: int) -> DatasetScale:
    if n_samples < 200:
        return DatasetScale.TINY
    if n_samples < 2000:
        return DatasetScale.SMALL
    if n_samples < 50000:
        return DatasetScale.MEDIUM
    return DatasetScale.LARGE


def _level_from_ir(imbalance_ratio: float) -> ImbalanceLevel:
    if imbalance_ratio <= 1.5:
        return ImbalanceLevel.BALANCED
    if imbalance_ratio <= 3.0:
        return ImbalanceLevel.MILD
    if imbalance_ratio <= 10.0:
        return ImbalanceLevel.MODERATE
    if imbalance_ratio <= 50.0:
        return ImbalanceLevel.SEVERE
    return ImbalanceLevel.CRITICAL


def _resolve_smote_k(minority_count: int, n_samples: int = 0) -> int:
    """Return a safe k_neighbors for SMOTE given dataset context.

    k must be strictly less than minority_count.  On tiny datasets we use
    fewer neighbours to reduce the risk of creating nearly identical synthetics.
    """
    k_max = max(1, int(minority_count) - 1)  # sklearn requires k < n_minority
    if n_samples > 0 and n_samples < 200:
        return min(3, k_max)  # conservative on tiny datasets
    return min(5, k_max)


def _smote_feasibility(
    binary: bool,
    minority_count: int,
    n_samples: int,
) -> tuple[bool, str | None]:
    """Return (feasible, reason) for SMOTE.

    Checks beyond the sklearn minimum (k < n_minority):
    - Tiny datasets: SMOTE with very few minority samples creates synthetic-data
      dominated training sets, inflating apparent performance.
    - High synthetic ratio: when IR > 10 and minority is small, >90 % of minority
      class after SMOTE would be synthetic → unreliable generalisation.
    """
    if not binary:
        return False, "requires_binary_target"
    if minority_count < 6:
        return False, f"requires_at_least_6_minority_samples (got {minority_count})"

    # On tiny datasets require at least 20 minority samples to keep synthesis meaningful
    if n_samples < 200 and minority_count < 20:
        return False, (
            f"tiny_dataset_smote_unstable: minority={minority_count}/{n_samples} — "
            "risk of synthetic-data dominance after equalisation"
        )

    # Synthetic ratio check: SMOTE (default strategy='auto') equalises classes.
    # It generates majority_count - minority_count new synthetic minority samples.
    # When that exceeds 10× the original minority count, the minority class becomes
    # mostly synthetic and the model learns the SMOTE manifold rather than real data.
    majority_estimate = n_samples - minority_count
    synthetic_to_generate = max(0, majority_estimate - minority_count)
    if minority_count > 0:
        synthetic_ratio = synthetic_to_generate / minority_count
        if synthetic_ratio > 10:
            return False, (
                f"synthetic_ratio_too_high ({synthetic_ratio:.1f}×): "
                "SMOTE would generate more than 10× the original minority count — "
                "use class_weight or threshold optimisation instead"
            )

    return True, None


def _undersampling_feasibility(
    minority_count: int,
    n_samples: int,
) -> tuple[bool, str | None]:
    """Return (feasible, reason) for random undersampling.

    Post-undersampling the dataset contains 2 × minority_count rows.
    We require that result to be at least 40 samples and that the dataset
    is not already tiny (losing majority data on a tiny dataset makes things worse).
    """
    if minority_count < 20:
        return False, f"requires_at_least_20_minority_samples (got {minority_count})"

    post_size = 2 * minority_count
    if post_size < 40:
        return False, f"post_undersampling_dataset_too_small ({post_size} samples)"

    # On tiny datasets undersampling discards too much of an already scarce majority
    if n_samples < 200:
        return False, (
            f"undersampling_not_recommended_on_tiny_dataset (n={n_samples}): "
            f"would reduce to {post_size} samples"
        )

    return True, None


def _recommended_ids(level: ImbalanceLevel, scale: DatasetScale) -> set[str]:
    """Return the set of strategy IDs that are recommended for this situation.

    Scale matters because resampling on tiny datasets risks worse outcomes
    than lightweight reweighting / threshold tuning.
    """
    if level == ImbalanceLevel.BALANCED:
        return {"none"}

    if scale == DatasetScale.TINY:
        # Tiny datasets: synthetic generation is fragile, prefer reweighting.
        if level == ImbalanceLevel.MILD:
            return {"class_weight"}
        # Moderate / Severe / Critical: add threshold optimisation for extra robustness
        return {"class_weight", "threshold_optimization"}

    # SMALL / MEDIUM / LARGE
    if level == ImbalanceLevel.MILD:
        return {"class_weight", "threshold_optimization"}
    if level == ImbalanceLevel.MODERATE:
        return {"smote", "class_weight", "threshold_optimization"}
    # SEVERE / CRITICAL
    return {"smote_tomek", "smote", "class_weight", "threshold_optimization"}


def _default_priority(level: ImbalanceLevel, scale: DatasetScale) -> list[str]:
    """Strategy IDs in descending priority for the default recommendation."""
    if scale == DatasetScale.TINY:
        return ["class_weight", "threshold_optimization", "smote", "random_undersampling"]
    if level == ImbalanceLevel.MILD:
        return ["class_weight", "threshold_optimization"]
    if level == ImbalanceLevel.MODERATE:
        return ["smote", "class_weight", "threshold_optimization", "random_undersampling"]
    # SEVERE / CRITICAL
    return ["smote_tomek", "smote", "class_weight", "threshold_optimization", "random_undersampling"]


def _build_strategies(
    *,
    binary: bool,
    minority_count: int,
    n_samples: int,
    level: ImbalanceLevel,
) -> list[AvailableStrategy]:
    scale = _infer_scale(n_samples)
    recommended_ids = _recommended_ids(level, scale)

    smote_ok, smote_reason = _smote_feasibility(binary, minority_count, n_samples)
    undersample_ok, undersample_reason = _undersampling_feasibility(minority_count, n_samples)

    return [
        AvailableStrategy(
            id="none",
            label="No balancing",
            description="Train on the original split without class rebalancing.",
            impact=StrategyImpact.LOW.value,
            recommended=("none" in recommended_ids),
            feasible=True,
            infeasible_reason=None,
        ),
        AvailableStrategy(
            id="class_weight",
            label="Class weight",
            description="Increase minority-class importance in the estimator loss.",
            impact=StrategyImpact.LOW.value,
            recommended=("class_weight" in recommended_ids),
            feasible=True,
            infeasible_reason=None,
        ),
        AvailableStrategy(
            id="smote",
            label="SMOTE",
            description="Create synthetic minority samples before model fitting.",
            impact=StrategyImpact.MEDIUM.value,
            recommended=("smote" in recommended_ids) and smote_ok,
            feasible=smote_ok,
            infeasible_reason=smote_reason,
        ),
        AvailableStrategy(
            id="smote_tomek",
            label="SMOTE + Tomek",
            description="SMOTE oversampling followed by Tomek links cleanup.",
            impact=StrategyImpact.HIGH.value,
            recommended=("smote_tomek" in recommended_ids) and smote_ok,
            feasible=smote_ok,
            infeasible_reason=smote_reason,
        ),
        AvailableStrategy(
            id="random_undersampling",
            label="Random undersampling",
            description="Downsample majority class to match minority class size.",
            impact=StrategyImpact.HIGH.value,
            recommended=("random_undersampling" in recommended_ids) and undersample_ok,
            feasible=undersample_ok,
            infeasible_reason=undersample_reason,
        ),
        AvailableStrategy(
            id="threshold_optimization",
            label="Threshold optimization",
            description="Keep data unchanged and optimize the probability threshold post-fit.",
            impact=StrategyImpact.LOW.value,
            recommended=("threshold_optimization" in recommended_ids),
            feasible=True,
            infeasible_reason=None,
        ),
    ]


def _default_recommendation(
    level: ImbalanceLevel,
    strategies: list[AvailableStrategy],
    scale: DatasetScale,
) -> str:
    if level == ImbalanceLevel.BALANCED:
        return "none"

    strategy_map = {s.id: s for s in strategies}

    # First pass: recommended AND feasible, in priority order
    for sid in _default_priority(level, scale):
        s = strategy_map.get(sid)
        if s is not None and s.feasible and s.recommended:
            return sid

    # Second pass: any feasible (non-none) strategy in priority order
    for sid in _default_priority(level, scale):
        s = strategy_map.get(sid)
        if s is not None and s.feasible and sid != "none":
            return sid

    return "class_weight"


def _build_metric_advice(level: ImbalanceLevel, needs_balancing: bool) -> list[str]:
    if level in {ImbalanceLevel.SEVERE, ImbalanceLevel.CRITICAL}:
        return [
            "Prefer PR-AUC/average_precision over accuracy for model selection.",
            "Track minority recall explicitly and consider threshold optimization.",
        ]
    if needs_balancing:
        return [
            "Monitor F1 and PR-AUC together on validation data.",
            "Use threshold optimization if recall/precision tradeoff matters.",
        ]
    return ["Dataset is balanced; standard F1/ROC-AUC monitoring is generally sufficient."]


def profile_binary_dataset(y: np.ndarray, X_shape: tuple[Any, ...]) -> DataProfile:
    y_arr = np.asarray(y)
    values, counts = np.unique(y_arr, return_counts=True)

    n_samples = int(len(y_arr))
    n_features = int(X_shape[1]) if len(X_shape) >= 2 else 0
    scale = _infer_scale(n_samples)

    warnings: list[str] = []
    if len(values) < 2:
        warnings.append("target_has_single_class")
    if len(values) > 2:
        warnings.append("target_is_not_binary")

    ordered = sorted(
        [(v, int(c)) for v, c in zip(values, counts)],
        key=lambda item: (item[1], str(item[0])),
    )
    if ordered:
        minority_label, minority_count = ordered[0]
        majority_label, majority_count = ordered[-1]
        minority_label = _to_python_scalar(minority_label)
        majority_label = _to_python_scalar(majority_label)
    else:
        minority_label, minority_count = 0, 0
        majority_label, majority_count = 0, 0

    total = max(1, int(majority_count + minority_count))
    minority_ratio_value = float(minority_count / total) if total > 0 else 0.0
    majority_ratio_value = float(majority_count / total) if total > 0 else 0.0
    if minority_count <= 0:
        imbalance_ratio_value = float("inf")
    else:
        imbalance_ratio_value = float(majority_count / minority_count)

    level = _level_from_ir(imbalance_ratio_value if np.isfinite(imbalance_ratio_value) else 999999.0)
    needs_balancing = bool(level != ImbalanceLevel.BALANCED)

    if not needs_balancing:
        warnings.append("dataset_is_already_balanced")

    # Scale-aware warnings
    if n_samples < 200:
        warnings.append("tiny_dataset_high_variance_risk")

    smote_ok, smote_reason = _smote_feasibility(len(values) == 2, int(minority_count), n_samples)
    if not smote_ok and needs_balancing:
        warnings.append(f"smote_infeasible: {smote_reason}")

    undersample_ok, undersample_reason = _undersampling_feasibility(int(minority_count), n_samples)
    if not undersample_ok and needs_balancing:
        warnings.append(f"random_undersampling_infeasible: {undersample_reason}")

    if level in {ImbalanceLevel.SEVERE, ImbalanceLevel.CRITICAL}:
        warnings.append("severe_imbalance_detected")

    strategies = _build_strategies(
        binary=(len(values) == 2),
        minority_count=int(minority_count),
        n_samples=n_samples,
        level=level,
    )
    default_rec = _default_recommendation(level, strategies, scale)

    if needs_balancing:
        summary_message = (
            f"Imbalance detected: minority '{minority_label}' has {minority_count}/{n_samples} samples "
            f"({minority_ratio_value:.2%}), imbalance ratio={imbalance_ratio_value:.2f} ({level.value})."
        )
    else:
        summary_message = (
            f"Dataset is balanced: majority/minority ratio={imbalance_ratio_value:.2f} "
            f"with minority share={minority_ratio_value:.2%}."
        )

    return DataProfile(
        n_samples=n_samples,
        n_features=n_features,
        scale=scale.value,
        majority=BinaryClassProfile(
            label=majority_label,
            count=int(majority_count),
            ratio=majority_ratio_value,
            role="majority",
        ),
        minority=BinaryClassProfile(
            label=minority_label,
            count=int(minority_count),
            ratio=minority_ratio_value,
            role="minority",
        ),
        imbalance_ratio=imbalance_ratio_value,
        minority_ratio=minority_ratio_value,
        imbalance_level=level,
        needs_balancing=needs_balancing,
        available_strategies=strategies,
        default_recommendation=default_rec,
        summary_message=summary_message,
        warnings=warnings,
        metric_advice=_build_metric_advice(level, needs_balancing),
    )
