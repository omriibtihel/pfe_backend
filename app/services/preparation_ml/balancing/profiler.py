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


def _recommended_ids(level: ImbalanceLevel) -> set[str]:
    if level == ImbalanceLevel.BALANCED:
        return {"none"}
    if level == ImbalanceLevel.MILD:
        return {"class_weight", "threshold_optimization"}
    if level == ImbalanceLevel.MODERATE:
        return {"smote", "class_weight", "threshold_optimization"}
    return {"smote_tomek", "smote", "class_weight", "threshold_optimization"}


def _build_strategies(
    *,
    binary: bool,
    minority_count: int,
    level: ImbalanceLevel,
) -> list[AvailableStrategy]:
    recommended_ids = _recommended_ids(level)
    smote_feasible = bool(binary and minority_count >= 7)
    undersampling_feasible = bool(minority_count >= 30)

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
            recommended=("smote" in recommended_ids),
            feasible=smote_feasible,
            infeasible_reason=None if smote_feasible else "requires_binary_target_and_minority_count_at_least_7",
        ),
        AvailableStrategy(
            id="smote_tomek",
            label="SMOTE + Tomek",
            description="SMOTE oversampling followed by Tomek links cleanup.",
            impact=StrategyImpact.HIGH.value,
            recommended=("smote_tomek" in recommended_ids),
            feasible=smote_feasible,
            infeasible_reason=None if smote_feasible else "requires_binary_target_and_minority_count_at_least_7",
        ),
        AvailableStrategy(
            id="random_undersampling",
            label="Random undersampling",
            description="Downsample majority class to match minority class size.",
            impact=StrategyImpact.HIGH.value,
            recommended=("random_undersampling" in recommended_ids),
            feasible=undersampling_feasible,
            infeasible_reason=None if undersampling_feasible else "requires_minority_count_at_least_30",
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


def _default_recommendation(level: ImbalanceLevel, strategies: list[AvailableStrategy]) -> str:
    if level == ImbalanceLevel.BALANCED:
        return "none"
    for strategy in strategies:
        if strategy.id == "none":
            continue
        if strategy.recommended and strategy.feasible:
            return strategy.id
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
    scale = _infer_scale(n_samples).value

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

    if minority_count < 6:
        warnings.append("smote_requires_minority_count_at_least_6")
    if minority_count < 30:
        warnings.append("random_undersampling_requires_minority_count_at_least_30")
    if level in {ImbalanceLevel.SEVERE, ImbalanceLevel.CRITICAL}:
        warnings.append("severe_imbalance_detected")
    if n_samples < 200:
        warnings.append("tiny_dataset_high_variance_risk")

    strategies = _build_strategies(
        binary=(len(values) == 2),
        minority_count=int(minority_count),
        level=level,
    )
    default_recommendation = _default_recommendation(level, strategies)

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
        scale=scale,
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
        default_recommendation=default_recommendation,
        summary_message=summary_message,
        warnings=warnings,
        metric_advice=_build_metric_advice(level, needs_balancing),
    )
