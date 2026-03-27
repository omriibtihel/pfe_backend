from __future__ import annotations

from dataclasses import dataclass, field

from .profiler import DataProfile, ImbalanceLevel


@dataclass
class BalancingDecision:
    strategy: str
    apply_threshold: bool
    threshold_strategy: str
    rationale: str
    refit_metric: str
    smote_k_neighbors: int | None
    class_weight_mode: str
    audit_flags: list[str]
    optimal_threshold: float
    random_state: int = 42
    min_recall_constraint: float | None = None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_smote_k(minority_count: int) -> int:
    return max(1, min(5, int(minority_count) - 2))


def resolve(
    profile: DataProfile,
    config: "BalancingConfig",  # type: ignore[name-defined]  # noqa: F821
    model_supports_class_weight: bool,
    model_supports_predict_proba: bool,
    model_supports_sample_weight: bool = False,
    random_state: int = 42,
) -> BalancingDecision:
    from app.services.training.config.schema import BalancingConfig  # local import to avoid circular at module load

    requested_strategy = str(config.strategy or "none").strip().lower()
    threshold_strategy = str(config.threshold_strategy or "maximize_f1").strip().lower()
    min_recall_constraint = config.min_recall_constraint
    apply_threshold = bool(config.apply_threshold) or requested_strategy == "threshold_optimization"
    refit_metric = (
        "average_precision"
        if profile.imbalance_level in {ImbalanceLevel.SEVERE, ImbalanceLevel.CRITICAL}
        else "f1"
    )

    # Dataset is balanced: skip resampling unless the user explicitly selected a non-"none" strategy.
    # The frontend is responsible for showing a warning modal before the user reaches this point.
    force_on_balanced = not profile.needs_balancing and requested_strategy != "none"
    if not profile.needs_balancing and not force_on_balanced:
        flags = ["none", "balanced_dataset_no_resampling"]
        if apply_threshold and not model_supports_predict_proba:
            flags.append("predict_proba_unavailable_threshold_fallback_expected")
        return BalancingDecision(
            strategy="none",
            apply_threshold=apply_threshold,
            threshold_strategy=threshold_strategy,
            rationale=(
                "Dataset is balanced, so no resampling/class reweighting was applied. "
                "Any threshold optimization setting is kept as requested."
            ),
            refit_metric=refit_metric,
            smote_k_neighbors=None,
            class_weight_mode="balanced",
            audit_flags=_dedupe(flags),
            optimal_threshold=0.5,
            random_state=random_state,
            min_recall_constraint=min_recall_constraint,
        )

    strategy = requested_strategy
    smote_k_neighbors: int | None = None
    flags: list[str] = [requested_strategy or "none"]
    if force_on_balanced:
        flags.append("forced_on_balanced_dataset")

    if requested_strategy == "none":
        flags.append("user_chose_none")
        rationale = "User explicitly selected no balancing; training data is left unchanged."
        strategy = "none"
    elif requested_strategy == "class_weight":
        if model_supports_class_weight:
            flags.append("class_weight_balanced")
            rationale = "User selected class_weight; class_weight='balanced' is applied to the estimator."
            strategy = "class_weight"
        elif model_supports_sample_weight:
            flags.extend(
                [
                    "fallback_sample_weight:model_no_class_weight",
                    "fallback_sample_weight_model_no_class_weight",
                ]
            )
            rationale = (
                "User selected class_weight, but estimator has no class_weight parameter; "
                "sample_weight='balanced' is applied during fit."
            )
            strategy = "sample_weight"
        else:
            flags.extend(
                [
                    "fallback_none:model_no_class_weight_or_sample_weight",
                    "fallback_none_model_no_class_weight_or_sample_weight",
                ]
            )
            rationale = (
                "User selected class_weight, but estimator supports neither class_weight nor sample_weight; "
                "no class reweighting was applied."
            )
            strategy = "none"
    elif requested_strategy == "smote":
        smote_k_neighbors = _resolve_smote_k(profile.minority.count)
        flags.extend(["smote", f"smote_k_neighbors_{smote_k_neighbors}"])
        rationale = (
            "User selected SMOTE; synthetic minority oversampling is applied on the training split "
            f"with k_neighbors={smote_k_neighbors}."
        )
        strategy = "smote"
    elif requested_strategy == "smote_tomek":
        smote_k_neighbors = _resolve_smote_k(profile.minority.count)
        flags.extend(["smote_tomek", f"smote_k_neighbors_{smote_k_neighbors}"])
        rationale = (
            "User selected SMOTE+Tomek; training data is oversampled then cleaned "
            f"with Tomek links (k_neighbors={smote_k_neighbors})."
        )
        strategy = "smote_tomek"
    elif requested_strategy == "random_undersampling":
        flags.extend(["random_undersampling", f"undersampling_target_{int(profile.minority.count)}"])
        rationale = (
            "User selected random undersampling; majority class is reduced to minority count "
            f"({int(profile.minority.count)} samples)."
        )
        strategy = "random_undersampling"
    elif requested_strategy == "threshold_optimization":
        flags.extend(["threshold_optimization", "no_resampling"])
        rationale = (
            "User selected threshold optimization only; no resampling or class reweighting is applied."
        )
        strategy = "none"
        apply_threshold = True
    else:
        flags.append("invalid_strategy_fallback_none")
        rationale = (
            f"Unknown strategy '{requested_strategy}' received; no balancing strategy was executed."
        )
        strategy = "none"

    if apply_threshold and not model_supports_predict_proba:
        flags.append("predict_proba_unavailable_threshold_fallback_expected")

    return BalancingDecision(
        strategy=strategy,
        apply_threshold=bool(apply_threshold),
        threshold_strategy=threshold_strategy,
        rationale=rationale.strip(),
        refit_metric=refit_metric,
        smote_k_neighbors=smote_k_neighbors,
        class_weight_mode="balanced",
        audit_flags=_dedupe(flags),
        optimal_threshold=0.5,
        random_state=random_state,
        min_recall_constraint=min_recall_constraint,
    )
