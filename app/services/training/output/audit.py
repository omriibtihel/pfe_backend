from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.training import BalancingAudit
from app.services.preparation_ml.balancing import BalancingDecision, DataProfile


def _profile_to_dict(profile: DataProfile) -> dict[str, Any]:
    return {
        "n_samples": int(profile.n_samples),
        "dataset_scale": str(profile.scale),
        "imbalance_ratio": float(profile.imbalance_ratio),
        "minority_ratio": float(profile.minority_ratio),
        "minority_count": int(profile.minority.count),
        "imbalance_level": str(profile.imbalance_level.value),
        "class_counts": {
            str(profile.majority.label): int(profile.majority.count),
            str(profile.minority.label): int(profile.minority.count),
        },
    }


def build_and_persist_audit(
    *,
    profile: DataProfile,
    decision: BalancingDecision,
    session_id: int | None,
    db: Session | None,
    smote_samples_added: int | None,
    optimal_threshold: float,
    threshold_f1_gain: float | None,
    postfit_warnings: list[str],
) -> dict[str, Any]:
    """Build the full audit payload and persist it in one single commit."""
    payload: dict[str, Any] = {
        **_profile_to_dict(profile),
        "strategy_applied": str(decision.strategy),
        "rationale": str(decision.rationale),
        "audit_flags": [str(f) for f in (decision.audit_flags or [decision.strategy])],
        "refit_metric": str(decision.refit_metric),
        "optimal_threshold": float(optimal_threshold),
        "threshold_f1_gain": threshold_f1_gain,
        "threshold_strategy": str(decision.threshold_strategy),
        "apply_threshold": bool(decision.apply_threshold),
        "smote_k_neighbors": decision.smote_k_neighbors,
        "smote_samples_added": smote_samples_added,
        "warnings": [
            *[
                str(w) for w in (profile.warnings or [])
                # Suppress "dataset_is_already_balanced" when a resampling strategy was actually forced
                # (strategy != "none" means the user explicitly chose one and it was applied).
                if not (str(w) == "dataset_is_already_balanced" and str(decision.strategy) not in {"none", "threshold_optimization"})
            ],
            *[str(w) for w in postfit_warnings],
        ],
    }

    if db is None or session_id is None:
        return payload

    audit = BalancingAudit(
        session_id=int(session_id),
        n_samples=int(payload["n_samples"]),
        dataset_scale=str(payload["dataset_scale"]),
        imbalance_ratio=float(payload["imbalance_ratio"]),
        minority_ratio=float(payload["minority_ratio"]),
        minority_count=int(payload["minority_count"]),
        imbalance_level=str(payload["imbalance_level"]),
        class_counts=dict(payload["class_counts"]),
        strategy_applied=str(payload["strategy_applied"]),
        rationale=str(payload["rationale"]),
        audit_flags=list(payload["audit_flags"]),
        refit_metric=str(payload["refit_metric"]),
        optimal_threshold=float(optimal_threshold),
        threshold_f1_gain=threshold_f1_gain,
        smote_k_neighbors=(
            int(payload["smote_k_neighbors"]) if payload["smote_k_neighbors"] is not None else None
        ),
        smote_samples_added=smote_samples_added,
        warnings=list(payload["warnings"]),
    )
    db.add(audit)
    db.commit()
    db.refresh(audit)
    payload["id"] = str(audit.id)
    return payload
