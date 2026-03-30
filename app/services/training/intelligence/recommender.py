"""
RecommendationEngine — Phase 2
Transforms a DatasetProfile into a TrainingRecommendation that can be
sent to the frontend and/or directly executed via TrainingConfigBuilder.

All decisions follow the product rules:
  - Prudent, fast, robust defaults.
  - At most 3–4 models recommended.
  - Explanations are human-readable and frontend-ready.
  - No side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.data.profiler import DatasetProfile
from app.services.training.intelligence.imbalance_handler import recommend_imbalance_strategy
from app.services.training.intelligence.metric_selector import select_metrics
from app.services.training.config.zero_shot import get_zero_shot_hyperparams


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRecommendation:
    mode: str                               # always "recommendation"
    recommended_models: list[str]
    recommended_resampling: str | None
    apply_threshold: bool
    recommended_metric: str
    secondary_metrics: list[str]
    recommended_cv_strategy: str           # "holdout" | "kfold" | "stratified_kfold"
    recommended_k_folds: int               # only relevant when cv != holdout
    recommended_search_type: str           # "none" | "grid" | "random"
    recommended_time_budget_s: int | None
    recommended_class_weight: str | None
    recommended_split: dict[str, int]      # {trainRatio, valRatio, testRatio} in %
    reasoning: dict[str, str]              # label → explanation (frontend-ready)
    training_config_payload: dict[str, Any]  # ready to pass to TrainingConfig.from_front()
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_models(profile: DatasetProfile) -> list[str]:
    task = profile.task_type
    size = profile.dataset_size_category
    ir = profile.imbalance_ratio or 1.0

    if task == "regression":
        if size == "large":
            return ["lightgbm", "ridge"]
        if size in {"tiny", "small"}:
            return ["ridge", "randomforest", "lightgbm"]
        return ["ridge", "randomforest", "lightgbm", "xgboost"]

    # Classification
    if size == "large":
        # Prioritise fast models
        return ["lightgbm", "logisticregression"]

    if size in {"tiny", "small"}:
        if ir > 5.0:
            return ["logisticregression", "lightgbm", "randomforest"]
        return ["logisticregression", "randomforest", "lightgbm"]

    # medium
    if ir > 5.0:
        return ["lightgbm", "randomforest", "logisticregression"]
    return ["logisticregression", "randomforest", "lightgbm", "xgboost"]


def _recommend_search_type(profile: DatasetProfile) -> tuple[str, int | None]:
    """Returns (search_type, time_budget_s_or_None)."""
    size = profile.dataset_size_category
    speed = profile.estimated_training_speed

    if size == "tiny":
        return "grid", None
    if size == "small":
        return "random", None
    if size == "medium":
        return "random", None
    # large → no HPO by default (too slow), user can activate
    return "none", None


def _recommend_split(profile: DatasetProfile) -> dict[str, int]:
    size = profile.dataset_size_category
    cv = profile.recommended_cv_strategy

    if cv != "holdout":
        # CV mode: no holdout test split by default
        return {"trainRatio": 80, "valRatio": 0, "testRatio": 0}
    if size in {"tiny", "small"}:
        return {"trainRatio": 70, "valRatio": 15, "testRatio": 15}
    return {"trainRatio": 70, "valRatio": 15, "testRatio": 15}


def _recommend_k_folds(profile: DatasetProfile) -> int:
    if profile.dataset_size_category in {"tiny", "small"}:
        return 10
    return 5


def _recommend_class_weight(profile: DatasetProfile, resampling: str | None) -> str | None:
    """Recommend class_weight only when resampling is none/threshold."""
    if profile.task_type == "regression":
        return None
    ir = profile.imbalance_ratio or 1.0
    if ir <= 1.5:
        return None
    if resampling in {None, "threshold_optimization"}:
        return "balanced"
    return None


def _build_reasoning(
    profile: DatasetProfile,
    models: list[str],
    resampling: str | None,
    metric: str,
    cv: str,
    search_type: str,
) -> dict[str, str]:
    size_label = {
        "tiny": "very small (< 500 rows)",
        "small": "small (500–2 000 rows)",
        "medium": "medium (2 000–50 000 rows)",
        "large": "large (> 50 000 rows)",
    }.get(profile.dataset_size_category, profile.dataset_size_category)

    ir = profile.imbalance_ratio
    ir_msg = (
        "dataset is balanced"
        if ir is None or ir <= 1.5
        else f"imbalance ratio={ir:.1f}"
    )

    search_msg = {
        "none": "No HPO (dataset is large; use Random or Grid search if time allows).",
        "grid": "Grid search (small dataset, exhaustive search is feasible).",
        "random": "Randomised search (balanced speed/quality tradeoff).",
    }.get(search_type, search_type)

    return {
        "dataset_size": f"Dataset is {size_label} — {profile.n_samples} samples, {profile.n_features} features.",
        "task_type": f"Detected task: {profile.task_type.replace('_', ' ')}.",
        "imbalance": f"Class balance: {ir_msg}.",
        "models": (
            f"Recommended {', '.join(models)} — diverse mix of linear and ensemble "
            f"methods suited to this dataset size."
        ),
        "resampling": (
            f"Recommended resampling: {resampling or 'none'} "
            f"({'needed for imbalanced data' if resampling else 'data is balanced, none needed'})."
        ),
        "metric": f"Primary metric: {metric} — chosen based on task type and class balance.",
        "cv_strategy": (
            f"Validation: {cv.replace('_', ' ')} "
            f"{'— best for small datasets to maximise data usage.' if cv != 'holdout' else '— efficient for large datasets.'}"
        ),
        "search": search_msg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Transforms a DatasetProfile into a TrainingRecommendation.

    Usage::

        engine = RecommendationEngine()
        rec = engine.recommend(profile)
    """

    def recommend(self, profile: DatasetProfile) -> TrainingRecommendation:
        models = _recommend_models(profile)
        search_type, time_budget = _recommend_search_type(profile)
        split = _recommend_split(profile)
        k_folds = _recommend_k_folds(profile)
        cv = profile.recommended_cv_strategy

        # Compute minority_count from profile for precise SMOTE feasibility checks
        _minority_count: int | None = None
        if profile.minority_ratio is not None and profile.n_samples > 0:
            _minority_count = max(1, int(round(profile.minority_ratio * profile.n_samples)))

        imbalance_rec = recommend_imbalance_strategy(
            profile,
            minority_count=_minority_count,
            n_samples=profile.n_samples,
        )
        resampling = imbalance_rec.strategy
        apply_threshold = imbalance_rec.apply_threshold

        class_weight = _recommend_class_weight(profile, resampling)

        metrics = select_metrics(profile)
        reasoning = _build_reasoning(profile, models, resampling, metrics.primary, cv, search_type)

        zero_shot_hp = get_zero_shot_hyperparams(
            size_cat=profile.dataset_size_category,
            task_type=profile.task_type,
            imbalance_ratio=profile.imbalance_ratio,
            models=models,
        )

        warnings: list[str] = []
        if profile.has_missing_values and profile.missing_ratio > 0.3:
            warnings.append(
                f"High missing-value ratio ({profile.missing_ratio:.0%}). "
                "Consider imputation before training."
            )
        if profile.dimensionality_ratio > 0.5:
            warnings.append(
                f"High dimensionality (features/samples={profile.dimensionality_ratio:.2f}). "
                "Consider feature selection to reduce overfitting."
            )
        if profile.dataset_size_category == "tiny":
            warnings.append(
                "Very small dataset: results may have high variance. "
                "Prefer cross-validation over a simple train/test split."
            )
        # Surface any SMOTE/undersampling feasibility notes from imbalance handler
        for note in imbalance_rec.feasibility_notes:
            warnings.append(note)

        # Choose threshold strategy based on imbalance severity.
        # For severe/critical imbalance, maximize_f2 (beta=2) penalises false negatives more
        # heavily than false positives — i.e., it prioritises not missing minority cases.
        # For mild/moderate we default to maximize_f1 (balanced precision/recall).
        if imbalance_rec.severity in {"severe", "critical"} and apply_threshold:
            threshold_strategy = "maximize_f2"
        else:
            threshold_strategy = "maximize_f1"

        # Build the unified payload dict (ready for TrainingConfig.from_front)
        balancing_payload: dict[str, Any] = {
            "strategy": resampling or "none",
            "apply_threshold": apply_threshold,
            "threshold_strategy": threshold_strategy,
        }

        payload: dict[str, Any] = {
            "taskType": _map_task_type(profile.task_type),
            "models": models,
            "metrics": [metrics.primary] + metrics.secondary,
            "splitMethod": cv,
            "kFolds": k_folds,
            "shuffle": True,
            "trainRatio": split["trainRatio"],
            "valRatio": split["valRatio"],
            "testRatio": split["testRatio"],
            "searchType": search_type,
            "balancing": balancing_payload,
            "modelHyperparams": zero_shot_hp,
            "configMode": "manual",
        }

        return TrainingRecommendation(
            mode="recommendation",
            recommended_models=models,
            recommended_resampling=resampling,
            apply_threshold=apply_threshold,
            recommended_metric=metrics.primary,
            secondary_metrics=metrics.secondary,
            recommended_cv_strategy=cv,
            recommended_k_folds=k_folds,
            recommended_search_type=search_type,
            recommended_time_budget_s=time_budget,
            recommended_class_weight=class_weight,
            recommended_split=split,
            reasoning=reasoning,
            training_config_payload=payload,
            warnings=warnings,
        )


def _map_task_type(full_task: str) -> str:
    """Convert DatasetProfile.task_type → TrainingConfig.task_type."""
    if full_task == "regression":
        return "regression"
    return "classification"
