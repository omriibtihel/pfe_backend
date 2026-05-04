"""
RecommendationEngine — Phase 2
Transforms a DatasetProfile into a TrainingRecommendation that can be
sent to the frontend and/or directly executed via TrainingConfigBuilder.
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
    mode: str
    recommended_models: list[str]
    recommended_resampling: str | None
    apply_threshold: bool
    recommended_metric: str
    secondary_metrics: list[str]
    recommended_cv_strategy: str
    recommended_k_folds: int
    recommended_search_type: str
    recommended_time_budget_s: int | None
    recommended_class_weight: str | None
    recommended_split: dict[str, int]
    reasoning: dict[str, str]
    training_config_payload: dict[str, Any]
    recommended_power_transform: str = "none"
    recommended_scaling: str = "none"
    recommended_preprocessing: dict[str, Any] = field(default_factory=dict)
    recommended_column_configs: dict[str, dict[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _map_task_type(full_task: str) -> str:
    return "regression" if full_task == "regression" else "classification"


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

    if size == "large":
        return ["lightgbm", "logisticregression"]
    if size in {"tiny", "small"}:
        if ir > 5.0:
            return ["logisticregression", "lightgbm", "randomforest"]
        return ["logisticregression", "randomforest", "lightgbm"]
    if ir > 5.0:
        return ["lightgbm", "randomforest", "logisticregression"]
    return ["logisticregression", "randomforest", "lightgbm", "xgboost"]


def _recommend_search_type(profile: DatasetProfile) -> tuple[str, int | None]:
    size = profile.dataset_size_category
    if size == "tiny":
        return "grid", None
    if size in {"small", "medium"}:
        return "halving_random", None
    return "none", None


def _recommend_split(profile: DatasetProfile) -> dict[str, int]:
    if profile.recommended_cv_strategy != "holdout":
        return {"trainRatio": 80, "valRatio": 0, "testRatio": 0}
    return {"trainRatio": 70, "valRatio": 15, "testRatio": 15}


def _recommend_k_folds(profile: DatasetProfile) -> int:
    return 10 if profile.dataset_size_category in {"tiny", "small"} else 5


def _col_power_transform(col_stat: dict[str, Any]) -> str:
    """Non-normal (practical) → yeo_johnson. Normal → none."""
    return "none" if col_stat["is_normal_practical"] else "yeo_johnson"


def _col_scaling(col_stat: dict[str, Any]) -> str:
    """
    After yeo_johnson the distribution is ~Gaussian → standard is optimal.
    For already-normal data: robust if |sk| ≥ 1.0 (outliers), else standard.
    """
    if not col_stat["is_normal_practical"]:
        return "standard"          # post-transform data is ~Gaussian
    if col_stat["abs_skewness"] >= 1.0:
        return "robust"            # normal but with notable outlier influence
    return "standard"


def _col_imputation(col_stat: dict[str, Any]) -> str:
    if not col_stat["has_missing"]:
        return "none"
    return "median" if col_stat["abs_skewness"] >= 0.75 else "mean"


def _recommend_transforms(profile: DatasetProfile) -> tuple[str, str]:
    """
    Returns (power_transform, scaling) global defaults via majority vote.
    - power_transform : "none" | "yeo_johnson"
    - scaling         : "none" | "standard" | "robust" | "minmax" | "maxabs"
    """
    if profile.column_distribution:
        pt_counts: dict[str, int] = {}
        sc_counts: dict[str, int] = {}
        for stat in profile.column_distribution.values():
            p = _col_power_transform(stat)
            s = _col_scaling(stat)
            pt_counts[p] = pt_counts.get(p, 0) + 1
            sc_counts[s] = sc_counts.get(s, 0) + 1
        power = max(pt_counts, key=lambda k: pt_counts[k])
        scaling = max(sc_counts, key=lambda k: sc_counts[k])
        return power, scaling

    nnr, sk = profile.non_normal_ratio, profile.avg_skewness
    if nnr == 0.0 and sk == 0.0:
        return "none", "none"
    if nnr > 0:                       # non-normal columns present
        return "yeo_johnson", "standard"
    if sk >= 1.0:                     # normal but outlier-prone
        return "none", "robust"
    return "none", "standard"


def _recommend_imputation(profile: DatasetProfile) -> tuple[str, str]:
    """(numeric_imputation, categorical_imputation) — majority vote; none if no missing values."""
    if not profile.has_missing_values:
        return "none", "none"

    if profile.column_distribution:
        counts: dict[str, int] = {}
        for stat in profile.column_distribution.values():
            if stat["has_missing"]:
                s = _col_imputation(stat)
                counts[s] = counts.get(s, 0) + 1
        numeric = max(counts, key=lambda k: counts[k]) if counts else "mean"
    else:
        numeric = "median" if profile.avg_skewness >= 0.75 else "mean"

    return numeric, "most_frequent"


def _recommend_class_weight(profile: DatasetProfile, resampling: str | None) -> str | None:
    if profile.task_type == "regression":
        return None
    ir = profile.imbalance_ratio or 1.0
    if ir <= 1.5:
        return None
    return "balanced" if resampling in {None, "threshold_optimization"} else None


def _build_reasoning(
    profile: DatasetProfile,
    models: list[str],
    resampling: str | None,
    metric: str,
    cv: str,
    search_type: str,
    power_transform: str,
    scaling: str,
    num_imputation: str,
) -> dict[str, str]:
    size_label = {
        "tiny":   "very small (< 500 rows)",
        "small":  "small (500–2 000 rows)",
        "medium": "medium (2 000–50 000 rows)",
        "large":  "large (> 50 000 rows)",
    }.get(profile.dataset_size_category, profile.dataset_size_category)

    ir = profile.imbalance_ratio
    ir_msg = (
        "dataset is balanced" if ir is None or ir <= 1.5
        else f"imbalance ratio={ir:.1f}"
    )

    search_msg = {
        "none":          "No HPO (dataset is large; activate Random or Grid search if time allows).",
        "grid":          "Grid search (small dataset, exhaustive search is feasible).",
        "random":        "Randomised search (balanced speed/quality tradeoff).",
        "halving_random":"Successive Halving (explores many candidates, eliminates poor ones early — 3–10× faster than random search).",
    }.get(search_type, search_type)

    pt_msg = {
        "yeo_johnson": (
            f"Yeo-Johnson power transform — {profile.non_normal_ratio:.0%} non-normal features "
            f"(mean |skewness|={profile.avg_skewness:.2f}). "
            "Finds optimal λ per feature to achieve normality."
        ),
        "box_cox": (
            "Box-Cox power transform — requires strictly positive values (X > 0). "
            "Yeo-Johnson is safer unless all features are strictly positive."
        ),
        "none": "No power transform — distribution is mostly Gaussian.",
    }.get(power_transform, f"Power transform: {power_transform}.")

    scaling_msg = {
        "standard": (
            f"Standard (z-score) — mean |skewness|={profile.avg_skewness:.2f}. "
            "Optimal after power transform or for already-Gaussian data."
        ),
        "robust": (
            f"Robust (IQR) — mean |skewness|={profile.avg_skewness:.2f}. "
            "Normal distribution but with notable outliers; IQR is resistant to their influence."
        ),
        "minmax": "MinMax — bounds features to [0, 1].",
        "maxabs": "MaxAbs — scales to [-1, 1] without centring.",
        "none":   "No linear scaler.",
    }.get(scaling, f"Scaling: {scaling}.")

    if num_imputation == "none":
        imputation_msg = "No imputation needed (no missing values)."
    elif num_imputation == "median":
        imputation_msg = "Numeric imputation: median — skewed distribution makes mean less robust."
    else:
        imputation_msg = "Numeric imputation: mean — distribution is approximately symmetric."

    return {
        "dataset_size": f"Dataset is {size_label} — {profile.n_samples} samples, {profile.n_features} features.",
        "task_type":    f"Detected task: {profile.task_type.replace('_', ' ')}.",
        "imbalance":    f"Class balance: {ir_msg}.",
        "models":       f"Recommended {', '.join(models)} — diverse mix suited to this dataset size.",
        "resampling":   f"Resampling: {resampling or 'none'} ({'needed for imbalanced data' if resampling else 'data is balanced'}).",
        "metric":       f"Primary metric: {metric} — chosen based on task type and class balance.",
        "cv_strategy":  (
            f"Validation: {cv.replace('_', ' ')} "
            f"{'— best for small datasets.' if cv != 'holdout' else '— efficient for large datasets.'}"
        ),
        "search":       search_msg,
        "preprocessing": f"{pt_msg} {scaling_msg} {imputation_msg}",
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

    def recommend(self, profile: DatasetProfile, target_column: str = "") -> TrainingRecommendation:
        models       = _recommend_models(profile)
        search_type, time_budget = _recommend_search_type(profile)
        split        = _recommend_split(profile)
        k_folds      = _recommend_k_folds(profile)
        cv           = profile.recommended_cv_strategy

        minority_count: int | None = None
        if profile.minority_ratio is not None and profile.n_samples > 0:
            minority_count = max(1, int(round(profile.minority_ratio * profile.n_samples)))

        imbalance_rec  = recommend_imbalance_strategy(profile, minority_count=minority_count, n_samples=profile.n_samples)
        resampling     = imbalance_rec.strategy
        apply_threshold = imbalance_rec.apply_threshold

        power_transform, scaling   = _recommend_transforms(profile)
        num_imputation, cat_imputation = _recommend_imputation(profile)
        class_weight               = _recommend_class_weight(profile, resampling)
        metrics                    = select_metrics(profile)

        reasoning = _build_reasoning(
            profile, models, resampling, metrics.primary, cv, search_type,
            power_transform=power_transform, scaling=scaling, num_imputation=num_imputation,
        )

        zero_shot_hp = get_zero_shot_hyperparams(
            size_cat=profile.dataset_size_category,
            task_type=profile.task_type,
            imbalance_ratio=profile.imbalance_ratio,
            models=models,
        )

        # Per-column configs — only columns that differ from the global default
        col_configs: dict[str, dict[str, str]] = {}
        for col, stat in profile.column_distribution.items():
            entry: dict[str, str] = {}
            col_pt = _col_power_transform(stat)
            col_s  = _col_scaling(stat)
            col_i  = _col_imputation(stat)
            if col_pt != power_transform:
                entry["numericPowerTransform"] = col_pt
            if col_s != scaling:
                entry["numericScaling"] = col_s
            if col_i != num_imputation:
                entry["numericImputation"] = col_i
            if entry:
                col_configs[col] = entry

        threshold_strategy = (
            "maximize_f2" if imbalance_rec.severity in {"severe", "critical"} and apply_threshold
            else "maximize_f1"
        )

        preprocessing_payload: dict[str, Any] = {
            "defaults": {
                "numericPowerTransform":  power_transform,
                "numericScaling":         scaling,
                "numericImputation":      num_imputation,
                "categoricalImputation":  cat_imputation,
                "categoricalEncoding":    "none",
            },
            "columns": {col: dict(cfg) for col, cfg in col_configs.items()},
        }

        payload: dict[str, Any] = {
            "targetColumn":  target_column,
            "taskType":      _map_task_type(profile.task_type),
            "models":        models,
            "metrics":       [metrics.primary] + metrics.secondary,
            "splitMethod":   cv,
            "kFolds":        k_folds,
            "shuffle":       True,
            "trainRatio":    split["trainRatio"],
            "valRatio":      split["valRatio"],
            "testRatio":     split["testRatio"],
            "searchType":    search_type,
            "balancing": {
                "strategy":           resampling or "none",
                "apply_threshold":    apply_threshold,
                "threshold_strategy": threshold_strategy,
            },
            "preprocessing": preprocessing_payload,
            "modelHyperparams": zero_shot_hp,
            "configMode":    "manual",
        }

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
        warnings.extend(imbalance_rec.feasibility_notes)

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
            recommended_power_transform=power_transform,
            recommended_scaling=scaling,
            recommended_preprocessing=preprocessing_payload,
            recommended_column_configs=col_configs,
            warnings=warnings,
        )
