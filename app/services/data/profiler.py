"""
DatasetProfiler — Phase 1
Analyses a raw DataFrame and produces a DatasetProfile used by the
RecommendationEngine.  Works for classification (binary / multiclass)
and regression.  Completely read-only: never mutates the input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .normality import NormalityAnalyzer


# ──────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetProfile:
    n_samples: int
    n_features: int
    n_classes: int | None                  # None for regression
    task_type: str                         # "binary_classification" | "multiclass_classification" | "regression"
    imbalance_ratio: float | None          # majority/minority count; None for regression or single class
    minority_ratio: float | None           # minority share in [0,1]; None for regression
    has_missing_values: bool
    missing_ratio: float                   # fraction of (row, col) cells that are NaN
    feature_types: dict[str, int]          # {"numeric": N, "categorical": N, "text": N}
    dimensionality_ratio: float            # n_features / n_samples
    dataset_size_category: str             # "tiny" | "small" | "medium" | "large"
    estimated_training_speed: str          # "fast" | "moderate" | "slow"
    recommended_cv_strategy: str          # "stratified_kfold" | "kfold" | "holdout"
    recommended_resampling: str | None    # None | one of BalancingStrategy values
    recommended_metric: str               # primary metric key
    meta_features: dict[str, Any] = field(default_factory=dict)
    # Distribution analysis — populated by NormalityAnalyzer
    non_normal_ratio: float = 0.0         # fraction of numeric cols that failed normality (practical)
    avg_skewness: float = 0.0             # mean |skewness| across tested numeric cols
    highly_skewed_count: int = 0          # number of cols with |skewness| ≥ 1.5
    columns_capped: bool = False          # True when numeric cols exceeded the 50-col cap
    columns_capped_count: int = 0         # number of cols that were dropped due to cap
    transform_suggestions: dict[str, str] = field(default_factory=dict)  # col → recommended_transform
    # Per-column distribution stats matching ColumnDistributionStat TypeScript interface
    column_distribution: dict[str, dict[str, Any]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _size_category(n: int) -> str:
    if n < 500:
        return "tiny"
    if n < 2_000:
        return "small"
    if n < 50_000:
        return "medium"
    return "large"


def _training_speed(size_cat: str, n_features: int) -> str:
    if size_cat in {"tiny", "small"}:
        return "fast"
    if size_cat == "medium" and n_features < 100:
        return "moderate"
    return "slow"


def _infer_task_type(y: pd.Series) -> tuple[str, int | None]:
    """Return (task_type, n_classes).  Uses dtype + cardinality heuristic."""
    if pd.api.types.is_float_dtype(y):
        unique_vals = y.dropna().unique()
        if len(unique_vals) > 20:
            return "regression", None
        n = len(unique_vals)
        if n == 2:
            return "binary_classification", 2
        return "multiclass_classification", n

    if pd.api.types.is_bool_dtype(y):
        return "binary_classification", 2

    if pd.api.types.is_integer_dtype(y):
        unique_vals = y.dropna().unique()
        n = len(unique_vals)
        if n == 2:
            return "binary_classification", 2
        if n <= 20:
            return "multiclass_classification", n
        return "regression", None

    # String / object / category
    unique_vals = y.dropna().unique()
    n = len(unique_vals)
    if n == 2:
        return "binary_classification", 2
    if n <= 50:
        return "multiclass_classification", n
    return "regression", None


def _imbalance_stats(y: pd.Series, task_type: str) -> tuple[float | None, float | None]:
    if task_type == "regression":
        return None, None
    counts = y.value_counts()
    if len(counts) < 2:
        return None, None
    minority = int(counts.min())
    majority = int(counts.max())
    total = int(counts.sum())
    ir = majority / minority if minority > 0 else float("inf")
    mr = minority / total if total > 0 else 0.0
    return ir, mr


def _feature_types(df: pd.DataFrame, target_col: str) -> dict[str, int]:
    features = df.drop(columns=[target_col], errors="ignore")
    numeric = int(features.select_dtypes(include=[np.number]).shape[1])
    _text_df = features.select_dtypes(include=["object", "string"])
    text = 0 if _text_df.empty else int(
        _text_df
        .apply(lambda s: s.dropna().str.split().str.len().median() or 0)
        .gt(5)
        .sum()
    )
    categorical = int(features.shape[1]) - numeric - text
    return {"numeric": numeric, "categorical": max(0, categorical), "text": text}


def _recommended_cv(size_cat: str, task_type: str, n_samples: int, imbalance_ratio: float | None) -> str:
    if size_cat == "large":
        return "holdout"
    if task_type in {"binary_classification", "multiclass_classification"}:
        return "stratified_kfold"
    return "kfold"


def _recommended_resampling(task_type: str, ir: float | None, minority_ratio: float | None) -> str | None:
    if task_type == "regression" or ir is None:
        return None
    if ir <= 1.5:
        return None
    if ir <= 3.0:
        return "class_weight"
    if ir <= 10.0:
        return "smote"
    return "smote_tomek"


def _recommended_metric(task_type: str, ir: float | None) -> str:
    if task_type == "regression":
        return "rmse"
    if ir is None or ir <= 1.5:
        return "roc_auc"
    if ir <= 5.0:
        return "f1"
    return "pr_auc"


def _missing_stats(df: pd.DataFrame) -> tuple[bool, float]:
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return False, 0.0
    missing = int(df.isnull().sum().sum())
    ratio = missing / total_cells
    return bool(missing > 0), float(ratio)


_COL_CAP = 50


def _distribution_stats(
    df: pd.DataFrame, target_col: str
) -> tuple[dict[str, dict[str, Any]], float, float, int, bool, int, dict[str, str]]:
    """
    Run NormalityAnalyzer on numeric feature columns (capped at _COL_CAP).

    Returns:
        column_distribution : {col: ColumnDistributionStat-compatible dict}
        non_normal_ratio    : fraction of tested cols that are not normal (practical)
        avg_skewness        : mean |skewness| across tested cols
        highly_skewed_count : cols with |skewness| ≥ 1.5
        columns_capped      : True when col count exceeded cap
        columns_capped_count: number of cols excluded
        transform_suggestions: {col: recommended_transform} for non-"none" transforms
    """
    features = df.drop(columns=[target_col], errors="ignore")
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        return {}, 0.0, 0.0, 0, False, 0, {}

    columns_capped = len(num_cols) > _COL_CAP
    columns_capped_count = max(0, len(num_cols) - _COL_CAP)
    num_cols = num_cols[:_COL_CAP]

    analyzer = NormalityAnalyzer()
    report = analyzer.analyze_columns(features[num_cols], num_cols)

    col_dist: dict[str, dict[str, Any]] = {}
    skewness_vals: list[float] = []
    non_normal_count = 0
    highly_skewed = 0
    transform_suggestions: dict[str, str] = {}

    for r in report.results:
        if r.error:
            continue
        has_missing = bool(features[r.col].isnull().any())
        abs_sk = abs(r.skewness) if r.skewness is not None else 0.0
        skewness_vals.append(abs_sk)
        if abs_sk >= 1.5:
            highly_skewed += 1
        if not r.is_normal_practical:
            non_normal_count += 1
        if r.recommended_transform != "none":
            transform_suggestions[r.col] = r.recommended_transform

        col_dist[r.col] = {
            "is_normal_practical": r.is_normal_practical,
            "skewness": r.skewness,
            "abs_skewness": abs_sk,
            "excess_kurtosis": r.excess_kurtosis,
            "n": r.n,
            "test_used": r.test_used,
            "p_value": r.p_value_corrected if r.p_value_corrected is not None else r.p_value,
            "has_missing": has_missing,
            "has_non_positive": r.has_non_positive,
            "skewness_level": r.skewness_level,
            "distribution_shape": r.distribution_shape,
            "recommended_transform": r.recommended_transform,
        }

    tested = len(skewness_vals)
    non_normal_ratio = non_normal_count / tested if tested > 0 else 0.0
    avg_skewness = float(np.mean(skewness_vals)) if skewness_vals else 0.0

    return (
        col_dist,
        non_normal_ratio,
        avg_skewness,
        highly_skewed,
        columns_capped,
        columns_capped_count,
        transform_suggestions,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Analyse a DataFrame and return a DatasetProfile.

    Usage::

        profiler = DatasetProfiler()
        profile = profiler.profile(df, target_column="diagnosis")
    """

    def profile(self, df: pd.DataFrame, target_column: str) -> DatasetProfile:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        df_clean = df.copy()
        y = df_clean[target_column].dropna()

        n_samples = int(len(y))
        feature_cols = [c for c in df_clean.columns if c != target_column]
        n_features = len(feature_cols)

        task_type, n_classes = _infer_task_type(y)
        ir, mr = _imbalance_stats(y, task_type)
        has_missing, missing_ratio = _missing_stats(df_clean)
        feat_types = _feature_types(df_clean, target_column)
        size_cat = _size_category(n_samples)
        dim_ratio = n_features / max(1, n_samples)
        speed = _training_speed(size_cat, n_features)
        cv_strategy = _recommended_cv(size_cat, task_type, n_samples, ir)
        resampling = _recommended_resampling(task_type, ir, mr)
        metric = _recommended_metric(task_type, ir)

        (
            col_dist,
            non_normal_ratio,
            avg_skewness,
            highly_skewed_count,
            columns_capped,
            columns_capped_count,
            transform_suggestions,
        ) = _distribution_stats(df_clean, target_column)

        meta: dict[str, Any] = {
            "class_distribution": (
                {str(k): int(v) for k, v in y.value_counts().items()}
                if task_type != "regression"
                else {}
            ),
            "target_dtype": str(y.dtype),
            "high_dimensionality": bool(dim_ratio > 0.5),
            "has_text_features": bool(feat_types["text"] > 0),
        }

        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            task_type=task_type,
            imbalance_ratio=ir,
            minority_ratio=mr,
            has_missing_values=has_missing,
            missing_ratio=float(missing_ratio),
            feature_types=feat_types,
            dimensionality_ratio=float(dim_ratio),
            dataset_size_category=size_cat,
            estimated_training_speed=speed,
            recommended_cv_strategy=cv_strategy,
            recommended_resampling=resampling,
            recommended_metric=metric,
            meta_features=meta,
            non_normal_ratio=non_normal_ratio,
            avg_skewness=avg_skewness,
            highly_skewed_count=highly_skewed_count,
            columns_capped=columns_capped,
            columns_capped_count=columns_capped_count,
            transform_suggestions=transform_suggestions,
            column_distribution=col_dist,
        )
