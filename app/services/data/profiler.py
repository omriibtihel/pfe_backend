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
        # Could still be integer-valued floats used as labels
        unique_vals = y.dropna().unique()
        if len(unique_vals) > 20:
            return "regression", None
        # Small number of float values → treat as categorical
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
        )
