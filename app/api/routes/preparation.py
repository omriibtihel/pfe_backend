# app/api/routes/preparation.py
"""
Endpoints for ML preparation steps:
  - GET  /capabilities       — column capabilities (imputation, scaling, encoding options)
  - POST /validate           — preprocessing + split config validation & preview
  - POST /analyze-balance    — class imbalance analysis
  - POST /profile            — dataset profiling (task type, missing values, feature types…)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_db
from app.schemas.preparation import (
    BalanceAnalysisIn,
    BalanceAnalysisResponse,
    DatasetProfileIn,
    DatasetProfileOut,
    FeatureEngineeringColumnsOut,
    FeatureEngineeringPreviewIn,
    FeatureEngineeringPreviewOut,
    FeaturePreviewResult,
    FeatureTypesOut,
)
from app.schemas.training import TrainingConfigIn, TrainingValidateOut
from app.services.preparation_ml.balancing.profiler import DataProfile, profile_binary_dataset
from app.services.training.config.schema import get_training_capabilities
from app.services.data.loader import load_dataframe, resolve_dataset_path
from app.api.utils_shared.versions import load_version_df
from app.services.data.profiler import DatasetProfiler
from app.services.data.preview import PreviewValidationError, build_validation_preview
from app.services.training.utils import to_python_scalar
from app.services.training.config.validation import validate_training_config_payload
from app.services.preparation_ml.feature_engineering.transformer import (
    _eval_expression,
    _validate_ast,
    validate_feature_defs,
)

_profiler = DatasetProfiler()

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_binary_profile(df, target_column: str) -> DataProfile:
    if target_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TARGET_COLUMN_NOT_FOUND",
                "message": f"Target column '{target_column}' was not found in the selected dataset version.",
            },
        )

    df_clean = df[df[target_column].notna()].copy()
    if df_clean.empty:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EMPTY_TARGET_AFTER_NA_FILTER",
                "message": "No rows remain after dropping target NaN values.",
            },
        )

    y = np.asarray(df_clean[target_column].values)
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "BINARY_CLASSIFICATION_ONLY",
                "message": "Balancing analysis currently supports binary classification only.",
                "detected_classes": [str(v) for v in unique_classes],
            },
        )

    X = df_clean.drop(columns=[target_column])
    return profile_binary_dataset(y=y, X_shape=X.shape)


def _profile_to_response(profile: DataProfile) -> dict[str, Any]:
    return {
        "needs_balancing": bool(profile.needs_balancing),
        "imbalance_level": str(profile.imbalance_level.value),
        "imbalance_ratio": float(profile.imbalance_ratio),
        "minority_ratio": float(profile.minority_ratio),
        "n_samples": int(profile.n_samples),
        "dataset_scale": str(profile.scale),
        "majority": {
            "label": to_python_scalar(profile.majority.label),
            "count": int(profile.majority.count),
            "ratio": float(profile.majority.ratio),
            "role": str(profile.majority.role),
        },
        "minority": {
            "label": to_python_scalar(profile.minority.label),
            "count": int(profile.minority.count),
            "ratio": float(profile.minority.ratio),
            "role": str(profile.minority.role),
        },
        "summary_message": str(profile.summary_message),
        "warnings": [str(w) for w in profile.warnings],
        "metric_advice": [str(m) for m in profile.metric_advice],
        "available_strategies": [
            {
                "id": str(s.id),
                "label": str(s.label),
                "description": str(s.description),
                "impact": str(s.impact),
                "recommended": bool(s.recommended),
                "feasible": bool(s.feasible),
                "infeasible_reason": s.infeasible_reason,
            }
            for s in profile.available_strategies
        ],
        "default_recommendation": str(profile.default_recommendation),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/capabilities")
def get_capabilities(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    return get_training_capabilities()


@router.post("/validate", response_model=TrainingValidateOut)
def validate_training(
    project_id: int,
    payload: TrainingConfigIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    version_id = payload.datasetVersionId
    if version_id is None:
        raise HTTPException(status_code=400, detail="datasetVersionId is required for validation")

    try:
        dataset_path, dataset_version_id = resolve_dataset_path(db, project_id, version_id)
        df = load_dataframe(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload_dict = payload.model_dump()
    result = validate_training_config_payload(payload_dict, df)
    include_payload = payload_dict.get("include") if isinstance(payload_dict.get("include"), dict) else {}
    include_preview = bool(include_payload.get("preview", False))
    if include_preview:
        try:
            result.update(
                build_validation_preview(
                    payload_dict,
                    df,
                    dataset_version_id=int(dataset_version_id),
                )
            )
        except PreviewValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": str(exc),
                    "error_details": exc.error_details,
                },
            ) from exc

    if isinstance(result.get("normalized_config"), dict):
        result["normalized_config"]["datasetVersionId"] = int(dataset_version_id)
    return result


@router.post("/analyze-balance", response_model=BalanceAnalysisResponse)
def analyze_balance(
    project_id: int,
    payload: BalanceAnalysisIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    try:
        dataset_path, _ = resolve_dataset_path(db, project_id, payload.version_id)
        df = load_dataframe(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = _build_binary_profile(df, str(payload.target_column))
    return _profile_to_response(profile)


@router.post("/profile", response_model=DatasetProfileOut)
def profile_dataset(
    project_id: int,
    payload: DatasetProfileIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Phase 1 — Profile the dataset.
    Returns a DatasetProfile describing size, task type, imbalance,
    missing values, feature types, and initial recommendations.
    """
    ensure_project_owner(db, project_id, current_user.id)
    try:
        df = load_version_df(db, project_id, payload.version_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        prof = _profiler.profile(df, str(payload.target_column))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DatasetProfileOut(
        n_samples=prof.n_samples,
        n_features=prof.n_features,
        n_classes=prof.n_classes,
        task_type=prof.task_type,
        imbalance_ratio=prof.imbalance_ratio,
        minority_ratio=prof.minority_ratio,
        has_missing_values=prof.has_missing_values,
        missing_ratio=prof.missing_ratio,
        feature_types=FeatureTypesOut(**prof.feature_types),
        dimensionality_ratio=prof.dimensionality_ratio,
        dataset_size_category=prof.dataset_size_category,
        estimated_training_speed=prof.estimated_training_speed,
        recommended_cv_strategy=prof.recommended_cv_strategy,
        recommended_resampling=prof.recommended_resampling,
        recommended_metric=prof.recommended_metric,
        meta_features=prof.meta_features,
        non_normal_ratio=prof.non_normal_ratio,
        avg_skewness=prof.avg_skewness,
        highly_skewed_count=prof.highly_skewed_count,
        column_distribution=prof.column_distribution,
    )


@router.get("/feature-engineering/columns", response_model=FeatureEngineeringColumnsOut)
def get_feature_columns(
    project_id: int,
    version_id: int,
    target_column: str = "",
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return the list of feature columns available for expression authoring.
    The target column is excluded so users don't accidentally reference it.
    """
    ensure_project_owner(db, project_id, current_user.id)
    try:
        df = load_version_df(db, project_id, version_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    columns = [c for c in df.columns if c != target_column]
    return FeatureEngineeringColumnsOut(columns=columns, n_rows=len(df))


@router.post("/feature-engineering/preview", response_model=FeatureEngineeringPreviewOut)
def preview_feature_engineering(
    project_id: int,
    payload: FeatureEngineeringPreviewIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Evaluate feature engineering expressions on a small sample of the dataset.
    Returns a preview of computed values (or an error message) for each feature.
    """
    ensure_project_owner(db, project_id, current_user.id)

    try:
        df = load_version_df(db, project_id, payload.version_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Drop the target column from the feature namespace to avoid accidental leakage.
    feature_cols = [c for c in df.columns if c != payload.target_column]
    available_columns = feature_cols

    # Take a small sample for preview.
    n_rows = min(payload.n_rows, len(df))
    df_sample = df[feature_cols].head(n_rows).copy()

    results: list[FeaturePreviewResult] = []
    # Accumulate computed FE columns so later features can reference earlier ones.
    df_running = df_sample.copy()

    for feat in payload.features:
        if not feat.enabled:
            continue

        name = feat.name.strip()
        expr = feat.expression.strip()

        if not name or not expr:
            results.append(FeaturePreviewResult(
                name=name or "(unnamed)",
                expression=expr,
                preview_values=[],
                error="Name or expression is empty.",
            ))
            continue

        try:
            tree = _validate_ast(expr)
            series = _eval_expression(expr, df_running, tree=tree)
            # Add to running df so subsequent features can depend on this one.
            df_running[name] = series
            preview_values = [
                None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                for v in series.tolist()
            ]
            results.append(FeaturePreviewResult(
                name=name,
                expression=expr,
                preview_values=preview_values,
            ))
        except Exception as exc:
            results.append(FeaturePreviewResult(
                name=name,
                expression=expr,
                preview_values=[],
                error=str(exc),
            ))

    return FeatureEngineeringPreviewOut(
        available_columns=available_columns,
        results=results,
    )
