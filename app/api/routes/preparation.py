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

from app.api.deps import get_db, get_current_user
from app.models.project import Project
from app.schemas.preparation import (
    BalanceAnalysisIn,
    BalanceAnalysisResponse,
    DatasetProfileIn,
    DatasetProfileOut,
    FeatureTypesOut,
)
from app.schemas.training import TrainingConfigIn, TrainingValidateOut
from app.services.preparation_ml.balancing.profiler import DataProfile, profile_binary_dataset
from app.services.training.config.schema import get_training_capabilities
from app.services.data.loader import load_dataframe, resolve_dataset_path
from app.services.data.profiler import DatasetProfiler
from app.services.data.preview import PreviewValidationError, build_validation_preview
from app.services.training.utils import to_python_scalar
from app.services.training.config.validation import validate_training_config_payload

_profiler = DatasetProfiler()

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_owned_project(db: Session, project_id: int, user_id: int) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return project


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
    _get_owned_project(db, project_id, current_user.id)
    return get_training_capabilities()


@router.post("/validate", response_model=TrainingValidateOut)
def validate_training(
    project_id: int,
    payload: TrainingConfigIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

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
    _get_owned_project(db, project_id, current_user.id)

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
    _get_owned_project(db, project_id, current_user.id)
    try:
        dataset_path, _ = resolve_dataset_path(db, project_id, payload.version_id)
        df = load_dataframe(dataset_path)
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
    )
