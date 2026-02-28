# app/api/routes/training.py
from __future__ import annotations

import io
import json
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.project import Project
from app.models.training import TrainingSession, TrainedModel
from app.schemas.training import (
    ActiveModelOut,
    BalanceAnalysisIn,
    BalanceAnalysisResponse,
    ManualPredictIn,
    TrainingConfigIn,
    TrainingValidateOut,
)
from app.services.training.balancing.profiler import DataProfile, profile_binary_dataset
from app.services.training.config import get_training_capabilities
from app.services.training.dataset_loader import load_dataframe, resolve_dataset_path
from app.services.training.predictor import (
    predict_rows_json,
    predict_to_csv,
    predict_with_trained_model,
    read_uploaded_dataframe,
)
from app.services.training.preview import PreviewValidationError, build_validation_preview
from app.services.training.validation import validate_training_config_payload
from app.services.training_service import run_training_session

router = APIRouter()


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


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _profile_to_response(profile: DataProfile) -> dict[str, Any]:
    return {
        "needs_balancing": bool(profile.needs_balancing),
        "imbalance_level": str(profile.imbalance_level.value),
        "imbalance_ratio": float(profile.imbalance_ratio),
        "minority_ratio": float(profile.minority_ratio),
        "n_samples": int(profile.n_samples),
        "dataset_scale": str(profile.scale),
        "majority": {
            "label": _json_safe_scalar(profile.majority.label),
            "count": int(profile.majority.count),
            "ratio": float(profile.majority.ratio),
            "role": str(profile.majority.role),
        },
        "minority": {
            "label": _json_safe_scalar(profile.minority.label),
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


def _model_to_front_result(m: TrainedModel) -> dict[str, Any]:
    metrics_all = m.metrics_json or {}
    artifacts = m.artifacts_json or {}

    test_metrics = metrics_all.get("test")
    metrics = test_metrics if isinstance(test_metrics, dict) else metrics_all
    metrics_legacy = metrics.get("legacy_flat") if isinstance(metrics.get("legacy_flat"), dict) else {}
    metrics_global = metrics.get("global") if isinstance(metrics.get("global"), dict) else {}
    metrics_binary = metrics.get("binary") if isinstance(metrics.get("binary"), dict) else {}

    def mget(k: str) -> Optional[float]:
        v = metrics.get(k)
        if v is None:
            v = metrics_legacy.get(k)
        if v is None:
            v = metrics_global.get(k)
        if v is None:
            v = metrics_binary.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    task_type = str(m.task_type or "").lower()
    primary = metrics_all.get("primary_score") if isinstance(metrics_all.get("primary_score"), dict) else {}
    primary_metric = primary.get("metric")

    # Legacy format contains primary_score. Refactor format may not, so infer robustly.
    if not primary_metric:
        candidates = (
            ["f1", "accuracy", "roc_auc", "pr_auc"]
            if task_type == "classification"
            else ["r2", "rmse", "mae", "mse"]
        )
        for key in candidates:
            if key in metrics:
                primary_metric = key
                break
        if not primary_metric and isinstance(metrics, dict) and metrics:
            primary_metric = next(iter(metrics.keys()))

    test_score = 0.0
    if isinstance(primary, dict) and primary.get("value") is not None:
        try:
            test_score = float(primary.get("value"))
        except Exception:
            test_score = 0.0
    elif primary_metric and isinstance(metrics, dict):
        try:
            test_score = float(metrics.get(primary_metric, 0.0))
        except Exception:
            test_score = 0.0

    train_score = 0.0
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_metric and primary_metric in train_block:
        try:
            train_score = float(train_block.get(primary_metric, 0.0))
        except Exception:
            train_score = 0.0

    try:
        training_time = float(metrics_all.get("training_time_sec", 0.0))
    except Exception:
        training_time = 0.0

    split_info = metrics_all.get("split_info") if isinstance(metrics_all.get("split_info"), dict) else None
    if split_info is None:
        split_info = artifacts.get("split_info") if isinstance(artifacts.get("split_info"), dict) else None

    gs = artifacts.get("grid_search") if isinstance(artifacts.get("grid_search"), dict) else {}
    best_params = gs.get("best_params") if isinstance(gs.get("best_params"), dict) else None
    cv_best_score = gs.get("best_score")
    cv_scoring = gs.get("scoring")

    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else None
    balancing = artifacts.get("balancing") if isinstance(artifacts.get("balancing"), dict) else None

    # CV-specific
    is_cv = bool(metrics_all.get("cv", False))
    cv_summary = metrics_all.get("cv_summary") if isinstance(metrics_all.get("cv_summary"), dict) else None
    cv_fold_results = metrics_all.get("fold_results") if isinstance(metrics_all.get("fold_results"), list) else None
    k_folds_used = metrics_all.get("k_folds")
    has_holdout_test = bool(metrics_all.get("has_holdout_test", False))
    cv_mean_metrics = metrics_all.get("cv_mean") if isinstance(metrics_all.get("cv_mean"), dict) else None
    holdout_test_metrics = (
        metrics_all.get("holdout_test_metrics")
        if isinstance(metrics_all.get("holdout_test_metrics"), dict)
        else None
    )

    # For CV: testScore = holdout test score when available, else CV mean val score.
    # metrics_all["test"] already holds the right value (set by orchestrator), but
    # we derive testScore from primary_metric for consistency with holdout logic.
    if is_cv and primary_metric:
        if has_holdout_test and isinstance(holdout_test_metrics, dict):
            # Prefer the genuine holdout test score
            raw = holdout_test_metrics.get(primary_metric)
            if raw is None:
                lf = holdout_test_metrics.get("legacy_flat") if isinstance(holdout_test_metrics.get("legacy_flat"), dict) else {}
                gl = holdout_test_metrics.get("global") if isinstance(holdout_test_metrics.get("global"), dict) else {}
                raw = lf.get(primary_metric) or gl.get(primary_metric)
            if raw is not None:
                try:
                    test_score = float(raw)
                except Exception:
                    pass
        elif isinstance(cv_summary, dict):
            cv_mean = cv_summary.get("mean", {})
            if isinstance(cv_mean, dict) and primary_metric in cv_mean:
                try:
                    test_score = float(cv_mean[primary_metric])
                except Exception:
                    pass

    return {
        "id": str(m.id),
        "modelType": m.model_type,
        "status": "completed",
        "metrics": {
            "accuracy": mget("accuracy"),
            "precision": mget("precision"),
            "recall": mget("recall"),
            "f1": mget("f1"),
            "roc_auc": mget("roc_auc"),

            # imbalance-friendly
            "pr_auc": mget("pr_auc"),
            "precision_pos": mget("precision_pos"),
            "recall_pos": mget("recall_pos"),
            "f1_pos": mget("f1_pos"),

            # macro (useful even for binary imbalance)
            "f1_macro": mget("f1_macro"),

            # regression (only meaningful when taskType=regression)
            "mse": mget("mse"),
            "rmse": mget("rmse"),
            "mae": mget("mae"),
            "r2": mget("r2"),
        },
        "trainScore": train_score,
        "testScore": test_score,
        "primaryMetric": primary_metric,
        "splitInfo": split_info,

        # Cross-validation fields (null when splitMethod=holdout)
        "isCV": is_cv,
        "cvFoldResults": cv_fold_results,
        "cvSummary": cv_summary,
        "kFoldsUsed": k_folds_used,
        "hasHoldoutTest": has_holdout_test,
        "cvMeanMetrics": cv_mean_metrics,
        "cvTestMetrics": holdout_test_metrics,

        "gridSearch": {
            "enabled": bool(gs.get("enabled", False)),
            "cvBestScore": float(cv_best_score) if cv_best_score is not None else None,
            "cvScoring": str(cv_scoring) if cv_scoring else None,
            "bestParams": best_params,
            "cvSplits": int(gs.get("cv_splits", 0)) if gs.get("cv_splits") else None,
        },

        "featureImportance": artifacts.get("feature_importance", []),
        "confusionMatrix": artifacts.get("confusion_matrix", []),
        "metricsDetailed": metrics,
        "metricsWarnings": metrics.get("warnings", []) if isinstance(metrics, dict) else [],
        "hyperparams": artifacts.get("hyperparams", None),

        # Step 1 visibility
        "classDistribution": artifacts.get("class_distribution", None),
        "baselineMajority": artifacts.get("baseline_majority", None),
        "splitDebug": artifacts.get("split_debug", None),
        "preprocessing": artifacts.get("preprocessing", None),

        # Balancing decision/audit visibility
        "balancing": balancing,
        "thresholding": thresholding,

        "trainingTime": training_time,
    }


def _session_to_front_session(db: Session, s: TrainingSession) -> dict[str, Any]:
    models = (
        db.query(TrainedModel)
        .filter(TrainedModel.session_id == s.id)
        .order_by(TrainedModel.id.asc())
        .all()
    )

    return {
        "id": str(s.id),
        "projectId": str(s.project_id),
        "datasetVersionId": str(s.dataset_version_id) if s.dataset_version_id else None,

        "status": s.status,
        "progress": int(s.progress or 0),
        "currentModel": s.current_model,
        "errorMessage": s.error_message,

        "config": s.config_json,
        "results": [_model_to_front_result(m) for m in models],

        "createdAt": s.created_at.isoformat() if s.created_at else None,
        "startedAt": s.started_at.isoformat() if s.started_at else None,
        "completedAt": s.finished_at.isoformat() if s.finished_at else None,
    }


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


@router.post("/versions/{version_id}/sessions", status_code=status.HTTP_201_CREATED)
def start_training_for_version(
    project_id: int,
    version_id: int,
    payload: TrainingConfigIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

    if not payload.models:
        raise HTTPException(status_code=400, detail="No model selected")
    if not payload.metrics:
        raise HTTPException(status_code=400, detail="No metric selected")

    if str(payload.taskType).strip().lower() == "classification":
        try:
            dataset_path, _ = resolve_dataset_path(db, project_id, version_id)
            df = load_dataframe(dataset_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        profile = _build_binary_profile(df, str(payload.targetColumn))
        balancing_cfg = payload.balancing

        available_by_id = {str(s.id): s for s in profile.available_strategies}
        chosen = available_by_id.get(str(balancing_cfg.strategy))
        if chosen is not None and not bool(chosen.feasible):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "STRATEGY_NOT_FEASIBLE",
                    "message": f"Selected strategy '{balancing_cfg.strategy}' is not feasible for this dataset.",
                    "strategy": str(balancing_cfg.strategy),
                    "reason": chosen.infeasible_reason,
                    "fallback_suggestion": str(profile.default_recommendation),
                },
            )

    s = TrainingSession(
        project_id=project_id,
        dataset_version_id=version_id,
        status="queued",
        progress=0,
        config_json={
            **payload.model_dump(),
            "datasetVersionId": int(version_id),
        },
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    background.add_task(run_training_session, s.id)
    return _session_to_front_session(db, s)


@router.get("/sessions")
def list_sessions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

    sessions = (
        db.query(TrainingSession)
        .filter(TrainingSession.project_id == project_id)
        .order_by(TrainingSession.id.desc())
        .all()
    )
    return [_session_to_front_session(db, s) for s in sessions]


@router.get("/sessions/{session_id}")
def get_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    return _session_to_front_session(db, s)


@router.post("/sessions/{session_id}/models/{model_id}/save")
def save_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    m = (
        db.query(TrainedModel)
        .filter(
            TrainedModel.id == model_id,
            TrainedModel.session_id == session_id,
            TrainedModel.project_id == project_id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    # Mark model as saved
    artifacts = m.artifacts_json or {}
    artifacts["saved"] = True
    m.artifacts_json = artifacts
    db.add(m)

    # Set as the project's active model
    previous_model_id = project.active_model_id
    project.active_model_id = model_id
    db.add(project)

    db.commit()

    return {
        "success": True,
        "message": "Modèle enregistré et activé avec succès",
        "isNowActive": True,
        "modelId": model_id,
        "previousActiveModelId": previous_model_id,
    }


@router.post("/sessions/{session_id}/models/{model_id}/predict")
async def predict_with_model(
    project_id: int,
    session_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

    m = (
        db.query(TrainedModel)
        .filter(
            TrainedModel.id == model_id,
            TrainedModel.session_id == session_id,
            TrainedModel.project_id == project_id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        result = predict_with_trained_model(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.get("/sessions/{session_id}/download")
def download_results(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _get_owned_project(db, project_id, current_user.id)

    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    payload = _session_to_front_session(db, s)
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_session_{session_id}.json"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Active model — project-level prediction endpoints
# ──────────────────────────────────────────────────────────────────────────────

def _get_active_model_or_404(db: Session, project_id: int) -> TrainedModel:
    """Load and return the project's active TrainedModel, or raise 404."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.active_model_id is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NO_ACTIVE_MODEL",
                "message": (
                    "Aucun modèle actif pour ce projet. "
                    "Entraînez un modèle puis cliquez sur 'Sauvegarder' pour l'activer."
                ),
            },
        )
    m = db.query(TrainedModel).filter(TrainedModel.id == project.active_model_id).first()
    if m is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ACTIVE_MODEL_NOT_FOUND",
                "message": "Le modèle actif référencé n'existe plus.",
            },
        )
    return m


@router.get("/active-model", response_model=ActiveModelOut)
def get_active_model(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return metadata about the project's active model (used by the prediction UI)."""
    _get_owned_project(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    artifacts = m.artifacts_json or {}
    training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names = training_schema.get("feature_names") or []

    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else {}
    threshold = 0.5
    raw_t = thresholding.get("optimal_threshold")
    if raw_t is not None:
        try:
            threshold = float(raw_t)
        except (TypeError, ValueError):
            pass

    trained_at = m.created_at.isoformat() if m.created_at else ""

    return ActiveModelOut(
        modelId=m.id,
        sessionId=m.session_id,
        modelType=m.model_type,
        taskType=m.task_type,
        featureNames=feature_names,
        threshold=threshold,
        trainedAt=trained_at,
    )


@router.post("/predict")
async def predict_active_model_file(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Run inference using the project's active model. Accepts a CSV/JSON/Parquet file."""
    _get_owned_project(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        result = predict_with_trained_model(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/predict/json")
def predict_active_model_json(
    project_id: int,
    payload: ManualPredictIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Run inference using the project's active model. Accepts JSON rows (manual input mode)."""
    _get_owned_project(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        result = predict_rows_json(m, payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/predict/export")
async def predict_active_model_export_csv(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Run inference and return results as a downloadable CSV file."""
    _get_owned_project(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        csv_str = predict_to_csv(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV export failed: {str(e)}")

    from datetime import datetime, timezone as _tz
    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{m.model_type}_{ts}.csv"

    return StreamingResponse(
        io.BytesIO(csv_str.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
