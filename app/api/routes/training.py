# app/api/routes/training.py
from __future__ import annotations

import io
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.project import Project
from app.models.training import TrainingSession, TrainedModel
from app.schemas.training import TrainingConfigIn, TrainingValidateOut
from app.services.training.config import get_training_capabilities
from app.services.training.dataset_loader import load_dataframe, resolve_dataset_path
from app.services.training.predictor import predict_with_trained_model, read_uploaded_dataframe
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
    smote = artifacts.get("smote") if isinstance(artifacts.get("smote"), dict) else None

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

        # NEW: robust behavior visibility
        "balancing": balancing,          # auto balance info (class_weight/spw decision)
        "thresholding": thresholding,    # tuned threshold + applied_on_test flag (holdout)
        "smote": smote,

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

    artifacts = m.artifacts_json or {}
    artifacts["saved"] = True
    m.artifacts_json = artifacts
    db.add(m)
    db.commit()

    return {"success": True, "message": "Modèle enregistré avec succès"}


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
