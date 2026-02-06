# app/api/routes/training.py
from __future__ import annotations

import io
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.project import Project
from app.models.training import TrainingSession, TrainedModel
from app.schemas.training import TrainingConfigIn
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

    # ✅ metrics peuvent être: {"test": {...}, "train": {...}, ...}
    test_metrics = metrics_all.get("test")
    if isinstance(test_metrics, dict):
        metrics = test_metrics
    else:
        # fallback ancien format
        metrics = metrics_all

    def mget(k: str) -> float:
        v = metrics.get(k)
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    # train score/test score (1 valeur)
    primary = metrics_all.get("primary_score") or {}
    primary_metric = primary.get("metric")
    try:
        test_score = float(primary.get("value", 0.0))
    except Exception:
        test_score = 0.0

    train_score = 0.0
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_metric and primary_metric in train_block:
        try:
            train_score = float(train_block.get(primary_metric, 0.0))
        except Exception:
            train_score = 0.0

    # ✅ training time vient de metrics_all["training_time_sec"]
    try:
        training_time = float(metrics_all.get("training_time_sec", 0.0))
    except Exception:
        training_time = 0.0

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
            "mse": mget("mse"),
            "rmse": mget("rmse"),
            "mae": mget("mae"),
            "r2": mget("r2"),
        },
        "trainScore": train_score,
        "testScore": test_score,
        "featureImportance": artifacts.get("feature_importance", []),
        "confusionMatrix": artifacts.get("confusion_matrix", []),
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
        config_json=payload.model_dump(),
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
