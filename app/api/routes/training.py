# app/api/routes/training.py
from __future__ import annotations

import io
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse as _StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_current_user_sse, get_db
from app.crud import training as crud_training
from app.models.training import TrainingSession
from app.schemas.training import (
    AutoMLConfigIn,
    DatasetProfileOut,
    FeatureTypesOut,
    RecommendIn,
    TrainingConfigIn,
    TrainingRecommendationOut,
)
from app.services.data.loader import load_dataframe, resolve_dataset_path
from app.services.data.profiler import DatasetProfiler
from app.services.preparation_ml.balancing.profiler import profile_binary_dataset
from app.services.training import presenter
from app.services.training.intelligence.recommender import RecommendationEngine
from app.services.training.notifier import training_notifier
from app.services.training.training_service import run_automl_session, run_training_session

_profiler = DatasetProfiler()
_engine = RecommendationEngine()

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Intelligent mode — recommendation
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=TrainingRecommendationOut)
def recommend_training_config(
    project_id: int,
    payload: RecommendIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    try:
        dataset_path, _ = resolve_dataset_path(db, project_id, payload.version_id)
        df = load_dataframe(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        prof = _profiler.profile(df, str(payload.target_column))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rec = _engine.recommend(prof)

    profile_out = DatasetProfileOut(
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

    return TrainingRecommendationOut(
        mode=rec.mode,
        recommended_models=rec.recommended_models,
        recommended_resampling=rec.recommended_resampling,
        apply_threshold=rec.apply_threshold,
        recommended_metric=rec.recommended_metric,
        secondary_metrics=rec.secondary_metrics,
        recommended_cv_strategy=rec.recommended_cv_strategy,
        recommended_k_folds=rec.recommended_k_folds,
        recommended_search_type=rec.recommended_search_type,
        recommended_time_budget_s=rec.recommended_time_budget_s,
        recommended_class_weight=rec.recommended_class_weight,
        recommended_split=rec.recommended_split,
        reasoning=rec.reasoning,
        training_config_payload=rec.training_config_payload,
        warnings=rec.warnings,
        profile=profile_out,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SSE — real-time training events
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/events")
async def stream_training_events(
    project_id: int,
    session_id: int,
    last_seq: int = -1,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_sse),
):
    ensure_project_owner(db, project_id, current_user.id)

    s = crud_training.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    return _StreamingResponse(
        training_notifier.subscribe(session_id, last_seq=last_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Launch training sessions
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/versions/{version_id}/sessions", status_code=status.HTTP_201_CREATED)
def start_training_for_version(
    project_id: int,
    version_id: int,
    payload: TrainingConfigIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

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

        target_col = str(payload.targetColumn)
        X = df.drop(columns=[target_col], errors="ignore")
        y = df[target_col] if target_col in df.columns else df.iloc[:, -1]
        balance_profile = profile_binary_dataset(y=y, X_shape=X.shape)

        balancing_cfg = payload.balancing
        available_by_id = {str(s.id): s for s in balance_profile.available_strategies}
        chosen = available_by_id.get(str(balancing_cfg.strategy))
        if chosen is not None and not bool(chosen.feasible):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "STRATEGY_NOT_FEASIBLE",
                    "message": f"Selected strategy '{balancing_cfg.strategy}' is not feasible for this dataset.",
                    "strategy": str(balancing_cfg.strategy),
                    "reason": chosen.infeasible_reason,
                    "fallback_suggestion": str(balance_profile.default_recommendation),
                },
            )

    s = crud_training.create_session(
        db,
        project_id=project_id,
        version_id=version_id,
        config_json={**payload.model_dump(), "datasetVersionId": int(version_id)},
    )
    background.add_task(run_training_session, s.id)

    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    models = crud_training.list_models_for_session(db, s.id)
    return presenter.session_to_front(s, models, active_model_id=active_model_id)


@router.post("/automl", status_code=status.HTTP_201_CREATED)
def start_automl_training(
    project_id: int,
    payload: AutoMLConfigIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Launch an AutoML (FLAML) training session."""
    ensure_project_owner(db, project_id, current_user.id)

    dv = crud_training.get_version(db, payload.datasetVersionId, project_id)
    if not dv:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    s = crud_training.create_session(
        db,
        project_id=project_id,
        version_id=payload.datasetVersionId,
        config_json={
            "datasetVersionId": payload.datasetVersionId,
            "targetColumn": payload.targetColumn,
            "taskType": payload.taskType,
            "timeBudget": payload.timeBudget,
            "metric": payload.metric,
            "testRatio": payload.testRatio,
            "positiveLabel": payload.positiveLabel,
            "configMode": "automl",
        },
    )
    background.add_task(run_automl_session, s.id)

    models = crud_training.list_models_for_session(db, s.id)
    return presenter.session_to_front(s, models)


# ──────────────────────────────────────────────────────────────────────────────
# Session CRUD
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/sessions")
def list_sessions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    sessions = crud_training.list_sessions_for_project(db, project_id)
    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    return [
        presenter.session_to_front(s, crud_training.list_models_for_session(db, s.id), active_model_id=active_model_id)
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
def get_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    s = crud_training.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    models = crud_training.list_models_for_session(db, s.id)
    return presenter.session_to_front(s, models, active_model_id=active_model_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    s = crud_training.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    crud_training.delete_session_and_files(db, s)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/sessions/{session_id}")
def rename_session(
    project_id: int,
    session_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    s = crud_training.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    crud_training.rename_session(db, s, name)
    return {"id": str(s.id), "name": name}


# ──────────────────────────────────────────────────────────────────────────────
# Saved models
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/saved-models")
def list_saved_models(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    active_id = int(project.active_model_id) if project.active_model_id is not None else None

    models, sessions_by_id, version_name_by_id = crud_training.get_saved_models_with_context(
        db, project_id, active_id
    )
    if not models:
        return []

    out = []
    for m in models:
        artifacts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
        is_saved = bool(getattr(m, "is_saved", None) or artifacts.get("saved", False))
        is_active = bool(active_id is not None and int(m.id) == active_id)
        if not is_saved and not is_active:
            continue

        session = sessions_by_id.get(int(m.session_id))
        dataset_version_id = (
            int(session.dataset_version_id)
            if session is not None and getattr(session, "dataset_version_id", None) is not None
            else None
        )

        front_result = presenter.model_to_front_result(m)
        out.append({
            "id": str(m.id),
            "modelType": str(m.model_type),
            "taskType": str(m.task_type),
            "sessionId": str(m.session_id),
            "datasetVersionId": str(dataset_version_id) if dataset_version_id is not None else None,
            "datasetVersionName": version_name_by_id.get(dataset_version_id) if dataset_version_id else None,
            "isActive": is_active,
            "isSaved": is_saved,
            "featureNames": presenter.extract_feature_names_for_prediction(artifacts),
            "threshold": presenter.extract_threshold(artifacts),
            "trainedAt": m.created_at.isoformat() if m.created_at else "",
            "testScore": front_result.get("testScore"),
            "primaryMetric": front_result.get("primaryMetric"),
            "trainingTime": front_result.get("trainingTime"),
        })

    out.sort(key=lambda item: str(item.get("trainedAt") or ""), reverse=True)
    out.sort(key=lambda item: not bool(item.get("isActive")))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model save / unsave / delete
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/models/{model_id}/save")
def save_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    m = crud_training.get_model(db, model_id, session_id=session_id, project_id=project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    previous_active_model_id = crud_training.save_model(db, m, project)
    return {
        "success": True,
        "message": "Modèle enregistré et activé avec succès",
        "isNowActive": True,
        "modelId": model_id,
        "previousActiveModelId": previous_active_model_id,
    }


@router.delete("/sessions/{session_id}/models/{model_id}/save", status_code=status.HTTP_204_NO_CONTENT)
def unsave_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    m = crud_training.get_model(db, model_id, session_id=session_id, project_id=project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    crud_training.unsave_model(db, m, project)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/sessions/{session_id}/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trained_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    m = crud_training.get_model(db, model_id, session_id=session_id, project_id=project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    crud_training.delete_model_and_file(db, m, project)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ──────────────────────────────────────────────────────────────────────────────
# Download session results
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/download")
def download_results(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    s = crud_training.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    models = crud_training.list_models_for_session(db, s.id)
    payload = presenter.session_to_front(s, models, active_model_id=active_model_id)
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    return _StreamingResponse(
        io.BytesIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_session_{session_id}.json"'},
    )
