# app/api/routes/training.py
from __future__ import annotations

import io
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Response
from fastapi.responses import StreamingResponse as _StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.dataset_version import DatasetVersion
from app.models.project import Project
from app.models.training import TrainingSession, TrainedModel
from app.schemas.training import (
    ActiveModelOut,
    AutoMLConfigIn,
    DatasetProfileOut,
    FeatureTypesOut,
    ManualPredictIn,
    RecommendIn,
    TrainingConfigIn,
    TrainingRecommendationOut,
    TrainingSessionOut,
)
from app.services.data.loader import load_dataframe, resolve_dataset_path
from app.services.data.profiler import DatasetProfiler
from app.services.training.notifier import training_notifier
from app.services.training.output.predictor import (
    predict_rows_json,
    predict_to_csv,
    predict_with_trained_model,
    read_uploaded_dataframe,
)
from app.services.training.intelligence.recommender import RecommendationEngine
from app.services.training.utils import to_python_scalar
from app.services.training.training_service import run_training_session, run_automl_session
from app.api.routes.preparation import _build_binary_profile

_profiler = DatasetProfiler()
_engine = RecommendationEngine()

router = APIRouter()


def _get_owned_project(db: Session, project_id: int, user_id: int) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return project


def _extract_primary_score(
    metrics_all: dict,
    metrics: dict,
    task_type: str,
    mget: Any,
) -> tuple[Optional[str], float, float, float]:
    """Returns (primary_metric, test_score, train_score, training_time)."""
    primary = metrics_all.get("primary_score") if isinstance(metrics_all.get("primary_score"), dict) else {}
    primary_metric = primary.get("metric")

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
            test_score = float(primary["value"])
        except Exception:
            pass
    elif primary_metric and isinstance(metrics, dict):
        try:
            test_score = float(metrics.get(primary_metric, 0.0))
        except Exception:
            pass

    train_score = 0.0
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_metric and primary_metric in train_block:
        try:
            train_score = float(train_block[primary_metric])
        except Exception:
            pass

    try:
        training_time = float(metrics_all.get("training_time_sec", 0.0))
    except Exception:
        training_time = 0.0

    return primary_metric, test_score, train_score, training_time


def _extract_automl_fields(metrics_all: dict, artifacts: dict) -> Optional[dict]:
    """Returns the automl result dict when the model was trained with AutoML, else None."""
    if not bool(metrics_all.get("automl", False)):
        return None
    automl_artifacts = artifacts.get("automl") if isinstance(artifacts.get("automl"), dict) else {}
    return {
        "isAutoML": True,
        "isBest": bool(automl_artifacts.get("is_best", True)),
        "bestEstimator": metrics_all.get("best_estimator"),
        "nIterations": metrics_all.get("n_iterations"),
        "totalTimeS": metrics_all.get("total_time_s"),
        "timeBudgetS": automl_artifacts.get("time_budget_s"),
        "metricOptimized": automl_artifacts.get("metric_optimized"),
    }


def _extract_cv_fields(
    metrics_all: dict,
    *,
    primary_metric: Optional[str],
    test_score: float,
) -> tuple[dict, float]:
    """Returns (cv_fields_dict, updated_test_score)."""
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

    # For CV: prefer holdout test score, fall back to CV mean val score.
    if is_cv and primary_metric:
        if has_holdout_test and isinstance(holdout_test_metrics, dict):
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

    cv_fields = {
        "isCV": is_cv,
        "nestedCv": bool(metrics_all.get("nested_cv", False)),
        "cvFoldResults": cv_fold_results,
        "cvSummary": cv_summary,
        "kFoldsUsed": k_folds_used,
        "hasHoldoutTest": has_holdout_test,
        "cvMeanMetrics": cv_mean_metrics,
        "cvTestMetrics": holdout_test_metrics,
    }
    return cv_fields, test_score


def _extract_tuning_fields(artifacts: dict) -> tuple[dict, Any]:
    """Returns (gridSearch_response_dict, gs_raw) for the tuning block."""
    gs = artifacts.get("grid_search") if isinstance(artifacts.get("grid_search"), dict) else {}
    best_params = gs.get("best_params") if isinstance(gs.get("best_params"), dict) else None
    cv_best_score = gs.get("best_score")
    cv_scoring = gs.get("scoring")

    # Normalize cv_results_summary: stored as mean_test_score, frontend expects mean_score.
    _raw = gs.get("cv_results_summary")
    cv_results_summary: Optional[list] = None
    if isinstance(_raw, list) and _raw:
        cv_results_summary = [
            {
                "params": dict(row.get("params") or {}),
                "mean_score": float(row.get("mean_score") or row.get("mean_test_score") or 0.0),
            }
            for row in _raw
            if isinstance(row, dict)
        ]

    grid_search_block = {
        "enabled": bool(gs.get("enabled", False)),
        "searchType": gs.get("search_type") or None,
        "cvBestScore": float(cv_best_score) if cv_best_score is not None else None,
        "cvScoring": str(cv_scoring) if cv_scoring else None,
        "bestParams": best_params,
        "cvSplits": int(gs.get("cv_splits", 0)) if gs.get("cv_splits") else None,
        "nCandidates": int(gs.get("n_candidates", 0)) if gs.get("n_candidates") else None,
        "cvResultsSummary": cv_results_summary,
    }
    return grid_search_block, gs


def _model_to_front_result(
    m: TrainedModel,
    *,
    is_saved: bool | None = None,
    is_active: bool = False,
) -> dict[str, Any]:
    metrics_all = m.metrics_json or {}
    artifacts = m.artifacts_json or {}
    if is_saved is None:
        is_saved = bool(getattr(m, "is_saved", None) or artifacts.get("saved", False))

    # Build the active metrics dict and a resolver closure.
    test_metrics = metrics_all.get("test")
    metrics = test_metrics if isinstance(test_metrics, dict) else metrics_all
    metrics_legacy = metrics.get("legacy_flat") if isinstance(metrics.get("legacy_flat"), dict) else {}
    metrics_global = metrics.get("global") if isinstance(metrics.get("global"), dict) else {}
    metrics_binary = metrics.get("binary") if isinstance(metrics.get("binary"), dict) else {}

    def mget(k: str) -> Optional[float]:
        for d in (metrics, metrics_legacy, metrics_global, metrics_binary):
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    task_type = str(m.task_type or "").lower()
    primary_metric, test_score, train_score, training_time = _extract_primary_score(
        metrics_all, metrics, task_type, mget
    )

    split_info = metrics_all.get("split_info") if isinstance(metrics_all.get("split_info"), dict) else None
    if split_info is None:
        split_info = artifacts.get("split_info") if isinstance(artifacts.get("split_info"), dict) else None

    grid_search_block, _ = _extract_tuning_fields(artifacts)
    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else None
    balancing = artifacts.get("balancing") if isinstance(artifacts.get("balancing"), dict) else None
    automl_result = _extract_automl_fields(metrics_all, artifacts)
    cv_fields, test_score = _extract_cv_fields(metrics_all, primary_metric=primary_metric, test_score=test_score)

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
            "pr_auc": mget("pr_auc"),
            "precision_pos": mget("precision_pos"),
            "recall_pos": mget("recall_pos"),
            "f1_pos": mget("f1_pos"),
            "f1_macro": mget("f1_macro"),
            "mse": mget("mse"),
            "rmse": mget("rmse"),
            "mae": mget("mae"),
            "r2": mget("r2"),
        },
        "trainScore": train_score,
        "testScore": test_score,
        "primaryMetric": primary_metric,
        "splitInfo": split_info,
        **cv_fields,
        "gridSearch": grid_search_block,
        "featureImportance": artifacts.get("feature_importance", []),
        "curves": artifacts.get("curves", None),
        "confusionMatrix": artifacts.get("confusion_matrix", []),
        "metricsDetailed": metrics,
        "metricsWarnings": metrics.get("warnings", []) if isinstance(metrics, dict) else [],
        "hyperparams": artifacts.get("hyperparams", None),
        "classDistribution": artifacts.get("class_distribution", None),
        "baselineMajority": artifacts.get("baseline_majority", None),
        "splitDebug": artifacts.get("split_debug", None),
        "preprocessing": artifacts.get("preprocessing", None),
        "balancing": balancing,
        "thresholding": thresholding,
        "trainingTime": training_time,
        "isSaved": bool(is_saved),
        "isActive": bool(is_active),
        "automl": automl_result,
    }


def _session_to_front_session(
    db: Session,
    s: TrainingSession,
    *,
    active_model_id: int | None = None,
) -> dict[str, Any]:
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
        "activeModelId": str(active_model_id) if active_model_id is not None else None,

        "config": s.config_json,
        "results": [
            _model_to_front_result(
                m,
                is_saved=bool((m.artifacts_json or {}).get("saved", False)),
                is_active=bool(active_model_id is not None and int(m.id) == int(active_model_id)),
            )
            for m in models
        ],

        "createdAt": s.created_at.isoformat() if s.created_at else None,
        "startedAt": s.started_at.isoformat() if s.started_at else None,
        "completedAt": s.finished_at.isoformat() if s.finished_at else None,
    }


def _extract_feature_names_for_prediction(artifacts: dict[str, Any]) -> list[str]:
    """
    Return the list of feature names the user should provide for manual prediction.

    Priority:
    1. artifacts["training_schema"]["feature_names"]  — most accurate (includes order)
    2. artifacts["columns"]["numeric"] + ["categorical"]  — fallback for older models
    3. Empty list                                         — nothing found

    AutoML interaction features (automl_feature_pairs) are excluded since the backend
    reconstructs them automatically at inference time.
    """
    training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names = training_schema.get("feature_names")

    if not isinstance(feature_names, list) or not feature_names:
        # Fallback: reconstruct from column type lists stored in artifacts
        columns_block = artifacts.get("columns") if isinstance(artifacts.get("columns"), dict) else {}
        numeric = columns_block.get("numeric") or []
        categorical = columns_block.get("categorical") or []
        if isinstance(numeric, list) and isinstance(categorical, list):
            feature_names = list(numeric) + [c for c in categorical if c not in numeric]
        else:
            feature_names = []

    # Exclude engineered AutoML interaction features — user shouldn't enter these manually.
    automl_pairs = artifacts.get("automl_feature_pairs")
    if isinstance(automl_pairs, list) and automl_pairs:
        engineered = {str(p["name"]) for p in automl_pairs if isinstance(p, dict) and "name" in p}
        feature_names = [f for f in feature_names if str(f) not in engineered]

    return [str(f) for f in feature_names]


def _extract_threshold(artifacts: dict[str, Any]) -> float:
    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else {}
    raw_t = thresholding.get("optimal_threshold")
    if raw_t is None:
        return 0.5
    try:
        return float(raw_t)
    except (TypeError, ValueError):
        return 0.5


def _copy_artifacts_json(artifacts_json: Any) -> dict[str, Any]:
    if isinstance(artifacts_json, dict):
        return dict(artifacts_json)
    return {}


def _get_saved_or_active_model_or_404(
    db: Session,
    *,
    project: Project,
    model_id: int,
) -> TrainedModel:
    model = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id, TrainedModel.project_id == project.id)
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    artifacts = model.artifacts_json if isinstance(model.artifacts_json, dict) else {}
    is_saved = bool(getattr(model, "is_saved", None) or artifacts.get("saved", False))
    is_active = bool(project.active_model_id is not None and int(project.active_model_id) == int(model.id))
    if not is_saved and not is_active:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_SAVED",
                "message": "Model is not saved for prediction.",
            },
        )
    return model


@router.post("/recommend", response_model=TrainingRecommendationOut)
def recommend_training_config(
    project_id: int,
    payload: RecommendIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Phase 2 — Generate a full TrainingRecommendation.
    Returns recommended models, balancing strategy, metric, CV strategy,
    HPO strategy, reasonings, and a ready-to-use training_config_payload.
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


@router.get("/sessions/{session_id}/events")
async def stream_training_events(
    project_id: int,
    session_id: int,
    last_seq: int = -1,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Phase 9 — Real-time training events via SSE.
    The client connects and receives Server-Sent Events as the training
    progresses (model complete, HPO progress, final complete, errors).

    Query params:
      last_seq: last event sequence number received (for resumption).
    """
    _get_owned_project(db, project_id, current_user.id)

    # Verify session belongs to project
    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
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


@router.post("/versions/{version_id}/sessions", status_code=status.HTTP_201_CREATED)
def start_training_for_version(
    project_id: int,
    version_id: int,
    payload: TrainingConfigIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

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
    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    return _session_to_front_session(db, s, active_model_id=active_model_id)


@router.post("/automl", status_code=status.HTTP_201_CREATED, response_model=TrainingSessionOut)
def start_automl_training(
    project_id: int,
    payload: AutoMLConfigIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Launch an AutoML (FLAML) training session. FLAML handles full pipeline: preprocessing, model selection, HPO."""
    _get_owned_project(db, project_id, current_user.id)

    dv = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == payload.datasetVersionId, DatasetVersion.project_id == project_id)
        .first()
    )
    if not dv:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    s = TrainingSession(
        project_id=project_id,
        dataset_version_id=payload.datasetVersionId,
        status="queued",
        progress=0,
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
    db.add(s)
    db.commit()
    db.refresh(s)

    background.add_task(run_automl_session, s.id)

    return {
        "id": s.id,
        "project_id": s.project_id,
        "dataset_version_id": s.dataset_version_id,
        "status": s.status,
        "progress": int(s.progress or 0),
        "config": s.config_json,
        "error_message": s.error_message,
        "created_at": s.created_at.isoformat() if s.created_at else "",
        "started_at": None,
        "finished_at": None,
    }


@router.get("/sessions")
def list_sessions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    sessions = (
        db.query(TrainingSession)
        .filter(TrainingSession.project_id == project_id)
        .order_by(TrainingSession.id.desc())
        .all()
    )
    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    return [_session_to_front_session(db, s, active_model_id=active_model_id) for s in sessions]


@router.get("/sessions/{session_id}")
def get_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    return _session_to_front_session(db, s, active_model_id=active_model_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Delete a training session and all its models (cascade)."""
    _get_owned_project(db, project_id, current_user.id)
    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete pkl files from disk before removing DB rows
    from pathlib import Path
    for m in s.models:
        try:
            arts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
            pkl = arts.get("model_pkl")
            if pkl:
                p = Path(pkl)
                if p.exists():
                    p.unlink()
        except Exception:
            pass  # Non-fatal

    db.delete(s)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/sessions/{session_id}")
def rename_session(
    project_id: int,
    session_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Rename a session by storing a `name` key in its config_json."""
    _get_owned_project(db, project_id, current_user.id)
    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    config = dict(s.config_json) if isinstance(s.config_json, dict) else {}
    config["name"] = name
    s.config_json = config
    db.commit()
    return {"id": str(s.id), "name": name}


@router.get("/saved-models")
def list_saved_models(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    active_id = int(project.active_model_id) if project.active_model_id is not None else None

    from sqlalchemy import or_
    query = db.query(TrainedModel).filter(TrainedModel.project_id == project_id)
    if active_id is not None:
        query = query.filter(
            or_(TrainedModel.is_saved == True, TrainedModel.id == active_id)  # noqa: E712
        )
    else:
        query = query.filter(TrainedModel.is_saved == True)  # noqa: E712

    models = query.order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc()).all()

    # Backward-compat: also include models saved only via artifacts_json (pre-migration rows).
    legacy_candidates = (
        db.query(TrainedModel)
        .filter(
            TrainedModel.project_id == project_id,
            TrainedModel.is_saved == False,  # noqa: E712
        )
        .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
        .all()
    )
    legacy_ids = {int(m.id) for m in models}
    for lm in legacy_candidates:
        artifacts_lm = lm.artifacts_json if isinstance(lm.artifacts_json, dict) else {}
        if bool(artifacts_lm.get("saved", False)) and int(lm.id) not in legacy_ids:
            models = list(models) + [lm]

    if not models:
        return []

    session_ids = list({int(m.session_id) for m in models})
    sessions = (
        db.query(TrainingSession)
        .filter(TrainingSession.id.in_(session_ids))
        .all()
        if session_ids
        else []
    )
    sessions_by_id = {int(s.id): s for s in sessions}

    version_ids = list(
        {
            int(s.dataset_version_id)
            for s in sessions
            if getattr(s, "dataset_version_id", None) is not None
        }
    )
    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id.in_(version_ids))
        .all()
        if version_ids
        else []
    )
    version_name_by_id = {int(v.id): str(v.name) for v in versions}

    out: list[dict[str, Any]] = []

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
        dataset_version_name = (
            version_name_by_id.get(dataset_version_id)
            if dataset_version_id is not None
            else None
        )

        feature_names = _extract_feature_names_for_prediction(artifacts)

        front_result = _model_to_front_result(m)

        out.append(
            {
                "id": str(m.id),
                "modelType": str(m.model_type),
                "taskType": str(m.task_type),
                "sessionId": str(m.session_id),
                "datasetVersionId": str(dataset_version_id) if dataset_version_id is not None else None,
                "datasetVersionName": dataset_version_name,
                "isActive": is_active,
                "isSaved": is_saved,
                "featureNames": [str(x) for x in feature_names],
                "threshold": _extract_threshold(artifacts),
                "trainedAt": m.created_at.isoformat() if m.created_at else "",
                "testScore": front_result.get("testScore"),
                "primaryMetric": front_result.get("primaryMetric"),
                "trainingTime": front_result.get("trainingTime"),
            }
        )

    out.sort(key=lambda item: str(item.get("trainedAt") or ""), reverse=True)
    out.sort(key=lambda item: not bool(item.get("isActive")))
    return out


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

    # Mark model as saved (column + artifacts mirror)
    artifacts = _copy_artifacts_json(m.artifacts_json)
    artifacts["saved"] = True
    m.artifacts_json = artifacts
    m.is_saved = True
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


@router.delete("/sessions/{session_id}/models/{model_id}/save", status_code=status.HTTP_204_NO_CONTENT)
def unsave_model(
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

    artifacts = _copy_artifacts_json(m.artifacts_json)
    artifacts["saved"] = False
    m.artifacts_json = artifacts
    m.is_saved = False
    db.add(m)

    if project.active_model_id is not None and int(project.active_model_id) == int(model_id):
        project.active_model_id = None
        db.add(project)

    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/sessions/{session_id}/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trained_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Permanently delete a TrainedModel record and its pkl file from disk."""
    from pathlib import Path

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

    # Unset active model if this one was active
    if project.active_model_id is not None and int(project.active_model_id) == int(model_id):
        project.active_model_id = None
        db.add(project)

    # Delete pkl file from disk
    artifacts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
    pkl_path_str = artifacts.get("model_pkl")
    if pkl_path_str:
        try:
            pkl_path = Path(str(pkl_path_str))
            if pkl_path.exists():
                pkl_path.unlink()
        except Exception:
            pass  # Non-fatal: DB record is deleted regardless

    db.delete(m)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


@router.post("/models/{model_id}/predict")
async def predict_with_saved_model_file(
    project_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        result = predict_with_trained_model(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/models/{model_id}/predict/json")
def predict_with_saved_model_json(
    project_id: int,
    model_id: int,
    payload: ManualPredictIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        result = predict_rows_json(m, payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/models/{model_id}/predict/export")
async def predict_with_saved_model_export_csv(
    project_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

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


@router.get("/sessions/{session_id}/download")
def download_results(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    s = (
        db.query(TrainingSession)
        .filter(TrainingSession.id == session_id, TrainingSession.project_id == project_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    active_model_id = int(project.active_model_id) if project.active_model_id is not None else None
    payload = _session_to_front_session(db, s, active_model_id=active_model_id)
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
    feature_names = _extract_feature_names_for_prediction(artifacts)

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
