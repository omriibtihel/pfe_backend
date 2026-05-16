from __future__ import annotations

import io
from datetime import datetime, timezone as _tz

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.models.project import Project
from app.models.training import TrainedModel
from app.schemas.training import ActiveModelOut, ManualPredictIn
from app.schemas.prediction import (
    PredictionResultsExportIn,
    CounterfactualIn,
    CounterfactualOut,
    FeatureRangesOut,
)
from app.services.training.output.predictor import (
    predict_rows_json,
    predict_to_csv,
    predict_with_trained_model,
    predict_with_explanations,
    read_uploaded_dataframe,
    compute_counterfactual_for_row,
    get_feature_ranges_for_model,
)
from app.services.training.presenter import (
    extract_feature_names_for_prediction,
    extract_threshold,
)

ExplainMethod = str  # "shap" | "lime" | "both" — validated below

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Active model metadata
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/active-model", response_model=ActiveModelOut)
def get_active_model(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return metadata about the project's active model (used by the prediction UI)."""
    ensure_project_owner(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    artifacts = m.artifacts_json or {}
    feature_names = extract_feature_names_for_prediction(artifacts)
    threshold = extract_threshold(artifacts)
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


# ──────────────────────────────────────────────────────────────────────────────
# Active-model prediction (project-level shortcuts)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_active_model_file(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Run inference using the project's active model. Accepts a CSV/JSON/Parquet file."""
    ensure_project_owner(db, project_id, current_user.id)
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
    ensure_project_owner(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        result = predict_rows_json(m, payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/predict/json/explain")
def predict_active_model_json_with_shap(
    project_id: int,
    payload: ManualPredictIn,
    method: ExplainMethod = Query("shap", regex="^(shap|lime|both)$"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Run inference with local explanations per row.

    Each row in the response includes one or both of:
      - "shap": [{feature, shap_value, data}]
      - "lime": [{feature, contribution, data}]

    Controlled by the `method` query parameter (default: "shap" for backward
    compatibility). Explanation computation is best-effort: if a method fails
    the corresponding key is simply absent.
    """
    ensure_project_owner(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        import pandas as pd
        raw_df = pd.DataFrame(payload.rows)
        result = predict_with_explanations(m, raw_df, method=method)
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
    ensure_project_owner(db, project_id, current_user.id)
    m = _get_active_model_or_404(db, project_id)

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        csv_str = predict_to_csv(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV export failed: {str(e)}")

    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{m.model_type}_{ts}.csv"

    return StreamingResponse(
        io.BytesIO(csv_str.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Saved-model prediction (explicit model_id)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/models/{model_id}/predict")
async def predict_with_saved_model_file(
    project_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
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
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        result = predict_rows_json(m, payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/models/{model_id}/predict/json/explain")
def predict_with_saved_model_json_explain(
    project_id: int,
    model_id: int,
    payload: ManualPredictIn,
    method: ExplainMethod = Query("shap", regex="^(shap|lime|both)$"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Run inference with local explanations per row (saved model variant).

    Each row in the response includes one or both of:
      - "shap": [{feature, shap_value, data}]
      - "lime": [{feature, contribution, data}]

    Controlled by the `method` query parameter (default: "shap" for backward
    compatibility). Explanation computation is best-effort: if a method fails
    the corresponding key is simply absent.
    """
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        import pandas as pd
        raw_df = pd.DataFrame(payload.rows)
        result = predict_with_explanations(m, raw_df, method=method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return result


@router.post("/models/{model_id}/predict/results/export")
def export_prediction_results_as_csv(
    project_id: int,
    model_id: int,
    payload: PredictionResultsExportIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Convert already-computed prediction rows to a downloadable CSV file.
    Accepts the rows returned by any /predict endpoint and streams back a CSV.
    No re-inference is performed.
    """
    import csv

    ensure_project_owner(db, project_id, current_user.id)

    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows to export.")

    buf = io.StringIO()
    input_keys = list(payload.rows[0].input_data.keys())
    writer = csv.writer(buf)
    writer.writerow(["row_index", "prediction", "score", *input_keys])
    for row in payload.rows:
        writer.writerow([
            row.row_index,
            row.prediction,
            "" if row.score is None else row.score,
            *[row.input_data.get(k, "") for k in input_keys],
        ])

    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{payload.model_type}_{ts}.csv"

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/models/{model_id}/predict/export")
async def predict_with_saved_model_export_csv(
    project_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        content = await file.read()
        raw_df = read_uploaded_dataframe(file.filename or "", content)
        csv_str = predict_to_csv(m, raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV export failed: {str(e)}")

    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{m.model_type}_{ts}.csv"

    return StreamingResponse(
        io.BytesIO(csv_str.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session-model prediction (session_id + model_id)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/models/{model_id}/predict")
async def predict_with_model(
    project_id: int,
    session_id: int,
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

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


# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual explanations (saved-model variant)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/models/{model_id}/feature-ranges",
    response_model=FeatureRangesOut,
)
def get_model_feature_ranges(
    project_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return per-feature min/max/mean/std (numeric) or categories (categorical)
    used to bound interactive counterfactual sliders on the frontend.

    Source: training-time column_stats stored in artifacts.
    """
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        ranges = get_feature_ranges_for_model(m)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load feature ranges: {str(e)}",
        )

    return {"model_id": int(m.id), "ranges": ranges}


@router.post(
    "/models/{model_id}/predict/json/counterfactual",
    response_model=CounterfactualOut,
)
def predict_counterfactual_json(
    project_id: int,
    model_id: int,
    payload: CounterfactualIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Generate counterfactual explanations for a single patient row.

    Answers "what would need to change to flip this prediction?".
    Only features listed in features_to_vary are eligible for modification.

    Response:
      - original_prediction / original_score: current model output for the row
      - counterfactual: [{feature, original_value, suggested_value, delta}]
        sorted by |delta| descending, or null if no flip was found
      - message: human-readable fallback when counterfactual is null
    """
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _get_saved_or_active_model_or_404(db, project=project, model_id=model_id)

    try:
        result = compute_counterfactual_for_row(
            m,
            payload.row,
            payload.features_to_vary,
            n_counterfactuals=payload.n_counterfactuals,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Counterfactual computation failed: {str(e)}",
        )

    return result
