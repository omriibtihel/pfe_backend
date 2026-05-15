"""Reporting API — generates a patient-readable report from one prediction."""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_db
from app.models.project import Project
from app.models.training import TrainedModel
from app.schemas.reporting import ReportGenerateIn
from app.services.reporting import ReportContextBuilder, ReportService
from app.services.reporting.cache import compute_request_hash, report_cache
from app.services.reporting.feature_glossary import feature_glossary
from app.services.reporting.model_context import build_reporting_model_context
from app.services.reporting.service import (
    REPORT_SCHEMA_VERSION,
    build_default_router,
    stream_existing_report,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _compute_cf_for_report(
    trained_model: TrainedModel,
    input_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Best-effort DiCE call for report enrichment.

    Returns ``[]`` on any failure (DiCE not converging, missing background,
    regression task, etc.) so the caller can pass an empty list to the prompt
    and the LLM yields an empty ``what_to_change`` section. Never raises.
    """
    task_type = str(trained_model.task_type or "classification").lower()
    if task_type != "classification":
        return []
    artifacts = trained_model.artifacts_json if isinstance(trained_model.artifacts_json, dict) else {}
    schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names = schema.get("feature_names") or []
    # Only vary numeric features by default — categorical CF reads poorly in
    # patient narratives ("change sex from male to female") and DiCE handles
    # them less reliably. A future iteration can lift this restriction.
    numeric_features = (schema.get("numeric_features") or [])
    features_to_vary = [f for f in numeric_features if f in feature_names] or feature_names
    if not features_to_vary:
        return []
    try:
        from app.services.training.output.predictor import compute_counterfactual_for_row
        result = compute_counterfactual_for_row(
            trained_model,
            input_data,
            features_to_vary,
            n_counterfactuals=1,
        )
    except Exception as exc:
        logger.info("reporting.cf_skipped: %s", exc)
        return []
    items = result.get("counterfactual") if isinstance(result, dict) else None
    return items if isinstance(items, list) else []


def _resolve_model(db: Session, project: Project, model_id: int) -> TrainedModel:
    """Allow either the project's active model or any saved model in scope."""
    m = (
        db.query(TrainedModel)
        .filter(TrainedModel.id == model_id, TrainedModel.project_id == project.id)
        .first()
    )
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    artifacts = m.artifacts_json if isinstance(m.artifacts_json, dict) else {}
    is_saved = bool(getattr(m, "is_saved", None) or artifacts.get("saved", False))
    is_active = bool(
        project.active_model_id is not None and int(project.active_model_id) == int(m.id)
    )
    if not is_saved and not is_active:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_SAVED",
                "message": "Model is not saved for prediction.",
            },
        )
    return m


@router.post("/predictions/models/{model_id}/report/stream")
def generate_report_stream(
    project_id: int,
    model_id: int,
    payload: ReportGenerateIn,
    lang: Literal["fr", "en"] = Query("fr"),
    force: bool = Query(False, description="Bypass cache and regenerate."),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Stream a generated report as Server-Sent Events.

    Body carries the prediction outcome (already computed by ``/predict/json/explain``).
    Response stream emits one ``chunk`` event per section, then a final ``done``.

    A DB cache keyed by ``(model_id, lang, sha256(input + lime + glossary))``
    short-circuits regeneration. Pass ``force=true`` to bypass the cache.
    """
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _resolve_model(db, project, model_id)

    lime_items = [item.model_dump() for item in payload.lime]
    request_hash = compute_request_hash(
        input_data=payload.input_data,
        lime=lime_items,
        lang=lang,
    )

    # ── Cache hit ────────────────────────────────────────────────────────────
    if not force:
        cached = report_cache.get(
            db, model_id=m.id, lang=lang, request_hash=request_hash,
        )
        # Only honour caches whose schema_version matches the current code.
        # Bumps in REPORT_SCHEMA_VERSION (regression handling, new sections,
        # post-process changes) silently invalidate stale rows without an
        # operator action — regeneration replaces them in place.
        cached_content = cached.content_json if cached is not None else None
        cached_schema = (
            cached_content.get("schema_version")
            if isinstance(cached_content, dict)
            else None
        )
        if (
            cached is not None
            and isinstance(cached_content, dict)
            and cached_schema == REPORT_SCHEMA_VERSION
        ):
            cached_meta = dict(cached.generation_event or {})
            cached_meta["cached"] = True
            return StreamingResponse(
                stream_existing_report(cached_content, cached_meta),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # ── Cache miss → generate ────────────────────────────────────────────────
    model_context = build_reporting_model_context(
        m,
        raw_prediction=payload.prediction,
        lang=lang,
        glossary=feature_glossary,
    )
    # Best-effort DiCE — feeds the "what_to_change" section. Empty list when
    # CF can't be computed (regression, no flip found, missing background).
    cf_items = _compute_cf_for_report(m, payload.input_data)

    builder = ReportContextBuilder()
    context = builder.build(
        prediction_id=f"p{project.id}-m{m.id}-r{payload.row_index}",
        lang=lang,
        prediction_value=model_context.display_prediction,
        score=payload.score,
        lime_items=lime_items,
        model_name=str(m.model_type or ""),
        model_version=str(m.id),
        input_data=payload.input_data,
        class_context=model_context.class_context,
        model_quality=model_context.model_quality,
        dataset_summary=model_context.dataset_summary,
        feature_metadata=model_context.feature_metadata,
        task_type=str(m.task_type or "classification"),
        glossary=feature_glossary,
        threshold_value=model_context.threshold_value,
        counterfactual_items=cf_items,
    )

    service = ReportService(router=build_default_router())
    report, meta = service.generate(context)

    # Persist for next time. Concurrent inserts are handled inside the cache.
    try:
        report_cache.put(
            db,
            model_id=m.id,
            lang=lang,
            request_hash=request_hash,
            content=report,
            meta=meta,
            llm_provider=str(meta.get("provider", "")),
            llm_model=str(m.model_type or ""),
        )
    except Exception:
        # Never let a cache write failure break the response — the report is
        # already in hand. The cache is best-effort.
        db.rollback()

    return StreamingResponse(
        stream_existing_report(report, meta),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/predictions/models/{model_id}/report/cache")
def invalidate_report_cache(
    project_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Force regeneration of all cached reports for a given model.

    Plan §6 — the operator path when prompts change, the glossary is bumped,
    or a stakeholder reports an issue with a specific output.
    """
    project = ensure_project_owner(db, project_id, current_user.id)
    m = _resolve_model(db, project, model_id)
    deleted = report_cache.delete_for_model(db, model_id=m.id)
    return {"deleted": deleted, "model_id": m.id}
