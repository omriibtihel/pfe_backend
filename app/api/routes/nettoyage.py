# [REFACTOR] Slimmed down: pandas/numpy and business logic moved to
# app/services/nettoyage/processing_ops.py; the raw db.query() in undo_last
# now goes through app/crud/nettoyage.get_last_operation. This module
# orchestrates HTTP only — auth checks, payload binding, exception mapping.
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.api.deps import ensure_project_owner, get_current_user, get_db
from app.api.utils_shared.datasets import get_dataset_or_404
from app.api.utils_shared.workspaces import (
    create_fresh_workspace_for_dataset,
    delete_workspace_dataset,
    find_active_dataset_workspace,
)
from app.crud import nettoyage as crud_nettoyage
from app.schemas.nettoyage import OperationIn, OperationOut
from app.services.nettoyage.df_utils import load_current_df, processed_path_for
from app.services.nettoyage.rebuild import rebuild_processed
from app.services.nettoyage import processing_ops as ops_service
from app.services.nettoyage.processing_ops import (
    AlertConfigOut,
    CleaningValidationError,
    SchemaDecisionIn,
    SchemaDecisionValidationError,
)

router = APIRouter()


# -----------------------------
# Validation wrappers
# -----------------------------
# [REFACTOR] Thin HTTP-aware shells around the pure-domain validators in
# processing_ops. Kept under their original names so existing tests that
# patch("app.api.routes.nettoyage._validate_cleaning_payload") still work.
def _validate_cleaning_payload(payload: OperationIn, df_current) -> str:
    try:
        return ops_service.validate_cleaning_payload(payload, df_current)
    except CleaningValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _validate_schema_decision(payload: SchemaDecisionIn, df_current) -> dict:
    try:
        return ops_service.validate_schema_decision(payload, df_current)
    except SchemaDecisionValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# Routes
# -----------------------------
@router.get("/datasets/{dataset_id}/nettoyage/alert-config", response_model=AlertConfigOut)
def get_alert_config(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    get_dataset_or_404(db, project_id, dataset_id)
    return AlertConfigOut()


@router.get("/datasets/{dataset_id}/nettoyage/operations", response_model=list[OperationOut])
def list_operations(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    return crud_nettoyage.list_operations(db, project_id, dataset_id)


@router.post("/datasets/{dataset_id}/nettoyage/operations", response_model=OperationOut)
def apply_operation(
    project_id: int,
    dataset_id: int,
    payload: OperationIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    df_current = load_current_df(ds.file_path, dataset_id)
    _validate_cleaning_payload(payload, df_current)

    # Step 1: write to disk FIRST — if this fails no DB record is created.
    try:
        op = crud_nettoyage.create_operation(
            db=db,
            project_id=project_id,
            dataset_id=dataset_id,
            user_id=current_user.id,
            op_type="cleaning",
            description=payload.description,
            columns=payload.columns or [],
            params=payload.params or {},
            flush_only=True,
        )
        rebuild_processed(db, project_id, dataset_id)
    except IOError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleaning could not be persisted to disk: {e}") from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Step 2: commit DB only after the file is safely on disk.
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        processed_path_for(ds.file_path, dataset_id).unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning written to disk but DB record failed — file cleaned up: {e}",
        ) from e

    db.refresh(op)
    return op


@router.post("/datasets/{dataset_id}/nettoyage/schema", response_model=OperationOut)
def apply_schema_decision(
    project_id: int,
    dataset_id: int,
    payload: SchemaDecisionIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)
    df_current = load_current_df(ds.file_path, dataset_id)

    params = _validate_schema_decision(payload, df_current)
    desc = ops_service.build_schema_decision_description(params)

    op = crud_nettoyage.create_operation(
        db=db,
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=current_user.id,
        op_type="schema",
        description=desc,
        columns=[params.get("column")] if params.get("column") else [],
        params=params,
    )

    db.refresh(op)
    return op


@router.get("/datasets/{dataset_id}/nettoyage/schema")
@router.get("/datasets/{dataset_id}/nettoyage/schema-state")
def get_schema_state(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    return crud_nettoyage.get_schema_state(db, project_id, dataset_id)


@router.post("/datasets/{dataset_id}/nettoyage/undo")
def undo_last(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    # [REFACTOR] Was an inline db.query(ProcessingOperation) — now via crud layer.
    last = crud_nettoyage.get_last_operation(db, project_id, dataset_id)
    if not last:
        return {"ok": False, "reason": "no_operations"}

    # Capturer l'identité avant suppression : après `db.delete(last)` + commit,
    # l'objet est détaché et accéder à `last.id` peut lever `DetachedInstanceError`.
    last_type = last.op_type
    last_id = last.id

    if last_type == "cleaning":
        # Transaction atomique : delete + rebuild (qui re-écrit le fichier sans
        # l'op supprimée), PUIS commit. Si le rebuild échoue, le rollback
        # restaure l'op — pas de perte de donnée et pas de divergence DB/fichier.
        db.delete(last)
        db.flush()

        try:
            rebuild_processed(db, project_id, dataset_id)
            db.commit()
        except IOError as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Undo failed at disk write — operation record preserved: {e}",
            ) from e
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Undo failed: {e}") from e
    else:
        # Les ops schema n'ont pas d'effet fichier — suppression directe.
        db.delete(last)
        db.commit()

    return {"ok": True, "undone_type": last_type, "undone_id": last_id}


@router.get("/datasets/{dataset_id}/nettoyage/preview")
def processing_preview(
    project_id: int,
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = load_current_df(dataset.file_path, dataset_id)
    return ops_service.build_preview_payload(df, page, page_size)


@router.get("/datasets/{dataset_id}/nettoyage/columns-meta")
def processing_columns_meta(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(dataset.file_path, dataset_id)

    payload = ops_service.build_columns_meta_payload(df)
    state = crud_nettoyage.get_schema_state(db, project_id, dataset_id)
    return ops_service.merge_kind_overrides(payload, state.get("kind_overrides", {}) or {})


@router.get("/datasets/{dataset_id}/nettoyage/column-distribution")
def get_column_distribution(
    project_id: int,
    dataset_id: int,
    column: str = Query(..., description="Column name"),
    max_bins: int = Query(default=20, ge=2, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(ds.file_path, dataset_id)
    try:
        return ops_service.compute_column_distribution(df, column, max_bins)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")


@router.get("/datasets/{dataset_id}/nettoyage/export")
def export_cleaned_dataset(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    ds = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(ds.file_path, dataset_id)

    tmp_path, download_name = ops_service.export_dataframe_to_csv(
        df, project_id, dataset_id, ds.original_name
    )
    return FileResponse(path=str(tmp_path), filename=download_name, media_type="text/csv")


@router.post("/datasets/{dataset_id}/nettoyage/workspace")
def create_dataset_workspace(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Crée (ou retourne) un workspace isolé pour le nettoyage du dataset brut.
    Le dataset source ne sera jamais modifié.
    """
    ensure_project_owner(db, project_id, current_user.id)
    ws = create_fresh_workspace_for_dataset(db, project_id, dataset_id, current_user.id)
    return {"workspace_dataset_id": ws.id}


@router.delete("/datasets/{dataset_id}/nettoyage/workspace")
def close_dataset_workspace(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Ferme explicitement le raw_workspace de ce dataset (cleanup côté client).

    Idempotent : retourne `ok=True` même si aucun workspace n'est actif. Permet
    au frontend d'appeler ça systématiquement en `unload` sans gérer le cas
    "déjà fermé".
    """
    ensure_project_owner(db, project_id, current_user.id)
    ws = find_active_dataset_workspace(db, project_id, dataset_id, current_user.id)
    if not ws:
        return {"ok": True, "closed": False}

    delete_workspace_dataset(db, ws)
    db.commit()
    return {"ok": True, "closed": True}


class SaveVersionBody(BaseModel):
    name: str = ""


@router.post("/datasets/{dataset_id}/nettoyage/save")
def save_cleaned_as_version(
    project_id: int,
    dataset_id: int,
    body: SaveVersionBody = Body(default={}),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    src = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(src.file_path, dataset_id)

    return ops_service.save_cleaned_as_version(
        db=db,
        project_id=project_id,
        dataset_id=dataset_id,
        src=src,
        df=df,
        body_name=body.name,
    )
