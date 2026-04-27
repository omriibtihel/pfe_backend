# app/api/routes/processing.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import json

from pathlib import Path
from uuid import uuid4

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.services.data.column_inference import infer_kind_for_series, detect_parasites, numeric_extra_stats
from app.api.utils_shared.datasets import get_dataset_or_404
from app.api.utils_shared.workspaces import create_fresh_workspace_for_dataset
from app.services.nettoyage.df_utils import load_current_df, processed_path_for
from app.schemas.nettoyage import OperationIn, OperationOut
from app.crud import nettoyage as crud_nettoyage
from app.services.nettoyage.rebuild import rebuild_processed
from app.models.dataset_version import DatasetVersion
from app.models.processing_operation import ProcessingOperation
from app.core.config import PROJECTS_PATH

router = APIRouter()


# -----------------------------
# Preview payload
# -----------------------------
def df_preview_payload(df: pd.DataFrame, page: int, page_size: int):
    page = max(1, page)
    page_size = min(max(1, page_size), 200)

    start = (page - 1) * page_size
    end = start + page_size

    chunk = df.iloc[start:end].copy()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.astype(object)
    chunk = chunk.where(pd.notnull(chunk), None)

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "rows": chunk.to_dict(orient="records"),
        "page": page,
        "page_size": page_size,
        "total_rows": int(df.shape[0]),
    }


# -----------------------------
# Columns meta
# -----------------------------


def _columns_meta_payload(df: pd.DataFrame):
    cols = [str(c) for c in df.columns]
    total_rows = int(df.shape[0])

    out_cols = []
    counts = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "binary": 0,
        "text": 0,
        "id": 0,
        "other": 0,
    }

    for c in cols:
        s = df[c]
        s2 = s.replace([np.inf, -np.inf], np.nan)

        missing = int(pd.isna(s2).sum())
        non_null = s2.dropna()
        unique = int(non_null.nunique(dropna=True))

        inferred, _ = infer_kind_for_series(c, s2)
        kind = inferred if inferred in counts else "other"
        counts[kind] += 1

        try:
            sample_vals = non_null.astype(str).head(5).tolist()
        except Exception:
            sample_vals = []

        from app.services.data.column_inference import _ALL_NONE
        extra = (
            numeric_extra_stats(non_null)
            if pd.api.types.is_numeric_dtype(s2)
            else dict(_ALL_NONE)
        )

        out_cols.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "kind": kind,
                "inferred_kind": inferred,
                "override_kind": None,
                "missing": missing,
                "unique": unique,
                "total": total_rows,
                "sample": sample_vals,
                "parasites": detect_parasites(s2),
                **extra,
            }
        )

    return {"columns": out_cols, "counts": counts, "total_rows": total_rows}


# -----------------------------
# CLEANING contract
# -----------------------------
ALLOWED_CLEANING_ACTIONS = {
    "drop_columns",
    "drop_duplicates",
    "drop_empty_rows",
    "drop_empty_cols",
    "rename_columns",
    "strip_whitespace",
    "substitute_values",
}


def _validate_cleaning_payload(payload: OperationIn, df_current: pd.DataFrame) -> str:
    if payload.type != "cleaning":
        raise HTTPException(
            status_code=400,
            detail=(
                "Ce module est limité au nettoyage (cleaning). "
                "Les étapes ML (imputation/encoding/normalization) se font dans l'entraînement (après split)."
            ),
        )

    params = payload.params or {}
    action = params.get("action")
    if action not in ALLOWED_CLEANING_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Cleaning: params.action doit être dans {sorted(ALLOWED_CLEANING_ACTIONS)}",
        )

    existing = set(map(str, df_current.columns))
    cols = [str(c) for c in (payload.columns or [])]
    missing = [c for c in cols if c not in existing]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes introuvables: {missing}")

    if action == "rename_columns":
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise HTTPException(status_code=400, detail="rename_columns: params.mapping doit être un dict non vide.")

        old_cols = [str(k) for k in mapping.keys()]
        old_missing = [c for c in old_cols if c not in existing]
        if old_missing:
            raise HTTPException(status_code=400, detail=f"rename_columns: colonnes introuvables: {old_missing}")

        new_names = [str(v) for v in mapping.values()]
        if len(set(new_names)) != len(new_names):
            raise HTTPException(status_code=400, detail="rename_columns: deux colonnes ne peuvent pas avoir le même nouveau nom.")

    if action == "drop_columns" and not cols:
        raise HTTPException(status_code=400, detail="drop_columns: payload.columns ne doit pas être vide.")

    # ✅ validate substitute_values
    if action == "substitute_values":
        if len(cols) != 1:
            raise HTTPException(status_code=400, detail="substitute_values: payload.columns doit contenir exactement 1 colonne.")
        treat_from_as_null = bool(params.get("treat_from_as_null", False))
        if not treat_from_as_null and params.get("from_value", None) is None:
            raise HTTPException(
                status_code=400,
                detail="substitute_values: 'from_value' est requis (ou coche treat_from_as_null=true).",
            )
        # case_sensitive optional; if present must be bool-ish
        if "case_sensitive" in params and not isinstance(params.get("case_sensitive"), (bool, int)):
            raise HTTPException(status_code=400, detail="substitute_values: 'case_sensitive' doit être un bool.")

    return action


# -----------------------------
# Alert configuration
# -----------------------------

class AlertConfigOut(BaseModel):
    missing_high: float = 0.20
    missing_low: float = 0.05
    high_cardinality_ratio: float = 0.90
    high_cardinality_min_uniq: int = 10
    outlier_high: float = 0.15
    outlier_moderate: float = 0.05
    skewness: float = 2.0


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


# -----------------------------
# Schema decisions persistence (unchanged)
# -----------------------------
ALLOWED_KINDS = {"numeric", "categorical", "datetime", "binary", "text", "id", "other"}
ALLOWED_SCHEMA_ACTIONS = {"set_kind", "clear_kind", "verify_categorical", "dismiss_alert"}


class SchemaDecisionIn(BaseModel):
    schema_action: str
    column: str | None = None
    kind: str | None = None
    alert_key: str | None = None
    verified: bool | None = None
    dismissed: bool | None = None


def _validate_schema_decision(payload: SchemaDecisionIn, df_current: pd.DataFrame) -> dict:
    action = (payload.schema_action or "").strip()
    if action not in ALLOWED_SCHEMA_ACTIONS:
        raise HTTPException(status_code=400, detail=f"schema_action doit être dans {sorted(ALLOWED_SCHEMA_ACTIONS)}")

    existing = set(map(str, df_current.columns))

    if action in {"set_kind", "clear_kind", "verify_categorical"}:
        if not payload.column or not payload.column.strip():
            raise HTTPException(status_code=400, detail="schema: 'column' est requis.")
        if payload.column not in existing:
            raise HTTPException(status_code=400, detail=f"schema: colonne introuvable: {payload.column}")

    if action == "set_kind":
        k = (payload.kind or "").strip().lower()
        if k not in ALLOWED_KINDS:
            raise HTTPException(status_code=400, detail=f"schema: 'kind' doit être dans {sorted(ALLOWED_KINDS)}")

    if action == "verify_categorical":
        if payload.verified is None:
            raise HTTPException(status_code=400, detail="schema: 'verified' est requis pour verify_categorical.")

    if action == "dismiss_alert":
        if not payload.alert_key or not payload.alert_key.strip():
            raise HTTPException(status_code=400, detail="schema: 'alert_key' est requis pour dismiss_alert.")
        if payload.dismissed is None:
            raise HTTPException(status_code=400, detail="schema: 'dismissed' est requis pour dismiss_alert.")

    params: dict = {"schema_action": action}
    if payload.column:
        params["column"] = payload.column
    if payload.kind:
        params["kind"] = (payload.kind or "").strip().lower()
    if payload.alert_key:
        params["alert_key"] = payload.alert_key
    if payload.verified is not None:
        params["verified"] = bool(payload.verified)
    if payload.dismissed is not None:
        params["dismissed"] = bool(payload.dismissed)
    return params


# -----------------------------
# Routes
# -----------------------------
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
            flush_only=True,  # staged in session, not yet committed
        )
        rebuild_processed(db, project_id, dataset_id)
    except IOError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleaning could not be persisted to disk: {e}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2: commit DB only after the file is safely on disk.
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        processed_path_for(ds.file_path, dataset_id).unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning written to disk but DB record failed — file cleaned up: {e}",
        )

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
    action = params["schema_action"]

    if action == "set_kind":
        desc = f"Schema: {params.get('column')} → {params.get('kind')}"
    elif action == "clear_kind":
        desc = f"Schema: clear override ({params.get('column')})"
    elif action == "verify_categorical":
        v = params.get("verified")
        desc = (
            f"Schema: verified categorical ({params.get('column')})"
            if v is True
            else f"Schema: unverified categorical ({params.get('column')})"
        )
    else:
        d = params.get("dismissed")
        desc = (
            f"Schema: dismissed alert ({params.get('alert_key')})"
            if d is True
            else f"Schema: undismissed alert ({params.get('alert_key')})"
        )

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

    last = (
        db.query(ProcessingOperation)
        .filter_by(project_id=project_id, dataset_id=dataset_id)
        .order_by(ProcessingOperation.created_at.desc())
        .first()
    )
    if not last:
        return {"ok": False, "reason": "no_operations"}

    if last.op_type == "cleaning":
        # Step 1: stage the deletion then rebuild the file WITHOUT this operation.
        # If the disk write fails, rollback restores the record — no data loss.
        db.delete(last)
        db.flush()

        try:
            rebuild_processed(db, project_id, dataset_id)
        except IOError as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Undo failed at disk write — operation record preserved: {e}",
            )

        # Step 2: commit only after the file is on disk and consistent.
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Undo written to disk but DB cleanup failed: {e}",
            )
    else:
        # Schema operations have no file side-effect — direct delete is safe.
        db.delete(last)
        db.commit()

    return {"ok": True, "undone_type": last.op_type, "undone_id": last.id}


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
    return df_preview_payload(df, page, page_size)


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

    payload = _columns_meta_payload(df)

    state = crud_nettoyage.get_schema_state(db, project_id, dataset_id)
    overrides = state.get("kind_overrides", {}) or {}

    for col in payload.get("columns", []):
        name = col.get("name")
        if name in overrides:
            col["override_kind"] = overrides[name]
            col["kind"] = overrides[name]

    return payload


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
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    series = df[column].dropna()
    total = len(series)
    n_unique = series.nunique()

    if n_unique <= max_bins or not pd.api.types.is_numeric_dtype(series):
        counts = series.astype(str).value_counts().head(max_bins)
        bars = [{"label": str(lbl), "count": int(cnt)} for lbl, cnt in counts.items()]
        return {"type": "categorical", "column": column, "total": total, "bars": bars}

    hist_counts, edges = np.histogram(series.astype(float).dropna(), bins=max_bins)
    bars = [
        {
            "label": f"{edges[i]:.2g}–{edges[i+1]:.2g}",
            "count": int(hist_counts[i]),
            "rangeMin": float(edges[i]),
            "rangeMax": float(edges[i + 1]),
        }
        for i in range(len(hist_counts))
    ]
    return {"type": "histogram", "column": column, "total": total, "bars": bars}


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

    export_dir = PROJECTS_PATH / str(project_id) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(ds.original_name).stem or f"dataset_{dataset_id}"
    download_name = f"{safe_stem}_cleaned.csv"

    tmp_path = export_dir / f"{uuid4().hex}.csv"
    df.to_csv(tmp_path, index=False)

    return FileResponse(
        path=str(tmp_path),
        filename=download_name,
        media_type="text/csv",
    )


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

    ops = crud_nettoyage.list_operations(db, project_id, dataset_id)
    ops_payload = []
    for o in ops:
        ops_payload.append(
            {
                "id": getattr(o, "id", None),
                "op_type": getattr(o, "op_type", None),
                "description": getattr(o, "description", None),
                "columns": getattr(o, "columns", None),
                "params": getattr(o, "params", None),
                "created_at": getattr(o, "created_at", None).isoformat() if getattr(o, "created_at", None) else None,
            }
        )

    versions_dir = PROJECTS_PATH / str(project_id) / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(src.original_name).stem or f"dataset_{dataset_id}"
    version_name = body.name.strip() if body.name and body.name.strip() else f"{safe_stem}_cleaned"

    stored_name = f"{uuid4().hex}.csv"
    dst_path = versions_dir / stored_name

    df.to_csv(dst_path, index=False)
    size_bytes = dst_path.stat().st_size

    target_value = getattr(src, "target_column", None)
    if target_value and target_value not in [str(c) for c in df.columns]:
        target_value = None
    can_predict = bool(target_value)

    operations_json = json.dumps(ops_payload, ensure_ascii=False, default=str)

    # Résoudre le source_dataset_id : si l'opération vient d'un workspace brut,
    # le vrai dataset source est workspace_source_dataset_id (pas le workspace lui-même).
    src_kind = getattr(src, "kind", "source")
    if src_kind == "raw_workspace" and getattr(src, "workspace_source_dataset_id", None):
        real_source_id = src.workspace_source_dataset_id
    else:
        real_source_id = src.id

    new_version = DatasetVersion(
        project_id=project_id,
        source_dataset_id=real_source_id,
        name=version_name,
        stored_name=stored_name,
        file_path=str(dst_path),
        content_type="text/csv",
        size_bytes=size_bytes,
        target_column=target_value,
        can_predict=can_predict,
        operations_json=operations_json,
    )

    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    return {
        "version_id": new_version.id,
        "project_id": new_version.project_id,
        "source_dataset_id": new_version.source_dataset_id,
        "name": new_version.name,
        "file_path": new_version.file_path,
        "can_predict": new_version.can_predict,
        "created_at": new_version.created_at.isoformat() if new_version.created_at else None,
    }
