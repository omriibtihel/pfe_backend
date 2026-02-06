from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import json


from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.processing_df import load_current_df
from app.schemas.processing import OperationIn, OperationOut
from app.crud import processing as crud_processing
from app.services.processing_rebuild import rebuild_processed
from app.models.dataset_version import DatasetVersion


from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.project import Project


router = APIRouter()


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


@router.get("/datasets/{dataset_id}/processing/operations", response_model=list[OperationOut])
def list_operations(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    return crud_processing.list_operations(db, project_id, dataset_id)


@router.post("/datasets/{dataset_id}/processing/operations", response_model=OperationOut)
def apply_operation(
    project_id: int,
    dataset_id: int,
    payload: OperationIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Phase Préparation (Dataiku-like Prepare):
    - Pas de split train/test ici (on le fera au moment du training).
    - On enregistre l'opération puis on rebuild le dataset traité de manière déterministe.
    """
    ensure_project_owner(db, project_id, current_user.id)

    ds = get_dataset_or_404(db, project_id, dataset_id)

    # Validation: les colonnes demandées doivent exister dans le dataset courant (après ops précédentes)
    df_current = load_current_df(ds.file_path, dataset_id)
    existing = set(map(str, df_current.columns))

    cols = payload.columns or []
    missing = [c for c in cols if str(c) not in existing]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes introuvables: {missing}")

    op = crud_processing.create_operation(
        db=db,
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=current_user.id,
        op_type=payload.type,
        description=payload.description,
        columns=cols,
        params=payload.params or {},
    )

    try:
        rebuild_processed(db, project_id, dataset_id)
    except Exception as e:
        db.rollback()  
        crud_processing.delete_last_operation(db, project_id, dataset_id)
        raise HTTPException(status_code=400, detail=str(e))
    
    db.refresh(op)
    return op


@router.post("/datasets/{dataset_id}/processing/undo")
def undo_last(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    deleted = crud_processing.delete_last_operation(db, project_id, dataset_id)
    if not deleted:
        return {"ok": False}

    rebuild_processed(db, project_id, dataset_id)
    return {"ok": True}


@router.get("/datasets/{dataset_id}/processing/preview")
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


@router.get("/datasets/{dataset_id}/processing/export")
def export_processed_dataset(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Sécurité
    ensure_project_owner(db, project_id, current_user.id)

    # Dataset + DF courant (traité)
    ds = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(ds.file_path, dataset_id)

    # Écriture fichier export (csv)
    export_dir = PROJECTS_PATH / str(project_id) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(ds.original_name).stem or f"dataset_{dataset_id}"
    download_name = f"{safe_stem}_processed.csv"

    tmp_path = export_dir / f"{uuid4().hex}.csv"
    df.to_csv(tmp_path, index=False)

    return FileResponse(
        path=str(tmp_path),
        filename=download_name,
        media_type="text/csv",
    )


@router.post("/datasets/{dataset_id}/processing/save")
def save_processed_as_version(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Sécurité
    ensure_project_owner(db, project_id, current_user.id)

    # Dataset source + DF courant (traité)
    src = get_dataset_or_404(db, project_id, dataset_id)
    df = load_current_df(src.file_path, dataset_id)

    # Récupérer la liste des opérations (pour l'historique)
    ops = crud_processing.list_operations(db, project_id, dataset_id)
    ops_payload = []
    for o in ops:
        if hasattr(o, "model_dump"):         # Pydantic v2
            ops_payload.append(o.model_dump())
        elif hasattr(o, "dict"):             # Pydantic v1
            ops_payload.append(o.dict())
        else:                                # fallback (ORM)
            ops_payload.append(
                {
                    "id": getattr(o, "id", None),
                    "type": getattr(o, "type", None),
                    "description": getattr(o, "description", None),
                    "columns": getattr(o, "columns", None),
                    "params": getattr(o, "params", None),
                    "created_at": getattr(o, "created_at", None).isoformat()
                    if getattr(o, "created_at", None)
                    else None,
                }
            )

    # Dossier versions
    versions_dir = PROJECTS_PATH / str(project_id) / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(src.original_name).stem or f"dataset_{dataset_id}"
    version_name = f"{safe_stem}_processed"

    stored_name = f"{uuid4().hex}.csv"   
    dst_path = versions_dir / stored_name

    # Écrire snapshot
    df.to_csv(dst_path, index=False)
    size_bytes = dst_path.stat().st_size

    # target_column / can_predict (utile pour la prédiction)
    target_value = getattr(src, "target_column", None)
    if target_value and target_value not in df.columns.astype(str).tolist():
        target_value = None
    can_predict = bool(target_value)

    # ✅ ton modèle attend operations_json (Text)
    operations_json = json.dumps(ops_payload, ensure_ascii=False, default=str)

    new_version = DatasetVersion(
        project_id=project_id,
        source_dataset_id=src.id,
        name=version_name,
        stored_name=stored_name,
        file_path=str(dst_path),
        content_type="text/csv",
        size_bytes=size_bytes,
        target_column=target_value,
        can_predict=can_predict,
        operations_json=operations_json,   # ✅ bon champ
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

