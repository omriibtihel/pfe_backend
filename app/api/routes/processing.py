from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.processing_df import load_current_df
from app.schemas.processing import OperationIn, OperationOut
from app.crud import processing as crud_processing
from app.services.processing_rebuild import rebuild_processed

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
        # rollback opération si rebuild échoue
        crud_processing.delete_last_operation(db, project_id, dataset_id)
        raise HTTPException(status_code=400, detail=str(e))

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
