from __future__ import annotations

import re
from uuid import uuid4
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.api.utils_shared.df import read_df
from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.api.utils_shared.datasets import get_dataset_or_404

from app.schemas.dataset import DatasetOut, DatasetPreviewOut, DatasetTargetIn, DatasetTargetOut
from app.schemas.dataset_active import ActiveDatasetIn, ActiveDatasetOut


router = APIRouter()

ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

_filename_safe_re = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = _filename_safe_re.sub("_", name).strip("._")
    return name or "file"


# -------------------------
# LIST
# -------------------------
@router.get("", response_model=list[DatasetOut])
def list_datasets(
    project_id: int,
    include_workspaces: bool = Query(False, description="Si true, inclut les workspaces"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    q = db.query(Dataset).filter(Dataset.project_id == project_id)

    if not include_workspaces:
        q = q.filter(Dataset.kind == "source")

    return q.order_by(Dataset.created_at.desc()).all()


# -------------------------
# UPLOAD
# -------------------------
@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

    original_name = sanitize_filename(file.filename or "file")
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    dataset_dir = PROJECTS_PATH / str(project_id) / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}{ext}"
    dst_path = dataset_dir / stored_name

    size = 0
    try:
        with dst_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                out.write(chunk)
    finally:
        await file.close()

    dataset = Dataset(
        project_id=project_id,
        original_name=original_name,
        stored_name=stored_name,
        file_path=str(dst_path),
        content_type=file.content_type,
        size_bytes=size,
        target_column=None,  # ✅ target par dataset
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # ✅ si le projet n'a pas encore de dataset actif => mettre celui-ci actif
    if getattr(project, "active_dataset_id", None) is None:
        project.active_dataset_id = dataset.id
        db.add(project)
        db.commit()

    # Créer une version initiale "brute" utilisable directement pour l'entraînement
    try:
        versions_dir = PROJECTS_PATH / str(project_id) / "dataset_versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        raw_stored_name = f"{uuid4().hex}.csv"
        raw_path = versions_dir / raw_stored_name

        df_raw = read_df(dst_path, nrows=None)
        df_raw.to_csv(raw_path, index=False)

        stem = Path(original_name).stem
        raw_version = DatasetVersion(
            project_id=project_id,
            source_dataset_id=dataset.id,
            name=stem,
            stored_name=raw_stored_name,
            file_path=str(raw_path),
            content_type="text/csv",
            size_bytes=raw_path.stat().st_size,
            target_column=None,
            can_predict=False,
            operations_json='[{"type": "original"}]',
        )
        db.add(raw_version)
        db.commit()
    except Exception:
        pass  # non-fatal : le dataset reste utilisable, juste sans version initiale

    return dataset


# -------------------------
# DOWNLOAD
# -------------------------
@router.get("/{dataset_id}/download")
def download_dataset(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    dataset = get_dataset_or_404(db, project_id, dataset_id)

    path = Path(dataset.file_path)
    return FileResponse(
        path=str(path),
        filename=dataset.original_name,
        media_type=dataset.content_type or "application/octet-stream",
    )


# -------------------------
# DELETE
# -------------------------
@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # remove file
    path = Path(dataset.file_path)
    if path.exists():
        path.unlink()

    # ✅ si on supprime le dataset actif => active_dataset_id = None
    if getattr(project, "active_dataset_id", None) == dataset.id:
        project.active_dataset_id = None
        db.add(project)

    db.delete(dataset)
    db.commit()
    return None


# -------------------------
# PREVIEW
# -------------------------
@router.get("/{dataset_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(
    project_id: int,
    dataset_id: int,
    rows: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path), nrows=rows)

    return {
        "dataset": dataset,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns.astype(str)),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "rows": df.fillna("").to_dict(orient="records"),
    }


# -------------------------
# OVERVIEW
# -------------------------
@router.get("/{dataset_id}/overview")
def dataset_overview(
    project_id: int,
    dataset_id: int,
    rows: int = Query(10, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path), nrows=None)

    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    columns = list(df.columns.astype(str))
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing = {str(col): int(df[col].isna().sum()) for col in df.columns}
    preview_rows = df.head(rows).fillna("").to_dict(orient="records")

    return {
        "dataset": dataset,
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "preview": preview_rows,
    }


# -------------------------
# ACTIVE DATASET (project-level)
# -------------------------
@router.get("/active", response_model=ActiveDatasetOut)
def get_active_dataset(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    return {"active_dataset_id": project.active_dataset_id}


@router.post("/active", response_model=ActiveDatasetOut)
def set_active_dataset(
    project_id: int,
    payload: ActiveDatasetIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

    ds = (
        db.query(Dataset)
        .filter(Dataset.id == payload.dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project.active_dataset_id = payload.dataset_id
    db.add(project)
    db.commit()
    db.refresh(project)
    return {"active_dataset_id": project.active_dataset_id}


# -------------------------
# TARGET (dataset-level ✅)
# -------------------------
@router.get("/{dataset_id}/target", response_model=DatasetTargetOut)
def get_dataset_target(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)
    return {"target_column": getattr(ds, "target_column", None)}


@router.put("/{dataset_id}/target", response_model=DatasetTargetOut)
def set_dataset_target(
    project_id: int,
    dataset_id: int,
    payload: DatasetTargetIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    # ✅ validation si target non vide: doit exister dans le dataset
    value = (payload.target_column or "").strip() or None
    if value:
        df = read_df(Path(ds.file_path), nrows=None)
        cols = df.columns.astype(str).tolist()
        if value not in cols:
            raise HTTPException(status_code=400, detail="Target column not found in this dataset")

    ds.target_column = value
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"target_column": ds.target_column}
