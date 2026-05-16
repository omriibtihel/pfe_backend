from __future__ import annotations

import re
from uuid import uuid4
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.api.utils_shared.df import AmbiguousCsvSeparatorError, read_df
from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.api.utils_shared.datasets import get_dataset_or_404

from app.schemas.dataset import DatasetOut, DatasetPreviewOut, DatasetTargetIn, DatasetTargetOut


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
    ensure_project_owner(db, project_id, current_user.id)

    # 409 si un dataset source existe déjà pour ce projet
    existing_source = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id, Dataset.kind == "source")
        .first()
    )
    if existing_source is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project already has a source dataset",
        )

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
        kind="source",
        target_column=None,
    )

    db.add(dataset)
    db.flush()

    # Si la création de la version initiale échoue, on rollback ET on supprime le fichier déjà écrit.
    versions_dir = PROJECTS_PATH / str(project_id) / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    raw_stored_name = f"{uuid4().hex}.csv"
    raw_path = versions_dir / raw_stored_name

    try:
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
    except AmbiguousCsvSeparatorError as exc:
        db.rollback()
        dst_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        dst_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create initial dataset version — upload aborted: {exc}",
        ) from exc

    db.refresh(dataset)
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
    ensure_project_owner(db, project_id, current_user.id)

    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = Path(dataset.file_path)
    if path.exists():
        path.unlink()

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
# TARGET (dataset-level)
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
