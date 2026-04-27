# app/api/utils_shared/datasets.py
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.core.config import PROJECTS_PATH

from app.models.dataset import Dataset


def _resolve_dataset_path(project_id: int, raw_path: str) -> Path:
    p = Path(raw_path)

    # 1) si déjà absolu
    if p.is_absolute():
        return p

    # 2) essayer quelques emplacements "standards"
    candidates = [
        PROJECTS_PATH / str(project_id) / p,
        PROJECTS_PATH / str(project_id) / "datasets" / p.name,
        PROJECTS_PATH / str(project_id) / "dataset_versions" / p.name,
        PROJECTS_PATH / str(project_id) / "workspaces" / p.name,
        PROJECTS_PATH / str(project_id) / "imports" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback (utile pour message d'erreur)
    return PROJECTS_PATH / str(project_id) / p


def get_dataset_or_404(db: Session, project_id: int, dataset_id: int) -> Dataset:
    ds = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    fp = _resolve_dataset_path(project_id, ds.file_path)

    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file missing on disk: {fp}")

    ds.file_path = str(fp)
    return ds


def get_project_source_dataset_or_404(db: Session, project_id: int) -> Dataset:
    """Retourne l'unique dataset source (kind='source') du projet, ou 404."""
    ds = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id, Dataset.kind == "source")
        .first()
    )
    if not ds:
        raise HTTPException(
            status_code=404,
            detail="No source dataset uploaded for this project",
        )

    fp = _resolve_dataset_path(project_id, ds.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file missing on disk: {fp}")

    ds.file_path = str(fp)
    return ds
