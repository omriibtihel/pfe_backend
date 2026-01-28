# app/api/utils/datasets.py
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.dataset import Dataset

def get_dataset_or_404(db: Session, project_id: int, dataset_id: int) -> Dataset:
    ds = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    fp = Path(ds.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing on disk")

    return ds
