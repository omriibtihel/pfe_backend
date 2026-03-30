from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import PROJECTS_PATH
from app.models.dataset_version import DatasetVersion


def write_df_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def resolve_version_path(project_id: int, version: DatasetVersion) -> Path:
    raw = getattr(version, "file_path", None) or getattr(version, "path", None) or getattr(version, "stored_name", None)
    if not raw:
        raise HTTPException(status_code=400, detail="Version file path missing")

    p = Path(str(raw))
    if p.is_absolute():
        return p

    return PROJECTS_PATH / str(project_id) / "dataset_versions" / p.name


def load_version_df(db: Session, project_id: int, version_id: int) -> pd.DataFrame:
    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .one_or_none()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    path = resolve_version_path(project_id, v)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Version file not found: {path}")

    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read version file: {e}")
