from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset_version import DatasetVersion


def resolve_dataset_path(db: Session, project_id: int, dataset_version_id: int | None) -> Tuple[Path, int]:
    if dataset_version_id is None:
        raise RuntimeError("datasetVersionId is required")

    dv = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.project_id == project_id, DatasetVersion.id == dataset_version_id)
        .first()
    )
    if not dv:
        raise RuntimeError("DatasetVersion not found for this project")

    path = Path(dv.file_path)
    if not path.exists():
        raise RuntimeError(f"Dataset file not found: {path}")

    return path, dv.id


def load_dataframe(path: Path) -> pd.DataFrame:
    # si tu supportes Excel plus tard: if suffix in (".xlsx", ...) => pd.read_excel
    return pd.read_csv(path)
