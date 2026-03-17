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


def _coerce_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns that use a comma as decimal separator to float.

    Only columns whose every non-null value matches the pattern ``digits,digits``
    are converted (e.g. "4,5" → 4.5).  Mixed columns or columns that contain
    commas for other reasons are left untouched.
    """
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        if not non_null.astype(str).str.match(r"^-?\d+,\d+$").all():
            continue
        df[col] = pd.to_numeric(
            series.str.replace(",", ".", regex=False), errors="coerce"
        )
    return df


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    df = pd.read_excel(path) if suffix in (".xlsx", ".xls") else pd.read_csv(path)
    return _coerce_decimal_commas(df)
