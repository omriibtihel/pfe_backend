from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.services.data.loader import resolve_dataset_path
from app.models.dataset_version import DatasetVersion
from app.services.nettoyage.df_utils import processed_path_for

logger = logging.getLogger(__name__)


def resolve_training_data_path(
    db: Session,
    project_id: int,
    version_id: int | None,
) -> tuple[Path, int, str]:
    """
    Return (path_to_load, resolved_dv_id, data_source).

    data_source is 'processed' when the ephemeral cleaned file was used,
    'RAW_ORIGINAL' when falling back to the DatasetVersion file.

    Prefers processed_dataset_<source_id>.csv over the raw DatasetVersion
    path when the file exists — consistent with run_training_session Fix C.
    All training-adjacent routes that load data for validation, preview, or
    pre-flight checks must go through this function so their data source
    matches what the actual training background task will use.
    """
    dataset_path, dv_id = resolve_dataset_path(db, project_id, version_id)

    dv = db.query(DatasetVersion).filter(DatasetVersion.id == dv_id).first()
    src_ds = dv.source_dataset if dv is not None else None
    if src_ds is not None:
        processed = processed_path_for(src_ds.file_path, src_ds.id)
        path_to_load = processed if processed.exists() else dataset_path
    else:
        path_to_load = dataset_path

    data_source = "processed" if path_to_load != dataset_path else "RAW_ORIGINAL"
    logger.info(
        "training_data_source | project_id=%s | dataset_version_id=%s | file=%s | source=%s",
        project_id, dv_id, Path(path_to_load).name, data_source,
    )
    return path_to_load, int(dv_id), data_source
