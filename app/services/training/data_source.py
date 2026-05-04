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

    data_source values:
      - "version_snapshot": the user-selected DatasetVersion has its own committed
        CSV (distinct from the raw source). This is the normal path — the version
        is the source of truth and may contain cleaning operations.
      - "processed": the version's snapshot was missing, so we fell back to the
        shared ephemeral processed_dataset_<source_id>.csv file.
      - "raw_original": the version's CSV path is literally the same physical
        file as the source dataset's raw file (no cleaning was committed for
        this version). Only this case warrants a "data not cleaned" warning.

    Prefers the version's own CSV over the shared processed file because the
    shared file is keyed by source_dataset_id and may contain data from other
    versions sharing the same source.

    Exception: when the version's CSV path *is* literally the source's raw file
    (raw_original case — no committed snapshot for this version), an unsaved
    cleaning workspace at processed_dataset_<src_id>.csv must take precedence.
    Otherwise the user's cleaning operations would be silently ignored.
    """
    dataset_path, dv_id = resolve_dataset_path(db, project_id, version_id)

    dv = db.query(DatasetVersion).filter(DatasetVersion.id == dv_id).first()
    src_ds = dv.source_dataset if dv is not None else None
    if src_ds is not None:
        processed = processed_path_for(src_ds.file_path, src_ds.id)
        is_raw_original = _same_file(Path(dataset_path), Path(src_ds.file_path))

        # raw_original + unsaved processed → prefer the cleaned workspace.
        if is_raw_original and processed.exists():
            path_to_load = processed
        elif Path(dataset_path).exists():
            path_to_load = dataset_path
        elif processed.exists():
            path_to_load = processed
        else:
            raise FileNotFoundError(
                f"No data file found for DatasetVersion id={dv_id}. "
                f"Expected: {dataset_path}"
            )
    else:
        path_to_load = dataset_path

    if path_to_load != dataset_path:
        data_source = "processed"
    elif src_ds is not None and _same_file(path_to_load, Path(src_ds.file_path)):
        data_source = "raw_original"
    else:
        data_source = "version_snapshot"

    logger.info(
        "training_data_source | project_id=%s | dataset_version_id=%s | file=%s | source=%s",
        project_id, dv_id, Path(path_to_load).name, data_source,
    )
    return path_to_load, int(dv_id), data_source


def _same_file(a: Path, b: Path) -> bool:
    """True when both paths resolve to the same file on disk."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)
