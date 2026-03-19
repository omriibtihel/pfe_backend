from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import pandas as pd
from pathlib import Path

from app.api.deps import get_current_user, get_db, ensure_project_owner
from app.models.dataset_version import DatasetVersion
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.nettoyage_df import load_current_df

from app.services.column_inference import infer_kind_for_series
from app.crud import version_column_schema as crud_schema
from app.schemas.version_column_schema import ColumnsMetaOut, ColumnKindsIn

from app.core.config import PROJECTS_PATH

router = APIRouter()


def _resolve_version_path(project_id: int, version: DatasetVersion) -> Path:
    # adapte selon tes champs réels: file_path / stored_name / path ...
    raw = getattr(version, "file_path", None) or getattr(version, "path", None) or getattr(version, "stored_name", None)
    if not raw:
        raise HTTPException(status_code=400, detail="Version file path missing")

    p = Path(str(raw))
    if p.is_absolute():
        return p

    # cas le plus courant: stored_name dans /projects/{id}/dataset_versions/
    candidate = PROJECTS_PATH / str(project_id) / "dataset_versions" / p.name
    return candidate


def _load_version_df(db: Session, project_id: int, version_id: int) -> pd.DataFrame:
    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .one_or_none()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    path = _resolve_version_path(project_id, v)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Version file not found: {path}")

    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read version file: {e}")


def _build_columns_meta(df: pd.DataFrame, schema_map: dict) -> dict:
    cols_out = []
    counts = {k: 0 for k in ["numeric", "categorical", "text", "binary", "datetime", "id", "other"]}
    total_rows = int(len(df))

    for c in list(df.columns):
        s = df[c]
        row = schema_map.get(str(c))
        inferred_kind = (getattr(row, "inferred_kind", None) or "other").lower()
        override_kind = getattr(row, "override_kind", None)
        effective = (override_kind or inferred_kind or "other").lower()

        if effective not in counts:
            effective = "other"
        counts[effective] += 1

        non_null = s.dropna()
        cols_out.append({
            "name": str(c),
            "dtype": str(s.dtype),
            "kind": effective,
            "inferred_kind": inferred_kind,
            "override_kind": override_kind,
            "confidence": float(getattr(row, "confidence", 0.0) or 0.0),

            "missing": int(s.isna().sum()),
            "unique": int(non_null.nunique(dropna=True)) if len(non_null) else 0,
            "total": int(len(s)),
            "sample": non_null.astype(str).head(12).tolist(),
        })

    return {"columns": cols_out, "counts": counts, "total_rows": total_rows}


@router.get("/{version_id}/columns-meta", response_model=ColumnsMetaOut)
def get_version_columns_meta(
    project_id: int,
    version_id: int,
    workspace_dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    # 1) df source:
    # - view mode => df from version file
    # - edit mode => df from workspace dataset (preview), but we persist schema ON THE VERSION
    if workspace_dataset_id is not None:
        ds = get_dataset_or_404(db, project_id, workspace_dataset_id)
        df = load_current_df(ds.file_path, workspace_dataset_id)
    else:
        df = _load_version_df(db, project_id, version_id)

    # 2) infer for each col (num/cat/text/binary/...)
    inferred = {}
    for c in df.columns:
        kind, conf = infer_kind_for_series(str(c), df[c])
        inferred[str(c)] = (kind, conf)

    # 3) persist inferred (keeps overrides)
    schema_map = crud_schema.upsert_many_inferred(
        db=db,
        project_id=project_id,
        dataset_version_id=version_id,
        inferred=inferred,
    )

    # 4) build payload (effective kind = override or inferred)
    return _build_columns_meta(df, schema_map)


@router.post("/{version_id}/column-kinds")
def set_version_column_kinds(
    project_id: int,
    version_id: int,
    payload: ColumnKindsIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    crud_schema.set_overrides(db, project_id, version_id, payload.overrides)
    return {"ok": True}
