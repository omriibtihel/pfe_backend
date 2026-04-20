from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd

from app.api.deps import get_current_user, get_db, ensure_project_owner
from app.api.utils_shared.datasets import get_dataset_or_404
from app.services.nettoyage.df_utils import load_current_df
from app.api.utils_shared.versions import load_version_df
from app.services.data.column_inference import infer_kind_for_series
from app.crud import version_column_schema as crud_schema
from app.schemas.version_column_schema import ColumnsMetaOut, ColumnKindsIn

router = APIRouter()


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
        df = load_version_df(db, project_id, version_id)

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


@router.get("/{version_id}/column-values")
def get_version_column_values(
    project_id: int,
    version_id: int,
    column: str = Query(..., description="Column name to get unique values for"),
    max_values: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return sorted unique non-null values for a specific column (used for positive label selector)."""
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
    unique_vals = sorted(
        df[column].dropna().astype(str).unique().tolist()
    )[:max_values]
    return {"column": column, "values": unique_vals, "total": len(unique_vals)}


@router.get("/{version_id}/column-distribution")
def get_version_column_distribution(
    project_id: int,
    version_id: int,
    column: str = Query(..., description="Column name to get distribution for"),
    max_bins: int = Query(default=20, ge=2, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Return value distribution for a column:
    - Categorical/low-cardinality: sorted value counts (up to max_bins entries).
    - Numeric/high-cardinality: histogram with max_bins buckets.
    """
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    series = df[column].dropna()
    total = len(series)
    n_unique = series.nunique()

    # Categorical / low-cardinality: return value counts
    if n_unique <= max_bins or not pd.api.types.is_numeric_dtype(series):
        counts = series.astype(str).value_counts().head(max_bins)
        bars = [{"label": str(label), "count": int(cnt)} for label, cnt in counts.items()]
        return {"type": "categorical", "column": column, "total": total, "bars": bars}

    # Numeric / high-cardinality: histogram
    counts, edges = np.histogram(series.astype(float).dropna(), bins=max_bins)
    bars = [
        {
            "label": f"{edges[i]:.2g}–{edges[i+1]:.2g}",
            "count": int(counts[i]),
            "rangeMin": float(edges[i]),
            "rangeMax": float(edges[i + 1]),
        }
        for i in range(len(counts))
    ]
    return {"type": "histogram", "column": column, "total": total, "bars": bars}


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
