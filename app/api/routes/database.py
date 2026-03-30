# app/api/routes/database.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.api.utils.df import read_df
from app.schemas.database import CorrelationOut
from app.schemas.dataset import DatasetPreviewOut, DatasetColumnOut
from fastapi.encoders import jsonable_encoder
from app.api.utils.datasets import get_dataset_or_404
import numpy as np




from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.models.dataset import Dataset
from app.schemas.database import (
    DatasetOverviewOut,
    DatasetProfileOut,
    TargetIn,
    TargetOut,
    ActiveDatasetIn,
    ActiveDatasetOut,
)

router = APIRouter()






@router.get("/datasets/{dataset_id}/columns", response_model=List[DatasetColumnOut])
def dataset_columns(
    project_id: int,
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path))
    return [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewOut)
def dataset_preview(
    project_id: int,
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    n: int | None = Query(None, ge=1, le=200),  # compat ancienne URL ?n=10
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    if n is not None:
        page = 1
        page_size = n

    df = read_df(Path(dataset.file_path))

    total_rows = int(df.shape[0])
    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}

    start = (page - 1) * page_size
    end = start + page_size


    chunk = df.iloc[start:end].copy()

    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    chunk = chunk.where(pd.notnull(chunk), None)


    rows = jsonable_encoder(chunk.to_dict(orient="records"))

    return {
        "dataset": dataset,
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "rows": rows,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
    }


@router.get("/datasets/{dataset_id}/overview", response_model=DatasetOverviewOut)
def dataset_overview(
    project_id: int,
    dataset_id: int,
    rows: int = Query(10, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path))

    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    columns = [str(c) for c in df.columns]
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing = {str(col): int(df[col].isna().sum()) for col in df.columns}
    preview = df.head(rows).fillna("").to_dict(orient="records")

    return {
        "dataset": dataset,
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "preview": preview,
    }


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfileOut)
def dataset_profile(
    project_id: int,
    dataset_id: int,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Profil simple type Dataiku:
    - numeric: count/mean/std/min/percentiles/max
    - categorical/text: unique + top values
    - datetime: unique + top values (string)
    """
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path))

    def _getf(desc, k):
        v = desc.get(k)
        try:
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profiles = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        name = str(col)
        dtype = str(s.dtype)

        missing = int(s.isna().sum())
        missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0

        unique = int(s.nunique(dropna=True))
        unique_pct = (unique / n_rows * 100.0) if n_rows else 0.0

        kind = "unknown"
        numeric = None
        categorical = None

        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            numeric = {
                "count": int(desc.get("count", 0) or 0),
                "mean": _getf(desc, "mean"),
                "std": _getf(desc, "std"),
                "min": _getf(desc, "min"),
                "p25": _getf(desc, "25%"),
                "p50": _getf(desc, "50%"),
                "p75": _getf(desc, "75%"),
                "max": _getf(desc, "max"),
            }

        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
            vc = (
                s.astype("datetime64[ns]", errors="ignore")
                .astype(str)
                .value_counts(dropna=True)
                .head(top_k)
            )
            categorical = {
                "unique": unique,
                "top_values": [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()],
            }

        else:
            ratio = (unique / max(1, n_rows)) if n_rows else 0
            kind = "categorical" if ratio < 0.5 else "text"
            vc = (
                s.astype(str)
                .fillna("")
                .replace("nan", "")
                .value_counts()
                .head(top_k)
            )
            categorical = {
                "unique": unique,
                "top_values": [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()],
            }

        profiles.append(
            {
                "name": name,
                "kind": kind,
                "dtype": dtype,

                "missing": missing,
                "missing_pct": round(missing_pct, 4),
                "unique": unique,
                "unique_pct": round(unique_pct, 4),

                "numeric": numeric,
                "categorical": categorical,
            }
        )

    return {
        "dataset": dataset,
        "shape": shape,
        "profiles": profiles,
    }

@router.get("/datasets/{dataset_id}/correlation", response_model=CorrelationOut)
def dataset_correlation(
    project_id: int,
    dataset_id: int,
    columns: list[str] = Query(..., min_length=2, max_length=20),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Matrice de corrélation (Pearson) pour colonnes numériques.
    Exemple:
      /correlation?columns=age&columns=chol&columns=trestbps
    """
    ensure_project_owner(db, project_id, current_user.id)
    dataset = get_dataset_or_404(db, project_id, dataset_id)

    df = read_df(Path(dataset.file_path))

    # normaliser colonnes (backend travaille en str)
    df_cols = df.columns.astype(str).tolist()
    cols = [str(c) for c in columns]

    # 1) vérifier existence
    missing_cols = [c for c in cols if c not in df_cols]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in dataset: {', '.join(missing_cols)}",
        )

    # 2) extraire + forcer numérique
    sub = df[cols].copy()

    # Si des colonnes sont "object" mais contiennent des nombres en texte,
    # on tente de convertir. Si ça échoue (trop de NaN), on refuse.
    for c in cols:
        if not pd.api.types.is_numeric_dtype(sub[c]):
            converted = pd.to_numeric(sub[c], errors="coerce")
            # si quasi tout devient NaN => pas une vraie colonne numérique
            if converted.notna().sum() < max(3, int(0.1 * len(converted))):
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{c}' is not numeric (cannot compute correlation)",
                )
            sub[c] = converted

    # 3) corr
    corr = sub.corr(method="pearson").fillna(0.0)

    # 4) output
    matrix = corr.to_numpy(dtype=float).tolist()
    return {"columns": cols, "matrix": matrix}

# ---------- Active dataset ----------
@router.get("/datasets/active", response_model=ActiveDatasetOut)
def get_active_dataset(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    return {"active_dataset_id": project.active_dataset_id}


@router.post("/datasets/active", response_model=ActiveDatasetOut)
def set_active_dataset(
    project_id: int,
    payload: ActiveDatasetIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

    ds = (
        db.query(Dataset)
        .filter(Dataset.id == payload.dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project.active_dataset_id = payload.dataset_id
    db.add(project)
    db.commit()
    db.refresh(project)

    return {"active_dataset_id": project.active_dataset_id}


# ---------- Target column ----------
@router.get("/target", response_model=TargetOut)
def get_target(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)
    return {"target_column": project.target_column}


@router.put("/target", response_model=TargetOut)
def set_target(
    project_id: int,
    payload: TargetIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = ensure_project_owner(db, project_id, current_user.id)

    # validation: si on a un active dataset, vérifier que la colonne existe
    if payload.target_column and project.active_dataset_id:
        ds = get_dataset_or_404(db, project_id, project.active_dataset_id)
        df = read_df(Path(ds.file_path))
        cols = {str(c) for c in df.columns}
        if payload.target_column not in cols:
            raise HTTPException(status_code=400, detail="Target column not found in active dataset")

    project.target_column = payload.target_column
    db.add(project)
    db.commit()
    db.refresh(project)

    return {"target_column": project.target_column}
