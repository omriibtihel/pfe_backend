from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from scipy import stats
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.models.dataset_version import DatasetVersion
from app.api.utils.versions import load_version_df

router = APIRouter()

AggFn = Literal["count", "sum", "avg", "min", "max", "median"]
SortOrder = Literal["asc", "desc"]
MAX_TOPK = 200
MAX_GROUPS = 1000
MAX_SAMPLE = 5000


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ops_to_tags(operations_json: str | None) -> list[str]:
    """Convertit operations_json (Text) en liste de tags (type/description)."""
    if not operations_json:
        return []
    try:
        data = json.loads(operations_json)
        if not isinstance(data, list):
            return []
        tags: list[str] = []
        for o in data:
            if isinstance(o, dict):
                t = o.get("type") or o.get("op_type") or o.get("opType")
                if t:
                    tags.append(str(t))
                else:
                    d = o.get("description")
                    if d:
                        tags.append(str(d))
            else:
                tags.append(str(o))
        return tags
    except Exception:
        return []


class OverwriteVersionPayload(BaseModel):
    content_base64: str
    content_type: str | None = None
    operations: list[dict] | None = None


@router.get("", response_model=list[dict])
def list_versions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.project_id == project_id)
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )

    return [
        {
            "id": v.id,
            "project_id": v.project_id,
            "source_dataset_id": v.source_dataset_id,
            "name": v.name,
            "stored_name": v.stored_name,
            "content_type": v.content_type,
            "size_bytes": v.size_bytes,
            "target_column": v.target_column,
            "can_predict": v.can_predict,
            "operations": _ops_to_tags(v.operations_json),
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


@router.get("/{version_id}/download")
def download_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    p = Path(v.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {p}")

    filename = f"{v.name}.csv"
    return FileResponse(path=str(p), filename=filename, media_type=v.content_type or "text/csv")


@router.post("/{version_id}/overwrite")
def overwrite_version(
    project_id: int,
    version_id: int,
    payload: OverwriteVersionPayload,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        content = base64.b64decode(payload.content_base64.encode("utf-8"))
        p = Path(v.file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

        v.size_bytes = len(content)
        if payload.content_type:
            v.content_type = payload.content_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid content: {e}")

    if payload.operations is not None:
        v.operations_json = json.dumps(payload.operations)

    db.add(v)
    db.commit()
    db.refresh(v)

    return {"ok": True, "version_id": v.id}


class RenameVersionPayload(BaseModel):
    name: str


@router.patch("/{version_id}", response_model=dict)
def rename_version(
    project_id: int,
    version_id: int,
    payload: RenameVersionPayload,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name cannot be empty")

    v.name = name
    db.add(v)
    db.commit()
    return {"ok": True, "id": v.id, "name": v.name}


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        Path(v.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(v)
    db.commit()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Analytics — version endpoints (mirrors /datasets/{id}/overview, profile, etc.)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{version_id}/overview")
def version_overview(
    project_id: int,
    version_id: int,
    rows: int = Query(10, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    columns = [str(c) for c in df.columns]
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing = {str(col): int(df[col].isna().sum()) for col in df.columns}
    head = df.head(rows).replace([np.inf, -np.inf], np.nan)
    preview = head.astype(object).where(pd.notnull(head), None).to_dict(orient="records")
    return {"shape": shape, "columns": columns, "dtypes": dtypes, "missing": missing, "preview": preview}


@router.get("/{version_id}/profile")
def version_profile(
    project_id: int,
    version_id: int,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)

    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    n_rows = len(df)
    profiles = []

    def _getf(desc, k):
        v = desc.get(k)
        try:
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    for col in df.columns:
        s = df[col]
        name = str(col)
        dtype = str(s.dtype)
        missing_count = int(s.isna().sum())
        missing_pct = (missing_count / n_rows * 100.0) if n_rows else 0.0
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
            vc = s.astype(str).value_counts(dropna=True).head(top_k)
            categorical = {
                "unique": unique,
                "top_values": [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()],
            }
        else:
            ratio = (unique / max(1, n_rows)) if n_rows else 0
            kind = "categorical" if ratio < 0.5 else "text"
            vc = s.astype(str).fillna("").replace("nan", "").value_counts().head(top_k)
            categorical = {
                "unique": unique,
                "top_values": [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()],
            }

        profiles.append({
            "name": name,
            "kind": kind,
            "dtype": dtype,
            "missing": missing_count,
            "missing_pct": round(missing_pct, 4),
            "unique": unique,
            "unique_pct": round(unique_pct, 4),
            "numeric": numeric,
            "categorical": categorical,
        })

    return {"shape": shape, "profiles": profiles}


@router.get("/{version_id}/correlation")
def version_correlation(
    project_id: int,
    version_id: int,
    columns: list[str] = Query(..., min_length=2, max_length=20),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    cols = [str(c) for c in columns]

    missing_cols = [c for c in cols if c not in df.columns.tolist()]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_cols)}")

    sub = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(sub[c]):
            converted = pd.to_numeric(sub[c], errors="coerce")
            if converted.notna().sum() < max(3, int(0.1 * len(converted))):
                raise HTTPException(status_code=400, detail=f"Column '{c}' is not numeric")
            sub[c] = converted

    corr = sub.corr(method="pearson").fillna(0.0)
    return {"columns": cols, "matrix": corr.to_numpy(dtype=float).tolist()}


@router.get("/{version_id}/charts/value-counts")
def version_value_counts(
    project_id: int,
    version_id: int,
    col: str = Query(...),
    top_k: int = Query(15, ge=1, le=MAX_TOPK),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    _ensure_cols(df, [col])

    s = df[col]
    s2 = s.dropna().astype(str) if dropna else s.where(~s.isna(), "__NULL__").astype(str)
    vc_all = s2.value_counts()
    total_count = int(vc_all.sum())
    vc_top = vc_all.head(top_k)
    others_count = max(0, total_count - int(vc_top.sum()))

    return {
        "col": col,
        "top_k": top_k,
        "total_count": total_count,
        "others_count": others_count,
        "rows": [{"value": str(k), "count": int(v)} for k, v in vc_top.items()],
    }


@router.get("/{version_id}/charts/aggregate")
def version_aggregate(
    project_id: int,
    version_id: int,
    x: str = Query(...),
    y: Optional[str] = Query(None),
    agg: Optional[AggFn] = Query(None),
    top_k: int = Query(15, ge=1, le=MAX_TOPK),
    order: SortOrder = Query("desc"),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)

    if agg is None:
        agg = "count" if not y else "avg"
    if agg != "count" and not y:
        raise HTTPException(status_code=400, detail="Parameter 'y' is required for this aggregation.")

    usecols = [x] if (agg == "count" or not y) else [x, y]
    _ensure_cols(df, usecols)
    sub = df[usecols].copy()

    if pd.api.types.is_numeric_dtype(sub[x]):
        raise HTTPException(status_code=400, detail="X must be categorical/text/datetime.")

    if dropna:
        sub = sub.dropna(subset=usecols)
    else:
        sub = sub.dropna(subset=[x])

    if sub.empty:
        return {"x": x, "y": y, "agg": agg, "top_k": top_k, "order": order, "rows": []}

    if agg == "count":
        s = sub.groupby(x, dropna=False).size()
    else:
        sub[y] = _to_num(sub[y])
        if dropna:
            sub = sub.dropna(subset=[x, y])
        g = sub.groupby(x, dropna=False)[y]
        s = {"sum": g.sum, "avg": g.mean, "min": g.min, "max": g.max, "median": g.median}[agg]()

    if len(s) > MAX_GROUPS:
        s = s.sort_values(ascending=False).head(MAX_GROUPS)
    s = s.sort_values(ascending=(order == "asc")).head(top_k)

    return {
        "x": x, "y": y, "agg": agg, "top_k": top_k, "order": order,
        "rows": [{"x": str(k), "y": float(v) if pd.notna(v) else 0.0} for k, v in s.items()],
    }


@router.get("/{version_id}/charts/hist")
def version_histogram(
    project_id: int,
    version_id: int,
    col: str = Query(...),
    bins: int = Query(20, ge=5, le=200),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    _ensure_cols(df, [col])

    s = _to_num(df[col])
    if dropna:
        s = s.dropna()
    if s.empty:
        return {"col": col, "bins": bins, "rows": []}

    x = s.to_numpy(dtype=float)
    if np.nanmin(x) == np.nanmax(x):
        v = float(np.nanmin(x))
        return {"col": col, "bins": 1, "rows": [{"x0": v - 0.5, "x1": v + 0.5, "count": int(len(x))}]}

    counts, edges = np.histogram(x, bins=bins)
    return {
        "col": col, "bins": bins,
        "rows": [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))],
    }


@router.get("/{version_id}/charts/sample")
def version_sample(
    project_id: int,
    version_id: int,
    cols: List[str] = Query(...),
    n: int = Query(400, ge=10, le=MAX_SAMPLE),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    if not cols:
        raise HTTPException(status_code=400, detail="Parameter 'cols' is required.")
    _ensure_cols(df, cols)

    sub = df[cols]
    if sub.empty:
        return {"cols": cols, "rows": []}

    k = min(n, len(sub))
    sampled = sub.sample(n=k, random_state=0) if len(sub) > k else sub
    return {"cols": cols, "rows": sampled.replace({np.nan: None}).to_dict(orient="records")}


@router.get("/{version_id}/charts/pearson")
def version_pearson_correlation(
    project_id: int,
    version_id: int,
    x: str = Query(...),
    y: str = Query(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    _ensure_cols(df, [x, y])

    sx = pd.to_numeric(df[x], errors="coerce").dropna()
    sy = pd.to_numeric(df[y], errors="coerce").dropna()
    common = sx.index.intersection(sy.index)
    sx, sy = sx.loc[common], sy.loc[common]

    n = len(sx)
    if n < 3:
        return {"x": x, "y": y, "r": None, "n": n}

    r, _ = stats.pearsonr(sx.to_numpy(dtype=float), sy.to_numpy(dtype=float))
    return {"x": x, "y": y, "r": float(r) if np.isfinite(r) else None, "n": n}


class VersionNormalityTestIn(BaseModel):
    columns: List[str]


@router.post("/{version_id}/normality-test")
def version_normality_test(
    project_id: int,
    version_id: int,
    body: VersionNormalityTestIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    if not body.columns:
        raise HTTPException(status_code=400, detail="At least one column is required.")

    df = load_version_df(db, project_id, version_id)
    df.columns = df.columns.astype(str)
    _ensure_cols(df, body.columns)

    results = []
    for col in body.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 3:
            raise HTTPException(status_code=400, detail=f"Column '{col}' has fewer than 3 non-null numeric values.")

        data = s.to_numpy(dtype=float)
        n = len(data)
        if n <= 5000:
            stat, p_value = stats.shapiro(data)
            test_used = "shapiro"
        else:
            stat, p_value = stats.normaltest(data)
            test_used = "dagostino"

        results.append({
            "col": col,
            "n": n,
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data) + 3),
            "test_used": test_used,
            "stat": float(stat),
            "p_value": float(p_value),
            "is_normal": bool(p_value > 0.05),
        })

    return {"results": results}
