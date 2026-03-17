# charts.py
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from scipy import stats
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_db
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.df import read_df

router = APIRouter()

MAX_TOPK = 200
MAX_GROUPS = 1000
MAX_SAMPLE = 5000

AggFn = Literal["count", "sum", "avg", "min", "max", "median"]
SortOrder = Literal["asc", "desc"]


def _ensure_cols_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in dataset: {missing}")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# -------------------------
# PIE / DOUGHNUT (Top K + Others)
# -------------------------
@router.get("/{dataset_id}/charts/value-counts")
def value_counts(
    project_id: int,
    dataset_id: int,
    col: str = Query(...),
    top_k: int = Query(15, ge=1, le=MAX_TOPK),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Dataiku-like Pie/Doughnut:
      - Always returns TOP K + Others
      - Others computed server-side accurately
    """
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    fp = Path(ds.file_path)
    df = read_df(fp, usecols=[col])
    df.columns = df.columns.astype(str)
    _ensure_cols_exist(df, [col])

    s = df[col]

    if dropna:
        s2 = s.dropna().astype(str)
    else:
        s2 = s.where(~s.isna(), "__NULL__").astype(str)

    vc_all = s2.value_counts()
    total_count = int(vc_all.sum())

    vc_top = vc_all.head(top_k)
    top_sum = int(vc_top.sum())
    others_count = max(0, total_count - top_sum)

    rows = [{"value": str(k), "count": int(v)} for k, v in vc_top.items()]

    return {
        "col": col,
        "top_k": top_k,
        "total_count": total_count,
        "others_count": others_count,
        "rows": rows,
    }


# -------------------------
# BAR / LINE / AREA (GroupBy + Agg)
# -------------------------
@router.get("/{dataset_id}/charts/aggregate")
def aggregate(
    project_id: int,
    dataset_id: int,
    x: str = Query(..., description="Group-by (categorical/text/datetime)"),
    y: Optional[str] = Query(None, description="Numeric column (optional when agg=count)"),
    agg: Optional[AggFn] = Query(None),
    top_k: int = Query(15, ge=1, le=MAX_TOPK),
    order: SortOrder = Query("desc"),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Bar/Line/Area style Dataiku:
      - group by X
      - aggregate Y with agg (optional: count if y missing)
      - sort by result
      - return top_k rows
    """
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    if agg is None:
        agg = "count" if not y else "avg"

    if agg != "count" and not y:
        raise HTTPException(status_code=400, detail="Parameter 'y' is required for this aggregation.")

    fp = Path(ds.file_path)

    usecols = [x] if (agg == "count" or not y) else [x, y]
    df = read_df(fp, usecols=usecols)
    df.columns = df.columns.astype(str)

    req_cols = [x] if (agg == "count" or not y) else [x, y]
    _ensure_cols_exist(df, req_cols)

    if _is_numeric(df[x]):
        raise HTTPException(status_code=400, detail="X must be categorical/text/datetime (numeric X not supported yet).")

    # dropna rules
    if dropna:
        df = df.dropna(subset=req_cols)
    else:
        df = df.dropna(subset=[x])

    if df.empty:
        return {"x": x, "y": y, "agg": agg, "top_k": top_k, "order": order, "rows": []}

    # compute aggregation
    if agg == "count":
        s = df.groupby(x, dropna=False).size()
    else:
        df[y] = _to_numeric(df[y])
        if dropna:
            df = df.dropna(subset=[x, y])

        g = df.groupby(x, dropna=False)[y]
        if agg == "sum":
            s = g.sum()
        elif agg == "avg":
            s = g.mean()
        elif agg == "min":
            s = g.min()
        elif agg == "max":
            s = g.max()
        elif agg == "median":
            s = g.median()
        else:
            raise HTTPException(status_code=400, detail="Invalid agg")

    # hard cap to avoid huge payloads
    if len(s) > MAX_GROUPS:
        s = s.sort_values(ascending=False).head(MAX_GROUPS)

    asc = order == "asc"
    s = s.sort_values(ascending=asc).head(top_k)

    rows = [{"x": str(k), "y": float(v) if pd.notna(v) else 0.0} for k, v in s.items()]
    return {"x": x, "y": y, "agg": agg, "top_k": top_k, "order": order, "rows": rows}


# -------------------------
# HISTOGRAM
# -------------------------
@router.get("/{dataset_id}/charts/hist")
def histogram(
    project_id: int,
    dataset_id: int,
    col: str = Query(...),
    bins: int = Query(20, ge=5, le=200),
    dropna: bool = Query(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    fp = Path(ds.file_path)
    df = read_df(fp, usecols=[col])
    df.columns = df.columns.astype(str)
    _ensure_cols_exist(df, [col])

    s = _to_numeric(df[col])
    if dropna:
        s = s.dropna()

    if s.empty:
        return {"col": col, "bins": bins, "rows": []}

    x = s.to_numpy(dtype=float)

    # if constant value, np.histogram is ok but edges collapse; we handle it
    if np.nanmin(x) == np.nanmax(x):
        v = float(np.nanmin(x))
        rows = [{"x0": v - 0.5, "x1": v + 0.5, "count": int(len(x))}]
        return {"col": col, "bins": 1, "rows": rows}

    counts, edges = np.histogram(x, bins=bins)
    rows = []
    for i in range(len(counts)):
        rows.append({"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])})

    return {"col": col, "bins": bins, "rows": rows}


# -------------------------
# SAMPLE (Scatter / Bubble)
# -------------------------
@router.get("/{dataset_id}/charts/sample")
def sample_rows(
    project_id: int,
    dataset_id: int,
    cols: List[str] = Query(..., description="Columns to fetch"),
    n: int = Query(400, ge=10, le=MAX_SAMPLE),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    if not cols:
        raise HTTPException(status_code=400, detail="Parameter 'cols' is required.")

    fp = Path(ds.file_path)
    df = read_df(fp, usecols=cols)
    df.columns = df.columns.astype(str)
    _ensure_cols_exist(df, cols)

    if df.empty:
        return {"cols": cols, "rows": []}

    # random sample for better representativity
    k = min(n, len(df))
    sampled = df.sample(n=k, random_state=0) if len(df) > k else df

    # return JSON-serializable values
    rows = sampled.replace({np.nan: None}).to_dict(orient="records")
    return {"cols": cols, "rows": rows}


# -------------------------
# NORMALITY TEST
# -------------------------
class NormalityTestIn(BaseModel):
    columns: List[str]


@router.post("/{dataset_id}/normality-test")
def normality_test(
    project_id: int,
    dataset_id: int,
    body: NormalityTestIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Test de normalité par colonne numérique.
    - Shapiro-Wilk pour n <= 5000 (scipy.stats.shapiro)
    - D'Agostino-Pearson pour n > 5000 (scipy.stats.normaltest)
    Retourne stat, p_value, is_normal (seuil α = 0.05) + descriptives.
    """
    ensure_project_owner(db, project_id, current_user.id)
    ds = get_dataset_or_404(db, project_id, dataset_id)

    if not body.columns:
        raise HTTPException(status_code=400, detail="At least one column is required.")

    fp = Path(ds.file_path)
    df = read_df(fp, usecols=body.columns)
    df.columns = df.columns.astype(str)
    _ensure_cols_exist(df, body.columns)

    results = []
    for col in body.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' has fewer than 3 non-null numeric values.",
            )

        data = s.to_numpy(dtype=float)
        n = len(data)

        if n <= 5000:
            stat, p_value = stats.shapiro(data)
            test_used = "shapiro"
        else:
            stat, p_value = stats.normaltest(data)
            test_used = "dagostino"

        skewness = float(stats.skew(data))
        # scipy kurtosis() retourne l'excès de kurtosis (Fisher) → on ajoute 3 pour Pearson
        kurtosis = float(stats.kurtosis(data) + 3)

        results.append({
            "col": col,
            "n": n,
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "test_used": test_used,
            "stat": float(stat),
            "p_value": float(p_value),
            "is_normal": bool(p_value > 0.05),
        })

    return {"results": results}
