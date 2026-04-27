# app/schemas/database.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal


from app.schemas.dataset import DatasetOut


from typing import Any
from pydantic import BaseModel, Field


class NumericStats(BaseModel):
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None


class TopValue(BaseModel):
    value: str
    count: int


class CategoricalStats(BaseModel):
    top_values: List[TopValue]



class ColumnProfile(BaseModel):
    name: str
    kind: str  # "numeric" | "categorical" | "text" | "datetime" | "unknown"
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float

    # spécifiques
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None


class CorrelationOut(BaseModel):
    columns: List[str] = Field(default_factory=list)
    matrix: List[List[float]] = Field(default_factory=list)

class ShapeOut(BaseModel):
    rows: int
    cols: int


class DatasetOverviewOut(BaseModel):
    dataset: DatasetOut
    shape: ShapeOut
    columns: List[str]
    dtypes: Dict[str, str]
    missing: Dict[str, int]
    preview: List[Dict[str, Any]]


# -------- Column profiling (simple) --------
class ColumnTopValue(BaseModel):
    value: str
    count: int


class NumericSummary(BaseModel):
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None


class CategoricalSummary(BaseModel):
    unique: int
    top_values: List[ColumnTopValue]


class ParasiteInfo(BaseModel):
    count: int
    distinct: List[str]
    convertible_ratio: float


class ColumnProfile(BaseModel):
    name: str
    kind: Literal["numeric", "categorical", "datetime", "text", "unknown"]
    missing: int
    missing_pct: float = 0.0
    unique: int = 0
    unique_pct: float = 0.0
    dtype: str
    numeric: Optional[NumericSummary] = None
    categorical: Optional[CategoricalSummary] = None
    parasites: Optional[ParasiteInfo] = None


class DatasetProfileOut(BaseModel):
    dataset: DatasetOut
    shape: dict
    profiles: List[ColumnProfile]


# -------- Target --------
class TargetOut(BaseModel):
    target_column: str | None = None


class TargetIn(BaseModel):
    target_column: str | None = None
