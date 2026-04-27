from pydantic import BaseModel
from typing import Dict, Optional, Literal, List

Kind = Literal["numeric", "categorical", "text", "binary", "datetime", "id", "other"]

class ColumnKindsIn(BaseModel):
    overrides: Dict[str, Optional[Kind]]

class ParasitesOut(BaseModel):
    count: int
    distinct: List[str]
    convertible_ratio: float

class ColumnMetaOut(BaseModel):
    name: str
    dtype: str

    kind: Kind
    inferred_kind: Kind
    override_kind: Optional[Kind] = None
    confidence: float

    missing: int
    unique: int
    total: int
    sample: List[str] = []

    parasites: Optional[ParasitesOut] = None
    skewness: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_ratio: Optional[float] = None
    has_negative: Optional[bool] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    q1_val: Optional[float] = None
    q3_val: Optional[float] = None

class ColumnsMetaOut(BaseModel):
    columns: List[ColumnMetaOut]
    counts: Dict[str, int] = {}
    total_rows: int
