from pydantic import BaseModel
from typing import Dict, Optional, Literal, List

Kind = Literal["numeric", "categorical", "text", "binary", "datetime", "id", "other"]

class ColumnKindsIn(BaseModel):
    # mapping: column_name -> kind
    # None => reset override
    overrides: Dict[str, Optional[Kind]]

class ColumnMetaOut(BaseModel):
    name: str
    dtype: str

    # effective kind (override or inferred)
    kind: Kind
    inferred_kind: Kind
    override_kind: Optional[Kind] = None
    confidence: float

    missing: int
    unique: int
    total: int
    sample: List[str] = []

class ColumnsMetaOut(BaseModel):
    columns: List[ColumnMetaOut]
    counts: Dict[str, int] = {}
    total_rows: int
