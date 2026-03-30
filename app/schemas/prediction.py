from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRowExport(BaseModel):
    """One row from a prediction response, sent back for CSV export."""
    row_index: int
    prediction: Any
    score: Optional[float] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)


class PredictionResultsExportIn(BaseModel):
    """Payload for exporting already-computed prediction results as CSV."""
    model_type: str = "model"
    rows: List[PredictionRowExport] = Field(..., min_length=1)
