from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LimeItemIn(BaseModel):
    """One LIME contribution as returned by /predict/json/explain."""
    feature: str
    contribution: float = 0.0
    data: Any | None = None


class ReportGenerateIn(BaseModel):
    """Body for the report generation endpoint.

    The frontend already has the prediction result in memory (it just got it
    from /predict/json/explain). Rather than persisting predictions just to
    re-fetch them here, the client posts the per-row payload directly.
    """
    prediction: Any
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    lime: list[LimeItemIn] = Field(default_factory=list)
    input_data: dict[str, Any] = Field(default_factory=dict)
    row_index: int = 0
