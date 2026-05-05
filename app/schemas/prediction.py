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


# ── Counterfactual schemas ─────────────────────────────────────────────────────

class CounterfactualItem(BaseModel):
    """One changed feature in a counterfactual explanation."""
    feature: str
    original_value: Optional[float] = None
    suggested_value: Optional[float] = None
    delta: Optional[float] = None


class CounterfactualIn(BaseModel):
    """Request body for the counterfactual endpoint."""
    row: Dict[str, Any]
    features_to_vary: List[str] = Field(..., min_length=1)
    n_counterfactuals: int = Field(default=3, ge=1, le=5)


class CounterfactualOut(BaseModel):
    """Response from the counterfactual endpoint."""
    model_id: int
    task_type: str
    original_prediction: Any
    original_score: Optional[float] = None
    counterfactual: Optional[List[CounterfactualItem]] = None
    message: Optional[str] = None


# ── Feature ranges (interactive counterfactual sliders) ────────────────────────

class FeatureRangeOut(BaseModel):
    """Per-feature statistics used to bound an interactive slider."""
    type: str  # "numeric" | "categorical" | "unknown"
    # Numeric:
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    # Categorical:
    categories: Optional[List[str]] = None


class FeatureRangesOut(BaseModel):
    model_id: int
    ranges: Dict[str, FeatureRangeOut]
