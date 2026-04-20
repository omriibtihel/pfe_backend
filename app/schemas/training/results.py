from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingSessionOut(BaseModel):
    id: int
    project_id: int
    dataset_version_id: Optional[int] = None
    status: str
    progress: int
    config: Dict[str, Any]
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    class Config:
        from_attributes = True


class TrainedModelOut(BaseModel):
    id: int
    session_id: int
    project_id: int
    model_type: str
    task_type: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    created_at: str

    class Config:
        from_attributes = True


class TrainingResultsOut(BaseModel):
    session: TrainingSessionOut
    models: List[TrainedModelOut]


class TrainingValidateOut(BaseModel):
    normalized_config: Dict[str, Any]
    effective_preprocessing_by_column: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    error_details: List[Dict[str, Any]] = Field(default_factory=list)
    previewTransformed: Optional[Dict[str, Any]] = None
    previewMeta: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Prediction schemas
# ──────────────────────────────────────────────────────────────────────────────

class ManualPredictIn(BaseModel):
    """Payload for manual (JSON rows) prediction."""
    rows: List[Dict[str, Any]] = Field(..., min_length=1)


class ActiveModelOut(BaseModel):
    """Active model info exposed for the prediction UI."""
    modelId: int
    sessionId: int
    modelType: str
    taskType: str
    featureNames: List[str]
    threshold: float
    trainedAt: str
