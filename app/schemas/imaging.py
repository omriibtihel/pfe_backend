"""
Schémas Pydantic pour le pipeline imagerie.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────────────
# Input schemas
# ──────────────────────────────────────────────────────────────────────────────

class AugmentationConfigIn(BaseModel):
    horizontalFlip: bool = True
    verticalFlip: bool = False
    rotationDegrees: int = Field(15, ge=0, le=180)
    brightnessLimit: float = Field(0.2, ge=0.0, le=1.0)
    contrastLimit: float = Field(0.2, ge=0.0, le=1.0)
    normalizeMean: List[float] = [0.485, 0.456, 0.406]
    normalizeStd: List[float] = [0.229, 0.224, 0.225]

    model_config = ConfigDict(populate_by_name=True)


class ImagingConfigIn(BaseModel):
    taskType: Literal["image_classification"] = "image_classification"
    models: List[str] = Field(..., min_length=1)
    imageSize: int = Field(224, ge=64, le=512)
    pretrained: bool = True
    epochs: int = Field(20, ge=1, le=300)
    batchSize: int = Field(32, ge=1, le=512)
    learningRate: float = Field(1e-4, gt=0)
    weightDecay: float = Field(1e-5, ge=0)
    freezeBackbone: bool = True
    unfreezeAfterEpoch: int = Field(5, ge=0)
    valSplit: float = Field(0.2, gt=0.05, lt=0.5)
    testSplit: float = Field(0.1, gt=0.0, lt=0.4)
    augmentation: AugmentationConfigIn = Field(default_factory=AugmentationConfigIn)

    model_config = ConfigDict(populate_by_name=True)


# ──────────────────────────────────────────────────────────────────────────────
# Output schemas
# ──────────────────────────────────────────────────────────────────────────────

class ImagingModelResultOut(BaseModel):
    id: int
    session_id: int
    project_id: int
    model_name: str
    task_type: str
    is_saved: bool
    metrics_json: Dict[str, Any]
    artifacts_json: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ImagingSessionOut(BaseModel):
    id: int
    project_id: int
    status: str
    progress: int
    current_model: Optional[str]
    config_json: Dict[str, Any]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    results: List[ImagingModelResultOut] = []

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @classmethod
    def from_orm_with_results(cls, session) -> "ImagingSessionOut":
        results = [
            ImagingModelResultOut.model_validate(m)
            for m in (session.models or [])
        ]
        data = {
            "id": session.id,
            "project_id": session.project_id,
            "status": session.status,
            "progress": session.progress,
            "current_model": session.current_model,
            "config_json": session.config_json or {},
            "error_message": session.error_message,
            "created_at": session.created_at,
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "results": results,
        }
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

class ImagingPredictionOut(BaseModel):
    model_id: int
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


# ──────────────────────────────────────────────────────────────────────────────
# Image listing
# ──────────────────────────────────────────────────────────────────────────────

class ImageClassInfo(BaseModel):
    name: str
    count: int


class ImageListOut(BaseModel):
    classes: List[ImageClassInfo]
    total_images: int
    images_dir: str
