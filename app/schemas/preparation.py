from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Types needed by preparation schemas (defined here to avoid circular imports)
ImpactLevel = Literal["low", "medium", "high"]
ImbalanceLevel = Literal["balanced", "mild", "moderate", "severe", "critical"]
DatasetScale = Literal["tiny", "small", "medium", "large"]
StrategyChoice = Literal[
    "none",
    "class_weight",
    "smote",
    "smote_tomek",
    "random_undersampling",
    "threshold_optimization",
]


# ──────────────────────────────────────────────────────────────────────────────
# Balance analysis schemas
# ──────────────────────────────────────────────────────────────────────────────

class BinaryClassProfileOut(BaseModel):
    label: Any
    count: int
    ratio: float
    role: Literal["majority", "minority"]


class AvailableStrategyOut(BaseModel):
    id: StrategyChoice
    label: str
    description: str
    impact: ImpactLevel
    recommended: bool
    feasible: bool
    infeasible_reason: str | None = None


class BalanceAnalysisResponse(BaseModel):
    needs_balancing: bool
    imbalance_level: ImbalanceLevel
    imbalance_ratio: float
    minority_ratio: float
    n_samples: int
    dataset_scale: DatasetScale
    majority: BinaryClassProfileOut
    minority: BinaryClassProfileOut
    summary_message: str
    warnings: List[str] = Field(default_factory=list)
    metric_advice: List[str] = Field(default_factory=list)
    available_strategies: List[AvailableStrategyOut] = Field(default_factory=list)
    default_recommendation: StrategyChoice


class BalanceAnalysisIn(BaseModel):
    version_id: int = Field(..., ge=1)
    target_column: str = Field(..., min_length=1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset profiling schemas
# ──────────────────────────────────────────────────────────────────────────────

class DatasetProfileIn(BaseModel):
    version_id: int = Field(..., ge=1)
    target_column: str = Field(..., min_length=1)


class FeatureTypesOut(BaseModel):
    numeric: int
    categorical: int
    text: int


class DatasetProfileOut(BaseModel):
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    task_type: str
    imbalance_ratio: Optional[float] = None
    minority_ratio: Optional[float] = None
    has_missing_values: bool
    missing_ratio: float
    feature_types: FeatureTypesOut
    dimensionality_ratio: float
    dataset_size_category: str
    estimated_training_speed: str
    recommended_cv_strategy: str
    recommended_resampling: Optional[str] = None
    recommended_metric: str
    meta_features: Dict[str, Any] = Field(default_factory=dict)
