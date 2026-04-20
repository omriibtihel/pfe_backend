from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.training.base import TaskType, DatasetProfileOut


class RecommendIn(BaseModel):
    """Input for the /recommend endpoint."""
    version_id: int = Field(..., ge=1)
    target_column: str = Field(..., min_length=1)
    # Optional: user may provide context hints
    task_type_hint: Optional[TaskType] = None


class TrainingRecommendationOut(BaseModel):
    """Output of the /recommend endpoint."""
    mode: str  # "recommendation"
    recommended_models: List[str]
    recommended_resampling: Optional[str] = None
    apply_threshold: bool
    recommended_metric: str
    secondary_metrics: List[str]
    recommended_cv_strategy: str
    recommended_k_folds: int
    recommended_search_type: str
    recommended_time_budget_s: Optional[int] = None
    recommended_class_weight: Optional[str] = None
    recommended_split: Dict[str, int]
    reasoning: Dict[str, str]
    training_config_payload: Dict[str, Any]
    recommended_power_transform: str = "none"
    recommended_scaling: str = "none"
    recommended_preprocessing: Dict[str, Any] = Field(default_factory=dict)
    recommended_column_configs: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    profile: DatasetProfileOut
