from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, conint, field_validator

from app.schemas.training.base import (
    TrainingMode,
    TaskType,
    SplitMethod,
    SearchType,
    PreviewSubset,
    PreviewMode,
    NumericImputationMethod,
    CategoricalImputationMethod,
    CategoricalEncodingMethod,
    NumericScalingMethod,
    ColumnType,
    ThresholdStrategy,
    ModelType,
    MetricType,
    StrategyChoice,
)


class PreprocessingDefaultsIn(BaseModel):
    numericImputation: NumericImputationMethod = "none"
    numericScaling: NumericScalingMethod = "none"
    categoricalImputation: CategoricalImputationMethod = "none"
    categoricalEncoding: CategoricalEncodingMethod = "none"


class PreprocessingColumnConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    use: Optional[bool] = None
    type: Optional[ColumnType] = None
    numericImputation: Optional[NumericImputationMethod] = None
    numericScaling: Optional[NumericScalingMethod] = None
    categoricalImputation: Optional[CategoricalImputationMethod] = None
    categoricalEncoding: Optional[CategoricalEncodingMethod] = None
    ordinalOrder: Optional[List[str]] = None


class PreprocessingConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    defaults: Optional[PreprocessingDefaultsIn] = None
    columns: Dict[str, PreprocessingColumnConfigIn] = Field(default_factory=dict)

    # Backward-compatible keys still accepted from older clients.
    numericImputation: Optional[NumericImputationMethod] = None
    numericScaling: Optional[NumericScalingMethod] = None
    categoricalImputation: Optional[CategoricalImputationMethod] = None
    categoricalEncoding: Optional[CategoricalEncodingMethod] = None
    imputation: Dict[str, Any] = Field(default_factory=dict)
    encoding: Dict[str, Any] = Field(default_factory=dict)
    scaling: Dict[str, Any] = Field(default_factory=dict)
    normalization: Dict[str, Any] = Field(default_factory=dict)


class FeatureDefIn(BaseModel):
    """Single user-defined feature to be created before preprocessing."""
    name: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1)
    enabled: bool = True


class FeatureEngineeringConfigIn(BaseModel):
    features: List[FeatureDefIn] = Field(default_factory=list)


class BalancingConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    strategy: StrategyChoice = "none"
    apply_threshold: bool = False
    threshold_strategy: ThresholdStrategy = "maximize_f1"
    min_recall_constraint: float | None = None
    f_beta: float = Field(default=2.0, ge=0.1, le=10.0)
    cost_fn: float = Field(default=1.0, ge=0.0, le=100.0)
    cost_fp: float = Field(default=1.0, ge=0.0, le=100.0)

    @field_validator("min_recall_constraint")
    @classmethod
    def _validate_min_recall(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not (0.0 < float(value) < 1.0):
            raise ValueError("min_recall_constraint must satisfy 0 < value < 1")
        return float(value)


class TrainingValidateIncludeIn(BaseModel):
    preview: bool = False


class TrainingValidatePreviewIn(BaseModel):
    subset: PreviewSubset = "train"
    mode: PreviewMode = "head"
    n: conint(ge=1, le=500) = 100
    seed: int = 42


class TrainingConfigIn(BaseModel):
    datasetVersionId: Optional[int] = None
    targetColumn: str = Field(..., min_length=1)
    taskType: TaskType
    models: List[ModelType] = Field(default_factory=list)
    useGridSearch: bool = False
    gridCvFolds: conint(ge=2, le=20) = 3
    gridScoring: str = "auto"
    useSmote: bool = False
    searchType: SearchType = "none"
    nIterRandomSearch: conint(ge=5, le=300) = 40
    balancing: BalancingConfigIn = Field(default_factory=BalancingConfigIn)
    splitMethod: SplitMethod = "holdout"

    trainRatio: conint(ge=1, le=98) = 70
    valRatio: conint(ge=0, le=98) = 15
    testRatio: conint(ge=0, le=98) = 15

    kFolds: conint(ge=2, le=20) = 5
    shuffle: bool = True
    nRepeats: conint(ge=1, le=20) = 3
    groupColumn: Optional[str] = None
    metrics: List[MetricType] = Field(default_factory=list)
    positiveLabel: Optional[Any] = None
    debug: bool = False
    trainingDebug: Optional[bool] = None
    preprocessing: PreprocessingConfigIn = Field(default_factory=PreprocessingConfigIn)
    modelHyperparams: Dict[str, Any] = Field(default_factory=dict)
    include: Optional[TrainingValidateIncludeIn] = None
    preview: Optional[TrainingValidatePreviewIn] = None

    featureEngineering: FeatureEngineeringConfigIn = Field(default_factory=FeatureEngineeringConfigIn)

    # Stored but never executed server side.
    customCode: str = ""

    # Mode tracking: "manual" (with or without auto-recommendations) | "automl" (FLAML)
    configMode: TrainingMode = "manual"
    # Keys pre-filled by the recommendation engine that the user may have adjusted
    userOverrides: List[str] = Field(default_factory=list)


class AutoMLConfigIn(BaseModel):
    """Payload for POST /projects/{id}/training/automl."""
    datasetVersionId: int = Field(..., ge=1)
    targetColumn: str = Field(..., min_length=1)
    taskType: TaskType
    timeBudget: int = Field(default=60, ge=10, le=3600)
    metric: Optional[MetricType] = None
    testRatio: float = Field(default=0.2, ge=0.0, le=0.4)
    positiveLabel: Optional[Any] = None
