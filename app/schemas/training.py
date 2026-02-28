from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, conint, field_validator

TaskType = Literal["classification", "regression"]
SplitMethod = Literal["holdout", "kfold", "stratified_kfold"]
PreviewSubset = Literal["train", "val", "test"]
PreviewMode = Literal["head", "random"]
NumericImputationMethod = Literal["none", "median", "mean", "most_frequent", "constant", "knn"]
CategoricalImputationMethod = Literal["none", "most_frequent", "constant"]
CategoricalEncodingMethod = Literal["none", "onehot", "label", "ordinal"]
NumericScalingMethod = Literal["none", "standard", "minmax", "robust", "maxabs"]
ColumnType = Literal["numeric", "categorical", "ordinal"]
StrategyChoice = Literal[
    "none",
    "class_weight",
    "smote",
    "smote_tomek",
    "random_undersampling",
    "threshold_optimization",
]
ThresholdStrategy = Literal["maximize_f1", "maximize_f2", "min_recall", "precision_recall_balance"]
ImbalanceLevel = Literal["balanced", "mild", "moderate", "severe", "critical"]
DatasetScale = Literal["tiny", "small", "medium", "large"]
ImpactLevel = Literal["low", "medium", "high"]

# frontend-side aliases: ModelType / MetricType
ModelType = Literal[
    "randomforest",
    "logisticregression",
    "logreg",
    "svm",
    "xgboost",
    "lightgbm",
    "knn",
    "naivebayes",
    "decisiontree",
]
MetricType = Literal[
    "accuracy",
    "f1",
    "precision",
    "recall",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "roc_auc",
    "pr_auc",
    "f1_pos",
    "confusion_matrix",
    "mae",
    "mse",
    "rmse",
    "r2",
]


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


class BalancingConfigIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    strategy: StrategyChoice = "none"
    apply_threshold: bool = False
    threshold_strategy: ThresholdStrategy = "maximize_f1"
    min_recall_constraint: float | None = None

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
    balancing: BalancingConfigIn = Field(default_factory=BalancingConfigIn)
    splitMethod: SplitMethod = "holdout"

    trainRatio: conint(ge=1, le=98) = 70
    valRatio: conint(ge=0, le=98) = 15
    testRatio: conint(ge=0, le=98) = 15

    kFolds: conint(ge=2, le=20) = 5
    shuffle: bool = True
    metrics: List[MetricType] = Field(default_factory=list)
    positiveLabel: Optional[Any] = None
    debug: bool = False
    trainingDebug: Optional[bool] = None
    preprocessing: PreprocessingConfigIn = Field(default_factory=PreprocessingConfigIn)
    modelHyperparams: Dict[str, Any] = Field(default_factory=dict)
    include: Optional[TrainingValidateIncludeIn] = None
    preview: Optional[TrainingValidatePreviewIn] = None

    # Stored but never executed server side.
    customCode: str = ""


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
