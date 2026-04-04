from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, conint, field_validator

TrainingMode = Literal["manual", "automl"]

TaskType = Literal["classification", "regression"]
SplitMethod = Literal[
    "holdout",
    "kfold",
    "stratified_kfold",
    "repeated_stratified_kfold",
    "group_kfold",
    "stratified_group_kfold",
    "loo",
]
SearchType = Literal["none", "grid", "random", "halving_random"]
PreviewSubset = Literal["train", "val", "test"]
PreviewMode = Literal["head", "random"]
NumericImputationMethod = Literal["none", "median", "mean", "most_frequent", "constant", "knn"]
CategoricalImputationMethod = Literal["none", "most_frequent", "constant"]
CategoricalEncodingMethod = Literal["none", "onehot", "label", "ordinal"]
NumericScalingMethod = Literal["none", "standard", "minmax", "robust", "maxabs"]
ColumnType = Literal["numeric", "categorical", "ordinal"]
ThresholdStrategy = Literal["maximize_f1", "maximize_f2", "min_recall", "precision_recall_balance"]

# Preparation-related types re-imported from preparation schemas
from app.schemas.preparation import (  # noqa: E402
    ImpactLevel,
    ImbalanceLevel,
    DatasetScale,
    StrategyChoice,
    BinaryClassProfileOut,
    AvailableStrategyOut,
    BalanceAnalysisResponse,
    BalanceAnalysisIn,
    DatasetProfileIn,
    FeatureTypesOut,
    DatasetProfileOut,
)

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
    "extratrees",
    "et",
    "gradientboosting",
    "gb",
    "gbm",
    "ridge",
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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset profiling & recommendation schemas (used by the manual wizard)
# ──────────────────────────────────────────────────────────────────────────────

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
    warnings: List[str] = Field(default_factory=list)
    profile: DatasetProfileOut
