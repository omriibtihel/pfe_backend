from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, conint

TaskType = Literal["classification", "regression"]
SplitMethod = Literal["holdout", "kfold"]
PreviewSubset = Literal["train", "val", "test"]
PreviewMode = Literal["head", "random"]
NumericImputationMethod = Literal["none", "median", "mean", "most_frequent", "constant", "knn"]
CategoricalImputationMethod = Literal["none", "most_frequent", "constant"]
CategoricalEncodingMethod = Literal["none", "onehot", "label", "ordinal"]
NumericScalingMethod = Literal["none", "standard", "minmax", "robust", "maxabs"]
ColumnType = Literal["numeric", "categorical", "ordinal"]

# côté frontend: ModelType / MetricType
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
    splitMethod: SplitMethod = "holdout"

    trainRatio: conint(ge=1, le=98) = 70
    valRatio: conint(ge=0, le=98) = 15
    testRatio: conint(ge=1, le=98) = 15

    kFolds: conint(ge=2, le=20) = 5
    metrics: List[MetricType] = Field(default_factory=list)
    positiveLabel: Optional[Any] = None
    debug: bool = False
    trainingDebug: Optional[bool] = None
    preprocessing: PreprocessingConfigIn = Field(default_factory=PreprocessingConfigIn)
    modelHyperparams: Dict[str, Any] = Field(default_factory=dict)
    include: Optional[TrainingValidateIncludeIn] = None
    preview: Optional[TrainingValidatePreviewIn] = None

    # ⚠️ on peut stocker le code, mais NE PAS l'exécuter côté serveur (risque sécurité)
    customCode: str = ""


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
