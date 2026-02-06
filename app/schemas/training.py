from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, conint

TaskType = Literal["classification", "regression"]
SplitMethod = Literal["holdout", "kfold"]

# côté frontend: ModelType / MetricType
ModelType = Literal[
    "randomforest",
    "logisticregression",
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
    "roc_auc",
    "mae",
    "mse",
    "rmse",
    "r2",
]


class TrainingConfigIn(BaseModel):
    targetColumn: str = Field(..., min_length=1)
    taskType: TaskType
    models: List[ModelType] = Field(default_factory=list)
    useGridSearch: bool = False
    useSmote: bool = False
    splitMethod: SplitMethod = "holdout"

    trainRatio: conint(ge=1, le=98) = 70
    valRatio: conint(ge=0, le=98) = 15
    testRatio: conint(ge=1, le=98) = 15

    kFolds: conint(ge=2, le=20) = 5
    metrics: List[MetricType] = Field(default_factory=list)

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
