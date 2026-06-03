from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────────────
# Legacy schemas (kept for backward compat — not used by routes)
# ──────────────────────────────────────────────────────────────────────────────

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


class ManualPredictIn(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1)


class ActiveModelOut(BaseModel):
    modelId: int
    sessionId: int
    modelType: str
    taskType: Literal["classification", "regression"]
    featureNames: List[str]
    threshold: float
    trainedAt: str


# ──────────────────────────────────────────────────────────────────────────────
# Sub-objects — ModelResultResponse (lightweight, used in list)
# ──────────────────────────────────────────────────────────────────────────────

class PrimaryMetric(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    value: Optional[float] = None
    displayName: str
    status: Literal["success", "not_applicable", "error"] = "success"
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"


class MetricsSummary(BaseModel):
    """Selected metrics for list views and comparison table."""
    model_config = ConfigDict(populate_by_name=True)

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    rocAuc: Optional[float] = None
    prAuc: Optional[float] = None
    balancedAccuracy: Optional[float] = None
    specificity: Optional[float] = None
    f1Pos: Optional[float] = None
    precisionPos: Optional[float] = None
    recallPos: Optional[float] = None
    precisionMacro: Optional[float] = None
    recallMacro: Optional[float] = None
    f1Macro: Optional[float] = None
    precisionWeighted: Optional[float] = None
    recallWeighted: Optional[float] = None
    f1Weighted: Optional[float] = None
    precisionMicro: Optional[float] = None
    recallMicro: Optional[float] = None
    f1Micro: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    mape: Optional[float] = None


class SplitSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    method: Optional[str] = None
    trainRows: Optional[int] = None
    valRows: Optional[int] = None
    testRows: Optional[int] = None


class AutoMLInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    isBest: bool = True
    bestEstimator: Optional[str] = None
    nIterations: Optional[int] = None
    totalTimeS: Optional[float] = None
    timeBudgetS: Optional[float] = None
    metricOptimized: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# ModelResultResponse — réponse légère (liste de modèles)
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationSource(BaseModel):
    type: Literal[
        "holdout_test", "cv_mean", "loo", "validation", "train_only", "unknown"
    ] = "unknown"
    label: str = "Inconnu"
    isIndependentTest: bool = False
    nSamples: Optional[int] = None


class ModelResultResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    modelType: str
    taskType: Literal["classification", "regression"]
    primaryMetric: PrimaryMetric
    metrics: MetricsSummary
    # trainScore is populated for ALL models, including train_only sessions.
    # For train_only sessions it is the sole numeric score (testScore is null);
    # for holdout sessions the train/test gap indicates overfitting.
    # Always check evaluationSource.type before interpreting this value.
    trainScore: Optional[float] = Field(
        default=None,
        description=(
            "Training-set metric value for the primary metric. "
            "Populated even when evaluationSource.type == 'train_only'; "
            "in that case testScore is null and this is the only score available. "
            "Check evaluationSource.type before treating this as a generalisation score."
        ),
    )
    testScore: Optional[float] = None
    trainingTime: float
    isSaved: bool
    isActive: bool
    isCV: bool = False
    hasHoldoutTest: bool = False
    testIsCvMean: bool = False
    testLabel: Optional[str] = None
    splitInfo: Optional[SplitSummary] = None
    automl: Optional[AutoMLInfo] = None
    evaluationSource: EvaluationSource = Field(default_factory=EvaluationSource)
    warnings: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-objects — ModelResultDetailResponse (full, used by /details endpoint)
# ──────────────────────────────────────────────────────────────────────────────

class CVInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kFoldsUsed: Optional[int] = None
    nestedCv: bool = False
    cvSummary: Optional[Dict[str, Any]] = None
    cvFoldResults: Optional[List[Any]] = None
    cvMeanMetrics: Optional[Dict[str, Any]] = None
    cvTestMetrics: Optional[Dict[str, Any]] = None


class ThresholdInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    strategy: Optional[str] = None
    optimalThreshold: Optional[float] = None
    improvementDelta: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class GridSearchInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    searchType: Optional[str] = None
    cvBestScore: Optional[float] = None
    cvScoring: Optional[str] = None
    bestParams: Optional[Dict[str, Any]] = None
    cvSplits: Optional[int] = None
    nCandidates: Optional[int] = None
    cvResultsSummary: Optional[List[Dict[str, Any]]] = None
    gridSearchFailed: bool = False
    gridSearchFailureReason: Optional[str] = None
    warnings: List[Dict[str, Any]] = Field(default_factory=list)


class BalancingInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    strategyApplied: Optional[str] = None
    refitMetric: Optional[str] = None
    imbalanceRatio: Optional[float] = None


class AnalysisBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    crossValidation: Optional[CVInfo] = None
    thresholding: Optional[ThresholdInfo] = None
    gridSearch: Optional[GridSearchInfo] = None
    residualAnalysis: Optional[Dict[str, Any]] = None
    confusionMatrix: Optional[List[Any]] = None
    classDistribution: Optional[Dict[str, Any]] = None
    baseline: Optional[Dict[str, Any]] = None
    metricsWarnings: List[str] = Field(default_factory=list)
    artifactWarnings: List[Dict[str, Any]] = Field(default_factory=list)


class ModelResultDetailResponse(ModelResultResponse):
    """Réponse complète retournée par GET /models/{id}/details."""
    metricsDetailed: Dict[str, Any] = Field(default_factory=dict)
    analysis: AnalysisBlock = Field(default_factory=AnalysisBlock)
    preprocessing: Optional[Dict[str, Any]] = None
    balancing: Optional[BalancingInfo] = None
    hyperparams: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Réponses des endpoints séparés /explainability et /curves
# ──────────────────────────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature: str
    importance: float


class ExplainabilityResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    featureImportance: List[FeatureImportanceItem] = Field(default_factory=list)
    permutationImportance: Optional[List[Dict[str, Any]]] = None
    shapGlobal: Optional[Dict[str, Any]] = None
    artifactWarnings: List[Dict[str, Any]] = Field(default_factory=list)


class CurvesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    roc: Optional[List[Any]] = None
    pr: Optional[List[Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    multiclassRoc: Optional[Dict[str, Any]] = None
    multiclassPr: Optional[Dict[str, Any]] = None
    learningCurves: Optional[Dict[str, Any]] = None
    artifactWarnings: List[Dict[str, Any]] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Session response
# ──────────────────────────────────────────────────────────────────────────────

class TrainingSessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    projectId: str
    datasetVersionId: Optional[str] = None
    name: Optional[str] = None
    status: str
    progress: int
    currentModel: Optional[str] = None
    errorMessage: Optional[str] = None
    activeModelId: Optional[str] = None
    config: Dict[str, Any]
    results: List[ModelResultResponse]
    createdAt: Optional[str] = None
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Saved models list response
# ──────────────────────────────────────────────────────────────────────────────

class SavedModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    modelType: str
    taskType: Literal["classification", "regression"]
    sessionId: str
    datasetVersionId: Optional[str] = None
    datasetVersionName: Optional[str] = None
    isActive: bool
    isSaved: bool
    featureNames: List[str]
    threshold: float
    trainedAt: str
    primaryMetric: Optional[PrimaryMetric] = None
    testScore: Optional[float] = None
    trainingTime: Optional[float] = None
