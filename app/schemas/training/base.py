from __future__ import annotations

from typing import Literal

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
NumericPowerTransformMethod = Literal["none", "log", "sqrt", "yeo_johnson", "box_cox"]
ColumnType = Literal["numeric", "categorical", "ordinal"]
ThresholdStrategy = Literal[
    "maximize_f1",
    "maximize_f2",
    "maximize_f_beta",
    "min_recall",
    "precision_recall_balance",
    "youden",
    "minimize_cost",
]

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
    "catboost",
    "cat",
    "cb",
    "ridge",
    "mlp",
    "elasticnet",
    "lasso",
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
    "mape",
    "r2",
]

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
