"""
Literal types, method lists, default constants, and static data structures
(PREPROCESSING_CAPABILITIES, MODEL_HP_SCHEMA).

No imports from other training modules — this file is a pure data definition.
"""
from __future__ import annotations

from typing import Any, Literal

# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------

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
ColumnType = Literal["numeric", "categorical", "ordinal"]
BalancingStrategy = Literal[
    "none",
    "class_weight",
    "smote",
    "smote_tomek",
    "random_undersampling",
    "threshold_optimization",
]
ThresholdStrategy = Literal[
    "maximize_f1",
    "maximize_f2",
    "maximize_f_beta",
    "min_recall",
    "precision_recall_balance",
    "youden",
    "minimize_cost",
]

# ---------------------------------------------------------------------------
# Preprocessing method lists and defaults
# ---------------------------------------------------------------------------

NUMERIC_IMPUTATION_METHODS = ["none", "median", "mean", "most_frequent", "constant", "knn"]
CATEGORICAL_IMPUTATION_METHODS = ["none", "most_frequent", "constant"]
CATEGORICAL_ENCODING_METHODS = ["none", "onehot", "label", "ordinal"]
NUMERIC_SCALING_METHODS = ["none", "standard", "minmax", "robust", "maxabs"]
NUMERIC_POWER_TRANSFORM_METHODS = ["none", "log", "sqrt", "yeo_johnson", "box_cox"]

DEFAULT_NUMERIC_IMPUTATION = "none"
DEFAULT_CATEGORICAL_IMPUTATION = "none"
DEFAULT_CATEGORICAL_ENCODING = "none"
DEFAULT_NUMERIC_SCALING = "none"
DEFAULT_NUMERIC_POWER_TRANSFORM = "none"

SUPPORTED_SPLIT_METHODS = [
    "holdout",
    "kfold",
    "stratified_kfold",
    "repeated_stratified_kfold",
    "group_kfold",
    "stratified_group_kfold",
    "loo",
]
CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
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
]
REGRESSION_METRICS = ["mae", "mse", "rmse", "r2"]
BALANCING_STRATEGIES = [
    "none",
    "class_weight",
    "smote",
    "smote_tomek",
    "random_undersampling",
    "threshold_optimization",
]
THRESHOLD_STRATEGIES = [
    "maximize_f1",
    "maximize_f2",
    "maximize_f_beta",
    "min_recall",
    "precision_recall_balance",
    "youden",
    "minimize_cost",
]

# Canonical frontend contract for Step3.
PREPROCESSING_CAPABILITIES = {
    "numericImputation": NUMERIC_IMPUTATION_METHODS,
    "numericPowerTransform": NUMERIC_POWER_TRANSFORM_METHODS,
    "numericScaling": NUMERIC_SCALING_METHODS,
    "categoricalImputation": CATEGORICAL_IMPUTATION_METHODS,
    "categoricalEncoding": CATEGORICAL_ENCODING_METHODS,
    "columnTypes": ["numeric", "categorical", "ordinal"],
    "supportsPerColumn": True,
    "defaults": {
        "numericImputation": DEFAULT_NUMERIC_IMPUTATION,
        "numericPowerTransform": DEFAULT_NUMERIC_POWER_TRANSFORM,
        "numericScaling": DEFAULT_NUMERIC_SCALING,
        "categoricalImputation": DEFAULT_CATEGORICAL_IMPUTATION,
        "categoricalEncoding": DEFAULT_CATEGORICAL_ENCODING,
    },
}

PREPROCESSING_EXECUTION_POLICY = {
    "stage": "after_split",
    "requiresSplit": True,
    "fitOn": "train_only",
    "transformOn": ["train", "val", "test", "predict"],
    "antiLeakage": True,
}

# ---------------------------------------------------------------------------
# Hyperparameter schema per model (static data — no runtime logic)
# ---------------------------------------------------------------------------

MODEL_HP_SCHEMA: dict[str, dict[str, dict[str, Any]]] = {
    "randomforest": {
        "n_estimators": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 2000,
            "grid_values": [50, 100, 200, 300, 500],
            "help": "Number of trees in the forest.",
        },
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "grid_values": [3, 5, 8, 10, 15, 20, None],
            "help": "Maximum depth of each tree. null means unlimited.",
        },
        "max_features": {
            "type": "enum",
            "enum": ["sqrt", "log2"],
            "default": "sqrt",
            "grid_values": ["sqrt", "log2"],
            "help": "Number of features to consider at each split.",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "grid_values": [1, 2, 4, 10],
            "help": "Minimum number of samples required to be at a leaf node.",
        },
        "min_samples_split": {
            "type": "int",
            "default": 2,
            "min": 2,
            "max": 100,
            "grid_values": [2, 5, 10],
            "help": "Minimum number of samples required to split an internal node.",
        },
        "class_weight": {
            "type": "enum_or_null",
            "enum": ["balanced", "balanced_subsample"],
            "default": None,
            "supported_in": ["classification"],
            "help": "Weight classes inversely proportional to frequency. 'balanced' is recommended for imbalanced datasets. 'balanced_subsample' recomputes weights per bootstrap. null = disabled (default).",
        },
    },
    "logisticregression": {
        "C": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "grid_values": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "help": "Inverse of regularization strength; must be > 0.",
        },
        "solver": {
            "type": "enum",
            "enum": ["saga", "liblinear", "lbfgs", "newton-cg", "newton-cholesky", "sag"],
            "default": "saga",
            "help": "Optimization algorithm. 'saga' supports all regularization types (l1_ratio 0–1).",
        },
        "l1_ratio": {
            "type": "float",
            "default": 0.0,
            "ge": 0.0,
            "le": 1.0,
            "grid_values": [0.0, 0.15, 0.5, 0.85, 1.0],
            "help": "Regularization mix: 0 = pure L2 (Ridge), 1 = pure L1 (Lasso). Replaces deprecated 'penalty' in sklearn ≥ 1.8.",
        },
        "max_iter": {
            "type": "int",
            "default": 4000,
            "min": 100,
            "max": 20000,
            "grid_values": [500, 1000, 2000, 4000],
            "help": "Maximum number of iterations.",
        },
        "class_weight": {
            "type": "enum_or_null",
            "enum": ["balanced"],
            "default": None,
            "supported_in": ["classification"],
            "help": "Weight classes inversely proportional to frequency. 'balanced' is recommended for imbalanced datasets. null = disabled (default).",
        },
    },
    "svm": {
        "C": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "grid_values": [0.01, 0.1, 1.0, 10.0, 100.0],
            "help": "Regularization strength; must be > 0.",
        },
        "kernel": {
            "type": "enum",
            "enum": ["linear", "rbf", "poly", "sigmoid"],
            "default": "rbf",
            "grid_values": ["rbf", "linear"],
            "help": "SVM kernel function.",
        },
        "gamma": {
            "type": "float_or_enum",
            "enum": ["scale", "auto"],
            "default": "scale",
            "gt": 0.0,
            "grid_values": ["scale", "auto", 0.001, 0.01, 0.1],
            "help": "Kernel coefficient. Ignored when kernel='linear'.",
        },
        "epsilon": {
            "type": "float",
            "default": 0.1,
            "ge": 0.0,
            "grid_values": [0.01, 0.1, 0.5],
            "supported_in": ["regression"],
            "help": "Epsilon-insensitive tube width (regression only). Errors smaller than this are not penalized.",
        },
        "class_weight": {
            "type": "enum_or_null",
            "enum": ["balanced"],
            "default": None,
            "supported_in": ["classification"],
            "help": "Weight classes inversely proportional to frequency. 'balanced' is recommended for imbalanced datasets. null = disabled (default).",
        },
    },
    "knn": {
        "n_neighbors": {
            "type": "int",
            "default": 5,
            "min": 1,
            "max": 500,
            "grid_values": [3, 5, 7, 10, 15, 20],
            "help": "Number of neighbors.",
        },
        "weights": {
            "type": "enum",
            "enum": ["uniform", "distance"],
            "default": "uniform",
            "grid_values": ["uniform", "distance"],
            "help": "Weight function used in prediction.",
        },
        "metric": {
            "type": "str",
            "default": "minkowski",
            "help": "Distance metric.",
        },
    },
    "naivebayes": {
        "var_smoothing": {
            "type": "float",
            "default": 1e-9,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            "help": "Portion of largest variance added to variances for stability.",
        },
    },
    "decisiontree": {
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "grid_values": [3, 5, 8, 10, 15, 20, None],
            "help": "Maximum depth of the tree. null means unlimited.",
        },
        "criterion": {
            "type": "enum",
            "enum": ["gini", "entropy", "log_loss"],
            "default": "gini",
            "grid_values": ["gini", "entropy"],
            "supported_in": ["classification"],
            "help": "Function to measure the quality of a split (classification only).",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "grid_values": [1, 4, 10],
            "help": "Minimum number of samples required to be at a leaf node.",
        },
        "class_weight": {
            "type": "enum_or_null",
            "enum": ["balanced"],
            "default": None,
            "supported_in": ["classification"],
            "help": "Weight classes inversely proportional to frequency. 'balanced' is recommended for imbalanced datasets. null = disabled (default).",
        },
    },
    "xgboost": {
        "n_estimators": {
            "type": "int",
            "default": 300,
            "min": 50,
            "max": 5000,
            "grid_values": [100, 200, 300, 400, 500],
            "help": "Number of boosting rounds.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.1,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.01, 0.05, 0.1, 0.15, 0.3],
            "help": "Step size shrinkage in each boosting step.",
        },
        "max_depth": {
            "type": "int",
            "default": 6,
            "min": 1,
            "max": 20,
            "grid_values": [3, 4, 6, 8, 10],
            "help": "Maximum tree depth.",
        },
        "subsample": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.6, 0.8, 0.9, 1.0],
            "help": "Subsample ratio of the training instances.",
        },
        "colsample_bytree": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.5, 0.7, 0.9, 1.0],
            "help": "Subsample ratio of columns when constructing each tree.",
        },
        "min_child_weight": {
            "type": "float",
            "default": 1.0,
            "ge": 0.0,
            "grid_values": [1, 3, 5],
            "help": "Minimum sum of instance weight (hessian) needed in a child node. Higher values reduce overfitting.",
        },
        "reg_alpha": {
            "type": "float",
            "default": 0.0,
            "ge": 0.0,
            "grid_values": [0, 0.1, 1.0],
            "help": "L1 regularization on leaf weights. Higher values reduce overfitting.",
        },
    },
    "lightgbm": {
        "n_estimators": {
            "type": "int",
            "default": 500,
            "min": 50,
            "max": 5000,
            "grid_values": [100, 200, 300, 500],
            "help": "Number of boosted trees.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.05,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.01, 0.05, 0.1, 0.2],
            "help": "Boosting learning rate.",
        },
        "num_leaves": {
            "type": "int",
            "default": 31,
            "min": 2,
            "max": 2048,
            "grid_values": [15, 31, 63, 127],
            "help": "Maximum number of leaves in one tree.",
        },
        "feature_fraction": {
            "type": "float",
            "default": 0.9,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.7, 0.8, 0.9, 1.0],
            "help": "Fraction of features used for each iteration.",
        },
        "bagging_fraction": {
            "type": "float",
            "default": 0.9,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.7, 0.8, 0.9, 1.0],
            "help": "Fraction of data sampled per tree (requires bagging_freq > 0, set automatically).",
        },
        "min_child_samples": {
            "type": "int",
            "default": 20,
            "min": 1,
            "max": 500,
            "grid_values": [10, 20, 30],
            "help": "Minimum number of samples in a leaf. Higher values reduce overfitting.",
        },
        "reg_alpha": {
            "type": "float",
            "default": 0.0,
            "ge": 0.0,
            "grid_values": [0, 0.1, 1.0],
            "help": "L1 regularization on leaf weights.",
        },
        "reg_lambda": {
            "type": "float",
            "default": 0.0,
            "ge": 0.0,
            "grid_values": [0.1, 1.0, 5.0],
            "help": "L2 regularization on leaf weights.",
        },
    },
    "extratrees": {
        "n_estimators": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 2000,
            "grid_values": [50, 100, 200, 300, 500],
            "help": "Number of trees in the forest.",
        },
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "grid_values": [5, 10, 20, None],
            "help": "Maximum depth of each tree. null means unlimited.",
        },
        "max_features": {
            "type": "enum",
            "enum": ["sqrt", "log2"],
            "default": "sqrt",
            "grid_values": ["sqrt", "log2"],
            "help": "Number of features to consider at each split.",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "grid_values": [1, 2, 5, 10],
            "help": "Minimum number of samples required to be at a leaf node.",
        },
        "min_samples_split": {
            "type": "int",
            "default": 2,
            "min": 2,
            "max": 100,
            "grid_values": [2, 5, 10],
            "help": "Minimum number of samples required to split an internal node.",
        },
        "class_weight": {
            "type": "enum_or_null",
            "enum": ["balanced", "balanced_subsample"],
            "default": None,
            "supported_in": ["classification"],
            "help": "Weight classes inversely proportional to frequency. null = disabled (default).",
        },
    },
    "gradientboosting": {
        "n_estimators": {
            "type": "int",
            "default": 200,
            "min": 50,
            "max": 2000,
            "grid_values": [100, 200, 300, 500],
            "help": "Number of boosting stages.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.1,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.01, 0.05, 0.1, 0.2],
            "help": "Step size shrinkage in each boosting stage.",
        },
        "max_depth": {
            "type": "int",
            "default": 3,
            "min": 1,
            "max": 20,
            "grid_values": [3, 5, 8, 10],
            "help": "Maximum depth of the individual estimators.",
        },
        "subsample": {
            "type": "float",
            "default": 0.8,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.6, 0.7, 0.8, 1.0],
            "help": "Fraction of samples used for fitting each tree. < 1 enables stochastic gradient boosting.",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "grid_values": [1, 5, 10, 20],
            "help": "Minimum number of samples required to be at a leaf node.",
        },
        "max_features": {
            "type": "enum_or_null",
            "enum": ["sqrt", "log2"],
            "default": None,
            "help": "Features considered at each split. null = all features (default). 'sqrt'/'log2' reduce overfitting.",
        },
    },
    "mlp": {
        "hidden_layer_sizes": {
            "type": "enum",
            "enum": ["100", "100,50", "50,50,50", "200", "200,100", "64,32", "128,64,32"],
            "default": "100",
            "grid_values": ["100", "100,50", "50,50,50", "200", "200,100"],
            "help": "Architecture du réseau : neurones par couche cachée, séparés par des virgules. Ex: '100,50' = 2 couches (100 puis 50 neurones).",
        },
        "activation": {
            "type": "enum",
            "enum": ["relu", "tanh", "logistic"],
            "default": "relu",
            "grid_values": ["relu", "tanh"],
            "help": "Fonction d'activation des neurones cachés. relu est recommandé en général.",
        },
        "alpha": {
            "type": "float",
            "default": 0.0001,
            "gt": 0.0,
            "grid_values": [0.0001, 0.001, 0.01, 0.1],
            "help": "Régularisation L2 — réduit le surapprentissage. Augmenter si le modèle overfite.",
        },
        "learning_rate_init": {
            "type": "float",
            "default": 0.001,
            "gt": 0.0,
            "grid_values": [0.0001, 0.001, 0.01],
            "help": "Taux d'apprentissage initial (solver=adam). Trop élevé → instabilité ; trop bas → convergence lente.",
        },
        "max_iter": {
            "type": "int",
            "default": 500,
            "min": 50,
            "max": 5000,
            "grid_values": [200, 500, 1000],
            "help": "Nombre maximal d'itérations d'optimisation.",
        },
    },
    "elasticnet": {
        "alpha": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "grid_values": [0.001, 0.01, 0.1, 1.0, 10.0],
            "help": "Force globale de régularisation. Augmenter pour réduire l'overfitting et la variance.",
        },
        "l1_ratio": {
            "type": "float",
            "default": 0.5,
            "ge": 0.0,
            "le": 1.0,
            "grid_values": [0.1, 0.3, 0.5, 0.7, 0.9],
            "help": "Mélange L1/L2 : 0 = Ridge pur (L2), 1 = Lasso pur (L1), 0.5 = mélange égal.",
        },
        "max_iter": {
            "type": "int",
            "default": 2000,
            "min": 100,
            "max": 20000,
            "grid_values": [500, 1000, 2000],
            "help": "Nombre maximal d'itérations pour la descente de coordonnées.",
        },
    },
    "lasso": {
        "alpha": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "grid_values": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "help": "Force de régularisation L1 — favorise la sélection de variables (coefficients mis à zéro).",
        },
        "max_iter": {
            "type": "int",
            "default": 2000,
            "min": 100,
            "max": 20000,
            "grid_values": [500, 1000, 2000],
            "help": "Nombre maximal d'itérations pour la descente de coordonnées.",
        },
    },
    "ridge": {
        "alpha": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "grid_values": [0.01, 0.1, 1.0, 10.0, 100.0],
            "supported_in": ["regression"],
            "help": "Force de régularisation L2. Plus alpha est élevé, plus les coefficients sont contraints (réduit l'overfitting).",
        },
    },
    "catboost": {
        "iterations": {
            "type": "int",
            "default": 500,
            "min": 50,
            "max": 5000,
            "grid_values": [100, 200, 300, 500],
            "help": "Number of boosting iterations.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.05,
            "gt": 0.0,
            "max": 1.0,
            "grid_values": [0.01, 0.03, 0.05, 0.1, 0.2],
            "help": "Step size shrinkage in each boosting step.",
        },
        "depth": {
            "type": "int",
            "default": 6,
            "min": 1,
            "max": 16,
            "grid_values": [4, 6, 8, 10],
            "help": "Depth of the trees. Typical range: 4–10.",
        },
        "l2_leaf_reg": {
            "type": "float",
            "default": 3.0,
            "gt": 0.0,
            "grid_values": [1.0, 3.0, 5.0, 10.0],
            "help": "L2 regularization coefficient. Higher values reduce overfitting.",
        },
    },
}
