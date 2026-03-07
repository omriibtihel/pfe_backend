from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .models import MODEL_REGISTRY, list_available_models

TaskType = Literal["classification", "regression"]
SplitMethod = Literal["holdout", "kfold", "stratified_kfold"]
ColumnType = Literal["numeric", "categorical", "ordinal"]
BalancingStrategy = Literal[
    "none",
    "class_weight",
    "smote",
    "smote_tomek",
    "random_undersampling",
    "threshold_optimization",
]
ThresholdStrategy = Literal["maximize_f1", "maximize_f2", "min_recall", "precision_recall_balance"]

NUMERIC_IMPUTATION_METHODS = ["none", "median", "mean", "most_frequent", "constant", "knn"]
CATEGORICAL_IMPUTATION_METHODS = ["none", "most_frequent", "constant"]
CATEGORICAL_ENCODING_METHODS = ["none", "onehot", "label", "ordinal"]
NUMERIC_SCALING_METHODS = ["none", "standard", "minmax", "robust", "maxabs"]

DEFAULT_NUMERIC_IMPUTATION = "none"
DEFAULT_CATEGORICAL_IMPUTATION = "none"
DEFAULT_CATEGORICAL_ENCODING = "none"
DEFAULT_NUMERIC_SCALING = "none"

SUPPORTED_SPLIT_METHODS = ["holdout", "kfold", "stratified_kfold"]
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
THRESHOLD_STRATEGIES = ["maximize_f1", "maximize_f2", "min_recall", "precision_recall_balance"]

# Canonical frontend contract for Step3.
PREPROCESSING_CAPABILITIES = {
    "numericImputation": NUMERIC_IMPUTATION_METHODS,
    "numericScaling": NUMERIC_SCALING_METHODS,
    "categoricalImputation": CATEGORICAL_IMPUTATION_METHODS,
    "categoricalEncoding": CATEGORICAL_ENCODING_METHODS,
    "columnTypes": ["numeric", "categorical", "ordinal"],
    "supportsPerColumn": True,
    "defaults": {
        "numericImputation": DEFAULT_NUMERIC_IMPUTATION,
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


MODEL_HP_SCHEMA: dict[str, dict[str, dict[str, Any]]] = {
    "randomforest": {
        "n_estimators": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 2000,
            "help": "Number of trees in the forest.",
        },
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "help": "Maximum depth of each tree. null means unlimited.",
        },
        "max_features": {
            "type": "enum",
            "enum": ["sqrt", "log2"],
            "default": "sqrt",
            "help": "Number of features to consider at each split.",
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
            "help": "Regularization mix: 0 = pure L2 (Ridge), 1 = pure L1 (Lasso). Replaces deprecated 'penalty' in sklearn ≥ 1.8.",
        },
        "max_iter": {
            "type": "int",
            "default": 4000,
            "min": 100,
            "max": 20000,
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
            "help": "Regularization strength; must be > 0.",
        },
        "kernel": {
            "type": "enum",
            "enum": ["linear", "rbf", "poly", "sigmoid"],
            "default": "rbf",
            "help": "SVM kernel function.",
        },
        "gamma": {
            "type": "float_or_enum",
            "enum": ["scale", "auto"],
            "default": "scale",
            "gt": 0.0,
            "help": "Kernel coefficient. Ignored when kernel='linear'.",
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
            "help": "Number of neighbors.",
        },
        "weights": {
            "type": "enum",
            "enum": ["uniform", "distance"],
            "default": "uniform",
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
            "help": "Portion of largest variance added to variances for stability.",
        },
    },
    "decisiontree": {
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "help": "Maximum depth of the tree. null means unlimited.",
        },
        "criterion": {
            "type": "enum",
            "enum": ["gini", "entropy", "log_loss"],
            "default": "gini",
            "help": "Function to measure the quality of a split.",
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
            "help": "Number of boosting rounds.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.1,
            "gt": 0.0,
            "max": 1.0,
            "help": "Step size shrinkage in each boosting step.",
        },
        "max_depth": {
            "type": "int",
            "default": 6,
            "min": 1,
            "max": 20,
            "help": "Maximum tree depth.",
        },
        "subsample": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "max": 1.0,
            "help": "Subsample ratio of the training instances.",
        },
        "colsample_bytree": {
            "type": "float",
            "default": 1.0,
            "gt": 0.0,
            "max": 1.0,
            "help": "Subsample ratio of columns when constructing each tree.",
        },
    },
    "lightgbm": {
        "n_estimators": {
            "type": "int",
            "default": 500,
            "min": 50,
            "max": 5000,
            "help": "Number of boosted trees.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.05,
            "gt": 0.0,
            "max": 1.0,
            "help": "Boosting learning rate.",
        },
        "num_leaves": {
            "type": "int",
            "default": 31,
            "min": 2,
            "max": 2048,
            "help": "Maximum number of leaves in one tree.",
        },
        "feature_fraction": {
            "type": "float",
            "default": 0.9,
            "gt": 0.0,
            "max": 1.0,
            "help": "Fraction of features used for each iteration.",
        },
        "bagging_fraction": {
            "type": "float",
            "default": 0.9,
            "gt": 0.0,
            "max": 1.0,
            "help": "Fraction of data sampled per tree (requires bagging_freq > 0, set automatically).",
        },
    },
    "extratrees": {
        "n_estimators": {
            "type": "int",
            "default": 200,
            "min": 10,
            "max": 2000,
            "help": "Number of trees in the forest.",
        },
        "max_depth": {
            "type": "int_or_none",
            "default": None,
            "min": 1,
            "max": 200,
            "help": "Maximum depth of each tree. null means unlimited.",
        },
        "max_features": {
            "type": "enum",
            "enum": ["sqrt", "log2"],
            "default": "sqrt",
            "help": "Number of features to consider at each split.",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "help": "Minimum number of samples required to be at a leaf node.",
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
            "help": "Number of boosting stages.",
        },
        "learning_rate": {
            "type": "float",
            "default": 0.1,
            "gt": 0.0,
            "max": 1.0,
            "help": "Step size shrinkage in each boosting stage.",
        },
        "max_depth": {
            "type": "int",
            "default": 3,
            "min": 1,
            "max": 20,
            "help": "Maximum depth of the individual estimators.",
        },
        "subsample": {
            "type": "float",
            "default": 0.8,
            "gt": 0.0,
            "max": 1.0,
            "help": "Fraction of samples used for fitting each tree. < 1 enables stochastic gradient boosting.",
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "help": "Minimum number of samples required to be at a leaf node.",
        },
        "max_features": {
            "type": "enum_or_null",
            "enum": ["sqrt", "log2"],
            "default": None,
            "help": "Features considered at each split. null = all features (default). 'sqrt'/'log2' reduce overfitting.",
        },
    },
}


def _legacy_preprocessing_capabilities() -> dict[str, Any]:
    return {
        "imputation": {
            "numeric": PREPROCESSING_CAPABILITIES["numericImputation"],
            "categorical": PREPROCESSING_CAPABILITIES["categoricalImputation"],
            "defaultNumeric": PREPROCESSING_CAPABILITIES["defaults"]["numericImputation"],
            "defaultCategorical": PREPROCESSING_CAPABILITIES["defaults"]["categoricalImputation"],
        },
        "encoding": {
            "categorical": PREPROCESSING_CAPABILITIES["categoricalEncoding"],
            "defaultCategorical": PREPROCESSING_CAPABILITIES["defaults"]["categoricalEncoding"],
        },
        "scaling": {
            "numeric": PREPROCESSING_CAPABILITIES["numericScaling"],
            "defaultNumeric": PREPROCESSING_CAPABILITIES["defaults"]["numericScaling"],
        },
        "normalization": {
            "numeric": PREPROCESSING_CAPABILITIES["numericScaling"],
            "defaultNumeric": PREPROCESSING_CAPABILITIES["defaults"]["numericScaling"],
        },
    }


def _build_model_hp_capabilities() -> dict[str, Any]:
    return copy.deepcopy(MODEL_HP_SCHEMA)


def _build_class_weight_capabilities() -> dict[str, Any]:
    """Expose per-model class_weight options for the frontend."""
    out: dict[str, Any] = {}
    for model_name, schema in MODEL_HP_SCHEMA.items():
        cw = schema.get("class_weight")
        if cw is None:
            continue
        out[model_name] = {
            "supported": True,
            "supportedIn": list(cw.get("supported_in", ["classification"])),
            "options": [None] + list(cw.get("enum", [])),
            "default": cw.get("default"),
            "help": cw.get("help", ""),
        }
    return out


def get_training_capabilities() -> dict[str, Any]:
    return {
        "engine": "training_service",
        "preprocessingCapabilities": PREPROCESSING_CAPABILITIES,
        # Keep legacy key for backward compatibility with existing UI code.
        "preprocessing": _legacy_preprocessing_capabilities(),
        "preprocessingExecution": PREPROCESSING_EXECUTION_POLICY,
        "supportedSplitMethods": SUPPORTED_SPLIT_METHODS,
        "splitMethodDefaults": {
            "kFolds": 5,
            "shuffle": True,
        },
        "availableModels": list_available_models(),
        "modelHyperparamsSchema": _build_model_hp_capabilities(),
        "classWeightCapabilities": _build_class_weight_capabilities(),
        "availableMetrics": {
            "classification": CLASSIFICATION_METRICS,
            "regression": REGRESSION_METRICS,
        },
        "balancingCapabilities": {
            "strategies": list(BALANCING_STRATEGIES),
            "thresholdStrategies": list(THRESHOLD_STRATEGIES),
            "requiresExplicitConfirmation": True,
        },
    }


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _norm_choice(raw: Any, allowed: list[str], default: str, aliases: dict[str, str] | None = None) -> str:
    v = str(raw or "").strip().lower()
    if not v:
        return default
    if aliases:
        v = aliases.get(v, v)
    return v if v in allowed else default


def _to_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return default
    # Unknown object types should never implicitly enable options.
    return default


def _to_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _to_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _normalize_model_hyperparams_payload(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for model_name_raw, params_raw in raw.items():
        model_name = MODEL_REGISTRY.normalize_name(str(model_name_raw or "").strip().lower())
        if not model_name:
            continue
        if not isinstance(params_raw, Mapping):
            out[model_name] = {}
            continue
        out[model_name] = {str(k): v for k, v in params_raw.items()}
    return out


def _coerce_scalar_with_schema(field_name: str, raw_value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
    field_type = str(schema.get("type", "")).strip().lower()
    enum_values = [str(v).strip().lower() for v in (schema.get("enum") or [])]
    default_value = schema.get("default")

    def _as_float(v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("bool is not accepted")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                raise ValueError("empty string")
            return float(txt)
        raise ValueError("not a float")

    def _as_int(v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("bool is not accepted")
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if not float(v).is_integer():
                raise ValueError("float must be integer-like")
            return int(v)
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                raise ValueError("empty string")
            val_float = float(txt)
            if not val_float.is_integer():
                raise ValueError("string float must be integer-like")
            return int(val_float)
        raise ValueError("not an int")

    def _check_bounds(v: float | int) -> str | None:
        min_value = schema.get("min")
        max_value = schema.get("max")
        gt_value = schema.get("gt")
        ge_value = schema.get("ge")
        lt_value = schema.get("lt")
        le_value = schema.get("le")

        if min_value is not None and v < min_value:
            return f"{field_name} must be >= {min_value}"
        if max_value is not None and v > max_value:
            return f"{field_name} must be <= {max_value}"
        if gt_value is not None and not (v > gt_value):
            return f"{field_name} must be > {gt_value}"
        if ge_value is not None and not (v >= ge_value):
            return f"{field_name} must be >= {ge_value}"
        if lt_value is not None and not (v < lt_value):
            return f"{field_name} must be < {lt_value}"
        if le_value is not None and not (v <= le_value):
            return f"{field_name} must be <= {le_value}"
        return None

    try:
        if field_type == "int":
            value = _as_int(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "int_or_none":
            if raw_value is None:
                return None, None
            if isinstance(raw_value, str) and raw_value.strip().lower() in {"none", "null"}:
                return None, None
            value = _as_int(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "float":
            value = _as_float(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "enum":
            value = str(raw_value or "").strip().lower()
            if value in enum_values:
                return value, None
            return default_value, f"{field_name} must be one of: {', '.join(enum_values)}"

        if field_type == "float_or_enum":
            if isinstance(raw_value, str):
                txt = raw_value.strip().lower()
                if txt in enum_values:
                    return txt, None
            value = _as_float(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "enum_or_null":
            if raw_value is None:
                return None, None
            if isinstance(raw_value, str) and raw_value.strip().lower() in {"none", "null", ""}:
                return None, None
            normalized = str(raw_value).strip().lower()
            if normalized in enum_values:
                return normalized, None
            allowed = ", ".join(f'"{v}"' for v in enum_values)
            return default_value, f"{field_name} must be one of: {allowed} or null"

        if field_type == "str":
            value = str(raw_value or "").strip()
            if value:
                return value, None
            return default_value, f"{field_name} must be a non-empty string"

        return raw_value, None
    except Exception:
        return default_value, f"{field_name} has an invalid type"


def _is_linear_kernel(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "linear"
    if isinstance(value, list) and value:
        normalized = [str(v).strip().lower() for v in value]
        return all(v == "linear" for v in normalized)
    return False


def normalize_model_hyperparams(
    model_key: str,
    params: dict[str, Any] | None,
    *,
    use_grid_search: bool,
    task_type: str | None = None,
) -> dict[str, Any]:
    model_name = MODEL_REGISTRY.normalize_name(str(model_key or "").strip().lower())
    schema = MODEL_HP_SCHEMA.get(model_name, {})
    source_raw = params if isinstance(params, dict) else {}
    schema_key_lookup = {str(field).strip().lower(): str(field) for field in schema.keys()}
    source: dict[str, Any] = {}
    unknown_fields: list[str] = []
    for raw_field, raw_value in source_raw.items():
        candidate = str(raw_field).strip()
        canonical_field = schema_key_lookup.get(candidate.lower())
        if canonical_field is None:
            unknown_fields.append(candidate)
            continue
        source[canonical_field] = raw_value

    effective: dict[str, Any] = {name: spec.get("default") for name, spec in schema.items()}
    estimator_params: dict[str, Any] = {name: spec.get("default") for name, spec in schema.items()}
    param_grid: dict[str, list[Any]] = {}
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    issue_keys: set[tuple[str, str, str]] = set()

    def _push_issue(severity: str, code: str, message: str) -> None:
        key = (severity, code, message)
        if key in issue_keys:
            return
        issue_keys.add(key)
        issues.append(
            {
                "model": model_name,
                "severity": severity,
                "code": code,
                "message": message,
            }
        )
        if severity == "error":
            errors.append(message)
        else:
            warnings.append(message)

    for unknown_field in unknown_fields:
        _push_issue(
            "warning",
            "HP_UNKNOWN",
            f"{model_name}.{unknown_field} is not supported and was ignored.",
        )

    for field, field_schema in schema.items():
        if field not in source:
            continue

        raw_value = source.get(field)
        if isinstance(raw_value, list):
            if len(raw_value) == 0:
                _push_issue(
                    "error",
                    "HP_INVALID",
                    f"{model_name}.{field} list cannot be empty.",
                )
                continue
            if use_grid_search:
                normalized_values: list[Any] = []
                for idx, item in enumerate(raw_value):
                    normalized_value, value_error = _coerce_scalar_with_schema(field, item, field_schema)
                    if value_error:
                        _push_issue(
                            "error",
                            "HP_INVALID",
                            f"{model_name}.{field}[{idx}] is invalid: {value_error}.",
                        )
                        continue
                    normalized_values.append(normalized_value)

                if normalized_values:
                    effective[field] = normalized_values
                    param_grid[field] = normalized_values
                continue

            _push_issue(
                "warning",
                "HP_LIST_IGNORED",
                f"{model_name}.{field} received a list while useGridSearch=false; using the first value.",
            )
            raw_value = raw_value[0]

        normalized_scalar, scalar_error = _coerce_scalar_with_schema(field, raw_value, field_schema)
        if scalar_error:
            _push_issue(
                "error",
                "HP_INVALID",
                f"{model_name}.{field} is invalid: {scalar_error}.",
            )
            continue

        effective[field] = normalized_scalar
        estimator_params[field] = normalized_scalar

    if model_name == "svm" and _is_linear_kernel(effective.get("kernel")) and "gamma" in schema:
        gamma_was_requested = "gamma" in source
        if gamma_was_requested or "gamma" in param_grid:
            _push_issue(
                "warning",
                "HP_SVM_GAMMA_IGNORED",
                "svm.gamma is ignored when svm.kernel='linear'; gamma was removed.",
            )
        effective.pop("gamma", None)
        estimator_params.pop("gamma", None)
        param_grid.pop("gamma", None)

    if model_name == "decisiontree" and str(task_type or "").strip().lower() == "regression":
        criterion_was_requested = "criterion" in source
        if criterion_was_requested or "criterion" in param_grid:
            _push_issue(
                "warning",
                "HP_TASK_INCOMPATIBLE",
                "decisiontree.criterion is classification-only and was removed for regression.",
            )
        effective.pop("criterion", None)
        estimator_params.pop("criterion", None)
        param_grid.pop("criterion", None)

    # class_weight is classification-only — strip it unconditionally in regression.
    if "class_weight" in schema and str(task_type or "").strip().lower() == "regression":
        if "class_weight" in source:
            _push_issue(
                "warning",
                "HP_TASK_INCOMPATIBLE",
                f"{model_name}.class_weight is classification-only and was removed for regression.",
            )
        effective.pop("class_weight", None)
        estimator_params.pop("class_weight", None)
        param_grid.pop("class_weight", None)

    for key in list(estimator_params.keys()):
        if key in param_grid and isinstance(param_grid.get(key), list):
            # Keep the estimator fixed for scalar params only; grid params are set by GridSearchCV.
            estimator_params.pop(key, None)

    return {
        "model": model_name,
        "effective": effective,
        "estimator_params": estimator_params,
        "param_grid": param_grid,
        "issues": issues,
        "warnings": warnings,
        "errors": errors,
    }


@dataclass(frozen=True)
class PreprocessingDefaults:
    numeric_imputation: str = DEFAULT_NUMERIC_IMPUTATION
    categorical_imputation: str = DEFAULT_CATEGORICAL_IMPUTATION
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING
    numeric_scaling: str = DEFAULT_NUMERIC_SCALING

    def as_dict(self) -> dict[str, str]:
        return {
            "numericImputation": self.numeric_imputation,
            "numericScaling": self.numeric_scaling,
            "categoricalImputation": self.categorical_imputation,
            "categoricalEncoding": self.categorical_encoding,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "imputation": {
                "numeric": self.numeric_imputation,
                "categorical": self.categorical_imputation,
            },
            "encoding": {
                "categorical": self.categorical_encoding,
            },
            "scaling": {
                "numeric": self.numeric_scaling,
            },
            "normalization": {
                "numeric": self.numeric_scaling,
            },
        }


def _norm_column_type(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip().lower()
    if value in {"numeric", "categorical", "ordinal"}:
        return value
    return fallback


def _to_optional_str_list(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out if out else None


def _copy_column_overrides(columns: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col, cfg in columns.items():
        cloned = dict(cfg)
        if isinstance(cloned.get("ordinalOrder"), list):
            cloned["ordinalOrder"] = [str(v) for v in cloned["ordinalOrder"]]
        out[str(col)] = cloned
    return out


@dataclass(frozen=True)
class PreprocessingConfig:
    defaults: PreprocessingDefaults = field(default_factory=PreprocessingDefaults)
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_front(payload: dict[str, Any]) -> "PreprocessingConfig":
        pp = _as_dict(payload.get("preprocessing"))
        defaults = _as_dict(pp.get("defaults"))
        imp = _as_dict(pp.get("imputation"))
        enc = _as_dict(pp.get("encoding"))
        scl = _as_dict(pp.get("scaling"))
        nrm = _as_dict(pp.get("normalization"))

        numeric_imputation = _norm_choice(
            defaults.get("numericImputation", pp.get("numericImputation", imp.get("numeric", payload.get("numericImputation")))),
            NUMERIC_IMPUTATION_METHODS,
            DEFAULT_NUMERIC_IMPUTATION,
        )
        categorical_imputation = _norm_choice(
            defaults.get(
                "categoricalImputation",
                pp.get("categoricalImputation", imp.get("categorical", payload.get("categoricalImputation"))),
            ),
            CATEGORICAL_IMPUTATION_METHODS,
            DEFAULT_CATEGORICAL_IMPUTATION,
        )
        categorical_encoding = _norm_choice(
            defaults.get(
                "categoricalEncoding",
                pp.get("categoricalEncoding", enc.get("categorical", payload.get("categoricalEncoding"))),
            ),
            CATEGORICAL_ENCODING_METHODS,
            DEFAULT_CATEGORICAL_ENCODING,
            aliases={
                "labelencoding": "label",
                "label_encoder": "label",
                "ordinalencoding": "ordinal",
            },
        )
        numeric_scaling = _norm_choice(
            defaults.get(
                "numericScaling",
                pp.get("numericScaling", scl.get("numeric", nrm.get("numeric", payload.get("numericScaling")))),
            ),
            NUMERIC_SCALING_METHODS,
            DEFAULT_NUMERIC_SCALING,
            aliases={
                "normalization": "standard",
                "normalisation": "standard",
            },
        )

        resolved_defaults = PreprocessingDefaults(
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            categorical_encoding=categorical_encoding,
            numeric_scaling=numeric_scaling,
        )

        columns_raw = _as_dict(pp.get("columns"))
        normalized_columns: dict[str, dict[str, Any]] = {}
        for raw_col, raw_cfg in columns_raw.items():
            col = str(raw_col or "").strip()
            if not col:
                continue

            cfg_col = _as_dict(raw_cfg)
            out: dict[str, Any] = {}

            if "use" in cfg_col:
                out["use"] = _to_bool(cfg_col.get("use"), default=True)

            if "type" in cfg_col:
                normalized_type = str(cfg_col.get("type") or "").strip().lower()
                if normalized_type in {"numeric", "categorical", "ordinal"}:
                    out["type"] = normalized_type

            if "numericImputation" in cfg_col:
                out["numericImputation"] = _norm_choice(
                    cfg_col.get("numericImputation"),
                    NUMERIC_IMPUTATION_METHODS,
                    resolved_defaults.numeric_imputation,
                )
            if "numericScaling" in cfg_col:
                out["numericScaling"] = _norm_choice(
                    cfg_col.get("numericScaling"),
                    NUMERIC_SCALING_METHODS,
                    resolved_defaults.numeric_scaling,
                    aliases={"normalization": "standard", "normalisation": "standard"},
                )
            if "categoricalImputation" in cfg_col:
                out["categoricalImputation"] = _norm_choice(
                    cfg_col.get("categoricalImputation"),
                    CATEGORICAL_IMPUTATION_METHODS,
                    resolved_defaults.categorical_imputation,
                )
            if "categoricalEncoding" in cfg_col:
                out["categoricalEncoding"] = _norm_choice(
                    cfg_col.get("categoricalEncoding"),
                    CATEGORICAL_ENCODING_METHODS,
                    resolved_defaults.categorical_encoding,
                    aliases={
                        "labelencoding": "label",
                        "label_encoder": "label",
                        "ordinalencoding": "ordinal",
                    },
                )

            if "ordinalOrder" in cfg_col:
                order = _to_optional_str_list(cfg_col.get("ordinalOrder"))
                if order is not None:
                    out["ordinalOrder"] = order

            if out:
                normalized_columns[col] = out

        return PreprocessingConfig(
            defaults=resolved_defaults,
            columns=normalized_columns,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults.as_dict(),
            "columns": _copy_column_overrides(self.columns),
        }

    @property
    def numeric_imputation(self) -> str:
        return self.defaults.numeric_imputation

    @property
    def numeric_scaling(self) -> str:
        return self.defaults.numeric_scaling

    @property
    def categorical_imputation(self) -> str:
        return self.defaults.categorical_imputation

    @property
    def categorical_encoding(self) -> str:
        return self.defaults.categorical_encoding

    def resolve_column(self, column: str, inferred_type: str) -> dict[str, Any]:
        inferred = _norm_column_type(inferred_type, "categorical")
        raw = self.columns.get(column, {})
        cfg = raw if isinstance(raw, dict) else {}

        use = bool(cfg.get("use", True))
        final_type = _norm_column_type(cfg.get("type"), inferred)
        if final_type not in {"numeric", "categorical", "ordinal"}:
            final_type = inferred

        resolved = {
            "use": use,
            "inferredType": inferred,
            "type": final_type,
            "numericImputation": _norm_choice(
                cfg.get("numericImputation"),
                NUMERIC_IMPUTATION_METHODS,
                self.defaults.numeric_imputation,
            ),
            "numericScaling": _norm_choice(
                cfg.get("numericScaling"),
                NUMERIC_SCALING_METHODS,
                self.defaults.numeric_scaling,
                aliases={"normalization": "standard", "normalisation": "standard"},
            ),
            "categoricalImputation": _norm_choice(
                cfg.get("categoricalImputation"),
                CATEGORICAL_IMPUTATION_METHODS,
                self.defaults.categorical_imputation,
            ),
            "categoricalEncoding": _norm_choice(
                cfg.get("categoricalEncoding"),
                CATEGORICAL_ENCODING_METHODS,
                self.defaults.categorical_encoding,
                aliases={
                    "labelencoding": "label",
                    "label_encoder": "label",
                    "ordinalencoding": "ordinal",
                },
            ),
        }
        if final_type == "ordinal":
            order = _to_optional_str_list(cfg.get("ordinalOrder"))
            if order is not None:
                resolved["ordinalOrder"] = order
        return resolved

    def defaults_dict(self) -> dict[str, str]:
        return {
            "numericImputation": self.defaults.numeric_imputation,
            "numericScaling": self.defaults.numeric_scaling,
            "categoricalImputation": self.defaults.categorical_imputation,
            "categoricalEncoding": self.defaults.categorical_encoding,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        return self.defaults.as_legacy_dict()


@dataclass(frozen=True)
class BalancingConfig:
    strategy: BalancingStrategy = "none"
    apply_threshold: bool = False
    threshold_strategy: ThresholdStrategy = "maximize_f1"
    min_recall_constraint: float | None = None

    @staticmethod
    def from_front(raw: Any, *, legacy_use_smote: bool = False) -> "BalancingConfig":
        data = _as_dict(raw)
        legacy_default_strategy = "smote" if legacy_use_smote else "none"

        strategy = _norm_choice(
            data.get("strategy"),
            BALANCING_STRATEGIES,
            legacy_default_strategy,
        )
        apply_threshold = _to_bool(
            data.get("apply_threshold", data.get("applyThreshold")),
            default=False,
        )
        threshold_strategy = _norm_choice(
            data.get("threshold_strategy", data.get("thresholdStrategy")),
            THRESHOLD_STRATEGIES,
            "maximize_f1",
        )
        min_recall_constraint = _to_optional_float(
            data.get("min_recall_constraint", data.get("minRecallConstraint"))
        )
        if min_recall_constraint is not None and not (0.0 < min_recall_constraint < 1.0):
            min_recall_constraint = None

        return BalancingConfig(
            strategy=strategy,  # type: ignore[arg-type]
            apply_threshold=apply_threshold,
            threshold_strategy=threshold_strategy,  # type: ignore[arg-type]
            min_recall_constraint=min_recall_constraint,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "apply_threshold": bool(self.apply_threshold),
            "threshold_strategy": str(self.threshold_strategy),
            "min_recall_constraint": self.min_recall_constraint,
        }


def _parse_search_type(payload: dict[str, Any]) -> str:
    """Parse search_type from payload with backward-compat fallback from useGridSearch."""
    raw = str(payload.get("searchType", "")).strip().lower()
    if raw in {"none", "grid", "random"}:
        return raw
    # Legacy fallback: useGridSearch=true → "grid"
    return "grid" if _to_bool(payload.get("useGridSearch"), default=False) else "none"


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str
    task_type: TaskType
    models: Sequence[str]
    metrics: Sequence[str]

    split_method: SplitMethod
    train_ratio: float
    val_ratio: float
    test_ratio: float
    k_folds: int

    use_grid_search: bool
    use_smote: bool
    shuffle: bool
    balancing: BalancingConfig
    preprocessing: PreprocessingConfig
    model_hyperparams: dict[str, dict[str, Any]] = field(default_factory=dict)
    positive_label: Any = None
    debug: bool = False
    random_state: int = 42
    dataset_version_id: int | None = None
    custom_code: str = ""
    search_type: str = "none"          # "none" | "grid" | "random"
    n_iter_random_search: int = 40     # only used when search_type == "random"

    @staticmethod
    def from_front(payload: dict[str, Any]) -> "TrainingConfig":
        def to_ratio(x: Any, default: float) -> float:
            try:
                v = float(x)
            except Exception:
                return default
            if v > 1:
                v = v / 100.0
            return max(0.0, min(v, 0.99))

        split_raw = str(payload.get("splitMethod", "holdout")).strip().lower()
        if split_raw == "stratified_kfold":
            split_method: SplitMethod = "stratified_kfold"
        elif split_raw == "kfold":
            split_method = "kfold"
        else:
            split_method = "holdout"
        legacy_use_smote = _to_bool(payload.get("useSmote"), default=False)
        balancing_cfg = BalancingConfig.from_front(
            payload.get("balancing"),
            legacy_use_smote=legacy_use_smote,
        )
        use_smote = bool(str(balancing_cfg.strategy) in {"smote", "smote_tomek"})

        return TrainingConfig(
            target_column=str(payload.get("targetColumn", "")).strip(),
            task_type=str(payload.get("taskType", "classification")).strip().lower(),  # type: ignore
            models=[str(m).strip().lower() for m in (payload.get("models") or [])],
            metrics=[str(m).strip().lower() for m in (payload.get("metrics") or [])],
            split_method=split_method,
            train_ratio=to_ratio(payload.get("trainRatio", 0.70), 0.70),
            val_ratio=to_ratio(payload.get("valRatio", 0.15), 0.15),
            # In CV mode the user may send testRatio=0 (no holdout test); default to 0.
            # In holdout mode the conventional default is 15%.
            test_ratio=to_ratio(
                payload.get("testRatio", 0.0 if split_method != "holdout" else 0.15),
                0.0 if split_method != "holdout" else 0.15,
            ),
            k_folds=int(payload.get("kFolds", 5) or 5),
            search_type=_parse_search_type(payload),
            use_grid_search=_parse_search_type(payload) != "none",
            n_iter_random_search=int(payload.get("nIterRandomSearch", 40) or 40),
            use_smote=use_smote,
            shuffle=_to_bool(payload.get("shuffle"), default=True),
            balancing=balancing_cfg,
            preprocessing=PreprocessingConfig.from_front(payload),
            model_hyperparams=_normalize_model_hyperparams_payload(payload.get("modelHyperparams")),
            positive_label=payload.get("positiveLabel", payload.get("positive_label")),
            debug=_to_bool(payload.get("debug", payload.get("trainingDebug")), default=False),
            random_state=int(payload.get("randomState", 42) or 42),
            dataset_version_id=_to_optional_int(payload.get("datasetVersionId")),
            custom_code=str(payload.get("customCode", "") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "datasetVersionId": self.dataset_version_id,
            "targetColumn": self.target_column,
            "taskType": self.task_type,
            "models": [str(m) for m in self.models],
            "metrics": [str(m) for m in self.metrics],
            "splitMethod": self.split_method,
            "trainRatio": self.train_ratio,
            "valRatio": self.val_ratio,
            "testRatio": self.test_ratio,
            "kFolds": self.k_folds,
            "shuffle": bool(self.shuffle),
            "searchType": self.search_type,
            "nIterRandomSearch": self.n_iter_random_search,
            "useGridSearch": bool(self.use_grid_search),  # backward compat
            "useSmote": bool(str(self.balancing.strategy) in {"smote", "smote_tomek"}),
            "balancing": self.balancing.as_dict(),
            "preprocessing": self.preprocessing.as_dict(),
            "modelHyperparams": copy.deepcopy(self.model_hyperparams),
            "positiveLabel": self.positive_label,
            "debug": bool(self.debug),
            "randomState": self.random_state,
            "customCode": self.custom_code,
        }
