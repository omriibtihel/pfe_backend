"""
config/schema — public re-exports.

All consumers that previously imported from the flat
`app.services.training.config.schema` module continue to work
unchanged through this package __init__.
"""
from .types import (
    # Literal types
    BalancingStrategy,
    ColumnType,
    SplitMethod,
    TaskType,
    ThresholdStrategy,
    # Method / strategy lists
    BALANCING_STRATEGIES,
    CATEGORICAL_ENCODING_METHODS,
    CATEGORICAL_IMPUTATION_METHODS,
    CLASSIFICATION_METRICS,
    NUMERIC_IMPUTATION_METHODS,
    NUMERIC_POWER_TRANSFORM_METHODS,
    NUMERIC_SCALING_METHODS,
    REGRESSION_METRICS,
    SUPPORTED_SPLIT_METHODS,
    THRESHOLD_STRATEGIES,
    # Default constants
    DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_CATEGORICAL_IMPUTATION,
    DEFAULT_NUMERIC_IMPUTATION,
    DEFAULT_NUMERIC_POWER_TRANSFORM,
    DEFAULT_NUMERIC_SCALING,
    # Static data structures
    MODEL_HP_SCHEMA,
    PREPROCESSING_CAPABILITIES,
    PREPROCESSING_EXECUTION_POLICY,
)
from .feature_def import (
    FeatureDef,
    FeatureEngineeringConfig,
)
from .preprocessing import (
    PreprocessingConfig,
    PreprocessingDefaults,
)
from .balancing import (
    BalancingConfig,
)
from .training_config import (
    TrainingConfig,
)
from .capabilities import (
    _legacy_preprocessing_capabilities,
    get_training_capabilities,
)
# normalize_model_hyperparams lives in config/normalization.py but is re-exported
# here for backward compatibility (consumers import it from config.schema).
from app.services.training.config.normalization import normalize_model_hyperparams

__all__ = [
    # types
    "BalancingStrategy",
    "ColumnType",
    "SplitMethod",
    "TaskType",
    "ThresholdStrategy",
    "BALANCING_STRATEGIES",
    "CATEGORICAL_ENCODING_METHODS",
    "CATEGORICAL_IMPUTATION_METHODS",
    "CLASSIFICATION_METRICS",
    "NUMERIC_IMPUTATION_METHODS",
    "NUMERIC_POWER_TRANSFORM_METHODS",
    "NUMERIC_SCALING_METHODS",
    "REGRESSION_METRICS",
    "SUPPORTED_SPLIT_METHODS",
    "THRESHOLD_STRATEGIES",
    "DEFAULT_CATEGORICAL_ENCODING",
    "DEFAULT_CATEGORICAL_IMPUTATION",
    "DEFAULT_NUMERIC_IMPUTATION",
    "DEFAULT_NUMERIC_POWER_TRANSFORM",
    "DEFAULT_NUMERIC_SCALING",
    "MODEL_HP_SCHEMA",
    "PREPROCESSING_CAPABILITIES",
    "PREPROCESSING_EXECUTION_POLICY",
    # feature_def
    "FeatureDef",
    "FeatureEngineeringConfig",
    # preprocessing
    "PreprocessingConfig",
    "PreprocessingDefaults",
    # balancing
    "BalancingConfig",
    # training_config
    "TrainingConfig",
    # capabilities
    "_legacy_preprocessing_capabilities",
    "get_training_capabilities",
    # normalization (re-exported for backward compat)
    "normalize_model_hyperparams",
]
