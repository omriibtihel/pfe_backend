"""
pipeline/models — public re-exports.

This package:
1. Re-exports MODEL_REGISTRY (created empty in registry.py).
2. Populates MODEL_REGISTRY by registering all specs in canonical order
   (identical to the original _build_defaults() order in models.py).
3. Re-exports all public helpers so that existing consumers that imported
   from `app.services.training.pipeline.models` continue to work unchanged.

Registration order (must match original _build_defaults to preserve
dict-insertion ordering used by list_available_models):
  randomforest, logisticregression, svm, knn, naivebayes, decisiontree,
  xgboost, lightgbm, extratrees, gradientboosting, catboost,
  ridge, mlp, elasticnet, lasso
"""
from app.services.training.pipeline.models.registry import (
    # Core
    MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    # Adaptive grids
    get_adaptive_model_grid,
    get_adaptive_n_iter,
    get_halving_n_candidates,
    # Public helpers
    build_model,
    get_model_capabilities,
    get_model_config,
    get_model_distributions,
    get_model_grid,
    get_model_spec,
    get_search_params,
    list_available_models,
    make_estimator,
    model_is_installed,
    # Constants / utilities
    log_randint,
    SEARCH_PROFILES,
    DEFAULT_SEARCH_LEVEL,
)

# ---------------------------------------------------------------------------
# Import spec factories and register in canonical order
# ---------------------------------------------------------------------------
from app.services.training.pipeline.models.specs.tree_models import (
    get_specs as _get_tree_specs,
)
from app.services.training.pipeline.models.specs.linear_models import (
    get_specs as _get_linear_specs,
)
from app.services.training.pipeline.models.specs.boosting import (
    get_specs as _get_boosting_specs,
)
from app.services.training.pipeline.models.specs.neighbors import (
    get_specs as _get_neighbor_specs,
)
from app.services.training.pipeline.models.specs.naive_bayes import (
    get_specs as _get_nb_specs,
)
from app.services.training.pipeline.models.specs.neural import (
    get_specs as _get_neural_specs,
)

# Build the spec lookup by name for ordered registration
_all_specs = {s.name: s for s in (
    _get_tree_specs()
    + _get_linear_specs()
    + _get_boosting_specs()
    + _get_neighbor_specs()
    + _get_nb_specs()
    + _get_neural_specs()
)}

# Register in the EXACT same order as the original _build_defaults():
_REGISTRATION_ORDER = [
    "randomforest",
    "logisticregression",
    "svm",
    "knn",
    "naivebayes",
    "decisiontree",
    "xgboost",
    "lightgbm",
    "extratrees",
    "gradientboosting",
    "catboost",
    "ridge",
    "mlp",
    "elasticnet",
    "lasso",
]

for _name in _REGISTRATION_ORDER:
    MODEL_REGISTRY._register(_all_specs[_name])

# Clean up registration temporaries
del _all_specs, _name, _REGISTRATION_ORDER

__all__ = [
    "MODEL_REGISTRY",
    "ModelRegistry",
    "ModelSpec",
    "get_adaptive_model_grid",
    "get_adaptive_n_iter",
    "get_halving_n_candidates",
    "build_model",
    "get_model_capabilities",
    "get_model_config",
    "get_model_distributions",
    "get_model_grid",
    "get_model_spec",
    "get_search_params",
    "list_available_models",
    "make_estimator",
    "model_is_installed",
    "log_randint",
    "SEARCH_PROFILES",
    "DEFAULT_SEARCH_LEVEL",
]
