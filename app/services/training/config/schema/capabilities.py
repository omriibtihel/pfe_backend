"""
get_training_capabilities() — neutral bridge between config/schema and pipeline/models.

This module deliberately imports from both:
  - config/schema/types.py  (pure data constants)
  - pipeline/models          (MODEL_REGISTRY, list_available_models)

It exists so that pure config submodules (types, preprocessing, balancing,
training_config) never have to import from pipeline/models.
"""
from __future__ import annotations

import copy
from typing import Any

from app.services.training.pipeline.models import list_available_models
from app.services.training.config.schema.types import (
    BALANCING_STRATEGIES,
    CLASSIFICATION_METRICS,
    MODEL_HP_SCHEMA,
    PREPROCESSING_CAPABILITIES,
    PREPROCESSING_EXECUTION_POLICY,
    REGRESSION_METRICS,
    SUPPORTED_SPLIT_METHODS,
    THRESHOLD_STRATEGIES,
)


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
            "numeric": PREPROCESSING_CAPABILITIES["numericPowerTransform"],
            "defaultNumeric": PREPROCESSING_CAPABILITIES["defaults"]["numericPowerTransform"],
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
            "nRepeats": 3,
            "groupColumn": None,
        },
        "splitMethodMeta": {
            "holdout": {"requiresGroupColumn": False, "requiresNRepeats": False, "maxSamples": None},
            "kfold": {"requiresGroupColumn": False, "requiresNRepeats": False, "maxSamples": None},
            "stratified_kfold": {"requiresGroupColumn": False, "requiresNRepeats": False, "maxSamples": None},
            "repeated_stratified_kfold": {"requiresGroupColumn": False, "requiresNRepeats": True, "maxSamples": None},
            "group_kfold": {"requiresGroupColumn": True, "requiresNRepeats": False, "maxSamples": None},
            "stratified_group_kfold": {"requiresGroupColumn": True, "requiresNRepeats": False, "maxSamples": None},
            "loo": {"requiresGroupColumn": False, "requiresNRepeats": False, "maxSamples": 500},
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
