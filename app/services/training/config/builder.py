"""
TrainingConfigBuilder
Central merge point for building a training-config payload.

Two paths:
  - build_from_recommendation(recommendation, user_overrides) → config pre-filled
    by the recommendation engine, with optional user adjustments applied on top.
    Always tagged configMode="manual" since the user validates the config
    in the wizard before launching.
  - build_manual(user_payload) → config as submitted by the user, verbatim.

The output is always a plain dict payload compatible with
``TrainingConfig.from_front()``.
"""
from __future__ import annotations

import copy
from typing import Any

from app.services.training.intelligence.recommender import TrainingRecommendation


ConfigMode = str  # "manual" | "automl"

# Keys the user can adjust when the wizard pre-fills from a recommendation.
_OVERRIDABLE_KEYS = {
    "models",
    "searchType",
    "kFolds",
    "shuffle",
    "trainRatio",
    "valRatio",
    "testRatio",
    "balancing",
    "modelHyperparams",
    "metrics",
    "positiveLabel",
    "preprocessing",
    "nIterRandomSearch",
}


class TrainingConfigBuilder:
    """
    Build a final training-config payload from a recommendation and/or
    user-supplied overrides.

    Usage (wizard pre-filled by recommendation, user accepted as-is)::

        builder = TrainingConfigBuilder()
        payload = builder.build_from_recommendation(recommendation)

    Usage (wizard pre-filled, user tweaked some fields)::

        payload = builder.build_from_recommendation(recommendation, user_overrides={
            "models": ["logisticregression"],
            "searchType": "grid",
        })

    Usage (fully manual)::

        payload = builder.build_manual(user_payload)
    """

    def build_from_recommendation(
        self,
        recommendation: TrainingRecommendation,
        user_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start from the recommendation payload and apply user overrides.
        Only keys in ``_OVERRIDABLE_KEYS`` are applied from overrides.
        Always tagged configMode="manual" — the recommendation is a pre-fill
        helper in the wizard, not a separate training mode.
        """
        payload = copy.deepcopy(recommendation.training_config_payload)
        payload["configMode"] = "manual"

        if user_overrides:
            for key, value in user_overrides.items():
                if key in _OVERRIDABLE_KEYS:
                    if key == "balancing" and isinstance(value, dict) and isinstance(payload.get("balancing"), dict):
                        # Deep merge balancing sub-object
                        payload["balancing"] = {**payload["balancing"], **value}
                    elif key == "modelHyperparams" and isinstance(value, dict) and isinstance(payload.get("modelHyperparams"), dict):
                        # Deep merge per-model hyperparams
                        merged: dict[str, Any] = copy.deepcopy(payload["modelHyperparams"])
                        for model, params in value.items():
                            merged[model] = {**(merged.get(model) or {}), **(params or {})}
                        payload["modelHyperparams"] = merged
                    else:
                        payload[key] = value
            payload["userOverrides"] = list(user_overrides.keys())

        return payload

    def build_manual(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Return the user payload with configMode forced to "manual".
        No recommendation values are injected.
        """
        payload = copy.deepcopy(user_payload)
        payload["configMode"] = "manual"
        return payload

    def resolve(
        self,
        *,
        recommendation: TrainingRecommendation | None = None,
        user_payload: dict[str, Any] | None = None,
        user_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Unified entry point.

        - recommendation provided → pre-fill from recommendation, apply overrides,
          inject metadata from user_payload (targetColumn, datasetVersionId, etc.).
        - recommendation=None → use user_payload as-is (fully manual).
        """
        if recommendation is None:
            if user_payload is None:
                raise ValueError("user_payload is required when no recommendation is provided.")
            return self.build_manual(user_payload)

        payload = self.build_from_recommendation(recommendation, user_overrides)

        # Inject non-overridable metadata from user_payload
        if user_payload:
            for meta_key in ("targetColumn", "datasetVersionId", "taskType", "positiveLabel"):
                if meta_key in user_payload and user_payload[meta_key] is not None:
                    payload[meta_key] = user_payload[meta_key]

        return payload
