"""
TrainingConfigBuilder — Phase 6
Central merge point between:
  - a system-generated recommendation (mode="intelligent"),
  - user overrides (mode="manual" or partial adjustments in "intelligent" mode).

The output is always a plain dict payload compatible with
``TrainingConfig.from_front()``.  The builder never calls from_front()
directly so that callers can inspect/validate the payload first.

Rules
-----
* mode="intelligent" (no overrides)  → use recommendation payload as-is.
* mode="intelligent" + user_overrides → deep-merge overrides on top.
* mode="manual"                       → use user payload as-is (system
  recommendations are shown in UI only, never injected silently).
"""
from __future__ import annotations

import copy
from typing import Any

from app.services.training.intelligence.recommender import TrainingRecommendation


ConfigMode = str  # "intelligent" | "manual"

# Keys that the user is allowed to override in intelligent mode.
# Everything else in the recommendation payload is authoritative.
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

    Usage (intelligent mode, user accepted recommendation as-is)::

        builder = TrainingConfigBuilder()
        payload = builder.build_intelligent(recommendation)

    Usage (intelligent mode, user tweaked some fields)::

        payload = builder.build_intelligent(recommendation, user_overrides={
            "models": ["logisticregression"],
            "searchType": "grid",
        })

    Usage (manual mode)::

        payload = builder.build_manual(user_payload)
    """

    def build_intelligent(
        self,
        recommendation: TrainingRecommendation,
        user_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start from the recommendation payload and apply user overrides.
        Only keys in ``_OVERRIDABLE_KEYS`` are applied from overrides to
        prevent the user silently breaking authoritative system decisions
        (task_type, target_column, datasetVersionId).
        """
        payload = copy.deepcopy(recommendation.training_config_payload)
        payload["configMode"] = "intelligent"

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
        mode: ConfigMode,
        *,
        recommendation: TrainingRecommendation | None = None,
        user_payload: dict[str, Any] | None = None,
        user_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Unified entry point.

        Parameters
        ----------
        mode:
            "intelligent" or "manual".
        recommendation:
            Required when mode="intelligent".
        user_payload:
            Required when mode="manual".
            Also used to supply target_column, datasetVersionId, etc. in
            intelligent mode (non-overridable metadata).
        user_overrides:
            Optional extra overrides applied on top of the recommendation
            in intelligent mode.
        """
        if mode == "manual":
            if user_payload is None:
                raise ValueError("user_payload is required in manual mode.")
            return self.build_manual(user_payload)

        # intelligent mode
        if recommendation is None:
            raise ValueError("recommendation is required in intelligent mode.")

        payload = self.build_intelligent(recommendation, user_overrides)

        # Inject non-overridable metadata from user_payload
        if user_payload:
            for meta_key in ("targetColumn", "datasetVersionId", "taskType", "positiveLabel"):
                if meta_key in user_payload and user_payload[meta_key] is not None:
                    payload[meta_key] = user_payload[meta_key]

        return payload
