"""
FeatureDef and FeatureEngineeringConfig frozen dataclasses.

No imports from other training modules — this file is self-contained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureDef:
    """Single engineered-feature definition (mirrors FeatureDefIn Pydantic schema)."""
    name: str
    expression: str
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    features: tuple[FeatureDef, ...] = field(default_factory=tuple)

    @staticmethod
    def from_front(payload: dict[str, Any] | None) -> "FeatureEngineeringConfig":
        if not payload:
            return FeatureEngineeringConfig()
        raw_features = payload.get("features") or []
        defs: list[FeatureDef] = []
        for item in raw_features:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            expr = str(item.get("expression", "")).strip()
            if not name or not expr:
                continue
            defs.append(FeatureDef(
                name=name,
                expression=expr,
                enabled=bool(item.get("enabled", True)),
            ))
        return FeatureEngineeringConfig(features=tuple(defs))

    def as_dict(self) -> dict[str, Any]:
        return {"features": [f.as_dict() for f in self.features]}

    def is_empty(self) -> bool:
        return not any(f.enabled for f in self.features)
