from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class DatasetVersionOut(BaseModel):
    id: int
    project_id: int
    source_dataset_id: int
    name: str
    created_at: str  # simple (ou datetime)
    can_predict: bool
    target_column: str | None = None
    operations: list[str] = []

    class Config:
        from_attributes = True


class DatasetVersionCreateIn(BaseModel):
    # si non fourni, on génère automatiquement
    name: str | None = None

    # opérations (string list) pour tracer ce qui a été appliqué
    operations: list[str] = []

    # est-ce que cette version est prête à prédire (ex: target existe + pas de NaN etc)
    can_predict: bool = False

    target_column: str | None = None


class DatasetVersionPreviewOut(BaseModel):
    version: DatasetVersionOut
    shape: dict[str, int]
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
