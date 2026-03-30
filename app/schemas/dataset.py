# app/schemas/dataset.py
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class DatasetOut(BaseModel):
    id: int
    project_id: int
    original_name: str
    stored_name: str
    content_type: str | None = None
    size_bytes: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetColumnOut(BaseModel):
    name: str
    dtype: str


class DatasetPreviewOut(BaseModel):
    dataset: DatasetOut
    shape: dict
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict]

    page: int = 1
    page_size: int = 50
    total_rows: int = 0


class DatasetTargetIn(BaseModel):
    target_column: str | None = None


class DatasetTargetOut(BaseModel):
    target_column: str | None = None
