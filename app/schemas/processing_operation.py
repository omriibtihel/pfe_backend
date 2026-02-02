# app/schemas/processing_operation.py
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel


OperationType = Literal["cleaning", "imputation", "normalization", "encoding", "other"]


class ProcessingOperationCreate(BaseModel):
    type: OperationType
    description: str
    columns: List[str] = []
    params: Dict[str, Any] = {}


class ProcessingOperationOut(BaseModel):
    id: int
    description: str
    timestamp: datetime  

    class Config:
        from_attributes = True
