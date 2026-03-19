from datetime import datetime
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Literal


OperationType = Literal["cleaning", "other"]

class OperationIn(BaseModel):
    type: OperationType
    description: str
    columns: List[str] = []
    params: Dict[str, Any] = {}

class OperationOut(BaseModel):
    id: int
    project_id: int
    dataset_id: int
    user_id: int | None = None
    op_type: str
    description: str
    columns: list
    params: dict
    created_at: datetime

    class Config:
        from_attributes = True

    @computed_field
    @property
    def result(self) -> dict | None:
        return (self.params or {}).get("__result")

