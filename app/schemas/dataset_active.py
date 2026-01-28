# app/schemas/dataset_active.py
from pydantic import BaseModel

class ActiveDatasetIn(BaseModel):
    dataset_id: int

class ActiveDatasetOut(BaseModel):
    active_dataset_id: int | None = None
