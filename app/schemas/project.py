from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    project_type: Literal["tabular"] = "tabular"

class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str | None
    owner_id: int
    active_dataset_id: int | None = None
    project_type: str = "tabular"

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
