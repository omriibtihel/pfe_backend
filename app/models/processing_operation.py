# app/models/processing_operation.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class ProcessingOperation(Base):
    __tablename__ = "processing_operations"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    op_type = Column(String, nullable=False)  # "cleaning" | "imputation" | ...
    description = Column(String, nullable=False)

    columns = Column(JSONB, nullable=False, server_default="[]")
    params = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # (optionnel)
    project = relationship("Project", lazy="joined")
    dataset = relationship("Dataset", lazy="joined")
    user = relationship("User", lazy="joined")
