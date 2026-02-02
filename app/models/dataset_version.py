from __future__ import annotations

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from sqlalchemy.types import JSON


from app.db.base_class import Base  


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    source_dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(1024), nullable=False)

    content_type = Column(String(255), nullable=True)
    size_bytes = Column(Integer, nullable=True)

    # utile pour prédiction
    target_column = Column(String(255), nullable=True)
    can_predict = Column(Boolean, nullable=False, default=False)

    # JSON string (simple) pour garder historique opérations
    operations_json = Column(Text, nullable=False, default="[]")
    

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # relationships (optionnel)
    source_dataset = relationship("Dataset", lazy="joined")
    project = relationship("Project", lazy="joined")
