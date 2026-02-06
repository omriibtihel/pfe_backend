from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

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

    target_column = Column(String(255), nullable=True)
    can_predict = Column(Boolean, nullable=False, default=False)

    operations_json = Column(Text, nullable=False, default="[]")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    source_dataset = relationship(
        "Dataset",
        back_populates="source_versions",
        foreign_keys=[source_dataset_id],
    )

    workspaces = relationship(
        "Dataset",
        back_populates="workspace_owner_version",
        foreign_keys="Dataset.workspace_owner_version_id",
    )
