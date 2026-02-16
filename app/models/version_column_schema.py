from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, UniqueConstraint
from app.db.base_class import Base

class VersionColumnSchema(Base):
    __tablename__ = "version_column_schemas"

    id = Column(Integer, primary_key=True)

    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=False)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), index=True, nullable=False)

    column_name = Column(String, nullable=False)

    # inferred always computed by backend
    inferred_kind = Column(String, nullable=False, default="other")  # numeric/categorical/text/binary/datetime/id/other
    confidence = Column(Float, nullable=False, default=0.0)

    # user override
    override_kind = Column(String, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "column_name", name="uq_version_column_schema"),
    )
