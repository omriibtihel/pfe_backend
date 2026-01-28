from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, JSON
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from sqlalchemy.orm import mapped_column, Mapped

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)

    content_type = Column(String(100), nullable=True)
    size_bytes = Column(BigInteger, nullable=False)


    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)


    project = relationship(
        "Project",
        back_populates="datasets",
        foreign_keys=[project_id],
    )

