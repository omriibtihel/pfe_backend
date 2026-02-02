from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text, DateTime, Column, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner = relationship("User", back_populates="projects")

    datasets = relationship(
        "Dataset",
        back_populates="project",
        cascade="all, delete-orphan",
        foreign_keys="Dataset.project_id",
        passive_deletes=True,

    )

    active_dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
    )


    active_dataset = relationship("Dataset", foreign_keys=[active_dataset_id])
