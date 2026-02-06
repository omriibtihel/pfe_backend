from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.db.base_class import Base


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

    # utile pour prédiction
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # =========================
    # Workspace support
    # =========================
    # "source" (upload normal) | "workspace" (clone temporaire pour version)
    kind = Column(String(20), nullable=False, default="source", index=True)

    # un workspace est rattaché à une version + un user (1 workspace / version / user)
    workspace_owner_version_id = Column(
        Integer,
        ForeignKey("dataset_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workspace_owner_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    is_workspace_active = Column(Boolean, nullable=False, default=True)
    workspace_expires_at = Column(DateTime, nullable=True)

    # =========================
    # relationships
    # =========================

    # ⚠️ IMPORTANT: préciser foreign_keys car Project a active_dataset_id -> datasets.id
    project = relationship(
        "Project",
        back_populates="datasets",
        foreign_keys=[project_id],
    )

    # ✅ versions dont ce dataset est la source (DatasetVersion.source_dataset_id)
    source_versions = relationship(
        "DatasetVersion",
        back_populates="source_dataset",
        foreign_keys="DatasetVersion.source_dataset_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ✅ relation workspace -> version (Dataset.workspace_owner_version_id)
    workspace_owner_version = relationship(
        "DatasetVersion",
        back_populates="workspaces",
        foreign_keys=[workspace_owner_version_id],
    )

    workspace_owner_user = relationship(
        "User",
        foreign_keys=[workspace_owner_user_id],
    )
