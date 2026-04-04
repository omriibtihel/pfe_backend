from __future__ import annotations

from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.types import JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class ImagingSession(Base):
    __tablename__ = "imaging_sessions"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status = Column(String(32), nullable=False, default="queued")  # queued|running|succeeded|failed
    progress = Column(Integer, nullable=False, default=0)           # 0..100
    current_model = Column(String(128), nullable=True)

    config_json = Column(JSON, nullable=False)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    models = relationship(
        "ImagingTrainedModel",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class ImagingTrainedModel(Base):
    __tablename__ = "imaging_trained_models"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        Integer,
        ForeignKey("imaging_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_name = Column(String(64), nullable=False)   # resnet50, efficientnet_b0, ...
    task_type = Column(String(64), nullable=False)    # image_classification

    is_saved = Column(Boolean, nullable=False, server_default="false", default=False)

    metrics_json = Column(JSON, nullable=False, default=dict)
    # {accuracy, f1_macro, f1_weighted, roc_auc, confusion_matrix,
    #  per_class, best_epoch, epoch_curves:[{epoch, train_loss, val_loss, val_acc}]}

    artifacts_json = Column(JSON, nullable=False, default=dict)
    # {model_pt: "path/.pt", class_names:[], image_size:int, pretrained:bool}

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("ImagingSession", back_populates="models")
