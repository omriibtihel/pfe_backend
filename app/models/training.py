from __future__ import annotations

import uuid

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from app.db.base_class import Base


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True, index=True)

    status = Column(String(32), nullable=False, default="queued")   # queued|running|succeeded|failed
    progress = Column(Integer, nullable=False, default=0)           # 0..100
    current_model = Column(String(128), nullable=True)              # e.g. "randomforest (2/4)"

    # config envoyée depuis le frontend (TrainingConfig)
    config_json = Column(JSON, nullable=False)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    models = relationship("TrainedModel", back_populates="session", cascade="all, delete-orphan")
    balancing_audits = relationship("BalancingAudit", back_populates="session", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    model_type = Column(String(64), nullable=False)  # randomforest, svm, xgboost, ...
    task_type = Column(String(32), nullable=False)   # classification|regression

    metrics_json = Column(JSON, nullable=False, default={})  # ex: {"accuracy":0.91,"f1":0.88}
    artifacts_json = Column(JSON, nullable=False, default={})  # chemins pkl, plots, etc

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("TrainingSession", back_populates="models")


class BalancingAudit(Base):
    __tablename__ = "balancing_audit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    n_samples = Column(Integer, nullable=False)
    dataset_scale = Column(String(10), nullable=False)
    imbalance_ratio = Column(Float, nullable=False)
    minority_ratio = Column(Float, nullable=False)
    minority_count = Column(Integer, nullable=False)
    imbalance_level = Column(String(20), nullable=False)
    class_counts = Column(JSONB, nullable=False, server_default="{}")

    strategy_applied = Column(String(30), nullable=False)
    rationale = Column(Text, nullable=False)
    audit_flags = Column(JSONB, nullable=False, server_default="[]")
    refit_metric = Column(String(30), nullable=False)

    optimal_threshold = Column(Float, nullable=True)
    threshold_f1_gain = Column(Float, nullable=True)
    smote_k_neighbors = Column(Integer, nullable=True)
    smote_samples_added = Column(Integer, nullable=True)
    warnings = Column(JSONB, nullable=False, server_default="[]")

    session = relationship("TrainingSession", back_populates="balancing_audits")
