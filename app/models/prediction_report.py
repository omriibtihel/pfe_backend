"""ORM model for the cached LLM-generated prediction report.

The cache is keyed by (model_id, lang, request_hash). The request_hash is a
stable digest of the input payload + LIME items + glossary file hash, so the
same row + same model + same glossary always hits the cache, and any glossary
edit transparently invalidates entries.
"""
from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.types import JSON

from app.db.base_class import Base


class PredictionReport(Base):
    __tablename__ = "prediction_reports"

    id = Column(Integer, primary_key=True, index=True)

    model_id = Column(
        Integer,
        ForeignKey("trained_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lang = Column(String(2), nullable=False)
    request_hash = Column(String(64), nullable=False)

    # The validated, post-processed report (canonical disclaimer included)
    content_json = Column(JSON, nullable=False)
    # Provider name (ollama|groq|template|...) and model identifier
    llm_provider = Column(String(20), nullable=False)
    llm_model = Column(String(120), nullable=False, default="")
    # Full ReportGenerationEvent + meta, for forensic debugging
    generation_event = Column(JSON, nullable=False)

    generated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "model_id", "lang", "request_hash",
            name="uq_prediction_reports_cache_key",
        ),
        # Composite index matches the cache lookup pattern (most queries
        # filter on all three columns at once).
        Index(
            "ix_prediction_reports_cache_lookup",
            "model_id", "lang", "request_hash",
        ),
    )
