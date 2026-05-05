"""add_prediction_reports_cache

Sprint 4 — adds the LLM-generated report cache table.

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-05-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prediction_reports",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "model_id",
            sa.Integer(),
            sa.ForeignKey("trained_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("lang", sa.String(length=2), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("content_json", sa.JSON(), nullable=False),
        sa.Column("llm_provider", sa.String(length=20), nullable=False),
        sa.Column("llm_model", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("generation_event", sa.JSON(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "model_id", "lang", "request_hash",
            name="uq_prediction_reports_cache_key",
        ),
    )
    op.create_index(
        "ix_prediction_reports_model_id",
        "prediction_reports",
        ["model_id"],
    )
    op.create_index(
        "ix_prediction_reports_cache_lookup",
        "prediction_reports",
        ["model_id", "lang", "request_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_reports_cache_lookup", table_name="prediction_reports")
    op.drop_index("ix_prediction_reports_model_id", table_name="prediction_reports")
    op.drop_table("prediction_reports")
