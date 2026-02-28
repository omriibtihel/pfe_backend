"""add balancing audit table

Revision ID: f2d9b6a1c8e4
Revises: 00cf2550110e
Create Date: 2026-02-24 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f2d9b6a1c8e4"
down_revision: Union[str, None] = "00cf2550110e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "balancing_audit",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("dataset_scale", sa.String(length=10), nullable=False),
        sa.Column("imbalance_ratio", sa.Float(), nullable=False),
        sa.Column("minority_ratio", sa.Float(), nullable=False),
        sa.Column("minority_count", sa.Integer(), nullable=False),
        sa.Column("imbalance_level", sa.String(length=20), nullable=False),
        sa.Column(
            "class_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("strategy_applied", sa.String(length=30), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column(
            "audit_flags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("refit_metric", sa.String(length=30), nullable=False),
        sa.Column("optimal_threshold", sa.Float(), nullable=True),
        sa.Column("threshold_f1_gain", sa.Float(), nullable=True),
        sa.Column("smote_k_neighbors", sa.Integer(), nullable=True),
        sa.Column("smote_samples_added", sa.Integer(), nullable=True),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["session_id"], ["training_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_balancing_audit_session_id"), "balancing_audit", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_balancing_audit_session_id"), table_name="balancing_audit")
    op.drop_table("balancing_audit")
