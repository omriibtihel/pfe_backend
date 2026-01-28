"""add dataset_splits

Revision ID: 7bf08c6f479a
Revises: cef5268e7b37
Create Date: 2026-01-27 23:14:10.081688

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7bf08c6f479a'
down_revision: Union[str, None] = 'cef5268e7b37'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_splits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(), server_default="holdout", nullable=False),
        sa.Column("test_size", sa.Float(), server_default="0.2", nullable=False),
        sa.Column("random_state", sa.Integer(), server_default="42", nullable=False),
        sa.Column("stratify", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "dataset_id", name="uq_dataset_split_project_dataset"),
    )

    op.create_index("ix_dataset_splits_project_dataset", "dataset_splits", ["project_id", "dataset_id"], unique=False)
    op.create_index(op.f("ix_dataset_splits_id"), "dataset_splits", ["id"], unique=False)
    op.create_index(op.f("ix_dataset_splits_project_id"), "dataset_splits", ["project_id"], unique=False)
    op.create_index(op.f("ix_dataset_splits_dataset_id"), "dataset_splits", ["dataset_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_dataset_splits_dataset_id"), table_name="dataset_splits")
    op.drop_index(op.f("ix_dataset_splits_project_id"), table_name="dataset_splits")
    op.drop_index(op.f("ix_dataset_splits_id"), table_name="dataset_splits")
    op.drop_index("ix_dataset_splits_project_dataset", table_name="dataset_splits")
    op.drop_table("dataset_splits")
