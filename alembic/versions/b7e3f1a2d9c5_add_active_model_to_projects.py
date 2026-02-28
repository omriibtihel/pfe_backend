"""add active_model_id to projects

Revision ID: b7e3f1a2d9c5
Revises: a1b2c3d4e5f6
Create Date: 2026-02-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7e3f1a2d9c5"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "active_model_id",
            sa.Integer(),
            sa.ForeignKey("trained_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_projects_active_model_id",
        "projects",
        ["active_model_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_projects_active_model_id", table_name="projects")
    op.drop_column("projects", "active_model_id")
