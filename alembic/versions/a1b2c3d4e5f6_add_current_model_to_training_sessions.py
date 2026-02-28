"""add current_model to training_sessions

Revision ID: a1b2c3d4e5f6
Revises: f2d9b6a1c8e4
Create Date: 2026-02-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f2d9b6a1c8e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_sessions",
        sa.Column("current_model", sa.String(128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_sessions", "current_model")
