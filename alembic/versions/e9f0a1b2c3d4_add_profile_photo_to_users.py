"""add profile_photo to users

Revision ID: e9f0a1b2c3d4
Revises: d1e2f3a4b5c6
Create Date: 2026-03-17 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9f0a1b2c3d4"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("profile_photo", sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "profile_photo")
