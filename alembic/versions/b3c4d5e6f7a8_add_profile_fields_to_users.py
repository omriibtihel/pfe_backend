"""add profile fields to users

Revision ID: b3c4d5e6f7a8
Revises: e9f0a1b2c3d4
Create Date: 2026-03-19 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "e9f0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("phone", sa.String(length=50), nullable=True))
    op.add_column("users", sa.Column("address", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("date_of_birth", sa.String(length=20), nullable=True))
    op.add_column("users", sa.Column("specialty", sa.String(length=120), nullable=True))
    op.add_column("users", sa.Column("hospital", sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "hospital")
    op.drop_column("users", "specialty")
    op.drop_column("users", "date_of_birth")
    op.drop_column("users", "address")
    op.drop_column("users", "phone")
