"""add preparation_config_json to dataset_versions

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # IF NOT EXISTS — idempotent si la colonne a déjà été ajoutée manuellement
    op.execute("ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS preparation_config_json TEXT")


def downgrade() -> None:
    op.drop_column("dataset_versions", "preparation_config_json")
