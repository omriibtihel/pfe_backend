"""add target_column to datasets

Revision ID: 15543835eb78
Revises: 59d7439e2a53
Create Date: 2026-01-24 17:28:11.109290

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '15543835eb78'
down_revision: Union[str, None] = '59d7439e2a53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("datasets", sa.Column("target_column", sa.String(length=255), nullable=True))

def downgrade():
    op.drop_column("datasets", "target_column")

