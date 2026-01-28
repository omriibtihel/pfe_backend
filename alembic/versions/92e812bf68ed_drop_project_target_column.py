"""drop project target_column

Revision ID: 92e812bf68ed
Revises: 15543835eb78
Create Date: 2026-01-24 18:01:57.545346

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '92e812bf68ed'
down_revision: Union[str, None] = '15543835eb78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.drop_column("projects", "target_column")



def downgrade():
    op.add_column("projects", sa.Column("target_column", sa.String(length=255), nullable=True))

