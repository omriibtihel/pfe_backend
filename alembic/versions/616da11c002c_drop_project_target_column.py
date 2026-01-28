"""drop project target_column

Revision ID: 616da11c002c
Revises: 92e812bf68ed
Create Date: 2026-01-24 18:02:40.370588

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '616da11c002c'
down_revision: Union[str, None] = '92e812bf68ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
