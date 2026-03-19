"""add params to processing_operations

Revision ID: cef5268e7b37
Revises: a96b219a7482
Create Date: 2026-01-27 10:12:35.929077

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'cef5268e7b37'
down_revision: Union[str, None] = 'a96b219a7482'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # ADD COLUMN IF NOT EXISTS : no-op sur DB fraîche (déjà créée par a96b219a7482),
    # applique le changement sur les anciennes DB locales où la colonne manquait.
    op.execute("""
        ALTER TABLE processing_operations
        ADD COLUMN IF NOT EXISTS params JSONB NOT NULL DEFAULT '{}'::jsonb
    """)

def downgrade():
    op.drop_column("processing_operations", "params")
