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
    # 1) ajouter nullable=True d'abord (sinon crash si table contient déjà des rows)
    op.add_column(
        "processing_operations",
        sa.Column("params", postgresql.JSONB(), nullable=True),
    )

    # 2) remplir les NULL existants
    op.execute("UPDATE processing_operations SET params='{}'::jsonb WHERE params IS NULL")

    # 3) rendre NOT NULL + default
    op.alter_column(
        "processing_operations",
        "params",
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )

def downgrade():
    op.drop_column("processing_operations", "params")
