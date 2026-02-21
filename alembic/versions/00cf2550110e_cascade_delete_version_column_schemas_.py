"""cascade delete version_column_schemas on dataset_version

Revision ID: 00cf2550110e
Revises: ac74d7522097
Create Date: 2026-02-18 08:57:44.016983

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '00cf2550110e'
down_revision: Union[str, None] = 'ac74d7522097'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.drop_constraint(
        "version_column_schemas_dataset_version_id_fkey",
        "version_column_schemas",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "version_column_schemas_dataset_version_id_fkey",
        "version_column_schemas",
        "dataset_versions",
        ["dataset_version_id"],
        ["id"],
        ondelete="CASCADE",
    )

def downgrade():
    op.drop_constraint(
        "version_column_schemas_dataset_version_id_fkey",
        "version_column_schemas",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "version_column_schemas_dataset_version_id_fkey",
        "version_column_schemas",
        "dataset_versions",
        ["dataset_version_id"],
        ["id"],
        ondelete=None,
    )
