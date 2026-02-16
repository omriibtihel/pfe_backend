"""add column_version_schema

Revision ID: ac74d7522097
Revises: b3e593a2649f
Create Date: 2026-02-13 21:54:54.150687

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ac74d7522097'
down_revision: Union[str, None] = 'b3e593a2649f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "version_column_schemas",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), sa.ForeignKey("dataset_versions.id"), nullable=False),
        sa.Column("column_name", sa.String(), nullable=False),
        sa.Column("inferred_kind", sa.String(), nullable=False, server_default="other"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("override_kind", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("dataset_version_id", "column_name", name="uq_version_column_schema"),
    )

    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_version_column_schemas_project_id
    ON version_column_schemas (project_id);
    """)

    op.execute("""
    CREATE INDEX IF NOT EXISTS ix_version_column_schemas_dataset_version_id
    ON version_column_schemas (dataset_version_id);
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_version_column_schemas_dataset_version_id;")
    op.execute("DROP INDEX IF EXISTS ix_version_column_schemas_project_id;")
    op.drop_table("version_column_schemas")

