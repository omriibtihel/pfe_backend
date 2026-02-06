"""datasets add workspace fields

Revision ID: c4c7517cdf46
Revises: 209a7b34fa17
Create Date: 2026-02-03 14:45:28.226034

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4c7517cdf46'
down_revision: Union[str, None] = '209a7b34fa17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("kind", sa.String(length=20), nullable=False, server_default="source"))
    op.add_column("datasets", sa.Column("workspace_owner_version_id", sa.Integer(), nullable=True))
    op.add_column("datasets", sa.Column("workspace_owner_user_id", sa.Integer(), nullable=True))
    op.add_column("datasets", sa.Column("is_workspace_active", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("datasets", sa.Column("workspace_expires_at", sa.DateTime(), nullable=True))

    op.create_index("ix_datasets_kind", "datasets", ["kind"])
    op.create_index("ix_datasets_workspace_owner_version_id", "datasets", ["workspace_owner_version_id"])
    op.create_index("ix_datasets_workspace_owner_user_id", "datasets", ["workspace_owner_user_id"])

    op.create_foreign_key(
        "fk_datasets_workspace_owner_version_id",
        "datasets", "dataset_versions",
        ["workspace_owner_version_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_datasets_workspace_owner_user_id",
        "datasets", "users",
        ["workspace_owner_user_id"], ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_datasets_workspace_owner_user_id", "datasets", type_="foreignkey")
    op.drop_constraint("fk_datasets_workspace_owner_version_id", "datasets", type_="foreignkey")

    op.drop_index("ix_datasets_workspace_owner_user_id", table_name="datasets")
    op.drop_index("ix_datasets_workspace_owner_version_id", table_name="datasets")
    op.drop_index("ix_datasets_kind", table_name="datasets")

    op.drop_column("datasets", "workspace_expires_at")
    op.drop_column("datasets", "is_workspace_active")
    op.drop_column("datasets", "workspace_owner_user_id")
    op.drop_column("datasets", "workspace_owner_version_id")
    op.drop_column("datasets", "kind")

