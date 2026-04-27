"""drop active_dataset_id, enforce one source dataset per project

Revision ID: c1d2e3f4a5b6
Revises: 0fb623d18f06, b2c3d4e5f6a7
Create Date: 2026-04-27 00:00:00.000000

Merge point: consolide les deux branches actives (0fb623d18f06 / b2c3d4e5f6a7).

Opérations :
  1. Supprime la colonne projects.active_dataset_id et sa FK.
  2. Crée un index unique partiel sur datasets(project_id) WHERE kind='source'
     pour garantir qu'un projet ne peut avoir qu'un seul dataset source.
  3. Ajoute datasets.workspace_source_dataset_id si la colonne n'existe pas encore.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, tuple, None] = ("0fb623d18f06", "b2c3d4e5f6a7")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Supprimer la FK + colonne active_dataset_id sur projects
    #    On cherche le nom de la contrainte dans information_schema pour être robuste.
    fk_name = conn.execute(
        sa.text(
            """
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = 'projects'
              AND tc.constraint_type = 'FOREIGN KEY'
              AND kcu.column_name = 'active_dataset_id'
            LIMIT 1
            """
        )
    ).scalar()

    if fk_name:
        op.drop_constraint(fk_name, "projects", type_="foreignkey")

    # Supprimer la colonne si elle existe
    col_exists = conn.execute(
        sa.text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'projects' AND column_name = 'active_dataset_id'
            """
        )
    ).scalar()

    if col_exists:
        op.drop_column("projects", "active_dataset_id")

    # 2. Index unique partiel — un seul dataset source par projet.
    #    Utilise IF NOT EXISTS pour être idempotent.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_datasets_one_source_per_project
        ON datasets (project_id)
        WHERE kind = 'source'
        """
    )

    # 3. Ajouter workspace_source_dataset_id si absent (colonne modèle sans migration)
    ws_col_exists = conn.execute(
        sa.text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'datasets'
              AND column_name = 'workspace_source_dataset_id'
            """
        )
    ).scalar()

    if not ws_col_exists:
        op.add_column(
            "datasets",
            sa.Column(
                "workspace_source_dataset_id",
                sa.Integer(),
                sa.ForeignKey("datasets.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )
        op.create_index(
            "ix_datasets_workspace_source_dataset_id",
            "datasets",
            ["workspace_source_dataset_id"],
        )


def downgrade() -> None:
    # Supprimer l'index unique partiel
    op.execute("DROP INDEX IF EXISTS uq_datasets_one_source_per_project")

    # Supprimer workspace_source_dataset_id
    op.drop_index("ix_datasets_workspace_source_dataset_id", table_name="datasets")
    op.drop_column("datasets", "workspace_source_dataset_id")

    # Recréer active_dataset_id
    op.add_column(
        "projects",
        sa.Column("active_dataset_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_projects_active_dataset_id",
        "projects",
        "datasets",
        ["active_dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )
