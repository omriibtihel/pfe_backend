"""unique partial index on active workspaces (one per owner)

Prévient la race condition de `get_or_create_workspace_for_version` /
`create_fresh_workspace_for_dataset` : deux requêtes concurrentes du même
utilisateur pouvaient toutes les deux ne rien trouver, puis créer chacune
un workspace actif → doublons en DB.

Deux index uniques **partiels** (PostgreSQL) :
  - un workspace ACTIF par (project, version, user) pour kind='workspace'
  - un workspace ACTIF par (project, source_dataset, user) pour kind='raw_workspace'

Les workspaces inactifs (terminés / committés) n'entrent pas dans la
contrainte — ils peuvent coexister sans conflit avec un nouveau workspace
actif sur la même cible.

Cette migration suppose que tous les doublons éventuels ont été nettoyés
au préalable (idempotence : la création réessaie au TTL de toute façon).
Si l'INSERT échoue à cause de cet index, l'appelant doit retomber sur un
`get_or_create` qui renverra le workspace concurrent existant.

Revision ID: b8f9a0c2d345
Revises: a7e8d9c1f234
Create Date: 2026-05-17
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "b8f9a0c2d345"
down_revision: Union[str, None] = "a7e8d9c1f234"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Noms d'index — préfixés `uq_` pour signaler l'unicité (convention SQLAlchemy)
IDX_VERSION_WORKSPACE = "uq_active_version_workspace_per_owner"
IDX_DATASET_WORKSPACE = "uq_active_dataset_workspace_per_owner"


def upgrade() -> None:
    # 0) Pré-nettoyage : si des doublons existent déjà (legacy), on garde le
    #    plus récent par (clé) et on désactive les autres. Sans ça, la création
    #    de l'index échouerait sur une DB historique.
    op.execute("""
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY project_id, workspace_owner_version_id, workspace_owner_user_id
                       ORDER BY id DESC
                   ) AS rn
            FROM datasets
            WHERE kind = 'workspace'
              AND is_workspace_active = TRUE
              AND workspace_owner_version_id IS NOT NULL
              AND workspace_owner_user_id IS NOT NULL
        )
        UPDATE datasets SET is_workspace_active = FALSE
        WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
    """)

    op.execute("""
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY project_id, workspace_source_dataset_id, workspace_owner_user_id
                       ORDER BY id DESC
                   ) AS rn
            FROM datasets
            WHERE kind = 'raw_workspace'
              AND is_workspace_active = TRUE
              AND workspace_source_dataset_id IS NOT NULL
              AND workspace_owner_user_id IS NOT NULL
        )
        UPDATE datasets SET is_workspace_active = FALSE
        WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
    """)

    # 1) Index unique partiel — un workspace de version actif par (proj, ver, user)
    op.execute(f"""
        CREATE UNIQUE INDEX {IDX_VERSION_WORKSPACE}
        ON datasets (project_id, workspace_owner_version_id, workspace_owner_user_id)
        WHERE kind = 'workspace' AND is_workspace_active = TRUE;
    """)

    # 2) Index unique partiel — un raw_workspace actif par (proj, dataset, user)
    op.execute(f"""
        CREATE UNIQUE INDEX {IDX_DATASET_WORKSPACE}
        ON datasets (project_id, workspace_source_dataset_id, workspace_owner_user_id)
        WHERE kind = 'raw_workspace' AND is_workspace_active = TRUE;
    """)


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {IDX_DATASET_WORKSPACE};")
    op.execute(f"DROP INDEX IF EXISTS {IDX_VERSION_WORKSPACE};")
