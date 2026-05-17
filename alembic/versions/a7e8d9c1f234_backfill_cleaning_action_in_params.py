"""backfill cleaning params.action from description (legacy ops)

Avant le refactor, les opérations `cleaning` historiques pouvaient être créées
sans `params.action`. Le code de rebuild s'appuyait alors sur une heuristique
fragile basée sur la description localisée (`_legacy_infer_action` dans
`rebuild.py`).

Cette migration backfille `params.action` pour toutes les ops `cleaning`
existantes qui n'en ont pas — en utilisant la même heuristique mais une seule
fois, en DB. Après quoi le fallback peut être supprimé du code applicatif.

Revision ID: a7e8d9c1f234
Revises: d4e5f6a7b8c9
Create Date: 2026-05-17
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7e8d9c1f234"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── Heuristique miroir de _legacy_infer_action (rebuild.py) ────────────────────
# Mots-clés FR + EN pour reconnaître l'action depuis une description libre.
# L'ordre dans la chaîne CASE compte : les patterns plus spécifiques d'abord
# (ex: "rename" avant "drop") pour éviter les collisions.
_BACKFILL_SQL = """
UPDATE processing_operations
SET params = params || jsonb_build_object('action',
    CASE
        WHEN LOWER(description) ~ '(rename|renomm)'         THEN 'rename_columns'
        WHEN LOWER(description) ~ '(doubl|duplicate)'       THEN 'drop_duplicates'
        WHEN LOWER(description) ~ '(vide|empty|blank)'      THEN 'drop_empty_rows'
        WHEN LOWER(description) ~ '(strip|trim|espaces)'    THEN 'strip_whitespace'
        WHEN LOWER(description) ~ '(substitut|replace)'     THEN 'substitute_values'
        WHEN LOWER(description) ~ '(supprim|drop)'          THEN 'drop_columns'
        ELSE NULL
    END
)
WHERE op_type = 'cleaning'
  AND (params ->> 'action') IS NULL
  AND CASE
        WHEN LOWER(description) ~ '(rename|renomm)'         THEN TRUE
        WHEN LOWER(description) ~ '(doubl|duplicate)'       THEN TRUE
        WHEN LOWER(description) ~ '(vide|empty|blank)'      THEN TRUE
        WHEN LOWER(description) ~ '(strip|trim|espaces)'    THEN TRUE
        WHEN LOWER(description) ~ '(substitut|replace)'     THEN TRUE
        WHEN LOWER(description) ~ '(supprim|drop)'          THEN TRUE
        ELSE FALSE
      END
"""


def upgrade() -> None:
    op.execute(sa.text(_BACKFILL_SQL))


def downgrade() -> None:
    # Reversal lossy : on ne sait plus distinguer une action backfillée d'une
    # action légitimement présente. On n'enlève rien — la présence de `action`
    # est désormais une invariante du schéma (toutes les nouvelles ops l'ont).
    pass
