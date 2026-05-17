"""Endpoints pour les workspaces de versions (édition isolée d'une version).

Cycle de vie :
  - POST   /{version_id}/workspace          → get-or-create un workspace
  - POST   /{version_id}/commit-workspace   → écraser la version avec le workspace
  - DELETE /{version_id}/workspace          → fermer (cleanup explicite par le client)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_db
from app.api.utils_shared.workspaces import (
    delete_workspace_dataset,
    find_active_version_workspace,
    get_or_create_workspace_for_version,
)
from app.crud import nettoyage as crud_nettoyage
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.services.nettoyage.df_utils import load_current_df

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────
class CommitWorkspaceIn(BaseModel):
    workspace_dataset_id: int = Field(..., gt=0, description="ID du workspace à committer")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _serialize_op(op) -> dict:
    return {
        "op_type": op.op_type,
        "description": op.description,
        "columns": op.columns,
        "params": op.params,
        "created_at": op.created_at.isoformat() if op.created_at else None,
    }


def _write_dataframe_atomic(df, target_path: str) -> int:
    """Écrit `df` en CSV atomiquement (temp + rename).

    Garantit que `target_path` n'est jamais dans un état partiellement écrit
    si le process crashe en plein milieu. Renvoie la taille en octets.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(target)  # atomique sur le même filesystem (POSIX & Windows ≥ Vista)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", tmp, e)
        raise
    return target.stat().st_size


def _load_owned_workspace(
    db: Session, project_id: int, version_id: int, workspace_dataset_id: int, user_id: int
) -> Dataset:
    """Charge un workspace en vérifiant qu'il appartient bien à (project, version, user)."""
    ws = (
        db.query(Dataset)
        .filter(
            Dataset.id == workspace_dataset_id,
            Dataset.project_id == project_id,
            Dataset.kind == "workspace",
            Dataset.workspace_owner_version_id == version_id,
            Dataset.workspace_owner_user_id == user_id,
            Dataset.is_workspace_active.is_(True),
        )
        .first()
    )
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found or not allowed")
    return ws


# ── Routes ───────────────────────────────────────────────────────────────────
@router.post("/{version_id}/workspace")
def create_or_get_workspace(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retourne le workspace actif pour la version, en crée un si nécessaire."""
    ensure_project_owner(db, project_id, current_user.id)
    try:
        ws = get_or_create_workspace_for_version(
            db=db, project_id=project_id, version_id=version_id, user_id=current_user.id,
        )
        return {"workspace_dataset_id": ws.id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Workspace creation failed for version %s", version_id)
        raise HTTPException(status_code=500, detail=f"Workspace error: {e}") from e


@router.delete("/{version_id}/workspace")
def close_workspace(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Ferme explicitement le workspace de cette version (cleanup côté client).

    No-op si aucun workspace actif — on retourne `ok=True` pour rester
    idempotent (le client peut appeler ça en `unload` sans risque).
    """
    ensure_project_owner(db, project_id, current_user.id)
    ws = find_active_version_workspace(db, project_id, version_id, current_user.id)
    if not ws:
        return {"ok": True, "closed": False}

    delete_workspace_dataset(db, ws)
    db.commit()
    return {"ok": True, "closed": True}


@router.post("/{version_id}/commit-workspace")
def commit_workspace(
    project_id: int,
    version_id: int,
    payload: CommitWorkspaceIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Écrase le fichier de la version par le contenu du workspace, puis cleanup.

    Ordre :
      1. Vérifier l'appartenance version + workspace
      2. Écrire la nouvelle data (atomique)
      3. Mettre à jour les métadonnées version (target, can_predict, ops_json)
      4. Commit DB
      5. Supprimer le workspace (best-effort sur les fichiers)
    """
    ensure_project_owner(db, project_id, current_user.id)

    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    ws = _load_owned_workspace(db, project_id, version_id, payload.workspace_dataset_id, current_user.id)

    # 1) Charger le dataset traité du workspace (déjà rebuild dans le workspace).
    df = load_current_df(ws.file_path, ws.id)

    # 2) Écrire le fichier de la version — atomiquement, AVANT toute mutation DB.
    #    Si l'écriture échoue, la version reste intacte et la DB n'est pas modifiée.
    try:
        size_bytes = _write_dataframe_atomic(df, version.file_path)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write version file: {e}",
        ) from e

    # 3) Mise à jour des métadonnées de la version.
    version.size_bytes = size_bytes

    target_value = version.target_column
    if target_value and target_value not in df.columns.astype(str).tolist():
        target_value = None
    version.target_column = target_value
    version.can_predict = bool(target_value)

    ops = crud_nettoyage.list_operations(db, project_id, ws.id)
    version.operations_json = json.dumps(
        [_serialize_op(o) for o in ops], ensure_ascii=False, default=str,
    )

    db.add(version)

    # 4) Cleanup workspace dans la même transaction — garantit que si le commit
    #    échoue, ni la version ni le workspace ne sont à moitié modifiés.
    delete_workspace_dataset(db, ws)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("commit_workspace failed for version %s", version_id)
        raise HTTPException(status_code=500, detail=f"Commit failed: {e}") from e

    return {"ok": True}
