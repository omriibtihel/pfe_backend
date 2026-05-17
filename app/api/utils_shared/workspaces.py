# app/api/utils_shared/workspaces.py
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.processing_operation import ProcessingOperation
from app.services.nettoyage.rebuild import rebuild_processed

logger = logging.getLogger(__name__)


def _workspace_dir(project_id: int) -> Path:
    return PROJECTS_PATH / str(project_id) / "datasets" / "workspaces"


def _clone_file(src_path: str, dst_path: str) -> None:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def _parse_ops_json(operations_json: str) -> list[dict]:
    if not operations_json:
        return []
    try:
        data = json.loads(operations_json)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _safe_remove_workspace_files(ws_path: str | None, ws_id: int) -> None:
    """Best-effort suppression du fichier source + processed associé.

    Les erreurs sont *logguées* mais n'interrompent pas l'appelant — la
    suppression est non-critique (TTL prendra le relais sinon).
    """
    if not ws_path:
        return
    try:
        if os.path.exists(ws_path):
            os.remove(ws_path)
    except OSError as e:
        logger.warning("Failed to remove workspace file %s: %s", ws_path, e)
    try:
        processed = Path(ws_path).parent / f"processed_dataset_{ws_id}.csv"
        if processed.exists():
            processed.unlink()
    except OSError as e:
        logger.warning("Failed to remove processed file for workspace %s: %s", ws_id, e)


def delete_workspace_dataset(db: Session, ws: Dataset) -> None:
    """Supprime un workspace dataset : ops associées + ligne DB + fichiers disque.

    Capture le `file_path` et le `id` **avant** la suppression — accéder à
    `ws.id` après `db.delete(ws) + db.commit()` peut lever `DetachedInstanceError`
    selon la config de la session.
    """
    ws_path = ws.file_path
    ws_id = ws.id

    db.query(ProcessingOperation).filter(
        ProcessingOperation.dataset_id == ws_id,
    ).delete(synchronize_session=False)
    db.delete(ws)
    db.flush()

    _safe_remove_workspace_files(ws_path, ws_id)


def _delete_workspaces_bulk(db: Session, workspaces: list[Dataset]) -> None:
    """Supprime un lot de workspaces (ops + DB + fichiers)."""
    if not workspaces:
        return
    snapshots = [(ws.file_path, ws.id) for ws in workspaces]
    for ws in workspaces:
        db.query(ProcessingOperation).filter(
            ProcessingOperation.dataset_id == ws.id,
        ).delete(synchronize_session=False)
        db.delete(ws)
    db.flush()
    for ws_path, ws_id in snapshots:
        _safe_remove_workspace_files(ws_path, ws_id)


def _cleanup_expired_workspaces(
    db: Session, project_id: int, version_id: int, user_id: int
) -> None:
    """Supprime les workspaces expirés (DB + fichiers) avant d'en créer un nouveau."""
    now = datetime.utcnow()
    expired = (
        db.query(Dataset)
        .filter(
            Dataset.project_id == project_id,
            Dataset.kind == "workspace",
            Dataset.workspace_owner_version_id == version_id,
            Dataset.workspace_owner_user_id == user_id,
            Dataset.workspace_expires_at <= now,
        )
        .all()
    )
    _delete_workspaces_bulk(db, expired)


def _purge_raw_workspaces(
    db: Session, project_id: int, dataset_id: int, user_id: int
) -> None:
    """Supprime TOUS les raw_workspaces existants pour (project_id, dataset_id, user_id)."""
    existing = (
        db.query(Dataset)
        .filter(
            Dataset.project_id == project_id,
            Dataset.kind == "raw_workspace",
            Dataset.workspace_source_dataset_id == dataset_id,
            Dataset.workspace_owner_user_id == user_id,
        )
        .all()
    )
    _delete_workspaces_bulk(db, existing)


def find_active_version_workspace(
    db: Session, project_id: int, version_id: int, user_id: int
) -> Dataset | None:
    """Retourne le workspace actif d'une version pour l'utilisateur, ou None."""
    return (
        db.query(Dataset)
        .filter(
            Dataset.project_id == project_id,
            Dataset.kind == "workspace",
            Dataset.workspace_owner_version_id == version_id,
            Dataset.workspace_owner_user_id == user_id,
            Dataset.is_workspace_active.is_(True),
        )
        .order_by(Dataset.id.desc())
        .first()
    )


def find_active_dataset_workspace(
    db: Session, project_id: int, dataset_id: int, user_id: int
) -> Dataset | None:
    """Retourne le raw_workspace actif d'un dataset pour l'utilisateur, ou None."""
    return (
        db.query(Dataset)
        .filter(
            Dataset.project_id == project_id,
            Dataset.kind == "raw_workspace",
            Dataset.workspace_source_dataset_id == dataset_id,
            Dataset.workspace_owner_user_id == user_id,
            Dataset.is_workspace_active.is_(True),
        )
        .order_by(Dataset.id.desc())
        .first()
    )


def create_fresh_workspace_for_dataset(
    db: Session,
    project_id: int,
    dataset_id: int,
    user_id: int,
    ttl_hours: int = 12,
) -> Dataset:
    """
    Crée toujours un nouveau workspace propre pour le nettoyage d'un dataset brut.
    Les workspaces précédents sont supprimés — chaque session repart de zéro.
    Le dataset source (kind='source') n'est jamais touché.

    Race-safe : si deux requêtes concurrentes appellent cette fonction, l'une
    crée un workspace et l'autre re-fetch le gagnant (via l'index unique partiel
    `uq_active_dataset_workspace_per_owner` de la migration b8f9a0c2d345).
    """
    _purge_raw_workspaces(db, project_id, dataset_id, user_id)

    raw = (
        db.query(Dataset)
        .filter(
            Dataset.id == dataset_id,
            Dataset.project_id == project_id,
            Dataset.kind == "source",
        )
        .first()
    )
    if not raw:
        raise ValueError(f"Dataset source introuvable (id={dataset_id})")

    src_suffix = Path(raw.file_path).suffix.lower() or ".csv"
    stored_name = f"rws_d{dataset_id}_u{user_id}_{uuid4().hex}"
    ws_path = str(_workspace_dir(project_id) / f"{stored_name}{src_suffix}")

    ws_dataset = Dataset(
        project_id=project_id,
        original_name=raw.original_name,
        stored_name=stored_name,
        file_path=ws_path,
        content_type=raw.content_type,
        size_bytes=raw.size_bytes or 0,
        target_column=raw.target_column,
        kind="raw_workspace",
        workspace_source_dataset_id=dataset_id,
        workspace_owner_user_id=user_id,
        is_workspace_active=True,
        workspace_expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
    )

    db.add(ws_dataset)
    try:
        db.flush()
    except IntegrityError:
        # Race : un autre thread a créé un workspace entre notre purge et notre
        # INSERT. On rollback et on retourne le gagnant — l'utilisateur perçoit
        # un workspace "frais" même s'il n'a pas été créé par cet appel précis.
        db.rollback()
        winner = find_active_dataset_workspace(db, project_id, dataset_id, user_id)
        if winner is None:
            raise
        return winner

    _clone_file(raw.file_path, ws_dataset.file_path)

    db.flush()
    rebuild_processed(db, project_id, ws_dataset.id)
    db.commit()
    db.refresh(ws_dataset)
    return ws_dataset


def _find_unexpired_version_workspace(
    db: Session, project_id: int, version_id: int, user_id: int
) -> Dataset | None:
    now = datetime.utcnow()
    return (
        db.query(Dataset)
        .filter(
            Dataset.project_id == project_id,
            Dataset.kind == "workspace",
            Dataset.workspace_owner_version_id == version_id,
            Dataset.workspace_owner_user_id == user_id,
            Dataset.is_workspace_active.is_(True),
            Dataset.workspace_expires_at > now,
        )
        .order_by(Dataset.id.desc())
        .first()
    )


def get_or_create_workspace_for_version(
    db: Session,
    project_id: int,
    version_id: int,
    user_id: int,
    ttl_hours: int = 12,
) -> Dataset:
    """
    Retourne le workspace actif et non expiré pour (project_id, version_id, user_id).
    Si aucun n'existe ou s'il est expiré, nettoie l'ancien et en crée un nouveau.

    Robuste aux conditions de course (cf. index `uq_active_version_workspace_per_owner`
    de la migration b8f9a0c2d345) : si deux requêtes concurrentes tentent de créer
    simultanément, l'une réussit et l'autre re-fetch le gagnant.
    """
    # 1) Fast path : un workspace actif existe déjà
    ws = _find_unexpired_version_workspace(db, project_id, version_id, user_id)
    if ws:
        return ws

    # 2) Cleanup des expirés avant création
    _cleanup_expired_workspaces(db, project_id, version_id, user_id)

    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not version:
        raise ValueError("Version not found")

    # 3) Tentative de création. En cas de race (l'index unique partiel rejette
    #    le second INSERT), on rollback et on re-fetch le workspace gagnant.
    stored_name = f"ws_v{version_id}_u{user_id}_{uuid4().hex}"
    ws_path = str(_workspace_dir(project_id) / f"{stored_name}.csv")

    ws_dataset = Dataset(
        project_id=project_id,
        original_name=f"workspace_version_{version_id}.csv",
        stored_name=stored_name,
        file_path=ws_path,
        content_type=getattr(version, "content_type", None),
        size_bytes=getattr(version, "size_bytes", 0) or 0,
        kind="workspace",
        workspace_owner_version_id=version_id,
        workspace_owner_user_id=user_id,
        is_workspace_active=True,
        workspace_expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
    )

    db.add(ws_dataset)
    try:
        db.flush()  # déclenche l'INSERT et donc la violation d'unique si race
    except IntegrityError:
        db.rollback()
        winner = _find_unexpired_version_workspace(db, project_id, version_id, user_id)
        if winner is None:
            # Ne devrait jamais arriver : l'INSERT concurrent a forcément
            # produit un workspace visible une fois committé.
            raise
        return winner

    # 4) Clone du fichier + seed des ops historiques
    _clone_file(version.file_path, ws_dataset.file_path)

    ops = _parse_ops_json(version.operations_json)
    for item in ops:
        op_type = item.get("op_type") or item.get("type") or "other"
        op = ProcessingOperation(
            project_id=project_id,
            dataset_id=ws_dataset.id,
            user_id=user_id,
            op_type=op_type,
            description=item.get("description") or "",
            columns=item.get("columns") or [],
            params=item.get("params") or {},
        )
        db.add(op)

    db.flush()
    rebuild_processed(db, project_id, ws_dataset.id)  # cohérence preview/charts

    db.commit()
    db.refresh(ws_dataset)
    return ws_dataset
