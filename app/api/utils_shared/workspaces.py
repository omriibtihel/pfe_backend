# app/api/utils_shared/workspaces.py
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.processing_operation import ProcessingOperation
from app.services.nettoyage.rebuild import rebuild_processed


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

    paths_to_delete = [(ws.file_path, ws.id) for ws in expired]

    for ws in expired:
        db.query(ProcessingOperation).filter(
            ProcessingOperation.dataset_id == ws.id,
        ).delete(synchronize_session=False)
        db.delete(ws)

    if expired:
        db.flush()

    for ws_path, ws_id in paths_to_delete:
        try:
            if ws_path and os.path.exists(ws_path):
                os.remove(ws_path)
            if ws_path:
                processed = Path(ws_path).parent / f"processed_dataset_{ws_id}.csv"
                if processed.exists():
                    processed.unlink()
        except Exception:
            pass


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

    paths_to_delete = [(ws.file_path, ws.id) for ws in existing]

    for ws in existing:
        db.query(ProcessingOperation).filter(
            ProcessingOperation.dataset_id == ws.id,
        ).delete(synchronize_session=False)
        db.delete(ws)

    if existing:
        db.flush()

    for ws_path, ws_id in paths_to_delete:
        try:
            if ws_path and os.path.exists(ws_path):
                os.remove(ws_path)
            if ws_path:
                processed = Path(ws_path).parent / f"processed_dataset_{ws_id}.csv"
                if processed.exists():
                    processed.unlink()
        except Exception:
            pass


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

    stored_name = f"rws_d{dataset_id}_u{user_id}_{uuid4().hex}"
    ws_path = str(_workspace_dir(project_id) / f"{stored_name}.csv")

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
    db.flush()

    _clone_file(raw.file_path, ws_dataset.file_path)

    db.flush()
    rebuild_processed(db, project_id, ws_dataset.id)
    db.commit()
    db.refresh(ws_dataset)
    return ws_dataset


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
    """
    now = datetime.utcnow()

    ws = (
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
    if ws:
        return ws

    # Nettoyer les workspaces expirés avant d'en créer un nouveau
    _cleanup_expired_workspaces(db, project_id, version_id, user_id)

    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not version:
        raise ValueError("Version not found")

    # dataset workspace
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
    db.flush()  # ws_dataset.id

    # clone file
    _clone_file(version.file_path, ws_dataset.file_path)

    # seed ops (version.operations_json -> ProcessingOperation rows)
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

    # ✅ rebuild pour que preview/charts soient cohérents dès le début
    rebuild_processed(db, project_id, ws_dataset.id)

    db.commit()
    db.refresh(ws_dataset)
    return ws_dataset
