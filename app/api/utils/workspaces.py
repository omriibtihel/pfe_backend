# app/api/utils/workspaces.py
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import PROJECTS_PATH
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.processing_operation import ProcessingOperation
from app.services.processing_rebuild import rebuild_processed


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


def get_or_create_workspace_for_version(
    db: Session,
    project_id: int,
    version_id: int,
    user_id: int,
    ttl_hours: int = 12,
) -> Dataset:
    """
    1 workspace par (project_id, version_id, user_id)
    """
    ws = (
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
    if ws:
        # (optionnel) si tu veux forcer un rebuild au retour, tu peux le faire ici
        return ws

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
