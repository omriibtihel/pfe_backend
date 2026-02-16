from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.models.dataset_version import DatasetVersion

router = APIRouter()


def _ops_to_tags(operations_json: str | None) -> list[str]:
    """Convertit operations_json (Text) en liste de tags (type/description)."""
    if not operations_json:
        return []
    try:
        data = json.loads(operations_json)
        if not isinstance(data, list):
            return []
        tags: list[str] = []
        for o in data:
            if isinstance(o, dict):
                t = o.get("type") or o.get("op_type") or o.get("opType")
                if t:
                    tags.append(str(t))
                else:
                    d = o.get("description")
                    if d:
                        tags.append(str(d))
            else:
                tags.append(str(o))
        return tags
    except Exception:
        return []


class OverwriteVersionPayload(BaseModel):
    content_base64: str
    content_type: str | None = None
    operations: list[dict] | None = None


@router.get("", response_model=list[dict])
def list_versions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.project_id == project_id)
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )

    return [
        {
            "id": v.id,
            "project_id": v.project_id,
            "source_dataset_id": v.source_dataset_id,
            "name": v.name,
            "stored_name": v.stored_name,
            "content_type": v.content_type,
            "size_bytes": v.size_bytes,
            "target_column": v.target_column,
            "can_predict": v.can_predict,
            "operations": _ops_to_tags(v.operations_json),
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


@router.get("/{version_id}/download")
def download_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    p = Path(v.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {p}")

    filename = f"{v.name}.csv"
    return FileResponse(path=str(p), filename=filename, media_type=v.content_type or "text/csv")


@router.post("/{version_id}/overwrite")
def overwrite_version(
    project_id: int,
    version_id: int,
    payload: OverwriteVersionPayload,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        content = base64.b64decode(payload.content_base64.encode("utf-8"))
        p = Path(v.file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

        v.size_bytes = len(content)
        if payload.content_type:
            v.content_type = payload.content_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid content: {e}")

    if payload.operations is not None:
        v.operations_json = json.dumps(payload.operations)

    db.add(v)
    db.commit()
    db.refresh(v)

    return {"ok": True, "version_id": v.id}


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_version(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    v = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        Path(v.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(v)
    db.commit()
    return None
