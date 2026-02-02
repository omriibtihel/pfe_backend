from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.models.dataset_version import DatasetVersion

router = APIRouter()


@router.get("", response_model=list[dict])
def list_versions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # sécurité: seul owner
    ensure_project_owner(db, project_id, current_user.id)

    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.project_id == project_id)
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )

    # dict simple (pas besoin de schemas pydantic pour l’instant)
    return [
        {
            "id": v.id,
            "project_id": v.project_id,
            "source_dataset_id": v.source_dataset_id,
            "name": v.name,
            "file_path": v.file_path,
            "operations": v.operations_json or [],
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


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

    db.delete(v)
    db.commit()
    return None
