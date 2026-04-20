from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from uuid import uuid4
import json
import os

from app.api.deps import get_db, get_current_user, ensure_project_owner
from app.services.nettoyage.df_utils import load_current_df
from app.api.utils_shared.workspaces import get_or_create_workspace_for_version
from app.crud import nettoyage as crud_nettoyage
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.processing_operation import ProcessingOperation
from app.core.config import PROJECTS_PATH

router = APIRouter()


@router.post("/{version_id}/workspace")
def create_or_get_workspace(
    project_id: int,
    version_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    try:
        ws = get_or_create_workspace_for_version(
            db=db,
            project_id=project_id,
            version_id=version_id,
            user_id=current_user.id,
        )
        return {"workspace_dataset_id": ws.id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workspace error: {e}")


@router.post("/{version_id}/commit-workspace")
def commit_workspace(
    project_id: int,
    version_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)

    workspace_dataset_id = payload.get("workspace_dataset_id")
    if not workspace_dataset_id:
        raise HTTPException(status_code=400, detail="workspace_dataset_id is required")

    # 1) Charger la version
    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.id == version_id,
            DatasetVersion.project_id == project_id,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    # 2) Charger le workspace dataset
    ws = (
        db.query(Dataset)
        .filter(
            Dataset.id == workspace_dataset_id,
            Dataset.project_id == project_id,
            Dataset.kind == "workspace",
            Dataset.workspace_owner_version_id == version_id,
            Dataset.workspace_owner_user_id == current_user.id,
            Dataset.is_workspace_active.is_(True),
        )
        .first()
    )
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found or not allowed")

    # 3) 🔑 LIRE LE DATASET TRAITÉ (IMPORTANT)
    df = load_current_df(ws.file_path, ws.id)

    # 4) Écraser le fichier de la version
    Path(version.file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(version.file_path, index=False)
    version.size_bytes = Path(version.file_path).stat().st_size

    # 5) target_column / can_predict
    target_value = version.target_column
    if target_value and target_value not in df.columns.astype(str).tolist():
        target_value = None
    version.target_column = target_value
    version.can_predict = bool(target_value)

    # 6) Recréer operations_json depuis le workspace
    ops = crud_nettoyage.list_operations(db, project_id, ws.id)

    ops_payload = []
    for o in ops:
        ops_payload.append(
            {
                "op_type": o.op_type,
                "description": o.description,
                "columns": o.columns,
                "params": o.params,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
        )

    version.operations_json = json.dumps(ops_payload, ensure_ascii=False, default=str)

    db.add(version)
    db.commit()

    # 7) 🧹 Nettoyage workspace
    db.query(ProcessingOperation).filter(
        ProcessingOperation.project_id == project_id,
        ProcessingOperation.dataset_id == ws.id,
    ).delete(synchronize_session=False)

    ws_path = ws.file_path
    db.delete(ws)
    db.commit()

    # supprimer fichiers workspace
    try:
        if ws_path and os.path.exists(ws_path):
            os.remove(ws_path)
        processed = Path(ws_path).parent / f"processed_dataset_{ws.id}.csv"
        if processed.exists():
            processed.unlink()
    except Exception:
        pass

    return {"ok": True}
