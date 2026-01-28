# app/crud/processing.py
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.processing_operation import ProcessingOperation


def list_operations(db: Session, project_id: int, dataset_id: int) -> List[ProcessingOperation]:
    return (
        db.query(ProcessingOperation)
        .filter(
            ProcessingOperation.project_id == project_id,
            ProcessingOperation.dataset_id == dataset_id,
        )
        .order_by(ProcessingOperation.created_at.asc())
        .all()
    )


def create_operation(
    db: Session,
    project_id: int,
    dataset_id: int,
    user_id: Optional[int],
    op_type: str,
    description: str,
    columns: list,
    params: dict,
) -> ProcessingOperation:
    obj = ProcessingOperation(
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=user_id,
        op_type=op_type,
        description=description,
        columns=columns or [],
        params=params or {},
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_last_operation(db: Session, project_id: int, dataset_id: int) -> bool:
    last = (
        db.query(ProcessingOperation)
        .filter(
            ProcessingOperation.project_id == project_id,
            ProcessingOperation.dataset_id == dataset_id,
        )
        .order_by(ProcessingOperation.created_at.desc())
        .first()
    )
    if not last:
        return False

    db.delete(last)
    db.commit()
    return True
