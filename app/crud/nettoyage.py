# app/crud/processing.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder

from app.models.processing_operation import ProcessingOperation


# -----------------------------
# Base operations
# -----------------------------
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


def list_operations_by_type(db: Session, project_id: int, dataset_id: int, op_type: str) -> List[ProcessingOperation]:
    return (
        db.query(ProcessingOperation)
        .filter(
            ProcessingOperation.project_id == project_id,
            ProcessingOperation.dataset_id == dataset_id,
            ProcessingOperation.op_type == op_type,
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


def pop_last_operation(
    db: Session,
    project_id: int,
    dataset_id: int,
    op_type: str | None = None,
) -> Optional[ProcessingOperation]:
    q = (
        db.query(ProcessingOperation)
        .filter(
            ProcessingOperation.project_id == project_id,
            ProcessingOperation.dataset_id == dataset_id,
        )
        .order_by(ProcessingOperation.created_at.desc())
    )

    if op_type:
        q = q.filter(ProcessingOperation.op_type == op_type)

    last = q.first()
    if not last:
        return None

    db.delete(last)
    db.commit()
    return last


def set_operation_result(db: Session, op_id: int, result: dict) -> None:
    """
    Store a JSON-safe '__result' payload inside op.params.
    Useful for returning detailed summaries per operation.
    """
    op = db.query(ProcessingOperation).filter(ProcessingOperation.id == op_id).first()
    if not op:
        return

    p = dict(op.params or {})
    p["__result"] = jsonable_encoder(result)
    op.params = p

    db.add(op)
    db.commit()


# -----------------------------
# Schema state (Option C)
# - We do NOT persist "alerts" themselves.
# - We persist ONLY user decisions as operations op_type='schema'.
#
# Supported schema actions stored in ProcessingOperation.params:
#   - schema_action: "set_kind" | "clear_kind" | "verify_categorical" | "dismiss_alert"
#
# Payloads:
#   set_kind:
#     { schema_action: "set_kind", column: "<col>", kind: "<kind>" }
#   clear_kind:
#     { schema_action: "clear_kind", column: "<col>" }
#   verify_categorical:
#     { schema_action: "verify_categorical", column: "<col>" }
#   dismiss_alert:
#     { schema_action: "dismiss_alert", alert_key: "<key>" }
# -----------------------------
def get_schema_state(db: Session, project_id: int, dataset_id: int) -> dict:
    ops = (
        db.query(ProcessingOperation)
        .filter(
            ProcessingOperation.project_id == project_id,
            ProcessingOperation.dataset_id == dataset_id,
            ProcessingOperation.op_type == "schema",
        )
        .order_by(ProcessingOperation.created_at.asc())
        .all()
    )

    kind_overrides: dict[str, str] = {}
    verified: set[str] = set()
    dismissed: set[str] = set()

    for op in ops:
        p = op.params or {}
        a = p.get("schema_action")

        if a == "set_kind":
            col = p.get("column")
            kind = p.get("kind")
            if col and kind:
                kind_overrides[str(col)] = str(kind)

        elif a == "clear_kind":
            col = p.get("column")
            if col:
                kind_overrides.pop(str(col), None)

        elif a == "verify_categorical":
            col = p.get("column")
            v = p.get("verified")
            if col:
                if v is True:
                    verified.add(str(col))
                else:
                    verified.discard(str(col))

        elif a == "dismiss_alert":
            key = p.get("alert_key")
            d = p.get("dismissed")
            if key:
                if d is True:
                    dismissed.add(str(key))
                else:
                    dismissed.discard(str(key))

    return {
        "kind_overrides": kind_overrides,
        "verified_categorical": sorted(verified),
        "dismissed_alert_keys": sorted(dismissed),
    }


