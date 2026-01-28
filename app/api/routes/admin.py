from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from pydantic import BaseModel

from app.schemas.admin import AdminStats
from app.models.user import User, AccountStatus, UserRole

from app.schemas.admin import (
    UserListItem,
    ApproveUserResponse,
    RejectUserRequest,
    RejectUserResponse,
)

class AdminStats(BaseModel):
    pending_users: int
    approved_users: int
    rejected_users: int
    doctors: int
    admins: int

router = APIRouter()


@router.get("/users/pending", response_model=list[UserListItem])
def list_pending_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    users = (
        db.query(User)
        .filter(User.status == AccountStatus.PENDING)
        .order_by(User.id.desc())
        .all()
    )
    return users


@router.post("/users/{user_id}/approve", response_model=ApproveUserResponse)
def approve_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.status = AccountStatus.APPROVED
    db.commit()

    # (Mock) email envoyé
    return ApproveUserResponse(
        message="User approved (mock email sent)",
        user_id=user.id,
        new_status=user.status,
    )


@router.post("/users/{user_id}/reject", response_model=RejectUserResponse)
def reject_user(
    user_id: int,
    payload: RejectUserRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.status = AccountStatus.REJECTED
    db.commit()

    # (Mock) email envoyé + reason
    return RejectUserResponse(
        message="User rejected (mock email sent)",
        user_id=user.id,
        new_status=user.status,
        reason=payload.reason,
    )




@router.get("/stats", response_model=AdminStats)
def admin_stats(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    pending = db.query(User).filter(User.status == AccountStatus.PENDING).count()
    approved = db.query(User).filter(User.status == AccountStatus.APPROVED).count()
    rejected = db.query(User).filter(User.status == AccountStatus.REJECTED).count()

    doctors = db.query(User).filter(User.role == UserRole.DOCTOR).count()
    admins = db.query(User).filter(User.role == UserRole.ADMIN).count()

    return AdminStats(
        pending_users=pending,
        approved_users=approved,
        rejected_users=rejected,
        doctors=doctors,
        admins=admins,
    )

