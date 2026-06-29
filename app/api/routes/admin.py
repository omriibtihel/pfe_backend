import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.models.user import User, AccountStatus, UserRole
from app.schemas.admin import (
    AdminStats,
    UserListItem,
    ApproveUserResponse,
    RejectUserRequest,
    RejectUserResponse,
    DeleteUserResponse,
)
from app.core.email import (
    send_approval_email,
    send_rejection_email,
    send_deletion_email,
    send_reactivation_email,
)

logger = logging.getLogger(__name__)

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


@router.get("/users/approved", response_model=list[UserListItem])
def list_approved_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    users = (
        db.query(User)
        .filter(User.status == AccountStatus.APPROVED)
        .order_by(User.id.desc())
        .all()
    )
    return users


@router.get("/users/rejected", response_model=list[UserListItem])
def list_rejected_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    users = (
        db.query(User)
        .filter(User.status == AccountStatus.REJECTED)
        .order_by(User.id.desc())
        .all()
    )
    return users


@router.post("/users/{user_id}/approve", response_model=ApproveUserResponse)
def approve_user(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Un compte précédemment rejeté est « réactivé » → email dédié.
    was_rejected = user.status == AccountStatus.REJECTED

    user.status = AccountStatus.APPROVED
    db.commit()

    if was_rejected:
        background_tasks.add_task(send_reactivation_email, user.email, user.full_name)
        message = "Utilisateur réactivé — email de notification envoyé."
    else:
        background_tasks.add_task(send_approval_email, user.email, user.full_name)
        message = "Utilisateur approuvé — email de confirmation envoyé."

    return ApproveUserResponse(
        message=message,
        user_id=user.id,
        new_status=user.status,
    )


@router.post("/users/{user_id}/reject", response_model=RejectUserResponse)
def reject_user(
    user_id: int,
    payload: RejectUserRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.status = AccountStatus.REJECTED
    db.commit()

    background_tasks.add_task(send_rejection_email, user.email, user.full_name, payload.reason)

    return RejectUserResponse(
        message="Utilisateur refusé — email de notification envoyé.",
        user_id=user.id,
        new_status=user.status,
        reason=payload.reason,
    )


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
def delete_user(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Vous ne pouvez pas supprimer votre propre compte.",
        )

    if user.role == UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Impossible de supprimer un compte administrateur.",
        )

    # On capture les coordonnées avant la suppression pour la notification.
    deleted_email = user.email
    deleted_name = user.full_name

    # Les projets de l'utilisateur sont supprimés en cascade (cf. relation User.projects).
    db.delete(user)
    db.commit()

    background_tasks.add_task(send_deletion_email, deleted_email, deleted_name)

    return DeleteUserResponse(
        message="Compte supprimé — email de notification envoyé.",
        user_id=user_id,
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
