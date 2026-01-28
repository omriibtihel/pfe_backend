from pydantic import BaseModel, EmailStr
from app.models.user import UserRole, AccountStatus


class UserListItem(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: UserRole
    status: AccountStatus

    class Config:
        from_attributes = True


class ApproveUserResponse(BaseModel):
    message: str
    user_id: int
    new_status: AccountStatus


class RejectUserRequest(BaseModel):
    reason: str | None = None


class RejectUserResponse(BaseModel):
    message: str
    user_id: int
    new_status: AccountStatus
    reason: str | None = None


class AdminStats(BaseModel):
    pending_users: int
    approved_users: int
    rejected_users: int
    doctors: int
    admins: int
