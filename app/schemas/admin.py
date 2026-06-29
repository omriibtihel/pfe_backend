from pydantic import BaseModel, EmailStr
from app.models.user import UserRole, AccountStatus


class UserListItem(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: UserRole
    status: AccountStatus
    phone: str | None = None
    address: str | None = None
    date_of_birth: str | None = None
    specialty: str | None = None
    hospital: str | None = None
    profile_photo: str | None = None

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


class DeleteUserResponse(BaseModel):
    message: str
    user_id: int


class AdminStats(BaseModel):
    pending_users: int
    approved_users: int
    rejected_users: int
    doctors: int
    admins: int
