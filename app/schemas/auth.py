from pydantic import BaseModel, EmailStr, Field
from app.models.user import UserRole, AccountStatus

class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    status: str
    user_id: int


class SignupResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: UserRole
    status: AccountStatus


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1)
    new_password: str = Field(min_length=6, max_length=72)


class MessageResponse(BaseModel):
    message: str
