from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    # Protection contre crash bcrypt (>72 bytes)
    safe_password = password[:72]
    return pwd_context.hash(safe_password)


def verify_password(password: str, password_hash: str) -> bool:
    safe_password = password[:72]
    return pwd_context.verify(safe_password, password_hash)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    exp = datetime.utcnow() + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": subject, "exp": exp},
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
