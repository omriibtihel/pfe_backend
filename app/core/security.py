from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
RESET_SCOPE = "pwd_reset"


def hash_password(password: str) -> str:
    # Protection contre crash bcrypt (>72 bytes)
    safe_password = password[:72]
    return pwd_context.hash(safe_password)


def verify_password(password: str, password_hash: str) -> bool:
    safe_password = password[:72]
    return pwd_context.verify(safe_password, password_hash)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    exp = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": subject, "exp": exp},
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )


def _reset_signing_key(password_hash: str) -> str:
    # Per-user key: once the password (hence its hash) changes, every previously
    # issued reset token becomes unverifiable → single-use, self-invalidating.
    return f"{settings.SECRET_KEY}:{password_hash}"


def create_password_reset_token(user_id: int, password_hash: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(
        minutes=settings.RESET_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": str(user_id), "scope": RESET_SCOPE, "exp": exp},
        _reset_signing_key(password_hash),
        algorithm=ALGORITHM,
    )


def get_reset_token_subject(token: str) -> str | None:
    """Read the 'sub' claim WITHOUT verifying signature/expiry, just to know
    which user to look up. The real verification happens in verify_* below."""
    try:
        return jwt.get_unverified_claims(token).get("sub")
    except Exception:
        return None


def verify_password_reset_token(token: str, password_hash: str) -> int | None:
    """Fully verify (signature + expiry + scope) against the user's current
    password hash. Returns the user id, or None if invalid/expired/already used."""
    try:
        payload = jwt.decode(
            token, _reset_signing_key(password_hash), algorithms=[ALGORITHM]
        )
    except JWTError:
        return None
    if payload.get("scope") != RESET_SCOPE:
        return None
    sub = payload.get("sub")
    try:
        return int(sub) if sub is not None else None
    except (TypeError, ValueError):
        return None
