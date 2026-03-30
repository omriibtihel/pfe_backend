import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.user import User, AccountStatus
from app.schemas.auth import LoginRequest, AuthResponse
from app.core.security import hash_password, verify_password, create_access_token
from fastapi.security import OAuth2PasswordRequestForm

PROFILES_DIR = os.path.join("storage", "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

router = APIRouter()

_ALLOWED_PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


async def _save_profile_photo(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in _ALLOWED_PHOTO_EXTS:
        raise HTTPException(status_code=400, detail="Format de photo non supporté")
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(PROFILES_DIR, filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return f"/static/profiles/{filename}"


# -------------------------
# SIGNUP
# -------------------------
@router.post("/signup", status_code=201)
async def signup(
    full_name: str = Form(..., min_length=2, max_length=120),
    email: str = Form(...),
    password: str = Form(..., min_length=6, max_length=72),
    phone: str | None = Form(None),
    address: str | None = Form(None),
    date_of_birth: str | None = Form(None),
    specialty: str | None = Form(None),
    hospital: str | None = Form(None),
    profile_photo: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    photo_path: str | None = None
    if profile_photo and profile_photo.filename:
        photo_path = await _save_profile_photo(profile_photo)

    user = User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password),
        status=AccountStatus.PENDING,
        profile_photo=photo_path,
        phone=phone or None,
        address=address or None,
        date_of_birth=date_of_birth or None,
        specialty=specialty or None,
        hospital=hospital or None,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "Account created successfully. Waiting for admin approval."}


# -------------------------
# LOGIN
# -------------------------
@router.post("/login", response_model=AuthResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # form_data.username contient l'email
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if user.status != AccountStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account status: {user.status}"
        )

    token = create_access_token(str(user.id))

    return AuthResponse(
        access_token=token,
        role=user.role.value,
        status=user.status.value,
        user_id=user.id
    )


# -------------------------
# CURRENT USER
# -------------------------
@router.put("/me")
async def update_me(
    full_name: str | None = Form(None),
    phone: str | None = Form(None),
    address: str | None = Form(None),
    date_of_birth: str | None = Form(None),
    specialty: str | None = Form(None),
    hospital: str | None = Form(None),
    profile_photo: UploadFile | None = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if full_name is not None:
        current_user.full_name = full_name
    if phone is not None:
        current_user.phone = phone or None
    if address is not None:
        current_user.address = address or None
    if date_of_birth is not None:
        current_user.date_of_birth = date_of_birth or None
    if specialty is not None:
        current_user.specialty = specialty or None
    if hospital is not None:
        current_user.hospital = hospital or None

    if profile_photo and profile_photo.filename:
        current_user.profile_photo = await _save_profile_photo(profile_photo)

    db.commit()
    db.refresh(current_user)

    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role.value,
        "status": current_user.status.value,
        "profile_photo": current_user.profile_photo,
        "phone": current_user.phone,
        "address": current_user.address,
        "date_of_birth": current_user.date_of_birth,
        "specialty": current_user.specialty,
        "hospital": current_user.hospital,
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role.value,
        "status": current_user.status.value,
        "profile_photo": current_user.profile_photo,
        "phone": current_user.phone,
        "address": current_user.address,
        "date_of_birth": current_user.date_of_birth,
        "specialty": current_user.specialty,
        "hospital": current_user.hospital,
    }
