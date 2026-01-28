from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User, AccountStatus
from app.schemas.auth import SignupRequest, LoginRequest, AuthResponse
from app.core.security import hash_password, verify_password, create_access_token

from app.api.deps import get_current_user
from app.models.user import User
from fastapi.security import OAuth2PasswordRequestForm



router = APIRouter()


# -------------------------
# SIGNUP
# -------------------------
@router.post("/signup", status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    # Vérifier si l'email existe déjà
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Créer l'utilisateur avec statut PENDING
    user = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        status=AccountStatus.PENDING
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "Account created successfully. Waiting for admin approval."
    }


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
@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role.value,
        "status": current_user.status.value,
    }
