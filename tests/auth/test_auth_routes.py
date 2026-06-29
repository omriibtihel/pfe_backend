from __future__ import annotations

"""
Tests for the auth route handlers in app/api/routes/auth.py.

Following the project convention (see tests/nettoyage/test_nettoyage_routes.py),
these call the handler functions directly with a mocked DB session — no real
database, no TestClient. We assert on returned payloads and raised HTTPExceptions.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.routes import auth as auth_routes
from app.models.user import AccountStatus, UserRole
from app.schemas.auth import ForgotPasswordRequest, ResetPasswordRequest

# signup() instantiates a real User ORM object, which forces SQLAlchemy to
# configure every mapper. Import the aggregator so related models
# (DatasetVersion, etc.) are registered before the mappers resolve.
import app.db.base  # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _db_with_user(user):
    """A MagicMock db whose query(...).filter(...).first() returns `user`."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = user
    return db


def _make_user(**kwargs):
    user = MagicMock()
    user.id = kwargs.get("id", 1)
    user.email = kwargs.get("email", "doc@hospital.org")
    user.full_name = kwargs.get("full_name", "Dr Who")
    user.password_hash = kwargs.get("password_hash", "hashed")
    user.status = kwargs.get("status", AccountStatus.APPROVED)
    user.role = kwargs.get("role", UserRole.DOCTOR)
    return user


# ---------------------------------------------------------------------------
# signup
# ---------------------------------------------------------------------------

def test_signup_creates_user_when_email_free():
    db = _db_with_user(None)  # no existing user

    with patch.object(auth_routes, "hash_password", return_value="HASHED"):
        result = asyncio.run(
            auth_routes.signup(
                full_name="Dr Who",
                email="new@hospital.org",
                password="secret1",
                phone=None,
                address=None,
                date_of_birth=None,
                specialty=None,
                hospital=None,
                profile_photo=None,
                db=db,
            )
        )

    assert "successfully" in result["message"].lower()
    db.add.assert_called_once()
    db.commit.assert_called_once()
    # New accounts must start PENDING (await admin approval).
    created = db.add.call_args[0][0]
    assert created.status == AccountStatus.PENDING


def test_signup_rejects_duplicate_email():
    db = _db_with_user(_make_user(email="dup@hospital.org"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            auth_routes.signup(
                full_name="Dr Who",
                email="dup@hospital.org",
                password="secret1",
                phone=None,
                address=None,
                date_of_birth=None,
                specialty=None,
                hospital=None,
                profile_photo=None,
                db=db,
            )
        )

    assert exc.value.status_code == 409
    db.add.assert_not_called()
    db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------

def _form(username, password):
    return MagicMock(username=username, password=password)


def test_login_success_returns_token_for_approved_user():
    user = _make_user(status=AccountStatus.APPROVED)
    db = _db_with_user(user)

    with patch.object(auth_routes, "verify_password", return_value=True), \
         patch.object(auth_routes, "create_access_token", return_value="JWT"):
        resp = auth_routes.login(form_data=_form(user.email, "pwd"), db=db)

    assert resp.access_token == "JWT"
    assert resp.user_id == user.id
    assert resp.status == AccountStatus.APPROVED.value


def test_login_unknown_user_returns_401():
    db = _db_with_user(None)
    with pytest.raises(HTTPException) as exc:
        auth_routes.login(form_data=_form("ghost@x.org", "pwd"), db=db)
    assert exc.value.status_code == 401


def test_login_wrong_password_returns_401():
    user = _make_user()
    db = _db_with_user(user)
    with patch.object(auth_routes, "verify_password", return_value=False):
        with pytest.raises(HTTPException) as exc:
            auth_routes.login(form_data=_form(user.email, "bad"), db=db)
    assert exc.value.status_code == 401


def test_login_unapproved_account_returns_403():
    user = _make_user(status=AccountStatus.PENDING)
    db = _db_with_user(user)
    with patch.object(auth_routes, "verify_password", return_value=True):
        with pytest.raises(HTTPException) as exc:
            auth_routes.login(form_data=_form(user.email, "pwd"), db=db)
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# forgot-password (anti-enumeration: generic response either way)
# ---------------------------------------------------------------------------

def test_forgot_password_existing_user_schedules_email():
    user = _make_user()
    db = _db_with_user(user)
    bg = MagicMock()

    with patch.object(auth_routes, "create_password_reset_token", return_value="TOK"):
        resp = auth_routes.forgot_password(
            payload=ForgotPasswordRequest(email=user.email),
            background_tasks=bg,
            db=db,
        )

    bg.add_task.assert_called_once()
    # The reset email task must carry the recipient and a URL containing the token.
    args = bg.add_task.call_args[0]
    assert auth_routes.send_password_reset_email in args
    assert any("TOK" in str(a) for a in args)
    assert "réinitialisation" in resp.message.lower()


def test_forgot_password_unknown_user_is_silent_but_generic():
    db = _db_with_user(None)
    bg = MagicMock()

    resp = auth_routes.forgot_password(
        payload=ForgotPasswordRequest(email="ghost@x.org"),
        background_tasks=bg,
        db=db,
    )

    bg.add_task.assert_not_called()
    # Same generic message — must not leak whether the account exists.
    assert "réinitialisation" in resp.message.lower()


# ---------------------------------------------------------------------------
# reset-password
# ---------------------------------------------------------------------------

def test_reset_password_success_updates_hash():
    user = _make_user(id=5, password_hash="OLD")
    db = _db_with_user(user)
    payload = ResetPasswordRequest(token="tok", new_password="brandnew")

    with patch.object(auth_routes, "get_reset_token_subject", return_value="5"), \
         patch.object(auth_routes, "verify_password_reset_token", return_value=5), \
         patch.object(auth_routes, "hash_password", return_value="NEWHASH"):
        resp = auth_routes.reset_password(payload=payload, db=db)

    assert user.password_hash == "NEWHASH"
    db.commit.assert_called_once()
    assert "succès" in resp.message.lower()


def test_reset_password_unreadable_token_returns_400():
    db = _db_with_user(None)
    payload = ResetPasswordRequest(token="garbage", new_password="brandnew")
    with patch.object(auth_routes, "get_reset_token_subject", return_value=None):
        with pytest.raises(HTTPException) as exc:
            auth_routes.reset_password(payload=payload, db=db)
    assert exc.value.status_code == 400
    db.commit.assert_not_called()


def test_reset_password_user_not_found_returns_400():
    db = _db_with_user(None)
    payload = ResetPasswordRequest(token="tok", new_password="brandnew")
    with patch.object(auth_routes, "get_reset_token_subject", return_value="999"):
        with pytest.raises(HTTPException) as exc:
            auth_routes.reset_password(payload=payload, db=db)
    assert exc.value.status_code == 400
    db.commit.assert_not_called()


def test_reset_password_failed_verification_returns_400():
    user = _make_user(id=5, password_hash="OLD")
    db = _db_with_user(user)
    payload = ResetPasswordRequest(token="tok", new_password="brandnew")

    # Token reads as user 5 but full verification fails (tampered / single-use).
    with patch.object(auth_routes, "get_reset_token_subject", return_value="5"), \
         patch.object(auth_routes, "verify_password_reset_token", return_value=None):
        with pytest.raises(HTTPException) as exc:
            auth_routes.reset_password(payload=payload, db=db)

    assert exc.value.status_code == 400
    assert user.password_hash == "OLD"  # unchanged
    db.commit.assert_not_called()
