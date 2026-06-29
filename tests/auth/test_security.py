from __future__ import annotations

"""
Unit tests for app/core/security.py — pure functions, no DB / no app needed.

Coverage:
  - password hashing / verification (+ bcrypt 72-byte truncation)
  - access token round-trip + expiry
  - password reset token: round-trip, scope check, single-use semantics
    (the per-user signing key embeds the password hash, so changing the
     password invalidates every previously issued reset token)
"""

import pytest
from jose import jwt, JWTError

from app.core import security
from app.core.config import settings


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------

def test_hash_password_is_verifiable():
    h = security.hash_password("hunter2")
    assert h != "hunter2"  # never store plaintext
    assert security.verify_password("hunter2", h) is True


def test_verify_password_rejects_wrong_password():
    h = security.hash_password("hunter2")
    assert security.verify_password("wrong", h) is False


def test_hash_password_truncates_at_72_bytes():
    # bcrypt only considers the first 72 bytes; a password that differs only
    # past byte 72 must still verify against a hash built from the prefix.
    base = "a" * 72
    h = security.hash_password(base)
    assert security.verify_password(base + "EXTRA-IGNORED", h) is True


# ---------------------------------------------------------------------------
# access token
# ---------------------------------------------------------------------------

def test_create_access_token_round_trip():
    token = security.create_access_token("42")
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[security.ALGORITHM])
    assert payload["sub"] == "42"
    assert "exp" in payload


def test_expired_access_token_raises_on_decode():
    token = security.create_access_token("42", expires_minutes=-1)
    with pytest.raises(JWTError):
        jwt.decode(token, settings.SECRET_KEY, algorithms=[security.ALGORITHM])


def test_access_token_wrong_secret_raises():
    token = security.create_access_token("42")
    with pytest.raises(JWTError):
        jwt.decode(token, "not-the-secret", algorithms=[security.ALGORITHM])


# ---------------------------------------------------------------------------
# password reset token
# ---------------------------------------------------------------------------

def test_reset_token_round_trip():
    pwd_hash = security.hash_password("current-pwd")
    token = security.create_password_reset_token(7, pwd_hash)
    assert security.verify_password_reset_token(token, pwd_hash) == 7


def test_reset_token_is_single_use_after_password_change():
    old_hash = security.hash_password("old-pwd")
    token = security.create_password_reset_token(7, old_hash)

    # User changes their password → hash changes → signing key changes.
    new_hash = security.hash_password("new-pwd")
    assert security.verify_password_reset_token(token, new_hash) is None


def test_reset_token_rejects_garbage():
    pwd_hash = security.hash_password("current-pwd")
    assert security.verify_password_reset_token("not.a.jwt", pwd_hash) is None


def test_reset_token_rejects_wrong_scope():
    # A plain access token (no "scope": pwd_reset claim) must not pass as a
    # reset token, even if signed with the matching per-user key.
    pwd_hash = security.hash_password("current-pwd")
    key = security._reset_signing_key(pwd_hash)
    from datetime import datetime, timedelta, timezone

    bad = jwt.encode(
        {"sub": "7", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        key,
        algorithm=security.ALGORITHM,
    )
    assert security.verify_password_reset_token(bad, pwd_hash) is None


def test_expired_reset_token_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "RESET_TOKEN_EXPIRE_MINUTES", -1)
    pwd_hash = security.hash_password("current-pwd")
    token = security.create_password_reset_token(7, pwd_hash)
    assert security.verify_password_reset_token(token, pwd_hash) is None


# ---------------------------------------------------------------------------
# get_reset_token_subject (unverified read)
# ---------------------------------------------------------------------------

def test_get_reset_token_subject_reads_sub_without_verification():
    # Even with the WRONG hash (so full verification would fail), the
    # unverified 'sub' must still be readable to locate the user.
    token = security.create_password_reset_token(99, security.hash_password("x"))
    assert security.get_reset_token_subject(token) == "99"


def test_get_reset_token_subject_returns_none_on_garbage():
    assert security.get_reset_token_subject("garbage") is None
