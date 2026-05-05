"""Sprint 4c — request hashing + DB cache CRUD.

Uses an in-memory SQLite DB with the table created from the SQLAlchemy
metadata so we don't depend on the alembic stack at test time.
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, Table, create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base_class import Base
from app.models.prediction_report import PredictionReport
from app.services.reporting.cache import (
    ReportCache,
    compute_request_hash,
    report_cache,
)

# Register a stub ``trained_models`` table on Base.metadata so the FK from
# prediction_reports resolves at schema-creation time. We deliberately avoid
# importing the real ``TrainedModel`` model — it uses Postgres-only JSONB,
# which SQLite cannot render. SQLite also doesn't enforce FK constraints by
# default, so cache CRUD tests don't need a real referenced row.
if "trained_models" not in Base.metadata.tables:
    Table("trained_models", Base.metadata, Column("id", Integer, primary_key=True))


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.tables["trained_models"].create(engine)
    PredictionReport.__table__.create(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── compute_request_hash ──────────────────────────────────────────────────────


def test_request_hash_is_deterministic():
    a = compute_request_hash(input_data={"a": 1}, lime=[], lang="fr")
    b = compute_request_hash(input_data={"a": 1}, lime=[], lang="fr")
    assert a == b
    assert len(a) == 64


def test_request_hash_changes_with_input():
    a = compute_request_hash(input_data={"a": 1}, lime=[], lang="fr")
    b = compute_request_hash(input_data={"a": 2}, lime=[], lang="fr")
    assert a != b


def test_request_hash_changes_with_lang():
    a = compute_request_hash(input_data={"a": 1}, lime=[], lang="fr")
    b = compute_request_hash(input_data={"a": 1}, lime=[], lang="en")
    assert a != b


def test_request_hash_stable_with_dict_key_order():
    """JSON serialization sorts keys, so {a:1, b:2} hashes the same as {b:2, a:1}."""
    a = compute_request_hash(input_data={"a": 1, "b": 2}, lime=[], lang="fr")
    b = compute_request_hash(input_data={"b": 2, "a": 1}, lime=[], lang="fr")
    assert a == b


def test_request_hash_changes_with_lime():
    a = compute_request_hash(input_data={"a": 1}, lime=[{"feature": "a", "contribution": 0.1, "data": 1}], lang="fr")
    b = compute_request_hash(input_data={"a": 1}, lime=[{"feature": "a", "contribution": 0.2, "data": 1}], lang="fr")
    assert a != b


# ── CRUD ──────────────────────────────────────────────────────────────────────


def test_cache_get_returns_none_on_miss(db_session):
    cache = ReportCache()
    assert cache.get(db_session, model_id=1, lang="fr", request_hash="h") is None


def test_cache_put_then_get(db_session):
    cache = ReportCache()
    row = cache.put(
        db_session,
        model_id=1,
        lang="fr",
        request_hash="h",
        content={"summary": "hi"},
        meta={"provider": "ollama"},
        llm_provider="ollama",
        llm_model="llama3.1:8b",
    )
    assert row.id is not None
    found = cache.get(db_session, model_id=1, lang="fr", request_hash="h")
    assert found is not None
    assert found.content_json == {"summary": "hi"}
    assert found.llm_provider == "ollama"
    assert found.generation_event == {"provider": "ollama"}


def test_cache_get_isolates_by_model_id(db_session):
    cache = ReportCache()
    cache.put(db_session, model_id=1, lang="fr", request_hash="h",
              content={"a": 1}, meta={}, llm_provider="ollama")
    assert cache.get(db_session, model_id=2, lang="fr", request_hash="h") is None
    assert cache.get(db_session, model_id=1, lang="fr", request_hash="h") is not None


def test_cache_get_isolates_by_lang(db_session):
    cache = ReportCache()
    cache.put(db_session, model_id=1, lang="fr", request_hash="h",
              content={"a": 1}, meta={}, llm_provider="ollama")
    assert cache.get(db_session, model_id=1, lang="en", request_hash="h") is None


def test_cache_unique_constraint_prevents_duplicates(db_session):
    """Concurrent put with same key collapses to a single row, returning the
    existing one rather than raising."""
    cache = ReportCache()
    first = cache.put(db_session, model_id=1, lang="fr", request_hash="h",
                      content={"v": 1}, meta={}, llm_provider="a")
    # Race: another caller tries to insert with the same key
    again = cache.put(db_session, model_id=1, lang="fr", request_hash="h",
                      content={"v": 2}, meta={}, llm_provider="b")
    assert again.id == first.id
    # The original row wins — we don't overwrite content silently.
    assert again.content_json == {"v": 1}


def test_delete_for_model_clears_all_langs(db_session):
    cache = ReportCache()
    cache.put(db_session, model_id=1, lang="fr", request_hash="h1",
              content={}, meta={}, llm_provider="x")
    cache.put(db_session, model_id=1, lang="en", request_hash="h2",
              content={}, meta={}, llm_provider="x")
    cache.put(db_session, model_id=2, lang="fr", request_hash="h3",
              content={}, meta={}, llm_provider="x")

    deleted = cache.delete_for_model(db_session, model_id=1)
    assert deleted == 2
    assert cache.get(db_session, model_id=1, lang="fr", request_hash="h1") is None
    assert cache.get(db_session, model_id=2, lang="fr", request_hash="h3") is not None


def test_delete_one_targets_single_entry(db_session):
    cache = ReportCache()
    cache.put(db_session, model_id=1, lang="fr", request_hash="h1",
              content={}, meta={}, llm_provider="x")
    cache.put(db_session, model_id=1, lang="fr", request_hash="h2",
              content={}, meta={}, llm_provider="x")
    assert cache.delete_one(db_session, model_id=1, lang="fr", request_hash="h1") is True
    assert cache.get(db_session, model_id=1, lang="fr", request_hash="h1") is None
    assert cache.get(db_session, model_id=1, lang="fr", request_hash="h2") is not None


def test_singleton_module_instance_works(db_session):
    """The module-level ``report_cache`` is a process singleton."""
    report_cache.put(db_session, model_id=42, lang="fr", request_hash="ok",
                     content={"x": 1}, meta={}, llm_provider="ollama")
    assert report_cache.get(db_session, model_id=42, lang="fr", request_hash="ok") is not None
