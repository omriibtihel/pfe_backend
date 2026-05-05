"""DB-backed report cache.

Cache key: ``(model_id, lang, request_hash)``. The ``request_hash`` is a
SHA-256 of:
    - the input row data (sorted keys for stability)
    - the LIME items
    - the lang
    - the glossary file's content hash

Including the glossary hash means any edit to ``feature_glossary.yaml``
invalidates every cached entry transparently — no manual flush needed.
The model identity is in the WHERE clause so it doesn't go into the hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.prediction_report import PredictionReport
from app.services.reporting.feature_glossary import _GLOSSARY_PATH

logger = logging.getLogger(__name__)

# Bump this when prompt/context semantics change. It prevents old low-quality
# reports from being served after the report generator becomes more specific.
REPORT_PROMPT_VERSION = "reporting-model-context-v4"


# ── Hashing ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _glossary_file_hash() -> str:
    """SHA-256 of the glossary file bytes. Cached for the process lifetime;
    operators that want to invalidate after editing the file should restart
    the server (consistent with how the loader caches the parsed dict)."""
    try:
        if not _GLOSSARY_PATH.exists():
            return "no-glossary"
        with _GLOSSARY_PATH.open("rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError as exc:  # pragma: no cover — defensive
        logger.warning("reporting.glossary_hash_failed: %s", exc)
        return "no-glossary"


def compute_request_hash(
    *,
    input_data: dict[str, Any],
    lime: list[dict[str, Any]],
    lang: str,
) -> str:
    """Stable cache key. See module docstring."""
    payload = json.dumps(
        {
            "input": input_data,
            "lime": lime,
            "lang": lang,
            "glossary": _glossary_file_hash(),
            "prompt_version": REPORT_PROMPT_VERSION,
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Store ─────────────────────────────────────────────────────────────────────


class ReportCache:
    """Thin wrapper around the ``prediction_reports`` table."""

    def get(
        self,
        db: Session,
        *,
        model_id: int,
        lang: str,
        request_hash: str,
    ) -> Optional[PredictionReport]:
        return (
            db.query(PredictionReport)
            .filter(
                PredictionReport.model_id == model_id,
                PredictionReport.lang == lang,
                PredictionReport.request_hash == request_hash,
            )
            .first()
        )

    def put(
        self,
        db: Session,
        *,
        model_id: int,
        lang: str,
        request_hash: str,
        content: dict[str, Any],
        meta: dict[str, Any],
        llm_provider: str,
        llm_model: str = "",
    ) -> PredictionReport:
        row = PredictionReport(
            model_id=model_id,
            lang=lang,
            request_hash=request_hash,
            content_json=content,
            llm_provider=llm_provider,
            llm_model=llm_model,
            generation_event=meta,
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            # Concurrent insert with same key — fetch the winner and use it.
            db.rollback()
            existing = self.get(
                db,
                model_id=model_id,
                lang=lang,
                request_hash=request_hash,
            )
            if existing is not None:
                return existing
            raise
        db.refresh(row)
        return row

    def delete_for_model(self, db: Session, *, model_id: int) -> int:
        """Force-regenerate path for a single model. Returns rows deleted."""
        deleted = (
            db.query(PredictionReport)
            .filter(PredictionReport.model_id == model_id)
            .delete(synchronize_session=False)
        )
        db.commit()
        return int(deleted or 0)

    def delete_one(
        self,
        db: Session,
        *,
        model_id: int,
        lang: str,
        request_hash: str,
    ) -> bool:
        deleted = (
            db.query(PredictionReport)
            .filter(
                PredictionReport.model_id == model_id,
                PredictionReport.lang == lang,
                PredictionReport.request_hash == request_hash,
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        return bool(deleted)


# Module-level singleton — the cache is stateless.
report_cache = ReportCache()


__all__ = ["ReportCache", "report_cache", "compute_request_hash"]
