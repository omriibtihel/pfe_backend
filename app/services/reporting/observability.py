"""Structured event logging for the reporting pipeline (plan §5).

Each report generation emits one ``ReportGenerationEvent`` to the
``reporting.events`` logger as a JSON-encoded line. Operators can scrape
these lines to monitor:

- Fallback rate to ``TemplateClient`` — signal of LLM degradation
- P95 latency per provider — signal of Groq throttling or network issues
- Blocklist hit frequency — signal that the system prompt needs reinforcement
- Validation-pass rate per provider — informs the prompt-engineering loop

The logger is **separate** from the application logger so events can be
routed independently (e.g., to a JSON file sink while application logs go
to stderr). Application code just emits, the deployment decides where it
goes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Dedicated logger so deployments can route reporting events to their own
# JSON sink without polluting the main app log.
event_logger = logging.getLogger("reporting.events")


@dataclass
class ReportGenerationEvent:
    """One row of structured telemetry for a report generation."""

    prediction_id: str
    lang: str
    provider_used: str               # "ollama" | "groq" | "template" | <other>
    model: str                       # the underlying model name (e.g. "llama3.1:8b")
    latency_ms: int
    retries: int                     # reinforced retries on the winning provider
    fallback_reason: Optional[str]   # last failure encountered before winning
    validation_passed: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


def log_event(event: ReportGenerationEvent) -> None:
    """Emit a single structured event line.

    We log at INFO level — these are operational events, not errors. If an
    operator wants to silence them they can set the ``reporting.events``
    logger to WARNING in their config.
    """
    event_logger.info("reporting.event %s", event.to_json())


def event_from_meta(
    *,
    prediction_id: str,
    lang: str,
    meta: dict[str, Any],
    model: str = "",
) -> ReportGenerationEvent:
    """Adapter: turn the ``meta`` dict produced by ``ReportService.generate``
    into a structured event. Keeps the service signature unchanged."""
    return ReportGenerationEvent(
        prediction_id=prediction_id,
        lang=lang,
        provider_used=str(meta.get("provider", "")),
        model=model,
        latency_ms=int(meta.get("latency_ms", 0) or 0),
        retries=int(meta.get("retries", 0) or 0),
        fallback_reason=meta.get("fallback_reason"),
        validation_passed=meta.get("provider") != "template" or meta.get("fallback_reason") is None,
        attempts=list(meta.get("attempts", []) or []),
    )


__all__ = ["ReportGenerationEvent", "event_from_meta", "log_event", "event_logger"]
