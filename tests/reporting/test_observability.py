"""Sprint 4b — observability event emission."""
from __future__ import annotations

import json
import logging

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.observability import (
    ReportGenerationEvent,
    event_from_meta,
    event_logger,
)
from app.services.reporting.service import ReportService


def _ctx():
    return ReportContext(
        prediction_id="p-test-1",
        lang="fr",
        label="X",
        confidence_text="élevée",
        score_pct="90 %",
        top_features=[FeatureContribution("a", "A", "1", "increase", 1.0)],
        model_name="random_forest",
        model_version="42",
    )


def test_event_serializes_to_json():
    event = ReportGenerationEvent(
        prediction_id="p1",
        lang="fr",
        provider_used="ollama",
        model="llama3.1:8b",
        latency_ms=1234,
        retries=0,
        fallback_reason=None,
        validation_passed=True,
    )
    payload = json.loads(event.to_json())
    assert payload["prediction_id"] == "p1"
    assert payload["provider_used"] == "ollama"
    assert payload["latency_ms"] == 1234
    assert payload["validation_passed"] is True
    # timestamp serialized as iso string by default=str
    assert isinstance(payload["timestamp"], str)


def test_event_from_meta_marks_template_fallback_invalid():
    """When provider is template AND fallback_reason is set, the LLM chain
    failed — validation_passed must be False so dashboards can count it."""
    meta = {
        "provider": "template",
        "latency_ms": 50,
        "retries": 1,
        "fallback_reason": "ollama:blocklist",
        "attempts": [{"client": "ollama", "reinforced": True, "outcome": "blocklist"}],
    }
    evt = event_from_meta(prediction_id="p1", lang="fr", meta=meta, model="rf")
    assert evt.validation_passed is False


def test_event_from_meta_marks_configured_template_valid():
    """When TemplateClient is the configured default (no LLM available),
    fallback_reason is None — that's a clean run, not a failure."""
    meta = {
        "provider": "template",
        "latency_ms": 5,
        "retries": 0,
        "fallback_reason": None,
        "attempts": [],
    }
    evt = event_from_meta(prediction_id="p1", lang="fr", meta=meta, model="rf")
    assert evt.validation_passed is True


def test_event_from_meta_marks_llm_success_valid():
    meta = {
        "provider": "ollama",
        "latency_ms": 4200,
        "retries": 0,
        "fallback_reason": None,
        "attempts": [{"client": "ollama", "reinforced": False, "outcome": "passed"}],
    }
    evt = event_from_meta(prediction_id="p1", lang="fr", meta=meta, model="llama3.1:8b")
    assert evt.validation_passed is True
    assert evt.provider_used == "ollama"


def test_service_emits_event_on_generate(caplog):
    """Smoke test: a single generate() call emits exactly one event line on
    the dedicated 'reporting.events' logger."""
    caplog.set_level(logging.INFO, logger="reporting.events")
    ReportService().generate(_ctx())  # default = TemplateClient

    matching = [r for r in caplog.records if r.name == "reporting.events"]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert msg.startswith("reporting.event ")

    # The JSON payload after the prefix must be parseable
    payload = json.loads(msg[len("reporting.event "):])
    assert payload["prediction_id"] == "p-test-1"
    assert payload["provider_used"] == "template"
    assert payload["lang"] == "fr"
    assert payload["model"] == "random_forest"
