"""Sprint 2 unit tests for OllamaClient — HTTP fully mocked.

Real Ollama integration is gated behind ``@pytest.mark.requires_ollama``
(plan §12); CI runs only the mocked variants.
"""
from __future__ import annotations

import json

import pytest
import requests

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.ollama import (
    OllamaClient,
    OllamaConfig,
    OllamaError,
)


def _ctx():
    return ReportContext(
        prediction_id="p1",
        lang="fr",
        label="X",
        confidence_text="élevée",
        score_pct="90 %",
        top_features=[
            FeatureContribution(
                raw_name="a",
                label="A",
                value="1",
                direction="increase",
                weight=1.0,
            )
        ],
        model_name="rf",
        model_version="1",
    )


def _cfg():
    return OllamaConfig(base_url="http://localhost:11434", model="llama3.1:8b", timeout_s=5)


class _FakeResponse:
    def __init__(self, *, status: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Minimal stand-in for requests.Session that records calls and returns
    pre-canned responses or raises pre-canned exceptions."""

    def __init__(self, *, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def post(self, url, *, json=None, timeout=None):  # noqa: A002 — match requests signature
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.response


# ── Happy path ───────────────────────────────────────────────────────────────


def test_ollama_returns_parsed_dict_on_success():
    valid = {
        "summary": "ok",
        "prediction": {"label": "X", "confidence_text": "élevée"},
        "key_factors": [],
        "context": "c",
        "limitations": "l",
        "next_steps": "n",
        "disclaimer": "PLACEHOLDER",
    }
    session = _FakeSession(
        response=_FakeResponse(payload={"response": json.dumps(valid)}),
    )
    client = OllamaClient(_cfg(), session=session)
    out = client.generate(_ctx())
    assert out == valid

    # Verify the request was shaped correctly
    sent = session.calls[0]
    assert sent["url"].endswith("/api/generate")
    assert sent["json"]["format"] == "json"
    assert sent["json"]["stream"] is False
    assert sent["json"]["model"] == "llama3.1:8b"
    assert "system" in sent["json"]
    assert "prompt" in sent["json"]


def test_ollama_passes_reinforced_flag_to_prompt_builder():
    """Ensures the retry path actually reaches the prompt builder's
    reinforced branch and the system prompt grows."""
    captured: list[str] = []

    class _Capturing(_FakeSession):
        def post(self, url, *, json=None, timeout=None):
            captured.append(json["system"])
            return _FakeResponse(payload={"response": '{"ok": 1}'})

    session = _Capturing(response=None)
    client = OllamaClient(_cfg(), session=session)
    try:
        client.generate(_ctx(), reinforced=True)
    except OllamaError:
        pass  # response_not_object, fine — we just want the captured prompt
    assert any("RAPPEL STRICT" in s for s in captured)


# ── Failure modes ────────────────────────────────────────────────────────────


def test_ollama_raises_on_timeout():
    session = _FakeSession(exc=requests.Timeout("slow"))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="timeout"):
        client.generate(_ctx())


def test_ollama_raises_on_network_error():
    session = _FakeSession(exc=requests.ConnectionError("refused"))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="network"):
        client.generate(_ctx())


def test_ollama_raises_on_http_error():
    session = _FakeSession(response=_FakeResponse(status=500, text="server down"))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="http_500"):
        client.generate(_ctx())


def test_ollama_raises_on_empty_response():
    session = _FakeSession(response=_FakeResponse(payload={"response": ""}))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="empty_response"):
        client.generate(_ctx())


def test_ollama_raises_on_non_json_response():
    session = _FakeSession(response=_FakeResponse(payload={"response": "not json {bad"}))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="non_json_response"):
        client.generate(_ctx())


def test_ollama_raises_on_array_response():
    """The schema is an object — top-level arrays are invalid."""
    session = _FakeSession(response=_FakeResponse(payload={"response": "[1,2,3]"}))
    client = OllamaClient(_cfg(), session=session)
    with pytest.raises(OllamaError, match="response_not_object"):
        client.generate(_ctx())


# ── Warmup ───────────────────────────────────────────────────────────────────


def test_warmup_returns_true_on_2xx():
    session = _FakeSession(response=_FakeResponse(payload={"response": "ok"}))
    assert OllamaClient(_cfg(), session=session).warmup() is True


def test_warmup_returns_false_when_unreachable():
    session = _FakeSession(exc=requests.ConnectionError("refused"))
    assert OllamaClient(_cfg(), session=session).warmup() is False
