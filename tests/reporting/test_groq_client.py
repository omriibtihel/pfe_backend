"""Sprint 4 — GroqClient HTTP envelope tests (mocked, no network)."""
from __future__ import annotations

import json

import pytest
import requests

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.groq import (
    GroqClient,
    GroqConfig,
    GroqError,
)


def _ctx():
    return ReportContext(
        prediction_id="p1",
        lang="fr",
        label="X",
        confidence_text="élevée",
        score_pct="90 %",
        top_features=[FeatureContribution("a", "A", "1", "increase", 1.0)],
        model_name="m",
        model_version="1",
    )


def _cfg(key="sk-test"):
    return GroqConfig(api_key=key, model="llama-3.3-70b-versatile", timeout_s=5)


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
    def __init__(self, *, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def post(self, url, *, json=None, headers=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.response


# ── Construction ─────────────────────────────────────────────────────────────


def test_constructor_raises_when_api_key_empty():
    with pytest.raises(GroqError, match="disabled"):
        GroqClient(_cfg(key=""))


# ── Happy path ───────────────────────────────────────────────────────────────


def test_groq_returns_parsed_dict_on_success():
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
        response=_FakeResponse(payload={
            "choices": [{"message": {"content": json.dumps(valid)}}],
        }),
    )
    client = GroqClient(_cfg(), session=session)
    out = client.generate(_ctx())
    assert out == valid

    sent = session.calls[0]
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    assert sent["json"]["response_format"]["type"] == "json_object"
    # Standard OpenAI message shape
    assert sent["json"]["messages"][0]["role"] == "system"
    assert sent["json"]["messages"][1]["role"] == "user"


def test_groq_passes_reinforced_flag_to_prompt_builder():
    captured = []

    class _Capturing(_FakeSession):
        def post(self, url, *, json=None, headers=None, timeout=None):
            captured.append(json["messages"][0]["content"])
            return _FakeResponse(payload={
                "choices": [{"message": {"content": "{\"ok\": 1}"}}],
            })

    session = _Capturing()
    client = GroqClient(_cfg(), session=session)
    try:
        client.generate(_ctx(), reinforced=True)
    except GroqError:
        pass  # response_not_object is fine; we just want the captured prompt
    assert any("RAPPEL STRICT" in s for s in captured)


# ── Failure modes ────────────────────────────────────────────────────────────


def test_groq_raises_on_timeout():
    session = _FakeSession(exc=requests.Timeout("slow"))
    with pytest.raises(GroqError, match="timeout"):
        GroqClient(_cfg(), session=session).generate(_ctx())


def test_groq_raises_on_network_error():
    session = _FakeSession(exc=requests.ConnectionError("refused"))
    with pytest.raises(GroqError, match="network"):
        GroqClient(_cfg(), session=session).generate(_ctx())


def test_groq_raises_on_http_401():
    session = _FakeSession(response=_FakeResponse(status=401, text="unauthorized"))
    with pytest.raises(GroqError, match="http_401"):
        GroqClient(_cfg(), session=session).generate(_ctx())


def test_groq_raises_on_missing_content():
    session = _FakeSession(response=_FakeResponse(payload={"choices": []}))
    with pytest.raises(GroqError, match="missing_content"):
        GroqClient(_cfg(), session=session).generate(_ctx())


def test_groq_raises_on_non_json_response():
    session = _FakeSession(response=_FakeResponse(payload={
        "choices": [{"message": {"content": "not json {bad"}}],
    }))
    with pytest.raises(GroqError, match="non_json_response"):
        GroqClient(_cfg(), session=session).generate(_ctx())
