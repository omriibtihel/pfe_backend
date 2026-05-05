"""Sprint 4 — LLMRouter chain semantics.

Asserts the §8 state machine when MULTIPLE clients are involved:
- ollama times out → groq tried → groq passes
- ollama returns invalid JSON → groq tried (no reinforced retry on ollama)
- ollama returns blocklist hit → ollama RETRIED with reinforced → if pass, return
- all real backends fail → TemplateClient floor always wins
"""
from __future__ import annotations

from typing import Any

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.router import LLMRouter
from app.services.reporting.llm_client.template import TemplateClient


def _ctx():
    return ReportContext(
        prediction_id="p1",
        lang="fr",
        label="X",
        confidence_text="élevée",
        score_pct="90 %",
        top_features=[
            FeatureContribution(
                raw_name="g",
                label="G",
                value="1",
                direction="increase",
                weight=1.0,
            )
        ],
        model_name="m",
        model_version="1",
    )


def _valid() -> dict:
    return {
        "summary": "ok",
        "prediction": {"label": "X", "confidence_text": "élevée"},
        "key_factors": [],
        "context": "c",
        "limitations": "l",
        "next_steps": "n",
        "disclaimer": "PLACEHOLDER",
    }


class _Scripted:
    """Each call pops the next item from `script`. Items are either values
    to return or exceptions to raise. Records every call's reinforced flag."""

    def __init__(self, name: str, script: list):
        self.name = name
        self.script = list(script)
        self.calls: list[bool] = []

    def generate(self, ctx, *, reinforced: bool = False) -> Any:  # noqa: ARG002
        self.calls.append(reinforced)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ── Single-client behavior (regression coverage) ──────────────────────────────


def test_router_returns_first_passing_client():
    a = _Scripted("a", [_valid()])
    b = _Scripted("b", [_valid()])  # never called
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "a"
    assert result.retries == 0
    assert result.fallback_reason is None
    assert b.calls == []


def test_router_appends_template_floor_automatically():
    """Even if you forget to add TemplateClient, the router ensures the
    chain has a deterministic floor."""
    only_failing = _Scripted("a", [RuntimeError("boom")])
    result = LLMRouter([only_failing]).generate(_ctx())
    assert result.provider == "template"


# ── Inter-client escalation (the new Sprint 4 path) ───────────────────────────


def test_router_escalates_to_next_on_exception():
    a = _Scripted("a", [RuntimeError("network")])
    b = _Scripted("b", [_valid()])
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "b"
    # Exception path: only ONE call to a (no reinforced retry on exceptions)
    assert a.calls == [False]
    # b was tried with a fresh prompt (not reinforced), since it's a new client
    assert b.calls == [False]
    assert result.fallback_reason and result.fallback_reason.startswith("a:exception:")


def test_router_escalates_to_next_on_invalid_json():
    """Plan §8: invalid_json on client X → escalate WITHOUT reinforced retry."""
    a = _Scripted("a", ["complete garbage"])
    b = _Scripted("b", [_valid()])
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "b"
    assert a.calls == [False]  # only one attempt
    assert result.fallback_reason and "invalid_json" in result.fallback_reason


def test_router_retries_same_client_on_blocklist_then_escalates():
    """Schema/blocklist failure → retry SAME client reinforced. If still
    failing → escalate to next client."""
    bad = _valid()
    bad["summary"] = "Vous êtes atteint de X."
    a = _Scripted("a", [bad, bad])
    b = _Scripted("b", [_valid()])
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "b"
    # Critical: a was tried twice — initial + reinforced
    assert a.calls == [False, True]


def test_router_returns_first_client_when_reinforced_retry_passes():
    bad = _valid()
    bad["summary"] = "Vous êtes atteint de X."
    a = _Scripted("a", [bad, _valid()])  # rejected, then passes on retry
    b = _Scripted("b", [_valid()])  # never reached
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "a"
    assert result.retries == 1
    assert a.calls == [False, True]
    assert b.calls == []


def test_router_falls_back_to_template_when_all_real_clients_fail():
    """End of chain — template always wins as last resort."""
    a = _Scripted("a", [RuntimeError("net")])
    b = _Scripted("b", ["garbage"])
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "template"
    assert result.fallback_reason and "b:" in result.fallback_reason


def test_router_records_full_attempt_trace():
    bad = _valid()
    bad["summary"] = "Vous êtes atteint."
    a = _Scripted("a", [bad, bad])  # 2 attempts, both rejected
    b = _Scripted("b", [RuntimeError("net")])  # 1 attempt, exception
    result = LLMRouter([a, b]).generate(_ctx())
    assert result.provider == "template"
    # Trace: a×2 (both blocklist) + b×1 (exception)
    assert len(result.attempts) == 3
    assert result.attempts[0].client == "a" and result.attempts[0].reinforced is False
    assert result.attempts[1].client == "a" and result.attempts[1].reinforced is True
    assert result.attempts[2].client == "b"
    assert result.attempts[2].outcome.startswith("exception:")


def test_router_does_not_validate_template_output():
    """TemplateClient is canonical — must short-circuit through the chain
    without going through the validator."""
    chain = [TemplateClient()]
    result = LLMRouter(chain).generate(_ctx())
    assert result.provider == "template"
    assert result.retries == 0
    assert result.fallback_reason is None
