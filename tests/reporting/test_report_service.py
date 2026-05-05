"""Sprint 1 integration tests for ReportService end-to-end.

Validates the data → render path that the SSE route exercises:
- Service swaps the LLM disclaimer placeholder for the canonical text
- Service falls back to TemplateClient when the configured client raises
- Streaming yields one SSE chunk per section, then a 'done' event
"""
from __future__ import annotations

import asyncio
import json

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.base import LLMReport
from app.services.reporting.service import ReportService


def _ctx():
    return ReportContext(
        prediction_id="p1",
        lang="fr",
        label="Diabète probable",
        confidence_text="élevée",
        score_pct="92 %",
        top_features=[
            FeatureContribution(
                raw_name="glucose",
                label="Glycémie",
                value="126 mg/dL",
                direction="increase",
                weight=1.0,
            )
        ],
        model_name="rf",
        model_version="1",
    )


# ── Disclaimer rewrite ────────────────────────────────────────────────────────

def test_service_rewrites_disclaimer_fr():
    report, meta = ReportService().generate(_ctx())
    assert "PLACEHOLDER" not in json.dumps(report, ensure_ascii=False)
    assert "professionnel de santé" in report["disclaimer"]
    assert meta["provider"] == "template"
    assert meta["lang"] == "fr"


def test_service_rewrites_disclaimer_en():
    ctx = _ctx()
    en_ctx = ReportContext(
        prediction_id=ctx.prediction_id,
        lang="en",
        label=ctx.label,
        confidence_text="high",
        score_pct=ctx.score_pct,
        top_features=ctx.top_features,
        model_name=ctx.model_name,
        model_version=ctx.model_version,
    )
    report, _ = ReportService().generate(en_ctx)
    assert "healthcare professional" in report["disclaimer"]


# ── Fallback when the configured client raises ────────────────────────────────

class _BrokenClient:
    name = "broken"

    def generate(self, context: ReportContext) -> LLMReport:  # noqa: ARG002
        raise RuntimeError("boom")


def test_service_falls_back_to_template_when_client_raises():
    svc = ReportService(client=_BrokenClient())
    report, meta = svc.generate(_ctx())
    assert meta["provider"] == "template"
    assert "summary" in report and report["summary"]


# ── Streaming output ──────────────────────────────────────────────────────────

def _collect(agen):
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out

    return asyncio.run(run())


def test_stream_yields_chunks_then_done():
    chunks = _collect(ReportService().generate_stream(_ctx()))
    # Each chunk is a fully-formed SSE message
    assert all(c.startswith("event: ") for c in chunks)
    types = [c.split("\n", 1)[0].split(": ", 1)[1] for c in chunks]
    assert types[-1] == "done"
    assert types.count("chunk") >= 5  # at least summary, prediction, factors, context, ...


def test_stream_chunk_payload_is_json():
    chunks = _collect(ReportService().generate_stream(_ctx()))
    for c in chunks:
        # SSE: "event: X\ndata: <json>\n\n"
        data_line = next(line for line in c.splitlines() if line.startswith("data: "))
        json.loads(data_line[len("data: "):])  # must parse


def test_stream_done_event_has_meta():
    chunks = _collect(ReportService().generate_stream(_ctx()))
    done = chunks[-1]
    data = json.loads(next(l for l in done.splitlines() if l.startswith("data: "))[6:])
    assert data["provider"] == "template"
    assert data["report_id"].startswith("rpt_")
    assert isinstance(data["latency_ms"], int)


# ── Sprint 2 — retry / validation / fallback (plan §8) ────────────────────────


def _valid_llm_report() -> LLMReport:
    return {
        "summary": "Le modèle suggère X.",
        "prediction": {"label": "X", "confidence_text": "élevée", "score_pct": "90 %"},
        "key_factors": [],
        "context": "c",
        "limitations": "l",
        "next_steps": "n",
        "disclaimer": "PLACEHOLDER",
    }


class _ScriptedClient:
    """LLMClient that returns a pre-baked sequence of values, recording each
    call's ``reinforced`` flag. Items in ``script`` may be exceptions; they
    are raised. Otherwise they are returned as-is."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[bool] = []

    def generate(self, context, *, reinforced: bool = False):  # noqa: ARG002
        self.calls.append(reinforced)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_service_uses_first_attempt_when_valid():
    client = _ScriptedClient([_valid_llm_report()])
    report, meta = ReportService(client=client).generate(_ctx())
    assert meta["provider"] == "scripted"
    assert meta["retries"] == 0
    assert meta["fallback_reason"] is None
    assert client.calls == [False]
    assert report["summary"] == "Le modèle suggère X."


def test_service_retries_with_reinforced_prompt_on_blocklist():
    bad = _valid_llm_report()
    bad["summary"] = "Vous êtes atteint de X."
    client = _ScriptedClient([bad, _valid_llm_report()])
    report, meta = ReportService(client=client).generate(_ctx())
    assert meta["provider"] == "scripted"
    assert meta["retries"] == 1
    # First call non-reinforced, second call reinforced
    assert client.calls == [False, True]


def test_service_falls_back_when_retry_also_fails():
    bad = _valid_llm_report()
    bad["summary"] = "Vous êtes atteint."
    client = _ScriptedClient([bad, bad])
    report, meta = ReportService(client=client).generate(_ctx())
    assert meta["provider"] == "template"
    # Sprint 4 router prefixes the reason with the client name for chain debug
    assert meta["fallback_reason"] == "scripted:blocklist"
    # Template output replaces the rejected one
    assert "Vous êtes atteint" not in report["summary"]


def test_service_skips_retry_on_invalid_json():
    """Plan §8: invalid JSON escalates immediately, no reinforced retry."""
    client = _ScriptedClient(["totally not json"])
    report, meta = ReportService(client=client).generate(_ctx())
    assert meta["provider"] == "template"
    assert meta["fallback_reason"] == "scripted:invalid_json"
    assert client.calls == [False]  # only one call, not two


def test_service_skips_retry_on_exception():
    """Network/timeout exceptions don't retry — fall back to template."""
    client = _ScriptedClient([RuntimeError("boom")])
    report, meta = ReportService(client=client).generate(_ctx())
    assert meta["provider"] == "template"
    assert meta["fallback_reason"] == "scripted:exception:RuntimeError"
    assert client.calls == [False]


def test_service_does_not_validate_template_output():
    """TemplateClient output is canonical — no schema check needed, and the
    fallback path must not loop on it."""
    report, meta = ReportService().generate(_ctx())
    assert meta["provider"] == "template"
    assert meta["retries"] == 0
    assert meta["fallback_reason"] is None


def test_service_repairs_blank_llm_factor_values_from_context():
    """The LLM may blank factual table fields; server-derived values win."""

    class _BlankFactorClient:
        name = "blanker"

        def generate(self, context, *, reinforced: bool = False):  # noqa: ARG002
            return {
                "summary": "ok",
                "prediction": {
                    "label": context.label,
                    "confidence_text": context.confidence_text,
                    "score_pct": context.score_pct,
                },
                "key_factors": [
                    {
                        "label": "num__Glucose",
                        "value": "—",
                        "direction": "increase",
                        "explanation": "texte LLM conserve",
                        "normal_range": None,
                    }
                ],
                "context": "c",
                "limitations": "l",
                "next_steps": "n",
                "disclaimer": "PLACEHOLDER",
            }

    ctx = _ctx()
    repaired_ctx = ReportContext(
        prediction_id=ctx.prediction_id,
        lang=ctx.lang,
        label=ctx.label,
        confidence_text=ctx.confidence_text,
        score_pct=ctx.score_pct,
        top_features=[
            FeatureContribution(
                raw_name="num__Glucose",
                label="Glycémie",
                value="148 mg/dL",
                direction="increase",
                weight=1.0,
                normal_range="70–100 mg/dL",
            )
        ],
        model_name=ctx.model_name,
        model_version=ctx.model_version,
    )
    report, meta = ReportService(client=_BlankFactorClient()).generate(repaired_ctx)
    assert meta["provider"] == "blanker"
    factor = report["key_factors"][0]
    assert factor["label"] == "Glycémie"
    assert factor["value"] == "148 mg/dL"
    assert factor["normal_range"] == "70–100 mg/dL"
    assert factor["explanation"] == "texte LLM conserve"
