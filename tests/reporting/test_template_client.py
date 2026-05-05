"""Sprint 1 unit tests for the deterministic TemplateClient.

The TemplateClient is the bottom of the LLM fallback chain. It must
*always* produce a schema-valid report, in either supported language,
regardless of how sparse the input context is.
"""
from __future__ import annotations

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.template import TemplateClient


def _ctx(lang="fr", *, with_features=True, score=0.92):
    feats = []
    if with_features:
        feats = [
            FeatureContribution(
                raw_name="glucose_fasting",
                label="Glycémie à jeun" if lang == "fr" else "Fasting glucose",
                value="126 mg/dL",
                direction="increase",
                weight=1.0,
                normal_range="70–100 mg/dL",
            ),
            FeatureContribution(
                raw_name="bmi",
                label="IMC" if lang == "fr" else "BMI",
                value="31 kg/m²",
                direction="increase",
                weight=0.6,
                normal_range="18.5–24.9 kg/m²",
            ),
        ]
    return ReportContext(
        prediction_id="p1",
        lang=lang,
        label="Diabète probable" if lang == "fr" else "Likely diabetes",
        confidence_text="élevée" if lang == "fr" else "high",
        score_pct=f"{int(round(score * 100))} %" if score is not None else None,
        top_features=feats,
        model_name="random_forest",
        model_version="42",
    )


def test_template_returns_full_schema_fr():
    report = TemplateClient().generate(_ctx("fr"))
    for key in (
        "summary",
        "prediction",
        "key_factors",
        "context",
        "limitations",
        "next_steps",
        "disclaimer",
    ):
        assert key in report, f"missing key {key!r}"


def test_template_returns_full_schema_en():
    report = TemplateClient().generate(_ctx("en"))
    assert "summary" in report
    assert isinstance(report["key_factors"], list)


def test_template_does_not_invent_label():
    """LLM hallucination guard: only labels from context appear in output."""
    ctx = _ctx("fr")
    report = TemplateClient().generate(ctx)
    for factor in report["key_factors"]:
        assert factor["label"] in {f.label for f in ctx.top_features}
        assert factor["value"] in {f.value for f in ctx.top_features}


def test_template_disclaimer_is_placeholder():
    """The service is responsible for the final disclaimer text — the
    client must emit the literal placeholder string so the post-processor
    can swap it deterministically."""
    report = TemplateClient().generate(_ctx("fr"))
    assert report["disclaimer"] == "PLACEHOLDER"


def test_template_handles_no_features():
    report = TemplateClient().generate(_ctx("fr", with_features=False))
    assert report["key_factors"] == []
    assert "Le modèle suggère" in report["summary"]


def test_template_handles_no_score():
    ctx = _ctx("fr", score=None)
    # When score is None, score_pct must be None per the dataclass contract,
    # mirroring regression / unquantified cases.
    ctx_no_score = ReportContext(
        prediction_id=ctx.prediction_id,
        lang=ctx.lang,
        label=ctx.label,
        confidence_text="non quantifiée",
        score_pct=None,
        top_features=ctx.top_features,
        model_name=ctx.model_name,
        model_version=ctx.model_version,
    )
    report = TemplateClient().generate(ctx_no_score)
    assert "score" not in report["summary"].lower() or "score :" not in report["summary"]
    assert report["prediction"]["score_pct"] is None


def test_template_uses_conditional_phrasing():
    """Plan §9: must use conditional, never assertive diagnosis verbs."""
    report = TemplateClient().generate(_ctx("fr"))
    assert "suggère" in report["summary"]
    forbidden = ["vous êtes atteint", "vous avez", "diagnostique", "confirme"]
    full_text = " ".join(
        [
            report["summary"],
            report["context"],
            report["limitations"],
            report["next_steps"],
        ]
    )
    for f in forbidden:
        assert f.lower() not in full_text.lower()


# ── Sprint 3 — semantic phrasing keyed by (position × direction) ─────────────


def _feat(direction="increase", position="unknown"):
    return FeatureContribution(
        raw_name="glucose",
        label="Glycémie",
        value="126 mg/dL",
        direction=direction,
        weight=1.0,
        normal_range="70–100 mg/dL",
        position_vs_normal=position,  # type: ignore[arg-type]
    )


def _ctx_with(feat: FeatureContribution, lang="fr"):
    return ReportContext(
        prediction_id="p1",
        lang=lang,
        label="Diabète probable" if lang == "fr" else "Likely diabetes",
        confidence_text="élevée" if lang == "fr" else "high",
        score_pct="92 %",
        top_features=[feat],
        model_name="rf",
        model_version="1",
    )


def test_template_uses_above_phrasing_when_value_high_and_increases_risk():
    feat = _feat(direction="increase", position="above")
    report = TemplateClient().generate(_ctx_with(feat))
    expl = report["key_factors"][0]["explanation"]
    assert "anormalement élevée" in expl
    assert "pousse" in expl


def test_template_uses_below_phrasing_when_value_low_and_decreases_risk():
    feat = _feat(direction="decrease", position="below")
    report = TemplateClient().generate(_ctx_with(feat))
    expl = report["key_factors"][0]["explanation"]
    assert "anormalement basse" in expl
    assert "éloigne" in expl


def test_template_uses_within_phrasing_for_in_range_values():
    feat = _feat(direction="increase", position="within")
    report = TemplateClient().generate(_ctx_with(feat))
    expl = report["key_factors"][0]["explanation"]
    assert "plage habituelle" in expl


def test_template_falls_back_to_direction_only_when_position_unknown():
    """Without a normal range, no clinical claim is made about the value."""
    feat = _feat(direction="increase", position="unknown")
    report = TemplateClient().generate(_ctx_with(feat))
    expl = report["key_factors"][0]["explanation"]
    assert "anormalement" not in expl
    assert "pousse" in expl


def test_template_english_above_phrasing():
    feat = FeatureContribution(
        raw_name="glucose",
        label="Glucose",
        value="126 mg/dL",
        direction="increase",
        weight=1.0,
        normal_range="70–100 mg/dL",
        position_vs_normal="above",
    )
    report = TemplateClient().generate(_ctx_with(feat, lang="en"))
    expl = report["key_factors"][0]["explanation"]
    assert "abnormally high" in expl


def test_template_neutral_direction_still_works_with_position():
    """Neutral contributions skip the position matrix even if position is known."""
    feat = _feat(direction="neutral", position="above")
    report = TemplateClient().generate(_ctx_with(feat))
    expl = report["key_factors"][0]["explanation"]
    assert "anormalement" not in expl


def test_template_client_is_pure():
    """Same input → same output. Important for caching and tests."""
    ctx = _ctx("fr")
    a = TemplateClient().generate(ctx)
    b = TemplateClient().generate(ctx)
    assert a == b
