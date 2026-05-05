"""Sprint 2 unit tests for PromptBuilder.

The prompt is the *only* thing the LLM sees. The plan's golden rule (no raw
floats reach the model) is enforced both at the dataclass layer and at the
prompt serialization layer; this file pins the prompt-layer guarantees:

- The serialized payload contains no ``weight`` field (an internal float).
- The serialized payload contains no ``raw_name`` (unsanitized column name).
- The prompt embeds the JSON schema and the few-shot examples.
- The prompt is bounded in length even with a very wide feature list.
"""
from __future__ import annotations

import json
import re

import pytest

from app.services.reporting.context_builder import (
    FeatureContribution,
    ModelQualitySignal,
    PredictionClassContext,
    ReportContext,
)
from app.services.reporting.prompts import PromptBuilder


def _ctx(lang="fr", n_features=2):
    feats = [
        FeatureContribution(
            raw_name=f"col_{i}",
            label=f"Feature {i}",
            value=f"{i} mg/dL",
            direction="increase" if i % 2 == 0 else "decrease",
            weight=1.0 / (i + 1),
            normal_range="70–100 mg/dL",
        )
        for i in range(n_features)
    ]
    return ReportContext(
        prediction_id="p1",
        lang=lang,
        label="Diabète probable",
        confidence_text="élevée" if lang == "fr" else "high",
        score_pct="92 %",
        top_features=feats,
        class_context=PredictionClassContext(
            raw_label="1",
            target_name="Outcome",
            positive_class="1",
            label_meaning="classe positive: risque de diabete suggere",
        ),
        model_quality=[
            ModelQualitySignal("Exactitude globale", "77.5 %", "part des predictions correctes"),
            ModelQualitySignal("Rappel classe positive", "71.6 %", "capacite a retrouver les cas positifs"),
        ],
        dataset_summary=["Jeu de donnees d'entrainement: 768 lignes"],
        model_name="rf",
        model_version="1",
    )


# ── Schema + few-shot embedding ──────────────────────────────────────────────

def test_prompt_contains_schema_and_few_shot():
    prompts = PromptBuilder().build(_ctx())
    user_payload = json.loads(prompts["user"])
    assert list(user_payload.keys())[0] == "context"
    assert "schema" in user_payload
    assert "few_shot" in user_payload
    assert isinstance(user_payload["few_shot"], list)
    assert len(user_payload["few_shot"]) == 2  # plan §9


def test_prompt_few_shot_outputs_use_placeholder_disclaimer():
    """Few-shot must teach the model the literal PLACEHOLDER string."""
    prompts = PromptBuilder().build(_ctx())
    payload = json.loads(prompts["user"])
    for example in payload["few_shot"]:
        assert example["output"]["disclaimer"] == "PLACEHOLDER"


# ── Serialization safety (no float, no raw_name) ─────────────────────────────

def test_prompt_omits_internal_weight_field():
    """The normalized ``weight`` is an ordering signal — order in the list
    already conveys it. Leaking the float would invite the LLM to
    parrot a number it shouldn't be reasoning about."""
    prompts = PromptBuilder().build(_ctx(n_features=4))
    payload = json.loads(prompts["user"])
    for feat in payload["context"]["top_features"]:
        assert "weight" not in feat


def test_prompt_omits_raw_column_name():
    """Raw column names may still contain unsanitized characters from the
    source dataset. The label is the LLM-safe surface."""
    prompts = PromptBuilder().build(_ctx(n_features=2))
    payload = json.loads(prompts["user"])
    for feat in payload["context"]["top_features"]:
        assert "raw_name" not in feat


def test_prompt_does_not_contain_bare_floats_in_serialized_context():
    """Hard rule from plan §14: the LLM never receives raw floats.
    ``score_pct`` is the only quantitative field, and it's a pre-formatted
    string ('92 %'). All feature values are strings."""
    prompts = PromptBuilder().build(_ctx(n_features=3))
    payload = json.loads(prompts["user"])
    ctx = payload["context"]
    # All values that would be tempting to emit as floats must be strings or null
    assert isinstance(ctx["score_pct"], (str, type(None)))
    for feat in ctx["top_features"]:
        for k in ("value", "direction", "label"):
            assert isinstance(feat[k], str)
        assert feat["normal_range"] is None or isinstance(feat["normal_range"], str)


def test_prompt_marks_evidence_type_for_features():
    """evidence_type must be one of the two supported values.

    LIME-derived features → "lime_contribution".
    Fallback observed-only features → "observed_value".
    """
    prompts = PromptBuilder().build(_ctx(n_features=2))
    payload = json.loads(prompts["user"])
    for feat in payload["context"]["top_features"]:
        assert feat["evidence_type"] in {"lime_contribution", "observed_value"}


def test_prompt_serializes_model_context():
    prompts = PromptBuilder().build(_ctx(n_features=1))
    ctx = json.loads(prompts["user"])["context"]
    assert ctx["class_context"]["raw_label"] == "1"
    assert ctx["class_context"]["target_name"] == "Outcome"
    assert ctx["model_quality"][0]["label"] == "Exactitude globale"
    assert ctx["dataset_summary"] == ["Jeu de donnees d'entrainement: 768 lignes"]


# ── Reinforcement clause ─────────────────────────────────────────────────────

def test_reinforced_prompt_appends_strict_reminder():
    base = PromptBuilder().build(_ctx(), reinforced=False)
    reinforced = PromptBuilder().build(_ctx(), reinforced=True)
    assert len(reinforced["system"]) > len(base["system"])
    assert "PLACEHOLDER" in reinforced["system"]


def test_reinforced_prompt_uses_correct_language():
    fr = PromptBuilder().build(_ctx("fr"), reinforced=True)
    en = PromptBuilder().build(_ctx("en"), reinforced=True)
    assert "RAPPEL STRICT" in fr["system"]
    assert "STRICT REMINDER" in en["system"]


# ── Length bounding ──────────────────────────────────────────────────────────

def test_prompt_length_is_bounded_with_many_features():
    """Even with an absurdly long top_features list, the prompt must stay
    under the cap so the context window is preserved."""
    ctx = _ctx(n_features=100)
    prompts = PromptBuilder().build(ctx)
    assert len(prompts["user"]) <= 32_000  # _MAX_PROMPT_CHARS (raised for Groq 128k context)


# ── System prompt contains the policy ────────────────────────────────────────

def test_system_prompt_states_strict_rules():
    prompts = PromptBuilder().build(_ctx("fr"))
    sys = prompts["system"]
    # Each strict rule from plan §9 must be present
    assert "B1" in sys
    assert "PLACEHOLDER" in sys
    # Forbidden phrases referenced in the policy
    assert "vous êtes atteint" in sys.lower()
    # JSON-only output
    assert "json" in sys.lower()


def test_system_prompt_language_matches_context():
    fr = PromptBuilder().build(_ctx("fr"))["system"]
    en = PromptBuilder().build(_ctx("en"))["system"]
    assert "français" in fr.lower() or "rédacteur" in fr.lower()
    assert "english" in en.lower() or "writer" in en.lower()
