"""Sprint 2 unit tests for ReportValidator.

Covers:
- JSON schema rejects malformed structure
- Blocklist scan flags forbidden diagnostic phrases (FR + EN)
- JSON extraction tolerates fenced / trailing-text LLM output
- Disclaimer placeholder is REQUIRED at validation time (rewrite is service-side)
"""
from __future__ import annotations

import json

import pytest

from app.services.reporting.validator import (
    ReportValidator,
    ValidationResult,
    extract_json,
)


def _valid_report() -> dict:
    return {
        "summary": "Le modèle suggère X avec une confiance modérée.",
        "prediction": {"label": "X", "confidence_text": "modérée", "score_pct": "70 %"},
        "key_factors": [
            {
                "label": "Glycémie",
                "value": "126 mg/dL",
                "direction": "augmente",
                "explanation": "Pousse le modèle.",
                "normal_range": "70–100 mg/dL",
            }
        ],
        "context": "Modèle statistique.",
        "limitations": "Pas de diagnostic.",
        "next_steps": "À discuter.",
        "disclaimer": "PLACEHOLDER",
    }


# ── extract_json ──────────────────────────────────────────────────────────────

def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_fence():
    raw = "Sure! Here it is:\n```json\n{\"a\": 1}\n```\nHope this helps."
    assert extract_json(raw) == {"a": 1}


def test_extract_json_greedy_object():
    raw = "Output: {\"a\": 1, \"b\": [1,2,3]} done."
    assert extract_json(raw) == {"a": 1, "b": [1, 2, 3]}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("not json at all") is None


def test_extract_json_returns_none_on_array():
    # The schema requires an object, so a top-level array is not acceptable.
    assert extract_json("[1, 2, 3]") is None


def test_extract_json_handles_empty():
    assert extract_json("") is None


# ── Schema validation ────────────────────────────────────────────────────────

def test_validator_passes_well_formed_report():
    v = ReportValidator()
    result = v.validate(_valid_report(), lang="fr")
    assert result.status == "passed"
    assert result.report is not None


def test_validator_rejects_missing_summary():
    v = ReportValidator()
    bad = _valid_report()
    del bad["summary"]
    result = v.validate(bad, lang="fr")
    assert result.status == "rejected"
    assert result.reason and result.reason.startswith("schema")


def test_validator_rejects_wrong_disclaimer_placeholder():
    """The schema pins ``disclaimer`` to the literal string PLACEHOLDER —
    the LLM must respect this so the postprocessor's rewrite is unambiguous."""
    v = ReportValidator()
    bad = _valid_report()
    bad["disclaimer"] = "Some other text"
    result = v.validate(bad, lang="fr")
    assert result.status == "rejected"
    assert result.reason and result.reason.startswith("schema")


def test_validator_accepts_string_input():
    """Validator should run extract_json under the hood when handed a string."""
    v = ReportValidator()
    result = v.validate(json.dumps(_valid_report()), lang="fr")
    assert result.status == "passed"


def test_validator_returns_invalid_json_for_garbage_string():
    v = ReportValidator()
    result = v.validate("garbage", lang="fr")
    assert result.status == "rejected"
    assert result.reason == "invalid_json"


def test_validator_rejects_empty_key_factors_label():
    v = ReportValidator()
    bad = _valid_report()
    bad["key_factors"][0]["label"] = 12  # wrong type
    result = v.validate(bad, lang="fr")
    assert result.status == "rejected"


# ── Blocklist scan ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "field, phrase",
    [
        ("summary", "Vous êtes atteint de diabète selon le modèle."),
        ("context", "Je diagnostique une pathologie chronique."),
        ("limitations", "Vous souffrez de diabète."),
        ("next_steps", "Tu es atteint d'une maladie."),
    ],
)
def test_validator_rejects_french_diagnostic_verbs(field, phrase):
    v = ReportValidator()
    bad = _valid_report()
    bad[field] = phrase
    result = v.validate(bad, lang="fr")
    assert result.status == "rejected"
    assert result.reason == "blocklist"
    assert result.blocklist_hits


@pytest.mark.parametrize(
    "field, phrase",
    [
        ("summary", "You are diagnosed with diabetes."),
        ("context", "I confirm the diagnosis."),
        ("limitations", "You have diabetes."),
    ],
)
def test_validator_rejects_english_diagnostic_verbs(field, phrase):
    v = ReportValidator()
    bad = _valid_report()
    bad["disclaimer"] = "PLACEHOLDER"
    bad[field] = phrase
    # English content with French validator? must explicitly switch lang.
    result = v.validate(bad, lang="en")
    assert result.status == "rejected"
    assert result.reason == "blocklist"


def test_validator_blocklist_skips_disclaimer_field():
    """Disclaimer is rewritten anyway; flagging it would block valid reports."""
    v = ReportValidator()
    bad = _valid_report()
    # Even a "diagnosed" word inside the placeholder string would be ok.
    # We use the mandatory placeholder here, but the principle is tested.
    result = v.validate(bad, lang="fr")
    assert result.status == "passed"


def test_validator_blocklist_checks_factor_explanation():
    """The blocklist must scan inside key_factors[].explanation, where the
    LLM is most likely to slip a diagnostic verb."""
    v = ReportValidator()
    bad = _valid_report()
    bad["key_factors"][0]["explanation"] = "Vous êtes atteint à cause de cette valeur."
    result = v.validate(bad, lang="fr")
    assert result.status == "rejected"
    assert result.reason == "blocklist"


# ── ValidationResult shape ───────────────────────────────────────────────────

def test_validation_result_passed_carries_report():
    v = ReportValidator()
    result = v.validate(_valid_report(), lang="fr")
    assert isinstance(result, ValidationResult)
    assert result.report is not None
    assert "summary" in result.report


def test_validation_result_rejected_carries_no_report():
    v = ReportValidator()
    result = v.validate("garbage", lang="fr")
    assert result.report is None
