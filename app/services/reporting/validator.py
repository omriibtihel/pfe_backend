"""ReportValidator — schema check + blocklist scan + JSON extraction.

The validator is *advisory*: it returns a structured result describing what
went wrong. The retry / fallback decision lives in ``ReportService``. This
keeps the validator pure and easy to test.

Plan §1 + §8 contract:
- Parse JSON from the raw LLM response (tolerant: handles trailing prose).
- Validate against the schema embedded in ``prompts._OUTPUT_SCHEMA``.
- Reject if any forbidden diagnostic phrase appears in user-visible text.
- Disclaimer rewrite happens AFTER validation, in the service.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from jsonschema import Draft202012Validator

from app.services.reporting.prompts import _OUTPUT_SCHEMA

ValidationStatus = Literal["passed", "rejected"]


# ── Blocklists (plan §9) ──────────────────────────────────────────────────────
# Literal substrings, lowercased before match. Tradeoff: false positives are
# acceptable (they trigger a retry, then template fallback), false negatives
# are not (would leak a diagnosis verb to a non-clinician reader).

_FORBIDDEN_FR = (
    "vous êtes atteint",
    "vous êtes diabétique",
    "vous êtes hypertendu",
    "vous souffrez de",
    "je diagnostique",
    "je confirme",
    "il est certain que",
    "tu es atteint",
)

_FORBIDDEN_EN = (
    "you are diagnosed",
    "i diagnose",
    "i confirm",
    "you suffer from",
    "you have diabetes",
    "you have cancer",
    "you have the disease",
    "diagnosed with",
)


def _forbidden_for(lang: str) -> tuple[str, ...]:
    return _FORBIDDEN_FR if lang == "fr" else _FORBIDDEN_EN


# ── JSON extraction (plan §8) ─────────────────────────────────────────────────
# Some LLMs prepend "Sure, here is the JSON:" or wrap output in ``` fences.
# We accept that gracefully rather than failing the whole stream.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON extraction from arbitrary LLM text. Returns None on failure."""
    if not raw:
        return None
    raw = raw.strip()
    # 1) Try the cleanest path first.
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    # 2) Strip ``` fences, take inner block.
    fence = _FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    # 3) Last resort: greediest object match.
    obj = _OBJECT_RE.search(raw)
    if obj:
        try:
            parsed = json.loads(obj.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return None


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    status: ValidationStatus
    report: Optional[dict[str, Any]]   # parsed report if status == "passed"
    reason: Optional[str]              # human-readable reason if rejected
    blocklist_hits: tuple[str, ...] = ()


# ── Validator ─────────────────────────────────────────────────────────────────


class ReportValidator:
    """Stateless validator. Cheap to instantiate per request."""

    def __init__(self) -> None:
        self._schema = Draft202012Validator(_OUTPUT_SCHEMA)

    def validate(self, raw: str | dict[str, Any], lang: str) -> ValidationResult:
        report = raw if isinstance(raw, dict) else extract_json(str(raw))
        if report is None:
            return ValidationResult(
                status="rejected",
                report=None,
                reason="invalid_json",
            )

        # Schema validation. We collect the first error to keep retry prompts
        # short; full error report bloats the reinforced prompt unnecessarily.
        errors = sorted(self._schema.iter_errors(report), key=lambda e: e.path)
        if errors:
            err = errors[0]
            path = ".".join(str(p) for p in err.path) or "<root>"
            return ValidationResult(
                status="rejected",
                report=None,
                reason=f"schema:{path}:{err.message[:120]}",
            )

        # Blocklist scan over user-visible text only. We deliberately skip the
        # ``disclaimer`` field because it will be overwritten with canonical
        # text by the post-processor and any LLM artifact there is harmless.
        hits = self._scan_blocklist(report, lang)
        if hits:
            return ValidationResult(
                status="rejected",
                report=None,
                reason="blocklist",
                blocklist_hits=hits,
            )

        return ValidationResult(status="passed", report=report, reason=None)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_text_fields(report: dict[str, Any]):
        """Yield every user-visible string in the report."""
        for key in ("summary", "context", "limitations", "next_steps"):
            v = report.get(key)
            if isinstance(v, str):
                yield v
        pred = report.get("prediction") or {}
        if isinstance(pred, dict):
            for k in ("label", "confidence_text"):
                v = pred.get(k)
                if isinstance(v, str):
                    yield v
        for factor in report.get("key_factors") or []:
            if not isinstance(factor, dict):
                continue
            for k in ("label", "value", "direction", "explanation"):
                v = factor.get(k)
                if isinstance(v, str):
                    yield v
        for change in report.get("what_to_change") or []:
            if not isinstance(change, dict):
                continue
            for k in ("factor", "current", "target", "change_text", "why_it_matters"):
                v = change.get(k)
                if isinstance(v, str):
                    yield v
    def _scan_blocklist(self, report: dict[str, Any], lang: str) -> tuple[str, ...]:
        forbidden = _forbidden_for(lang)
        hits: list[str] = []
        for text in self._iter_text_fields(report):
            low = text.lower()
            for needle in forbidden:
                if needle in low:
                    hits.append(needle)
        return tuple(sorted(set(hits)))


__all__ = ["ReportValidator", "ValidationResult", "extract_json"]
