"""Deterministic, always-available report generator.

Used as the bottom of the fallback chain: when both Ollama and Groq fail (or
when neither is configured), this client still produces a valid report from
the same ``ReportContext`` the LLM clients would have consumed. Output shape
matches the JSON schema in plan §2.2 verbatim, so the validator and the
frontend treat the response identically regardless of backend.
"""
from __future__ import annotations

from typing import Any

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)
from app.services.reporting.llm_client.base import LLMReport


# ── Localized phrasing tables ────────────────────────────────────────────────
# Each entry covers both languages so the function bodies stay short. Keep
# wording at B1 reading level and in the conditional ("le modèle suggère…").

_DIRECTION_VERB = {
    ("fr", "increase"): "augmente",
    ("fr", "decrease"): "diminue",
    ("fr", "neutral"): "neutre",
    ("en", "increase"): "increase",
    ("en", "decrease"): "decrease",
    ("en", "neutral"): "neutral",
}

_DIRECTION_SENTENCE = {
    ("fr", "increase"): "Cette mesure pousse l'estimation du modèle vers « {label} ».",
    ("fr", "decrease"): "Cette mesure éloigne l'estimation du modèle de « {label} ».",
    ("fr", "neutral"): "Cette mesure a un effet faible sur l'estimation du modèle.",
    ("en", "increase"): "This measurement pushes the model toward \"{label}\".",
    ("en", "decrease"): "This measurement moves the model away from \"{label}\".",
    ("en", "neutral"): "This measurement has little effect on the model output.",
}

# Richer phrasing keyed by (lang, position, direction). When a normal range is
# known and the value is comparable, we say "anormalement élevée" /
# "anormalement basse" / "dans la plage habituelle" rather than the
# generic direction-only sentence above.
_POSITION_DIRECTION_SENTENCE: dict[tuple[str, str, str], str] = {
    # ── French ────────────────────────────────────────────────────────────
    ("fr", "above", "increase"):
        "Cette valeur est anormalement élevée, ce qui pousse le modèle vers « {label} ».",
    ("fr", "above", "decrease"):
        "Cette valeur est anormalement élevée, mais elle éloigne le modèle de « {label} ».",
    ("fr", "below", "increase"):
        "Cette valeur est anormalement basse, ce qui pousse le modèle vers « {label} ».",
    ("fr", "below", "decrease"):
        "Cette valeur est anormalement basse, ce qui éloigne le modèle de « {label} ».",
    ("fr", "within", "increase"):
        "Bien que dans la plage habituelle, cette valeur contribue vers « {label} ».",
    ("fr", "within", "decrease"):
        "Cette valeur, dans la plage habituelle, éloigne le modèle de « {label} ».",
    # ── English ───────────────────────────────────────────────────────────
    ("en", "above", "increase"):
        "This value is abnormally high, which pushes the model toward \"{label}\".",
    ("en", "above", "decrease"):
        "This value is abnormally high, but it moves the model away from \"{label}\".",
    ("en", "below", "increase"):
        "This value is abnormally low, which pushes the model toward \"{label}\".",
    ("en", "below", "decrease"):
        "This value is abnormally low, which moves the model away from \"{label}\".",
    ("en", "within", "increase"):
        "Although within the usual range, this value contributes to \"{label}\".",
    ("en", "within", "decrease"):
        "This value, within the usual range, moves the model away from \"{label}\".",
}

_CONTEXT_TEXT = {
    "fr": (
        "Les modèles statistiques exploitent des corrélations apprises sur des "
        "patients similaires. Ils ne tiennent pas compte du contexte clinique "
        "complet (antécédents, examen physique, autres examens)."
    ),
    "en": (
        "Statistical models exploit correlations learned from similar patients. "
        "They do not consider the full clinical context (history, physical "
        "examination, other tests)."
    ),
}

_LIMITATIONS_TEXT = {
    "fr": (
        "Ce modèle ne pose pas de diagnostic. Il s'appuie uniquement sur les "
        "données fournies et peut être moins fiable sur des profils peu "
        "représentés dans les données d'entraînement."
    ),
    "en": (
        "This model does not provide a diagnosis. It relies solely on the data "
        "provided and may be less reliable for profiles under-represented in "
        "the training data."
    ),
}

_NEXT_STEPS_TEXT = {
    "fr": (
        "Ces résultats doivent être discutés avec un professionnel de santé "
        "qualifié, qui pourra les replacer dans votre contexte médical."
    ),
    "en": (
        "These results should be discussed with a qualified healthcare "
        "professional, who can interpret them within your medical context."
    ),
}


def _direction_label(lang: str, direction: str) -> str:
    return _DIRECTION_VERB.get((lang, direction), direction)


def _factor_explanation(lang: str, label: str, feat: FeatureContribution) -> str:
    """Pick the most informative phrasing available.

    When ``position_vs_normal`` is known and the contribution has a clear
    direction, use the (position × direction) matrix for richer text. Falls
    back to direction-only phrasing for ``unknown``/``neutral`` cases."""
    if feat.position_vs_normal != "unknown" and feat.direction != "neutral":
        key = (lang, feat.position_vs_normal, feat.direction)
        template = _POSITION_DIRECTION_SENTENCE.get(key)
        if template is not None:
            return template.format(label=label)
    template = _DIRECTION_SENTENCE[(lang, feat.direction)]
    return template.format(label=label)


def _summary(ctx: ReportContext) -> str:
    if ctx.lang == "fr":
        score_part = f" (score : {ctx.score_pct})" if ctx.score_pct else ""
        n = len(ctx.top_features)
        if n == 0:
            return (
                f"Le modèle suggère « {ctx.label} » avec une confiance "
                f"{ctx.confidence_text}{score_part}. Aucun facteur explicatif "
                "détaillé n'est disponible pour cette estimation."
            )
        return (
            f"Le modèle suggère « {ctx.label} » avec une confiance "
            f"{ctx.confidence_text}{score_part}. Cette estimation s'appuie "
            f"principalement sur {n} facteur{'s' if n > 1 else ''} identifié"
            f"{'s' if n > 1 else ''} à partir de vos données."
        )
    score_part = f" (score: {ctx.score_pct})" if ctx.score_pct else ""
    n = len(ctx.top_features)
    if n == 0:
        return (
            f"The model suggests \"{ctx.label}\" with {ctx.confidence_text} "
            f"confidence{score_part}. No detailed explanatory factors are "
            "available for this estimate."
        )
    return (
        f"The model suggests \"{ctx.label}\" with {ctx.confidence_text} "
        f"confidence{score_part}. This estimate is mainly driven by {n} "
        f"factor{'s' if n > 1 else ''} identified in your data."
    )


def _context_text(ctx: ReportContext) -> str:
    base = _CONTEXT_TEXT[ctx.lang]
    if not ctx.model_quality:
        return base
    metrics = ", ".join(f"{m.label}: {m.value}" for m in ctx.model_quality[:3])
    if ctx.lang == "fr":
        return f"{base} Qualité mesurée du modèle: {metrics}."
    return f"{base} Measured model quality: {metrics}."


def _build_key_factors(ctx: ReportContext) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feat in ctx.top_features:
        out.append(
            {
                "label": feat.label,
                "value": feat.value,
                "direction": _direction_label(ctx.lang, feat.direction),
                "explanation": _factor_explanation(ctx.lang, ctx.label, feat),
                "normal_range": feat.normal_range,
            }
        )
    return out


class TemplateClient:
    """Always-available deterministic backend for the report pipeline."""

    name = "template"

    def generate(self, context: ReportContext, *, reinforced: bool = False) -> LLMReport:
        # ``reinforced`` is ignored — the template output is already canonical.
        del reinforced
        return {
            "summary": _summary(context),
            "prediction": {
                "label": context.label,
                "confidence_text": context.confidence_text,
                "score_pct": context.score_pct,
            },
            "key_factors": _build_key_factors(context),
            "context": _context_text(context),
            "limitations": _LIMITATIONS_TEXT[context.lang],
            "next_steps": _NEXT_STEPS_TEXT[context.lang],
            "disclaimer": "PLACEHOLDER",
        }
