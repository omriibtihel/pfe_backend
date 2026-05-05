"""PromptBuilder — turns a ReportContext into LLM-ready system + user prompts.

Hard contracts (plan §9):
- The LLM never sees a raw float. The dataclass already pre-formats values; here
  we serialize only string-safe fields and never include the normalized weight.
- The JSON schema is embedded verbatim in the prompt so backends without
  function-calling support still produce parseable output.
- Two few-shot examples are pinned (positive/high confidence + negative/low).
"""
from __future__ import annotations

import json
from typing import Any

from app.services.reporting.context_builder import (
    FeatureContribution,
    ReportContext,
)

# Safety cap on the serialized user payload. The previous limit of 8000 was
# sized for Ollama small models (7–8 B, ~4 k context). With Groq + Llama 3.3
# 70B (128 k context) the real limit is orders of magnitude larger; 32000
# chars (~8 k tokens) is a practical ceiling that lets the rich few-shot
# examples fit while still guarding against runaway dataset cell values.
_MAX_PROMPT_CHARS = 32_000


# ── JSON output schema (embedded in the prompt) ───────────────────────────────
# Mirrors plan §2.2. Keep field names stable — the validator parses against this.

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "summary",
        "prediction",
        "key_factors",
        "context",
        "limitations",
        "next_steps",
        "disclaimer",
    ],
    "properties": {
        "summary": {"type": "string"},
        "prediction": {
            "type": "object",
            "required": ["label", "confidence_text"],
            "properties": {
                "label": {"type": "string"},
                "confidence_text": {"type": "string"},
                "score_pct": {"type": ["string", "null"]},
            },
        },
        "key_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "value", "direction", "explanation"],
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "direction": {"type": "string"},
                    "explanation": {"type": "string"},
                    "normal_range": {"type": ["string", "null"]},
                },
            },
        },
        "context": {"type": "string"},
        "limitations": {"type": "string"},
        "next_steps": {"type": "string"},
        "disclaimer": {"type": "string", "const": "PLACEHOLDER"},
    },
}


# ── System prompts (FR / EN) ──────────────────────────────────────────────────
# Keep terse; long policy preambles dilute instruction-following on small
# local models (Llama 3.1 8B / Qwen 7B target).

_SYSTEM_FR = """Tu es un rédacteur médical pédagogique. Tu n'es PAS clinicien.

Règles strictes:
- Utilise UNIQUEMENT les valeurs présentes dans le contexte JSON ci-dessous.
- Le rapport doit être spécifique au patient: cite la prédiction, le score et les facteurs fournis.
- Si top_features n'est pas vide, key_factors doit contenir un item pour chaque facteur.
- Interdit les phrases vagues comme « une issue », « des données », « certains facteurs » si une valeur concrète existe.
- N'invente JAMAIS de chiffre, de seuil, de plage ni de nom de variable.
- Utilise le conditionnel: « le modèle suggère », « pourrait indiquer ».
- Niveau de lecture B1, phrases courtes, vocabulaire courant.
- INTERDIT: « vous êtes atteint », « vous avez la maladie », « je diagnostique », « confirme ».
- La langue de sortie est: français.
- Le champ "disclaimer" doit valoir EXACTEMENT la chaîne "PLACEHOLDER".
- Réponds avec un objet JSON STRICT conforme au schéma fourni, sans texte hors-JSON."""

_SYSTEM_EN = """You are a pedagogical medical writer. You are NOT a clinician.

Strict rules:
- Use ONLY the values from the JSON context below.
- The report must be patient-specific: mention the prediction, score, and provided factors.
- If top_features is not empty, key_factors must contain one item for each factor.
- Avoid vague phrases such as "an outcome", "some data", or "certain factors" when a concrete value exists.
- NEVER invent numbers, thresholds, ranges, or variable names.
- Use the conditional: "the model suggests", "may indicate".
- B1 reading level, short sentences, plain vocabulary.
- FORBIDDEN: "you have", "you are diagnosed", "I diagnose", "confirms".
- Output language: English.
- The "disclaimer" field MUST be exactly the string "PLACEHOLDER".
- Reply with a STRICT JSON object matching the schema, no prose outside JSON."""


# ── Few-shot examples (frozen, plan §9) ───────────────────────────────────────
# Two cases: confident positive, uncertain negative. Values inside are
# self-consistent so the model learns to never invent.

_FEWSHOT_FR = [
    # ── Cas 1 : prédiction positive, confiance élevée ─────────────────────────
    # Montre : summary ancré sur le score + seuil, explanation avec training_reference
    # + global_importance, context avec ROC AUC + taille dataset, limitations avec
    # rappel chiffré, next_steps avec marge d'incertitude concrète.
    {
        "context": {
            "label": "risque de diabète suggéré",
            "confidence_text": "élevée",
            "score_pct": "92 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "classe positive : risque de diabète suggéré",
            },
            "model_quality": [
                {"label": "ROC AUC", "value": "86 %",
                 "interpretation": "capacité à séparer les classes"},
                {"label": "Rappel classe positive", "value": "74 %",
                 "interpretation": "capacité à retrouver les cas positifs"},
                {"label": "Seuil de décision", "value": "50 %",
                 "interpretation": "score à partir duquel la classe positive est suggérée"},
            ],
            "dataset_summary": [
                "Jeu de données d'entraînement : 614 lignes",
                "Validation : holdout",
            ],
            "top_features": [
                {
                    "label": "Glycémie à jeun",
                    "value": "126 mg/dL",
                    "direction": "increase",
                    "normal_range": "70–100 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "moyenne entraînement : 98 mg/dL ; intervalle observé : 44–199 mg/dL",
                    "global_importance": "rang global 1/7, importance relative 32 %",
                    "evidence_type": "local_explanation",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "moyenne entraînement : 26 kg/m²",
                    "global_importance": "rang global 2/7, importance relative 18 %",
                    "evidence_type": "local_explanation",
                },
            ],
        },
        "output": {
            "summary": (
                "Le modèle suggère « risque de diabète » avec une confiance élevée (92 %). "
                "Ce score dépasse largement le seuil de décision (50 %), ce qui renforce la fiabilité "
                "de cette estimation. Deux facteurs principaux ont orienté ce résultat."
            ),
            "prediction": {
                "label": "risque de diabète suggéré",
                "confidence_text": "élevée",
                "score_pct": "92 %",
            },
            "key_factors": [
                {
                    "label": "Glycémie à jeun",
                    "value": "126 mg/dL",
                    "direction": "augmente",
                    "explanation": (
                        "Votre glycémie (126 mg/dL) dépasse la plage habituelle (70–100 mg/dL) et "
                        "la moyenne observée dans les données d'entraînement (98 mg/dL). "
                        "C'est le facteur le plus déterminant de cette estimation (rang 1 sur 7)."
                    ),
                    "normal_range": "70–100 mg/dL",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "31 kg/m²",
                    "direction": "augmente",
                    "explanation": (
                        "Un IMC de 31 kg/m² est au-dessus de la plage habituelle (18,5–24,9 kg/m²) "
                        "et de la moyenne d'entraînement (26 kg/m²). "
                        "Ce facteur est le 2e plus influent sur l'estimation."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "Ce modèle a été entraîné sur 614 patients (validation holdout). "
                "Il obtient un ROC AUC de 86 %, ce qui indique une bonne capacité à distinguer "
                "les profils à risque de ceux sans risque. Un score personnel de 92 % est "
                "nettement au-dessus du seuil de 50 % utilisé pour classer les résultats."
            ),
            "limitations": (
                "Le modèle détecte environ 74 % des cas positifs — environ 1 cas sur 4 peut être "
                "manqué. Il ne tient pas compte des antécédents médicaux, des symptômes ni de "
                "l'examen physique."
            ),
            "next_steps": (
                "Un score de 92 % est élevé, mais une marge d'incertitude subsiste. "
                "Ces résultats doivent être confirmés par un professionnel de santé, "
                "notamment par une analyse biologique dédiée."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Cas 2 : prédiction négative, confiance faible ──────────────────────────
    # Montre : summary qui alerte sur la faible confiance, explanation avec
    # training_reference + rang, context qui relie ROC AUC à l'incertitude,
    # limitations avec rappel comme risque spécifique, next_steps actionnable.
    {
        "context": {
            "label": "risque de diabète non suggéré",
            "confidence_text": "faible",
            "score_pct": "55 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "0",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "classe négative : risque de diabète non suggéré",
            },
            "model_quality": [
                {"label": "ROC AUC", "value": "86 %",
                 "interpretation": "capacité à séparer les classes"},
                {"label": "Rappel classe positive", "value": "74 %",
                 "interpretation": "capacité à retrouver les cas positifs"},
                {"label": "Seuil de décision", "value": "50 %",
                 "interpretation": "score à partir duquel la classe positive est suggérée"},
            ],
            "dataset_summary": [
                "Jeu de données d'entraînement : 614 lignes",
                "Validation : holdout",
            ],
            "top_features": [
                {
                    "label": "Âge",
                    "value": "32 ans",
                    "direction": "decrease",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "moyenne entraînement : 45 ans ; intervalle observé : 21–81 ans",
                    "global_importance": "rang global 4/7",
                    "evidence_type": "local_explanation",
                },
            ],
        },
        "output": {
            "summary": (
                "Le modèle suggère « risque de diabète non suggéré », mais avec une confiance "
                "faible (55 %). Ce score est très proche du seuil de décision (50 %), "
                "ce qui signifie que l'estimation est peu fiable."
            ),
            "prediction": {
                "label": "risque de diabète non suggéré",
                "confidence_text": "faible",
                "score_pct": "55 %",
            },
            "key_factors": [
                {
                    "label": "Âge",
                    "value": "32 ans",
                    "direction": "diminue",
                    "explanation": (
                        "Votre âge (32 ans) est inférieur à la moyenne des patients d'entraînement "
                        "(45 ans), ce qui éloigne légèrement l'estimation vers la classe négative. "
                        "Ce facteur a une influence modeste sur ce résultat (rang 4 sur 7)."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "Ce modèle a été entraîné sur 614 patients. Il atteint un ROC AUC de 86 %, "
                "mais avec un score personnel de 55 %, le modèle hésite entre les deux issues. "
                "Cette zone d'incertitude (50–65 %) est celle où les prédictions sont les moins fiables."
            ),
            "limitations": (
                "Un score aussi proche de 50 % rend ce résultat peu fiable. "
                "Par ailleurs, le modèle ne détecte que 74 % des cas positifs — "
                "certains profils à risque peuvent être classés à tort comme négatifs."
            ),
            "next_steps": (
                "La faible confiance de cette estimation rend la consultation médicale "
                "particulièrement recommandée. Ce résultat ne doit pas être utilisé seul "
                "comme garantie d'absence de risque."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
]

_FEWSHOT_EN = [
    # ── Case 1: positive prediction, high confidence ──────────────────────────
    {
        "context": {
            "label": "diabetes risk suggested",
            "confidence_text": "high",
            "score_pct": "92 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "positive class: diabetes risk suggested",
            },
            "model_quality": [
                {"label": "ROC AUC", "value": "86 %",
                 "interpretation": "ability to separate classes"},
                {"label": "Positive-class recall", "value": "74 %",
                 "interpretation": "ability to find positive cases"},
                {"label": "Decision threshold", "value": "50 %",
                 "interpretation": "score from which the positive class is suggested"},
            ],
            "dataset_summary": [
                "Training dataset: 614 rows",
                "Validation: holdout",
            ],
            "top_features": [
                {
                    "label": "Fasting glucose",
                    "value": "126 mg/dL",
                    "direction": "increase",
                    "normal_range": "70–100 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "training mean: 98 mg/dL; observed range: 44–199 mg/dL",
                    "global_importance": "global rank 1/7, relative importance 32 %",
                    "evidence_type": "local_explanation",
                },
                {
                    "label": "Body Mass Index",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "training mean: 26 kg/m²",
                    "global_importance": "global rank 2/7, relative importance 18 %",
                    "evidence_type": "local_explanation",
                },
            ],
        },
        "output": {
            "summary": (
                "The model suggests \"diabetes risk\" with high confidence (92%). "
                "This score is well above the decision threshold (50%), which strengthens "
                "the reliability of this estimate. Two main factors drove this result."
            ),
            "prediction": {
                "label": "diabetes risk suggested",
                "confidence_text": "high",
                "score_pct": "92 %",
            },
            "key_factors": [
                {
                    "label": "Fasting glucose",
                    "value": "126 mg/dL",
                    "direction": "increase",
                    "explanation": (
                        "Your fasting glucose (126 mg/dL) is above the usual range (70–100 mg/dL) "
                        "and above the training average (98 mg/dL). "
                        "This is the most influential factor in this estimate (rank 1 out of 7)."
                    ),
                    "normal_range": "70–100 mg/dL",
                },
                {
                    "label": "Body Mass Index",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "explanation": (
                        "A BMI of 31 kg/m² is above the usual range (18.5–24.9 kg/m²) "
                        "and above the training average (26 kg/m²). "
                        "This factor is the 2nd most influential in the estimate."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "This model was trained on 614 patients (holdout validation). "
                "It achieves a ROC AUC of 86%, indicating a good ability to distinguish "
                "at-risk profiles from lower-risk ones. A personal score of 92% is "
                "well above the 50% decision threshold."
            ),
            "limitations": (
                "The model detects approximately 74% of positive cases — "
                "about 1 in 4 may be missed. It does not account for medical history, "
                "symptoms, or physical examination."
            ),
            "next_steps": (
                "A score of 92% is high, but some uncertainty remains. "
                "These results should be confirmed by a healthcare professional, "
                "ideally through dedicated biological testing."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Case 2: negative prediction, low confidence ───────────────────────────
    {
        "context": {
            "label": "diabetes risk not suggested",
            "confidence_text": "low",
            "score_pct": "55 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "0",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "negative class: diabetes risk not suggested",
            },
            "model_quality": [
                {"label": "ROC AUC", "value": "86 %",
                 "interpretation": "ability to separate classes"},
                {"label": "Positive-class recall", "value": "74 %",
                 "interpretation": "ability to find positive cases"},
                {"label": "Decision threshold", "value": "50 %",
                 "interpretation": "score from which the positive class is suggested"},
            ],
            "dataset_summary": [
                "Training dataset: 614 rows",
                "Validation: holdout",
            ],
            "top_features": [
                {
                    "label": "Age",
                    "value": "32 years",
                    "direction": "decrease",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "training mean: 45 years; observed range: 21–81 years",
                    "global_importance": "global rank 4/7",
                    "evidence_type": "local_explanation",
                },
            ],
        },
        "output": {
            "summary": (
                "The model suggests \"diabetes risk not suggested\", but with low confidence (55%). "
                "This score is very close to the decision threshold (50%), "
                "meaning the estimate is unreliable."
            ),
            "prediction": {
                "label": "diabetes risk not suggested",
                "confidence_text": "low",
                "score_pct": "55 %",
            },
            "key_factors": [
                {
                    "label": "Age",
                    "value": "32 years",
                    "direction": "decrease",
                    "explanation": (
                        "Your age (32 years) is below the training average (45 years), "
                        "slightly pulling the estimate toward the negative class. "
                        "This factor has a modest influence on this result (rank 4 out of 7)."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "This model was trained on 614 patients and achieves a ROC AUC of 86%. "
                "However, with a personal score of 55%, the model is undecided between "
                "the two outcomes. Predictions in the 50–65% range are the least reliable."
            ),
            "limitations": (
                "A score this close to 50% makes this result unreliable. "
                "Furthermore, the model only detects 74% of positive cases — "
                "some at-risk profiles may be incorrectly classified as negative."
            ),
            "next_steps": (
                "The low confidence of this estimate makes medical consultation "
                "particularly important. Do not use this result alone as a guarantee "
                "of no risk."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
]


# ── Builder ───────────────────────────────────────────────────────────────────


class PromptBuilder:
    """Stateless. Produces ``{system, user}`` strings ready for any chat-style
    completion API. Plain text path (no tool calls) keeps the surface portable
    across Ollama, Groq, OpenAI-compatible endpoints, etc."""

    def build(self, context: ReportContext, *, reinforced: bool = False) -> dict[str, str]:
        system = _SYSTEM_FR if context.lang == "fr" else _SYSTEM_EN
        if reinforced:
            system = system + "\n\n" + _reinforcement_clause(context.lang)

        # Put the actual patient context first. Earlier versions placed the
        # long schema/few-shot block first; small models tended to copy generic
        # examples and ignore the row-specific values.
        payload = {
            "context": _serialize_context(context),
            "instruction": _INSTRUCTION_FR if context.lang == "fr" else _INSTRUCTION_EN,
            "schema": _OUTPUT_SCHEMA,
            "few_shot": _FEWSHOT_FR if context.lang == "fr" else _FEWSHOT_EN,
        }
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        # Never truncate the JSON mid-object. If the fixed examples make the
        # payload too long, drop examples first; the context + schema are more
        # important for factual, patient-specific output.
        if len(user) > _MAX_PROMPT_CHARS:
            payload["few_shot"] = []
            user = json.dumps(payload, ensure_ascii=False, indent=2)
        return {"system": system, "user": user}


# ── Helpers ───────────────────────────────────────────────────────────────────

_INSTRUCTION_FR = (
    "Génère un rapport JSON strict, concret et utile pour le contexte ci-dessus. "
    "Le résumé doit reprendre exactement le label, la confiance et le score_pct si disponible. "
    "Utilise model_quality pour nuancer la conclusion: distingue score patient et performance globale. "
    "Utilise dataset_summary seulement pour situer la fiabilité du modèle, sans allonger inutilement. "
    "Pour chaque top_feature, crée un key_factor qui reprend son label, sa valeur, sa plage normale "
    "et explique sobrement ce que cela signifie. Mentionne training_reference et global_importance quand ils existent. "
    "Si evidence_type vaut observed_value, précise que la valeur est observée (pas une contribution LIME). "
    "Ne reproduis aucun chiffre ou mot des exemples dans le rapport final s'il n'apparaît pas dans le contexte."
)
_INSTRUCTION_EN = (
    "Generate a strict JSON report that is concrete and useful for the context above. "
    "The summary must restate the exact label, confidence_text, and score_pct when available. "
    "Use model_quality to qualify the conclusion: distinguish patient score from global model performance. "
    "Use dataset_summary only to frame reliability, without making the report verbose. "
    "For each top_feature, create one key_factor preserving its label, value, normal range, "
    "and a plain explanation. Mention training_reference and global_importance when present. "
    "If evidence_type is observed_value, note the value is observed, not a LIME model contribution. "
    "Do not reuse numbers or words from the examples in the final report if they do not appear in the context."
)


def _reinforcement_clause(lang: str) -> str:
    if lang == "fr":
        return (
            "RAPPEL STRICT: la réponse précédente était invalide. "
            "Réponds UNIQUEMENT avec un objet JSON conforme au schéma, sans "
            "préfixe ni suffixe, sans texte explicatif. Le champ 'disclaimer' "
            "doit être exactement 'PLACEHOLDER'."
        )
    return (
        "STRICT REMINDER: the previous response was invalid. "
        "Reply ONLY with a JSON object matching the schema, no prefix or "
        "suffix, no explanatory text. The 'disclaimer' field must be exactly "
        "'PLACEHOLDER'."
    )


def _serialize_feature(feat: FeatureContribution) -> dict[str, Any]:
    """Project a FeatureContribution to the LLM-visible subset.

    Excludes ``raw_name`` (unsafe column name, already mapped to label) and
    ``weight`` (a normalized float — the order in the list already conveys
    importance, no need to leak the number).

    Includes ``position_vs_normal`` so the LLM can write "abnormally high"
    without re-deriving it from the value + range itself. That derivation is
    deterministic and lives in the context builder."""
    return {
        "label": feat.label,
        "value": feat.value,
        "direction": feat.direction,
        "normal_range": feat.normal_range,
        "position_vs_normal": feat.position_vs_normal,
        "training_reference": feat.training_reference,
        "global_importance": feat.global_importance,
        # "lime_contribution" when the value comes from a local LIME explanation;
        # "observed_value" when no explanation was available and we fell back to
        # raw input data. The distinction comes from the builder, not direction.
        "evidence_type": "observed_value" if feat.is_fallback else "lime_contribution",
    }


def _serialize_context(context: ReportContext) -> dict[str, Any]:
    return {
        "label": context.label,
        "confidence_text": context.confidence_text,
        "score_pct": context.score_pct,
        "task_type": context.task_type,
        "class_context": {
            "raw_label": context.class_context.raw_label,
            "target_name": context.class_context.target_name,
            "positive_class": context.class_context.positive_class,
            "label_meaning": context.class_context.label_meaning,
        },
        "model_quality": [
            {
                "label": m.label,
                "value": m.value,
                "interpretation": m.interpretation,
            }
            for m in context.model_quality[:7]
        ],
        "dataset_summary": context.dataset_summary[:4],
        # Keep the LLM focused. The builder already defaults to top 5, but
        # tests and future callers may pass a wider list.
        "top_features": [_serialize_feature(f) for f in context.top_features[:8]],
    }


__all__ = ["PromptBuilder"]
