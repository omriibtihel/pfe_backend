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

_SYSTEM_FR = """Tu es un rédacteur médical dont les rapports sont lus par des patients. Tu n'es PAS clinicien.

Règles strictes:
- Utilise UNIQUEMENT les valeurs présentes dans le contexte JSON.
- Le rapport doit être spécifique au profil: cite le label, le score_pct et le threshold_distance_pct si disponible.
- Si top_features n'est pas vide, key_factors doit contenir un item pour CHAQUE facteur listé.
- Interdit les phrases vagues: cite TOUJOURS les valeurs concrètes du contexte.
- N'invente JAMAIS de chiffre, de plage ou de nom de variable.
- Utilise le conditionnel: « l'analyse suggère », « pourrait indiquer ».
- Niveau de lecture A2–B1, phrases courtes, vocabulaire du quotidien.
- INTERDIT: « vous êtes atteint », « vous avez la maladie », « je diagnostique », « confirme ».
- La langue de sortie est: français.

Vocabulaire INTERDIT → remplacement obligatoire:
- "ROC AUC" → utilise directement la formulation de model_quality.interpretation (ex: "dans X cas sur 100, l'outil distingue correctement les profils préoccupants")
- "LIME", "contribution", "importance relative", "poids" → jamais mentionner ces termes
- "entraîné sur", "jeu de données", "dataset", "training" → utilise directement dataset_summary (ex: "basé sur X profils médicaux")
- "holdout", "cross-validation", "folds" → utilise directement dataset_summary (ex: "fiabilité vérifiée sur des données distinctes")
- "classe positive / négative" → "résultat préoccupant" / "résultat rassurant"
- "seuil de décision" → "niveau d'alerte" ou "limite de détection"
- "rang global X/Y" → utilise directement global_importance (ex: "facteur le plus influent")
- "moyenne d'entraînement", "valeur d'entraînement" → utilise directement training_reference (ex: "valeur habituelle parmi les profils de référence")

Longueur attendue par section:
- summary: 2–3 phrases courtes. Cite le label, le score_pct, et dit en langage courant si le résultat est fiable ou incertain (en lien avec threshold_distance_pct).
- key_factors.explanation: 2–3 phrases. Compare la valeur à la plage normale si disponible. Cite training_reference en clair. Cite global_importance en clair. Évite tout jargon.
- context: 2–3 phrases. Cite model_quality.interpretation pour la fiabilité globale. Cite dataset_summary pour la base de référence. Si threshold_distance_pct < 20 %, souligne l'incertitude.
- limitations: 1–2 phrases. Exprime le taux de détection en "X cas sur 100" depuis model_quality (recall_pos). Cite ce que l'outil ne voit pas (antécédents, symptômes).
- next_steps: 1–2 phrases concrètes et actionnables. Ne répète pas les limitations.

- Le champ "disclaimer" doit valoir EXACTEMENT la chaîne "PLACEHOLDER".
- Réponds avec un objet JSON STRICT conforme au schéma fourni, sans texte hors-JSON."""

_SYSTEM_EN = """You are a medical writer whose reports are read by patients. You are NOT a clinician.

Strict rules:
- Use ONLY the values from the JSON context below.
- The report must be profile-specific: mention the label, score_pct, and threshold_distance_pct when available.
- If top_features is not empty, key_factors must contain one item for EACH listed factor.
- Avoid vague phrases: ALWAYS cite the concrete values from the context.
- NEVER invent numbers, ranges, or variable names.
- Use the conditional: "the analysis suggests", "may indicate".
- A2–B1 reading level, short sentences, everyday vocabulary.
- FORBIDDEN: "you have", "you are diagnosed", "I diagnose", "confirms".
- Output language: English.

FORBIDDEN vocabulary → mandatory replacement:
- "ROC AUC" → use the model_quality.interpretation wording directly (e.g., "in X out of 100 cases, the tool correctly identifies concerning profiles")
- "LIME", "contribution", "relative importance", "weight" → never mention these terms
- "trained on", "dataset", "training data" → use dataset_summary directly (e.g., "based on X medical profiles")
- "holdout", "cross-validation", "folds" → use dataset_summary directly (e.g., "reliability verified on a separate dataset")
- "positive / negative class" → "concerning result" / "reassuring result"
- "decision threshold" → "alert level" or "detection limit"
- "global rank X/Y" → use global_importance directly (e.g., "most influential factor")
- "training mean", "training reference" → use training_reference directly (e.g., "typical value among reference profiles")

Expected length per section:
- summary: 2–3 short sentences. Cite the label, score_pct, and say in plain language whether the result is reliable or uncertain (using threshold_distance_pct).
- key_factors.explanation: 2–3 sentences. Compare the value to the normal range if available. Cite training_reference in plain language. Cite global_importance in plain language. No jargon.
- context: 2–3 sentences. Use model_quality.interpretation for overall reliability. Use dataset_summary for the reference base. If threshold_distance_pct < 20 %, highlight the uncertainty.
- limitations: 1–2 sentences. Express the detection rate as "X out of 100" from model_quality (recall_pos). State what the tool does not see (history, symptoms).
- next_steps: 1–2 concrete, actionable sentences. Do not repeat the limitations.

- The "disclaimer" field MUST be exactly the string "PLACEHOLDER".
- Reply with a STRICT JSON object matching the schema, no prose outside JSON."""


# ── Few-shot examples (frozen, plan §9) ───────────────────────────────────────
# Two cases: confident positive, uncertain negative. Values inside are
# self-consistent so the model learns to never invent.

_FEWSHOT_FR = [
    # ── Cas 1 : résultat préoccupant, confiance élevée ────────────────────────
    # Montre le vocabulaire patient : pas de "ROC AUC", "holdout", "rang global",
    # "moyenne entraînement" — tous reformulés en langage courant.
    {
        "context": {
            "label": "risque de diabète suggéré",
            "confidence_text": "élevée",
            "score_pct": "92 %",
            "threshold_distance_pct": "42 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "resultat preoccupant : risque de diabete suggere",
            },
            "model_quality": [
                {
                    "label": "Fiabilite globale de l'outil",
                    "value": "86 %",
                    "interpretation": "sur 100 comparaisons, l'outil distingue correctement les profils preoccupants dans 86 cas",
                },
                {
                    "label": "Taux de detection des cas preoccupants",
                    "value": "74 %",
                    "interpretation": "sur 100 profils vraiment preoccupants, l'outil en detecte 74 — les autres peuvent passer inapercus",
                },
                {
                    "label": "Niveau d'alerte",
                    "value": "50 %",
                    "interpretation": "au-dela de ce score, l'outil signale un resultat preoccupant",
                },
            ],
            "dataset_summary": [
                "Base sur l'analyse de 614 profils medicaux",
                "Fiabilite verifiee sur des donnees distinctes",
            ],
            "top_features": [
                {
                    "label": "Glycémie à jeun",
                    "value": "126 mg/dL",
                    "direction": "increase",
                    "normal_range": "70–100 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 98 mg/dL ; ecart observe : de 44 mg/dL a 199 mg/dL",
                    "global_importance": "1er facteur le plus influent parmi 7 (poids : 32 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 26 kg/m² ; ecart observe : de 15 kg/m² a 67 kg/m²",
                    "global_importance": "2e facteur le plus influent parmi 7 (poids : 18 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
        },
        "output": {
            "summary": (
                "L'analyse suggère un risque de diabète avec un score de 92 %, "
                "nettement au-dessus du niveau d'alerte — ce résultat est dans la zone de forte fiabilité de l'outil. "
                "Deux éléments de votre profil ont particulièrement pesé dans ce résultat."
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
                        "Votre glycémie (126 mg/dL) est au-dessus de la plage habituelle (70–100 mg/dL). "
                        "Chez les personnes de référence, la valeur habituelle est de 98 mg/dL. "
                        "C'est l'élément qui a le plus pesé dans ce résultat."
                    ),
                    "normal_range": "70–100 mg/dL",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "31 kg/m²",
                    "direction": "augmente",
                    "explanation": (
                        "Un IMC de 31 kg/m² est au-dessus de la plage habituelle (18,5–24,9 kg/m²). "
                        "Chez les personnes de référence, la valeur habituelle est de 26 kg/m². "
                        "C'est le 2e élément le plus influent dans ce résultat."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "Cet outil a été testé sur 614 profils médicaux, avec une fiabilité vérifiée sur des données distinctes. "
                "Dans 86 cas sur 100, il distingue correctement les profils préoccupants des autres. "
                "Un score de 92 % est à 42 % au-dessus du niveau d'alerte, ce qui rend ce résultat fiable."
            ),
            "limitations": (
                "Sur 100 profils réellement préoccupants, l'outil en détecte environ 74 — certains peuvent passer inaperçus. "
                "Il ne tient pas compte de vos antécédents médicaux, de vos symptômes ni des résultats d'examens cliniques."
            ),
            "next_steps": (
                "Consultez un médecin pour confirmer ce résultat, "
                "notamment avec une prise de sang incluant une glycémie à jeun."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Cas 2 : résultat rassurant, confiance faible ──────────────────────────
    # Montre : summary expliquant la zone d'incertitude en clair,
    # explanation sans jargon, context et limitations en "X sur 100".
    {
        "context": {
            "label": "risque de diabète non suggéré",
            "confidence_text": "faible",
            "score_pct": "55 %",
            "threshold_distance_pct": "5 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "0",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "resultat rassurant : risque de diabete non suggere",
            },
            "model_quality": [
                {
                    "label": "Fiabilite globale de l'outil",
                    "value": "86 %",
                    "interpretation": "sur 100 comparaisons, l'outil distingue correctement les profils preoccupants dans 86 cas",
                },
                {
                    "label": "Taux de detection des cas preoccupants",
                    "value": "74 %",
                    "interpretation": "sur 100 profils vraiment preoccupants, l'outil en detecte 74 — les autres peuvent passer inapercus",
                },
                {
                    "label": "Niveau d'alerte",
                    "value": "50 %",
                    "interpretation": "au-dela de ce score, l'outil signale un resultat preoccupant",
                },
            ],
            "dataset_summary": [
                "Base sur l'analyse de 614 profils medicaux",
                "Fiabilite verifiee sur des donnees distinctes",
            ],
            "top_features": [
                {
                    "label": "Âge",
                    "value": "32 ans",
                    "direction": "decrease",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "valeur habituelle parmi les profils de reference : 45 ans ; ecart observe : de 21 ans a 81 ans",
                    "global_importance": "4e facteur le plus influent parmi 7",
                    "evidence_type": "lime_contribution",
                },
            ],
        },
        "output": {
            "summary": (
                "L'analyse ne suggère pas de risque de diabète, avec un score de 55 %. "
                "Ce score n'est qu'à 5 % du niveau d'alerte — il se trouve dans la zone la plus incertaine de l'outil. "
                "Ce résultat doit être pris avec beaucoup de précautions."
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
                        "Votre âge (32 ans) est inférieur à la valeur habituelle des personnes de référence (45 ans). "
                        "Cette différence a légèrement orienté le résultat vers la zone rassurante. "
                        "C'est le 4e élément le plus influent, mais son poids reste modéré."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "Cet outil a été testé sur 614 profils médicaux. "
                "Dans 86 cas sur 100, il distingue correctement les profils préoccupants. "
                "Mais avec un score de 55 %, à seulement 5 % du niveau d'alerte, ce résultat est dans la zone la moins fiable."
            ),
            "limitations": (
                "Sur 100 profils réellement préoccupants, l'outil en détecte 74 — certains peuvent passer inaperçus, surtout dans cette zone d'incertitude. "
                "Il ne prend pas en compte vos antécédents familiaux, vos symptômes ni votre alimentation."
            ),
            "next_steps": (
                "En raison de la faible fiabilité de ce résultat, une consultation médicale reste recommandée. "
                "Ne pas utiliser ce résultat seul pour écarter tout risque."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
]

_FEWSHOT_EN = [
    # ── Case 1: concerning result, high confidence ────────────────────────────
    # Shows plain-language vocabulary: no "ROC AUC", "holdout", "global rank",
    # "training mean" — all rephrased for a non-technical reader.
    {
        "context": {
            "label": "diabetes risk suggested",
            "confidence_text": "high",
            "score_pct": "92 %",
            "threshold_distance_pct": "42 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "concerning result: diabetes risk suggested",
            },
            "model_quality": [
                {
                    "label": "Overall tool reliability",
                    "value": "86 %",
                    "interpretation": "out of 100 comparisons, the tool correctly identifies concerning profiles in 86 cases",
                },
                {
                    "label": "At-risk case detection rate",
                    "value": "74 %",
                    "interpretation": "out of 100 truly concerning profiles, the tool detects 74 — the rest may go unnoticed",
                },
                {
                    "label": "Alert threshold",
                    "value": "50 %",
                    "interpretation": "above this score, the tool signals a concerning result",
                },
            ],
            "dataset_summary": [
                "Based on the analysis of 614 medical profiles",
                "Reliability verified on a separate held-out dataset",
            ],
            "top_features": [
                {
                    "label": "Fasting glucose",
                    "value": "126 mg/dL",
                    "direction": "increase",
                    "normal_range": "70–100 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 98 mg/dL; observed spread: from 44 mg/dL to 199 mg/dL",
                    "global_importance": "1st most influential factor out of 7 (weight: 32 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Body Mass Index",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 26 kg/m²; observed spread: from 15 kg/m² to 67 kg/m²",
                    "global_importance": "2nd most influential factor out of 7 (weight: 18 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
        },
        "output": {
            "summary": (
                "The analysis suggests a diabetes risk with a score of 92%, "
                "well above the alert level — this result falls in the tool's high-reliability zone. "
                "Two elements of your profile played the biggest role in this result."
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
                        "Your fasting glucose (126 mg/dL) is above the usual range (70–100 mg/dL). "
                        "Among reference profiles, the typical value is 98 mg/dL. "
                        "This is the element that weighed most in this result."
                    ),
                    "normal_range": "70–100 mg/dL",
                },
                {
                    "label": "Body Mass Index",
                    "value": "31 kg/m²",
                    "direction": "increase",
                    "explanation": (
                        "A BMI of 31 kg/m² is above the usual range (18.5–24.9 kg/m²). "
                        "Among reference profiles, the typical value is 26 kg/m². "
                        "This is the 2nd most influential element in this result."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "This tool was tested on 614 medical profiles, with reliability verified on a separate dataset. "
                "In 86 out of 100 cases, it correctly distinguishes concerning profiles from others. "
                "A score of 92% is 42% above the alert level, making this result highly reliable."
            ),
            "limitations": (
                "Out of 100 truly concerning profiles, the tool detects about 74 — some may go unnoticed. "
                "It does not take into account your medical history, symptoms, or the results of clinical examinations."
            ),
            "next_steps": (
                "See a doctor to confirm this result, "
                "ideally with a blood test including a fasting glucose measurement."
            ),
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Case 2: reassuring result, low confidence ─────────────────────────────
    {
        "context": {
            "label": "diabetes risk not suggested",
            "confidence_text": "low",
            "score_pct": "55 %",
            "threshold_distance_pct": "5 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "0",
                "target_name": "Outcome",
                "positive_class": "1",
                "label_meaning": "reassuring result: diabetes risk not suggested",
            },
            "model_quality": [
                {
                    "label": "Overall tool reliability",
                    "value": "86 %",
                    "interpretation": "out of 100 comparisons, the tool correctly identifies concerning profiles in 86 cases",
                },
                {
                    "label": "At-risk case detection rate",
                    "value": "74 %",
                    "interpretation": "out of 100 truly concerning profiles, the tool detects 74 — the rest may go unnoticed",
                },
                {
                    "label": "Alert threshold",
                    "value": "50 %",
                    "interpretation": "above this score, the tool signals a concerning result",
                },
            ],
            "dataset_summary": [
                "Based on the analysis of 614 medical profiles",
                "Reliability verified on a separate held-out dataset",
            ],
            "top_features": [
                {
                    "label": "Age",
                    "value": "32 years",
                    "direction": "decrease",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "typical value among reference profiles: 45 years; observed spread: from 21 years to 81 years",
                    "global_importance": "4th most influential factor out of 7",
                    "evidence_type": "lime_contribution",
                },
            ],
        },
        "output": {
            "summary": (
                "The analysis does not suggest a diabetes risk, with a score of 55%. "
                "This score is only 5% from the alert level — it falls in the tool's most uncertain zone. "
                "This result should be treated with great caution."
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
                        "Your age (32 years) is below the typical value among reference profiles (45 years). "
                        "This difference slightly pushed the result toward the reassuring side. "
                        "It is the 4th most influential element, but its weight remains moderate."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "This tool was tested on 614 medical profiles. "
                "In 86 out of 100 cases, it correctly distinguishes concerning profiles. "
                "But with a score of 55% — only 5% from the alert level — this result is in the tool's least reliable zone."
            ),
            "limitations": (
                "Out of 100 truly concerning profiles, the tool detects 74 — some may go unnoticed, especially in this uncertain zone. "
                "It does not take into account your family history, symptoms, or lifestyle habits."
            ),
            "next_steps": (
                "Given the low reliability of this result, a medical consultation is still recommended. "
                "Do not use this result alone to rule out any risk."
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
    "Génère un rapport JSON en français courant, lisible par toute personne sans formation médicale.\n"
    "summary: cite le label et le score_pct. Si threshold_distance_pct est fourni, exprime en clair si le résultat est fiable "
    "(ex: 'ce score est nettement au-dessus du niveau d'alerte') ou incertain ('ce score est très proche du niveau d'alerte'). Pas de jargon.\n"
    "key_factors: pour chaque top_feature, cite sa valeur et normal_range. "
    "Reformule training_reference en langage naturel ('chez les personnes de référence, la valeur habituelle est…'). "
    "Reformule global_importance naturellement ('c'est le facteur qui a le plus pesé dans ce résultat'). "
    "Si position_vs_normal=above, dis que la valeur est 'au-dessus de la normale'. "
    "Si below, 'en dessous de la normale'. Si within, 'dans les valeurs habituelles'.\n"
    "context: reformule dataset_summary (ex: 'cet outil a été testé sur X profils médicaux'). "
    "Reformule model_quality.interpretation pour la fiabilité globale en clair ('dans X cas sur 100, il distingue correctement les profils préoccupants'). "
    "Si threshold_distance_pct < 20 %, explique que le résultat est dans la zone la moins fiable.\n"
    "limitations: reformule recall_pos depuis model_quality: 'sur 100 profils réellement préoccupants, l'outil en détecte environ X'. "
    "Cite 1–2 éléments que l'outil ne prend pas en compte (antécédents, symptômes, examens cliniques).\n"
    "next_steps: 1–2 phrases concrètes ('consultez un médecin', 'demandez un bilan…'). Pas de répétition des limitations.\n"
    "Ne reproduis aucun chiffre ou mot des exemples si absent du contexte."
)
_INSTRUCTION_EN = (
    "Generate a JSON report in plain English, readable by anyone without medical training.\n"
    "summary: cite the label and score_pct. If threshold_distance_pct is provided, express clearly whether the result is reliable "
    "('this score is well above the alert level') or uncertain ('this score is very close to the alert level'). No jargon.\n"
    "key_factors: for each top_feature, cite its value and normal_range. "
    "Rephrase training_reference naturally ('among reference profiles, the typical value is…'). "
    "Rephrase global_importance naturally ('this is the factor that weighed most in this result'). "
    "If position_vs_normal=above, say the value is 'above the normal range'. "
    "If below, 'below the normal range'. If within, 'within the usual range'.\n"
    "context: rephrase dataset_summary (e.g., 'this tool was tested on X medical profiles'). "
    "Rephrase model_quality.interpretation for overall reliability ('in X out of 100 cases, it correctly identifies concerning profiles'). "
    "If threshold_distance_pct < 20 %, explain the result is in the least reliable zone.\n"
    "limitations: rephrase recall_pos from model_quality: 'out of 100 truly concerning profiles, the tool detects about X'. "
    "Cite 1–2 things the tool does not capture (history, symptoms, clinical examination).\n"
    "next_steps: 1–2 concrete sentences ('consult a doctor', 'ask for a blood test…'). Do not repeat the limitations.\n"
    "Do not reuse any number or word from the examples if absent from the context."
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
    out: dict[str, Any] = {
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
        "dataset_summary": context.dataset_summary[:5],
        # Keep the LLM focused. The builder already defaults to top 5, but
        # tests and future callers may pass a wider list.
        "top_features": [_serialize_feature(f) for f in context.top_features[:8]],
    }

    # Pre-compute threshold distance so the LLM can reference it in the summary
    # without having to derive it from floating-point arithmetic.
    if context.score is not None:
        threshold = context.threshold_value if context.threshold_value is not None else 0.5
        distance_pct = round(abs(float(context.score) - float(threshold)) * 100)
        out["threshold_distance_pct"] = f"{distance_pct} %"

    return out


__all__ = ["PromptBuilder"]
