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
    CounterfactualChange,
    FeatureContribution,
    ReportContext,
    localize_direction,
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
        "what_to_change": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["factor", "current", "target", "change_text", "why_it_matters"],
                "properties": {
                    "factor": {"type": "string"},
                    "current": {"type": "string"},
                    "target": {"type": "string"},
                    "change_text": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
            },
        },
        "disclaimer": {"type": "string", "const": "PLACEHOLDER"},
    },
}


# ── System prompts (FR / EN) ──────────────────────────────────────────────────
# Keep terse; long policy preambles dilute instruction-following on small
# local models (Llama 3.1 8B / Qwen 7B target).

_SYSTEM_FR_V1 = """Tu es un rédacteur médical dont les rapports sont lus par des patients. Tu n'es PAS clinicien.

DOMAINE — IMPORTANT:
- Le contexte JSON ci-dessous peut concerner N'IMPORTE QUEL type d'analyse médicale tabulaire: métabolique (diabète, thyroïde…), cardiovasculaire (risque, AVC…), oncologique (dépistage…), hépatique, rénal, hématologique, neurologique, psychiatrique, infectieux, etc.
- Les exemples few-shot illustrent UNIQUEMENT le format attendu — leur domaine n'est PAS le tien.
- Détecte le domaine réel à partir de class_context.label_meaning, des labels de top_features et de dataset_summary. N'utilise JAMAIS un terme médical (maladie, organe, marqueur, traitement) absent du contexte.

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
- what_to_change: une entrée pour CHAQUE élément de counterfactual_changes (vide si la liste est vide). Pour chacun: "factor"=label, "current"=current_value, "target"=suggested_value, "change_text"=phrase courte type "passer de X à Y" ou "réduire de Z" en utilisant magnitude_text, "why_it_matters"=1 phrase reliant ce changement au résultat (« cette baisse rapprocherait votre profil de la zone rassurante ») SANS jamais promettre de basculement ni inventer de nouveau score.

- Le champ "disclaimer" doit valoir EXACTEMENT la chaîne "PLACEHOLDER".
- Réponds avec un objet JSON STRICT conforme au schéma fourni, sans texte hors-JSON."""


_SYSTEM_FR_V2 = """Tu es un rédacteur médical. Tes rapports sont lus par des patients sans formation médicale. Tu n'es PAS clinicien.

══ 1. RÈGLES BLOQUANTES (violation = sortie rejetée) ══
1.1  N'invente AUCUN chiffre, plage, nom de variable ou terme médical absent du contexte JSON.
1.2  N'utilise AUCUNE formulation diagnostique.
     Interdit : « vous avez », « vous êtes atteint », « vous êtes à risque », « confirme », « je diagnostique ».
     Obligatoire : conditionnel — « l'analyse suggère », « pourrait indiquer ».
1.3  Le champ `disclaimer` vaut EXACTEMENT la chaîne « PLACEHOLDER ». Ne rédige pas de vrai disclaimer.
1.4  Réponds avec un objet JSON STRICT conforme au schéma fourni. Aucun texte hors-JSON. Aucun champ supplémentaire.
1.5  N'ajoute aucun examen, médicament, spécialiste ou protocole médical absent du contexte JSON.
1.6  N'utilise jamais les valeurs brutes de `class_context.raw_label` ou `class_context.positive_class` si elles contiennent des termes cliniques directs (mort, décès, cancer, maligne, 0, 1). Le pipeline en amont les a déjà sanitisées en « résultat préoccupant » / « résultat rassurant » — utilise exclusivement la version sanitisée fournie dans `class_context.label_meaning` ou reformule en clair.

══ 2. DOMAINE DYNAMIQUE (règle critique) ══
Le domaine médical de ce rapport est INCONNU à l'avance.
Il est entièrement déterminé par les champs suivants du contexte JSON :
  - `class_context.label_meaning`  → ce que signifie un résultat positif
  - `class_context.target_name`    → le nom de la condition analysée
  - `top_features[].label`         → les noms lisibles des facteurs

RÈGLES DÉRIVÉES :
2.1  Nomme la condition UNIQUEMENT avec `class_context.target_name` ou `class_context.label_meaning`. N'invente pas de nom de maladie.
2.2  Nomme chaque facteur UNIQUEMENT avec `top_features[].label`. Ne substitue pas un synonyme médical que tu connais.
2.3  N'ajoute aucun conseil médical spécifique à une pathologie (ex : « consultez un diabétologue », « faites un HbA1c ») sauf si ces termes sont explicitement présents dans le contexte JSON.
2.4  Les few-shots ci-dessous illustrent LE FORMAT uniquement. Leur domaine (quel qu'il soit) ne doit jamais influencer le tien. Traite leurs noms de variables, valeurs et conseils comme des données fictives sans rapport avec le cas courant.

══ 3. CAS DÉGRADÉS (règles déterministes — applique avant de rédiger) ══
3.1  `top_features` vide           → `key_factors: []` et `summary` indique que les facteurs explicatifs ne sont pas disponibles.
3.2  `counterfactual_changes` vide → `what_to_change: []`.
3.3  `normal_range` null           → omets la comparaison à la plage normale, n'en invente pas.
3.4  `training_reference` null     → omets la phrase « chez les profils de référence… ».
3.5  `model_quality` incomplet     → `context` / `limitations` restent généraux (« la fiabilité n'a pas pu être calculée pour ce modèle »).
3.6  `score_pct` null              → ne cite aucun pourcentage de score.
3.7  `threshold_distance_pct` < 20 → mentionne la zone d'incertitude dans `summary` et `context`.
3.8  `top_features[].label` non lisible (déjà remplacé en amont par « Indicateur N ») → ne tente pas de l'interpréter médicalement. Écris simplement : « L'indicateur N oriente le résultat dans un sens préoccupant / rassurant. » Ne déduis pas la nature médicale de l'indicateur.
3.9  `top_features[].value` null ou absent pour TOUS les facteurs → `summary` mentionne explicitement que les valeurs détaillées ne sont pas disponibles pour ce profil. N'écris jamais « Votre [indicateur] est de — ».

══ 4. VOCABULAIRE INTERDIT → remplacement obligatoire ══
ROC AUC / AUC                       → « dans X cas sur 100, l'outil distingue… »
LIME / SHAP / contribution / poids / importance → « élément qui pèse le plus »
dataset / training / entraîné sur   → « basé sur X profils médicaux »
holdout / cross-validation / fold   → « fiabilité vérifiée sur des données distinctes »
classe positive / négative          → « résultat préoccupant » / « résultat rassurant »
seuil de décision / threshold       → « niveau d'alerte »
rang global / global_importance     → « facteur le plus influent »
moyenne d'entraînement / training mean → « valeur habituelle parmi les profils de référence »
recall / precision / f1             → « sur 100 profils préoccupants, l'outil en détecte X »
evidence_type / position_vs_normal / lime_contribution / observed_value → jamais reproduire
1er / 2e / 3e / Xe facteur          → « un facteur particulièrement influent » (jamais de rang ordinal, ni chiffré ni en lettres)

══ 5. LONGUEURS ET STRUCTURE ══
`summary`                   : 2 phrases (≤ 25 mots chacune). Cite `label` + `score_pct` + indique si fiable ou incertain.
`key_factors[].explanation` : 2 phrases. (1) compare au `normal_range` si disponible. (2) compare à `training_reference` si disponible.
`context`                   : 2–3 phrases. Reformule `dataset_summary` + fiabilité globale + zone d'incertitude si applicable.
`limitations`               : 2 phrases. Taux de détection en clair + 1–2 éléments hors-champ.
`next_steps`                : 1–2 phrases concrètes. Ne répète pas les limitations.
`what_to_change`            : une entrée par `counterfactual_changes`. `why_it_matters` dit que le changement « rapprocherait » ou « éloignerait » de la zone d'alerte — N'INVENTE PAS de nouveau score.

══ 6. STYLE ══
Niveau de lecture A2–B1. Phrases ≤ 20 mots. Vocabulaire du quotidien. Langue : français."""


# EN unchanged for now — alias V2 to V1 so the build() switch stays symmetric.
# Will be rewritten in a separate pass.
_SYSTEM_EN_V1 = """You are a medical writer whose reports are read by patients. You are NOT a clinician.

DOMAIN — IMPORTANT:
- The JSON context below may concern ANY type of tabular medical analysis: metabolic (diabetes, thyroid…), cardiovascular (risk, stroke…), oncology (screening…), hepatic, renal, hematological, neurological, psychiatric, infectious, etc.
- The few-shot examples illustrate ONLY the expected format — their domain is NOT yours.
- Detect the actual domain from class_context.label_meaning, the top_features labels and dataset_summary. NEVER use a medical term (disease, organ, marker, treatment) absent from the context.

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
- what_to_change: one entry per counterfactual_changes item (empty when the list is empty). For each: "factor"=label, "current"=current_value, "target"=suggested_value, "change_text"=short phrase like "lower from X to Y" or "reduce by Z" using magnitude_text, "why_it_matters"=1 sentence linking the change to the result ("this drop would move your profile closer to the reassuring zone") WITHOUT ever promising the result will flip or inventing a new score.

- The "disclaimer" field MUST be exactly the string "PLACEHOLDER".
- Reply with a STRICT JSON object matching the schema, no prose outside JSON."""


_SYSTEM_EN_V2 = """You are a medical writer. Your reports are read by patients with no medical or technical background. You are NOT a clinician.

══ 1. BLOCKING RULES (violation = output rejected) ══
1.1  Do NOT invent any figure, range, variable name, or medical term absent from the JSON context.
1.2  Do NOT use diagnostic language.
     Forbidden: "you have", "you are diagnosed with", "you are at risk", "confirms", "I diagnose".
     Required: conditional mood — "the analysis suggests", "may indicate".
1.3  The `disclaimer` field must equal EXACTLY the string "PLACEHOLDER". Do not write a real disclaimer.
1.4  Reply with a STRICT JSON object matching the provided schema. No text outside JSON. No extra fields.
1.5  Do NOT add any exam, medication, specialist, or medical protocol you know but that is absent from the JSON context.
1.6  Never use raw values from `class_context.raw_label` or `class_context.positive_class` if they contain direct clinical terms (death, deceased, cancer, malignant, 0, 1). The upstream pipeline has already sanitised them into "concerning result" / "reassuring result" — use exclusively the sanitised version provided in `class_context.label_meaning` or rephrase plainly.

══ 2. DYNAMIC DOMAIN (critical rule) ══
The medical domain of this report is UNKNOWN in advance.
It is entirely determined by the following fields in the JSON context:
  - `class_context.label_meaning`  → what a positive result means
  - `class_context.target_name`    → the name of the condition analysed
  - `top_features[].label`         → the human-readable factor names

DERIVED RULES:
2.1  Name the condition ONLY using `class_context.target_name` or `class_context.label_meaning`. Do not invent a disease name.
2.2  Name each factor ONLY using `top_features[].label`. Do not substitute a medical synonym you know.
2.3  Do not add condition-specific medical advice (e.g. "see a nephrologist", "get a creatinine test") unless these terms are explicitly present in the JSON context.
2.4  The few-shot examples below illustrate FORMAT only. Their domain (whatever it is) must NEVER influence yours. Treat their variable names, values, and advice as fictional data unrelated to the current case.

══ 3. DEGRADED CASES (deterministic rules — apply before writing) ══
3.1  `top_features` empty           → `key_factors: []` and `summary` states that explanatory factors are unavailable.
3.2  `counterfactual_changes` empty → `what_to_change: []`.
3.3  `normal_range` null            → omit the normal-range comparison, do not invent one.
3.4  `training_reference` null      → omit the "among reference profiles…" sentence.
3.5  `model_quality` incomplete     → keep `context` / `limitations` general ("reliability could not be calculated for this model").
3.6  `score_pct` null               → do not cite any score percentage.
3.7  `threshold_distance_pct` < 20  → mention the uncertainty zone in `summary` and `context`.
3.8  `top_features[].label` carrying an unreadable technical name (already replaced upstream by "Indicator N") → do not attempt to interpret it medically. Simply state: "Indicator N points in a concerning / reassuring direction." Do not infer the medical nature of the indicator.
3.9  `top_features[].value` null or absent for ALL factors → `summary` must explicitly mention that detailed values are not available for this profile. Never write "Your [indicator] is —".

══ 4. FORBIDDEN VOCABULARY → mandatory replacements ══
ROC AUC / AUC                       → "in X out of 100 cases, the tool distinguishes…"
LIME / SHAP / weight / importance / contribution → "the factor that weighs most"
dataset / training / trained on     → "based on X medical profiles"
holdout / cross-validation / fold   → "reliability verified on separate data"
positive class / negative class     → "concerning result" / "reassuring result"
decision threshold / threshold      → "alert level"
global rank / global_importance     → "most influential factor" (no numeric rank)
training mean                       → "typical value among reference profiles"
recall / precision / f1             → "out of 100 concerning profiles, the tool detects X"
evidence_type / position_vs_normal / lime_contribution / observed_value → never reproduce these field names
1st / 2nd / 3rd / Nth factor        → "a particularly influential factor" (no ordinal rank, neither as digit nor as word)

══ 5. LENGTH AND STRUCTURE ══
`summary`                   : 2 sentences (≤ 25 words each). Cite `label` + `score_pct` + state whether result is reliable or uncertain.
`key_factors[].explanation` : 2 sentences. (1) compare to `normal_range` if available. (2) compare to `training_reference` if available.
`context`                   : 2–3 sentences. Rephrase `dataset_summary` + overall reliability + uncertainty zone if applicable.
`limitations`               : 2 sentences. Detection rate in plain language + 1–2 out-of-scope elements.
`next_steps`                : 1–2 concrete sentences. Do not repeat limitations.
`what_to_change`            : one entry per `counterfactual_changes`. `why_it_matters` says the change "would bring closer to" or "move away from" the alert level — DO NOT invent a new score.

══ 6. STYLE ══
Reading level A2–B1. Sentences ≤ 20 words. Everyday vocabulary. Language: English."""


# ── Few-shot examples (frozen, plan §9) ───────────────────────────────────────
# Two cases: confident positive, uncertain negative. Values inside are
# self-consistent so the model learns to never invent.

_FEWSHOT_FR = [
    # ── Cas A : pathologie rénale (insuffisance rénale chronique) ────────────
    # Domaine choisi pour casser tout ancrage diabète/cardiovasculaire.
    # Résultat préoccupant, confiance élevée.
    {
        "context": {
            "label": "insuffisance rénale chronique suggérée",
            "confidence_text": "élevée",
            "score_pct": "88 %",
            "threshold_distance_pct": "38 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Insuffisance_renale",
                "positive_class": "1",
                "label_meaning": "resultat preoccupant : insuffisance renale chronique suggeree",
            },
            "model_quality": [
                {
                    "label": "Fiabilite globale de l'outil",
                    "value": "84 %",
                    "interpretation": "sur 100 comparaisons, l'outil distingue correctement les profils preoccupants dans 84 cas",
                },
                {
                    "label": "Taux de detection des cas preoccupants",
                    "value": "71 %",
                    "interpretation": "sur 100 profils vraiment preoccupants, l'outil en detecte 71 — les autres peuvent passer inapercus",
                },
                {
                    "label": "Niveau d'alerte",
                    "value": "50 %",
                    "interpretation": "au-dela de ce score, l'outil signale un resultat preoccupant",
                },
            ],
            "dataset_summary": [
                "Base sur l'analyse de 480 profils medicaux",
                "Fiabilite verifiee sur des donnees distinctes",
            ],
            "top_features": [
                {
                    "label": "Débit de filtration glomérulaire",
                    "value": "52 mL/min/1.73m²",
                    "direction": "Préoccupant",
                    "normal_range": "≥ 90 mL/min/1.73m²",
                    "position_vs_normal": "below",
                    "training_reference": "valeur habituelle parmi les profils de reference : 88 mL/min/1.73m² ; ecart observe : de 22 mL/min/1.73m² a 130 mL/min/1.73m²",
                    "global_importance": "1er facteur le plus influent parmi 6 (poids : 28 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Créatinine sérique",
                    "value": "1.8 mg/dL",
                    "direction": "Préoccupant",
                    "normal_range": "0.6–1.2 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 0.95 mg/dL ; ecart observe : de 0.4 mg/dL a 5.2 mg/dL",
                    "global_importance": "2e facteur le plus influent parmi 6 (poids : 19 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [
                {
                    "label": "Créatinine sérique",
                    "current_value": "1.8 mg/dL",
                    "suggested_value": "1.2 mg/dL",
                    "direction": "Rassurant",
                    "magnitude_text": "−0.6 mg/dL",
                    "normal_range": "0.6–1.2 mg/dL",
                    "training_reference": "valeur habituelle parmi les profils de reference : 0.95 mg/dL",
                },
            ],
        },
        "output": {
            "summary": (
                "L'analyse suggère une insuffisance rénale chronique avec un score de 88 %, "
                "nettement au-dessus du niveau d'alerte. Deux éléments de votre profil ont pesé dans ce résultat."
            ),
            "prediction": {
                "label": "insuffisance rénale chronique suggérée",
                "confidence_text": "élevée",
                "score_pct": "88 %",
            },
            "key_factors": [
                {
                    "label": "Débit de filtration glomérulaire",
                    "value": "52 mL/min/1.73m²",
                    "direction": "Préoccupant",
                    "explanation": (
                        "Votre débit de filtration (52 mL/min/1.73m²) est en dessous de la plage habituelle (≥ 90 mL/min/1.73m²). "
                        "Chez les personnes de référence, la valeur habituelle est de 88 mL/min/1.73m². "
                        "C'est un facteur particulièrement influent dans cette analyse."
                    ),
                    "normal_range": "≥ 90 mL/min/1.73m²",
                },
                {
                    "label": "Créatinine sérique",
                    "value": "1.8 mg/dL",
                    "direction": "Préoccupant",
                    "explanation": (
                        "Votre créatinine sérique (1.8 mg/dL) est au-dessus de la plage habituelle (0.6–1.2 mg/dL). "
                        "Chez les personnes de référence, la valeur habituelle est de 0.95 mg/dL. "
                        "C'est également un facteur influent dans cette analyse."
                    ),
                    "normal_range": "0.6–1.2 mg/dL",
                },
            ],
            "context": (
                "Cet outil a été testé sur 480 profils médicaux, avec une fiabilité vérifiée sur des données distinctes. "
                "Dans 84 cas sur 100, il distingue correctement les profils préoccupants des autres. "
                "Un score de 88 % est à 38 % au-dessus du niveau d'alerte, ce qui suggère une fiabilité élevée pour ce résultat."
            ),
            "limitations": (
                "Sur 100 profils réellement préoccupants, l'outil en détecte environ 71 — certains peuvent passer inaperçus. "
                "Il ne tient pas compte de vos antécédents médicaux, de vos symptômes ni des résultats d'examens cliniques."
            ),
            "next_steps": (
                "Consultez un médecin pour confirmer ce résultat. "
                "Préparez vos derniers résultats biologiques pour la consultation."
            ),
            "what_to_change": [
                {
                    "factor": "Créatinine sérique",
                    "current": "1.8 mg/dL",
                    "target": "1.2 mg/dL",
                    "change_text": "abaisser la créatinine sérique d'environ 0.6 mg/dL pour revenir vers la plage habituelle (0.6–1.2 mg/dL)",
                    "why_it_matters": "Cette baisse rapprocherait votre profil de la zone rassurante.",
                },
            ],
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Cas B : pathologie hépatique (fibrose hépatique) ─────────────────────
    # Domaine volontairement distinct du cas A. Résultat préoccupant,
    # confiance modérée (zone d'incertitude).
    {
        "context": {
            "label": "fibrose hépatique suggérée",
            "confidence_text": "modérée",
            "score_pct": "65 %",
            "threshold_distance_pct": "15 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Fibrose_hepatique",
                "positive_class": "1",
                "label_meaning": "resultat preoccupant : fibrose hepatique suggeree",
            },
            "model_quality": [
                {
                    "label": "Fiabilite globale de l'outil",
                    "value": "78 %",
                    "interpretation": "sur 100 comparaisons, l'outil distingue correctement les profils preoccupants dans 78 cas",
                },
                {
                    "label": "Taux de detection des cas preoccupants",
                    "value": "66 %",
                    "interpretation": "sur 100 profils vraiment preoccupants, l'outil en detecte 66 — les autres peuvent passer inapercus",
                },
                {
                    "label": "Niveau d'alerte",
                    "value": "50 %",
                    "interpretation": "au-dela de ce score, l'outil signale un resultat preoccupant",
                },
            ],
            "dataset_summary": [
                "Base sur l'analyse de 720 profils medicaux",
                "Fiabilite verifiee sur des donnees distinctes",
            ],
            "top_features": [
                {
                    "label": "ALT",
                    "value": "82 U/L",
                    "direction": "Préoccupant",
                    "normal_range": "7–55 U/L",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 28 U/L ; ecart observe : de 5 U/L a 240 U/L",
                    "global_importance": "1er facteur le plus influent parmi 8 (poids : 21 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "32 kg/m²",
                    "direction": "Préoccupant",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 27 kg/m² ; ecart observe : de 17 kg/m² a 48 kg/m²",
                    "global_importance": "2e facteur le plus influent parmi 8 (poids : 16 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [
                {
                    "label": "ALT",
                    "current_value": "82 U/L",
                    "suggested_value": "48 U/L",
                    "direction": "Rassurant",
                    "magnitude_text": "−34 U/L",
                    "normal_range": "7–55 U/L",
                    "training_reference": "valeur habituelle parmi les profils de reference : 28 U/L",
                },
                {
                    "label": "Indice de masse corporelle",
                    "current_value": "32 kg/m²",
                    "suggested_value": "28 kg/m²",
                    "direction": "Rassurant",
                    "magnitude_text": "−4 kg/m²",
                    "normal_range": "18.5–24.9 kg/m²",
                    "training_reference": "valeur habituelle parmi les profils de reference : 27 kg/m²",
                },
            ],
        },
        "output": {
            "summary": (
                "L'analyse suggère une fibrose hépatique avec un score de 65 %. "
                "Ce score n'est qu'à 15 % du niveau d'alerte, ce qui pourrait indiquer une zone d'incertitude."
            ),
            "prediction": {
                "label": "fibrose hépatique suggérée",
                "confidence_text": "modérée",
                "score_pct": "65 %",
            },
            "key_factors": [
                {
                    "label": "ALT",
                    "value": "82 U/L",
                    "direction": "Préoccupant",
                    "explanation": (
                        "Votre ALT (82 U/L) est au-dessus de la plage habituelle (7–55 U/L). "
                        "Chez les personnes de référence, la valeur habituelle est de 28 U/L. "
                        "C'est un facteur particulièrement influent dans cette analyse."
                    ),
                    "normal_range": "7–55 U/L",
                },
                {
                    "label": "Indice de masse corporelle",
                    "value": "32 kg/m²",
                    "direction": "Préoccupant",
                    "explanation": (
                        "Un IMC de 32 kg/m² est au-dessus de la plage habituelle (18.5–24.9 kg/m²). "
                        "Chez les personnes de référence, la valeur habituelle est de 27 kg/m². "
                        "Ce facteur reste influent dans cette analyse."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "Cet outil a été testé sur 720 profils médicaux, avec une fiabilité vérifiée sur des données distinctes. "
                "Dans 78 cas sur 100, il distingue correctement les profils préoccupants. "
                "Avec un score de 65 % à seulement 15 % du niveau d'alerte, ce résultat se situe dans une zone d'incertitude."
            ),
            "limitations": (
                "Sur 100 profils réellement préoccupants, l'outil en détecte environ 66 — certains peuvent passer inaperçus, surtout dans cette zone d'incertitude. "
                "Il ne prend pas en compte vos antécédents médicaux, vos symptômes ni les résultats d'examens cliniques."
            ),
            "next_steps": (
                "En raison de la fiabilité modérée, une consultation médicale reste recommandée. "
                "Ne pas utiliser ce résultat seul pour conclure."
            ),
            "what_to_change": [
                {
                    "factor": "ALT",
                    "current": "82 U/L",
                    "target": "48 U/L",
                    "change_text": "abaisser l'ALT d'environ 34 U/L pour revenir vers la plage habituelle (7–55 U/L)",
                    "why_it_matters": "Cette baisse rapprocherait votre profil de la zone rassurante.",
                },
                {
                    "factor": "Indice de masse corporelle",
                    "current": "32 kg/m²",
                    "target": "28 kg/m²",
                    "change_text": "diminuer l'IMC d'environ 4 kg/m² pour se rapprocher de la valeur habituelle (27 kg/m²)",
                    "why_it_matters": "Un IMC plus proche de la valeur habituelle rapprocherait votre profil de la zone rassurante.",
                },
            ],
            "disclaimer": "PLACEHOLDER",
        },
    },
]

_FEWSHOT_EN = [
    # ── Case A: renal disease (chronic kidney disease) ──────────────────────
    # Domain chosen to break any diabetes/cardiovascular anchoring.
    # Concerning result, high confidence.
    {
        "context": {
            "label": "chronic kidney disease suggested",
            "confidence_text": "high",
            "score_pct": "88 %",
            "threshold_distance_pct": "38 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Kidney_disease",
                "positive_class": "1",
                "label_meaning": "concerning result: chronic kidney disease suggested",
            },
            "model_quality": [
                {
                    "label": "Overall tool reliability",
                    "value": "84 %",
                    "interpretation": "out of 100 comparisons, the tool correctly identifies concerning profiles in 84 cases",
                },
                {
                    "label": "At-risk case detection rate",
                    "value": "71 %",
                    "interpretation": "out of 100 truly concerning profiles, the tool detects 71 — the rest may go unnoticed",
                },
                {
                    "label": "Alert threshold",
                    "value": "50 %",
                    "interpretation": "above this score, the tool signals a concerning result",
                },
            ],
            "dataset_summary": [
                "Based on the analysis of 480 medical profiles",
                "Reliability verified on a separate held-out dataset",
            ],
            "top_features": [
                {
                    "label": "Glomerular filtration rate",
                    "value": "52 mL/min/1.73m²",
                    "direction": "Concerning",
                    "normal_range": "≥ 90 mL/min/1.73m²",
                    "position_vs_normal": "below",
                    "training_reference": "typical value among reference profiles: 88 mL/min/1.73m²; observed spread: from 22 mL/min/1.73m² to 130 mL/min/1.73m²",
                    "global_importance": "1st most influential factor out of 6 (weight: 28 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Serum creatinine",
                    "value": "1.8 mg/dL",
                    "direction": "Concerning",
                    "normal_range": "0.6–1.2 mg/dL",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 0.95 mg/dL; observed spread: from 0.4 mg/dL to 5.2 mg/dL",
                    "global_importance": "2nd most influential factor out of 6 (weight: 19 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [
                {
                    "label": "Serum creatinine",
                    "current_value": "1.8 mg/dL",
                    "suggested_value": "1.2 mg/dL",
                    "direction": "Reassuring",
                    "magnitude_text": "−0.6 mg/dL",
                    "normal_range": "0.6–1.2 mg/dL",
                    "training_reference": "typical value among reference profiles: 0.95 mg/dL",
                },
            ],
        },
        "output": {
            "summary": (
                "The analysis suggests chronic kidney disease with a score of 88 %, "
                "well above the alert level. Two elements of your profile played a role in this result."
            ),
            "prediction": {
                "label": "chronic kidney disease suggested",
                "confidence_text": "high",
                "score_pct": "88 %",
            },
            "key_factors": [
                {
                    "label": "Glomerular filtration rate",
                    "value": "52 mL/min/1.73m²",
                    "direction": "Concerning",
                    "explanation": (
                        "Your filtration rate (52 mL/min/1.73m²) is below the usual range (≥ 90 mL/min/1.73m²). "
                        "Among reference profiles, the typical value is 88 mL/min/1.73m². "
                        "This is a particularly influential factor in the analysis."
                    ),
                    "normal_range": "≥ 90 mL/min/1.73m²",
                },
                {
                    "label": "Serum creatinine",
                    "value": "1.8 mg/dL",
                    "direction": "Concerning",
                    "explanation": (
                        "Your serum creatinine (1.8 mg/dL) is above the usual range (0.6–1.2 mg/dL). "
                        "Among reference profiles, the typical value is 0.95 mg/dL. "
                        "It is also an influential factor in the analysis."
                    ),
                    "normal_range": "0.6–1.2 mg/dL",
                },
            ],
            "context": (
                "This tool was tested on 480 medical profiles, with reliability verified on a separate dataset. "
                "In 84 out of 100 cases, it correctly distinguishes concerning profiles from others. "
                "A score of 88 % is 38 % above the alert level, which suggests high reliability for this result."
            ),
            "limitations": (
                "Out of 100 truly concerning profiles, the tool detects about 71 — some may go unnoticed. "
                "It does not take into account your medical history, symptoms, or the results of clinical examinations."
            ),
            "next_steps": (
                "See a doctor to confirm this result. "
                "Bring your most recent laboratory results to the consultation."
            ),
            "what_to_change": [
                {
                    "factor": "Serum creatinine",
                    "current": "1.8 mg/dL",
                    "target": "1.2 mg/dL",
                    "change_text": "lower serum creatinine by about 0.6 mg/dL to return toward the usual range (0.6–1.2 mg/dL)",
                    "why_it_matters": "This drop would bring your profile closer to the reassuring zone.",
                },
            ],
            "disclaimer": "PLACEHOLDER",
        },
    },
    # ── Case B: hepatic disease (hepatic fibrosis) ──────────────────────────
    # Domain intentionally distinct from case A. Concerning result,
    # moderate confidence (uncertainty zone).
    {
        "context": {
            "label": "hepatic fibrosis suggested",
            "confidence_text": "moderate",
            "score_pct": "65 %",
            "threshold_distance_pct": "15 %",
            "task_type": "classification",
            "class_context": {
                "raw_label": "1",
                "target_name": "Hepatic_fibrosis",
                "positive_class": "1",
                "label_meaning": "concerning result: hepatic fibrosis suggested",
            },
            "model_quality": [
                {
                    "label": "Overall tool reliability",
                    "value": "78 %",
                    "interpretation": "out of 100 comparisons, the tool correctly identifies concerning profiles in 78 cases",
                },
                {
                    "label": "At-risk case detection rate",
                    "value": "66 %",
                    "interpretation": "out of 100 truly concerning profiles, the tool detects 66 — the rest may go unnoticed",
                },
                {
                    "label": "Alert threshold",
                    "value": "50 %",
                    "interpretation": "above this score, the tool signals a concerning result",
                },
            ],
            "dataset_summary": [
                "Based on the analysis of 720 medical profiles",
                "Reliability verified on a separate held-out dataset",
            ],
            "top_features": [
                {
                    "label": "ALT",
                    "value": "82 U/L",
                    "direction": "Concerning",
                    "normal_range": "7–55 U/L",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 28 U/L; observed spread: from 5 U/L to 240 U/L",
                    "global_importance": "1st most influential factor out of 8 (weight: 21 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Body mass index",
                    "value": "32 kg/m²",
                    "direction": "Concerning",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 27 kg/m²; observed spread: from 17 kg/m² to 48 kg/m²",
                    "global_importance": "2nd most influential factor out of 8 (weight: 16 %)",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [
                {
                    "label": "ALT",
                    "current_value": "82 U/L",
                    "suggested_value": "48 U/L",
                    "direction": "Reassuring",
                    "magnitude_text": "−34 U/L",
                    "normal_range": "7–55 U/L",
                    "training_reference": "typical value among reference profiles: 28 U/L",
                },
                {
                    "label": "Body mass index",
                    "current_value": "32 kg/m²",
                    "suggested_value": "28 kg/m²",
                    "direction": "Reassuring",
                    "magnitude_text": "−4 kg/m²",
                    "normal_range": "18.5–24.9 kg/m²",
                    "training_reference": "typical value among reference profiles: 27 kg/m²",
                },
            ],
        },
        "output": {
            "summary": (
                "The analysis suggests hepatic fibrosis with a score of 65 %. "
                "This score is only 15 % above the alert level, which may indicate an uncertainty zone."
            ),
            "prediction": {
                "label": "hepatic fibrosis suggested",
                "confidence_text": "moderate",
                "score_pct": "65 %",
            },
            "key_factors": [
                {
                    "label": "ALT",
                    "value": "82 U/L",
                    "direction": "Concerning",
                    "explanation": (
                        "Your ALT (82 U/L) is above the usual range (7–55 U/L). "
                        "Among reference profiles, the typical value is 28 U/L. "
                        "This is a particularly influential factor in the analysis."
                    ),
                    "normal_range": "7–55 U/L",
                },
                {
                    "label": "Body mass index",
                    "value": "32 kg/m²",
                    "direction": "Concerning",
                    "explanation": (
                        "A BMI of 32 kg/m² is above the usual range (18.5–24.9 kg/m²). "
                        "Among reference profiles, the typical value is 27 kg/m². "
                        "It remains an influential factor in the analysis."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
            ],
            "context": (
                "This tool was tested on 720 medical profiles, with reliability verified on a separate dataset. "
                "In 78 out of 100 cases, it correctly distinguishes concerning profiles. "
                "With a score of 65 % only 15 % above the alert level, this result lies in an uncertainty zone."
            ),
            "limitations": (
                "Out of 100 truly concerning profiles, the tool detects about 66 — some may go unnoticed, especially in this uncertainty zone. "
                "It does not take into account your medical history, symptoms, or the results of clinical examinations."
            ),
            "next_steps": (
                "Given the moderate reliability, a medical consultation is still recommended. "
                "Do not use this result alone to draw conclusions."
            ),
            "what_to_change": [
                {
                    "factor": "ALT",
                    "current": "82 U/L",
                    "target": "48 U/L",
                    "change_text": "lower ALT by about 34 U/L to return toward the usual range (7–55 U/L)",
                    "why_it_matters": "This drop would bring your profile closer to the reassuring zone.",
                },
                {
                    "factor": "Body mass index",
                    "current": "32 kg/m²",
                    "target": "28 kg/m²",
                    "change_text": "reduce BMI by about 4 kg/m² to get closer to the typical value (27 kg/m²)",
                    "why_it_matters": "A BMI closer to the typical value would bring your profile closer to the reassuring zone.",
                },
            ],
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
        is_regression = context.task_type == "regression"
        system = _SYSTEM_FR_V2 if context.lang == "fr" else _SYSTEM_EN_V2
        if is_regression:
            system = system + (
                _SYSTEM_REGRESSION_APPENDIX_FR
                if context.lang == "fr"
                else _SYSTEM_REGRESSION_APPENDIX_EN
            )
        if reinforced:
            system = system + "\n\n" + _reinforcement_clause(context.lang)

        if is_regression:
            instruction = (
                _INSTRUCTION_REGRESSION_FR
                if context.lang == "fr"
                else _INSTRUCTION_REGRESSION_EN
            )
            few_shot = (
                _FEWSHOT_REGRESSION_FR
                if context.lang == "fr"
                else _FEWSHOT_REGRESSION_EN
            )
        else:
            instruction = _INSTRUCTION_FR_V2 if context.lang == "fr" else _INSTRUCTION_EN_V2
            few_shot = _FEWSHOT_FR if context.lang == "fr" else _FEWSHOT_EN

        # Context first, instruction LAST as a closing reminder. Small models
        # weight the most recent tokens more heavily, so placing the short
        # checklist at the end of the user payload reinforces the bloquant
        # rules without re-spending the system prompt budget.
        payload = {
            "context": _serialize_context(context),
            "schema": _OUTPUT_SCHEMA,
            "few_shot": few_shot,
            "instruction": instruction,
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

_INSTRUCTION_FR_V1 = (
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
    "what_to_change: si counterfactual_changes est vide, renvoie une liste vide []. Sinon, une entrée par changement avec factor/current/target/change_text/why_it_matters. N'invente JAMAIS de nouveau score; dis seulement que ce changement 'rapprocherait' ou 'éloignerait' du niveau d'alerte.\n"
    "RAPPEL FINAL: les exemples few-shot couvrent un domaine (diabète, cardio) qui n'est probablement PAS le tien. Ton domaine est dicté par class_context.label_meaning, top_features et dataset_summary du contexte ci-dessus. N'utilise AUCUN terme médical (maladie, organe, marqueur, examen, traitement) absent du contexte courant."
)

_INSTRUCTION_FR_V2 = """Génère le rapport JSON pour le contexte ci-dessus.
Rappel :
- Conditionnel obligatoire (« suggère », « pourrait »).
- Aucun terme de la liste interdite.
- Phrases ≤ 20 mots dans summary.
- `disclaimer` = chaîne littérale « PLACEHOLDER ».
- Aucun champ supplémentaire hors schéma."""

_INSTRUCTION_EN_V1 = (
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
    "what_to_change: if counterfactual_changes is empty, return an empty list []. Otherwise, one entry per change with factor/current/target/change_text/why_it_matters. NEVER invent a new score; only say the change would 'move closer to' or 'further from' the alert level.\n"
    "FINAL REMINDER: the few-shot examples cover a domain (diabetes, cardio) that is probably NOT yours. Your domain is dictated by class_context.label_meaning, top_features and dataset_summary in the context above. Use NO medical term (disease, organ, marker, exam, treatment) that is absent from the current context."
)

# EN V2 aliased to V1 until the EN rewrite lands.
_INSTRUCTION_EN_V2 = _INSTRUCTION_EN_V1


# ── Regression overrides ─────────────────────────────────────────────────────
# Appended to the system prompt only when ``context.task_type == "regression"``.
# Replaces every classification-only assumption (score in %, alert level,
# positive/negative class, recall/precision) without rewriting the rest.
_SYSTEM_REGRESSION_APPENDIX_FR = """

MODE RÉGRESSION (s'applique car task_type == "regression"):
- Le résultat n'est PAS une probabilité ni un score en %. C'est une VALEUR ESTIMÉE (ex: glycémie estimée à 132 mg/dL).
- INTERDIT en plus du vocabulaire ci-dessus: "score", "score_pct", "niveau d'alerte", "résultat préoccupant / rassurant", "classe positive / négative", "détection", "recall", "precision".
- Remplacements obligatoires:
  - "score_pct" → cite directement la valeur estimée et son unité (ex: "estimation : 132 mg/dL").
  - "fiabilité globale" / "ROC AUC" → utilise model_quality.interpretation pour R² (« la part de la variation expliquée »), MAE / RMSE (« écart moyen attendu ») et MAPE.
  - "rapprocher / éloigner du niveau d'alerte" → parle d'« écart par rapport à la valeur réelle ».
- summary: cite la valeur estimée avec son unité, puis dit en clair si l'estimation est fiable (R² élevé) ou approximative (R² faible). PAS de seuil.
- context: reformule R², MAE/RMSE et MAPE en clair. Exemple: « en moyenne, l'estimation s'écarte de 12 mg/dL de la valeur réelle ».
- limitations: cite l'écart typique attendu (RMSE/MAE) et ce que l'outil ne voit pas. PAS de "100 profils détectés".
- next_steps: 1–2 phrases concrètes (« demandez une mesure de confirmation »).
- what_to_change: TOUJOURS une liste vide [] en régression (DiCE n'est pas utilisé)."""
_SYSTEM_REGRESSION_APPENDIX_EN = """

REGRESSION MODE (applies because task_type == "regression"):
- The result is NOT a probability nor a percentage score. It is an ESTIMATED VALUE (e.g., estimated glucose of 132 mg/dL).
- ALSO FORBIDDEN on top of the above: "score", "score_pct", "alert level", "concerning / reassuring result", "positive / negative class", "detection", "recall", "precision".
- Mandatory replacements:
  - "score_pct" → cite the estimated value with its unit directly (e.g., "estimate: 132 mg/dL").
  - "overall reliability" / "ROC AUC" → use model_quality.interpretation for R² ("share of variation explained"), MAE / RMSE ("typical expected gap"), and MAPE.
  - "move closer to / further from the alert level" → talk about "the gap from the real value".
- summary: cite the estimated value with its unit, then say plainly whether the estimate is reliable (high R²) or approximate (low R²). NO threshold.
- context: rephrase R², MAE/RMSE and MAPE in plain language. Example: "on average, the estimate differs from the real value by 12 mg/dL".
- limitations: cite the typical expected gap (RMSE/MAE) and what the tool does not see. NO "100 profiles detected".
- next_steps: 1–2 concrete sentences ("ask for a confirmation measurement").
- what_to_change: ALWAYS an empty list [] in regression (DiCE is not used)."""

_INSTRUCTION_REGRESSION_FR = (
    "Génère un rapport JSON en français courant, lisible par toute personne sans formation médicale.\n"
    "summary: cite la valeur estimée avec son unité (depuis class_context.label_meaning ou label) et dis en clair si l'estimation est fiable, modérée ou approximative selon confidence_text.\n"
    "key_factors: pour chaque top_feature, cite sa valeur et normal_range. "
    "Reformule training_reference en langage naturel. Reformule global_importance naturellement. "
    "Si position_vs_normal=above, dis que la valeur est 'au-dessus de la normale'. Si below, 'en dessous'. Si within, 'dans les valeurs habituelles'.\n"
    "context: reformule dataset_summary et model_quality.interpretation pour R², MAE/RMSE et MAPE. "
    "Exprime la fiabilité en clair (« part de la variation expliquée », « écart moyen attendu »). PAS de seuil ni de niveau d'alerte.\n"
    "limitations: cite l'écart typique attendu et 1–2 éléments que l'outil ne voit pas (antécédents, symptômes, examens cliniques).\n"
    "next_steps: 1–2 phrases concrètes (consultation, confirmation par un examen). Pas de répétition des limitations.\n"
    "what_to_change: TOUJOURS une liste vide [] en régression.\n"
    "RAPPEL FINAL: les exemples few-shot couvrent un domaine qui n'est probablement PAS le tien. Ton domaine est dicté par class_context.label_meaning, top_features et dataset_summary. N'utilise AUCUN terme médical absent du contexte courant."
)
_INSTRUCTION_REGRESSION_EN = (
    "Generate a JSON report in plain English, readable by anyone without medical training.\n"
    "summary: cite the estimated value with its unit (from class_context.label_meaning or label) and state plainly whether the estimate is reliable, moderate or approximate based on confidence_text.\n"
    "key_factors: for each top_feature, cite its value and normal_range. "
    "Rephrase training_reference naturally. Rephrase global_importance naturally. "
    "If position_vs_normal=above, say the value is 'above the normal range'. If below, 'below'. If within, 'within the usual range'.\n"
    "context: rephrase dataset_summary and model_quality.interpretation for R², MAE/RMSE and MAPE. "
    "Express reliability in plain language ('share of variation explained', 'typical expected gap'). NO threshold, NO alert level.\n"
    "limitations: cite the typical expected gap and 1–2 things the tool does not see (history, symptoms, clinical exam).\n"
    "next_steps: 1–2 concrete sentences (consultation, confirmation by a test). Do not repeat limitations.\n"
    "what_to_change: ALWAYS an empty list [] in regression.\n"
    "FINAL REMINDER: the few-shot examples cover a domain that is probably NOT yours. Your domain is dictated by class_context.label_meaning, top_features and dataset_summary. Use NO medical term absent from the current context."
)

# One regression few-shot per language. Self-consistent values; the LLM should
# pattern-match the SHAPE (no score_pct, no threshold, R²-anchored phrasing)
# rather than copy the domain.
_FEWSHOT_REGRESSION_FR = [
    {
        "context": {
            "label": "tension systolique estimee a 138 mmHg",
            "confidence_text": "modérée",
            "score_pct": None,
            "task_type": "regression",
            "class_context": {
                "raw_label": "138.0",
                "target_name": "Systolic_BP",
                "positive_class": "",
                "label_meaning": "valeur numerique estimee : 138",
            },
            "model_quality": [
                {
                    "label": "Part de la variation expliquee par l'outil",
                    "value": "62.0 %",
                    "interpretation": "plus le pourcentage est eleve, plus l'estimation suit les variations reelles",
                },
                {
                    "label": "Ecart moyen entre l'estimation et la valeur reelle",
                    "value": "8.4",
                    "interpretation": "en moyenne, l'estimation s'ecarte de la valeur reelle de cette quantite",
                },
                {
                    "label": "Ecart moyen en pourcentage",
                    "value": "6.5 %",
                    "interpretation": "en moyenne, l'estimation s'ecarte de la valeur reelle de ce pourcentage",
                },
            ],
            "dataset_summary": [
                "Base sur l'analyse de 540 profils medicaux",
                "Fiabilite verifiee sur des donnees distinctes",
            ],
            "top_features": [
                {
                    "label": "Indice de masse corporelle",
                    "value": "29.4 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "valeur habituelle parmi les profils de reference : 26.1 kg/m²",
                    "global_importance": "1er facteur le plus influent parmi 9 (poids : 22.0 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Age",
                    "value": "58 ans",
                    "direction": "increase",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "valeur habituelle parmi les profils de reference : 49 ans",
                    "global_importance": "2e facteur le plus influent parmi 9",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [],
        },
        "report": {
            "summary": (
                "L'estimation est de 138 mmHg pour la tension systolique. "
                "Avec une fiabilité modérée, ce chiffre doit être confirmé par une mesure réelle."
            ),
            "key_factors": [
                {
                    "label": "Indice de masse corporelle",
                    "value": "29.4 kg/m²",
                    "direction": "augmente",
                    "explanation": (
                        "Votre IMC (29.4 kg/m²) est au-dessus de la plage habituelle (18.5–24.9 kg/m²). "
                        "Chez les profils de référence, la valeur habituelle est de 26.1 kg/m². "
                        "C'est le facteur qui a le plus pesé dans cette estimation."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
                {
                    "label": "Age",
                    "value": "58 ans",
                    "direction": "augmente",
                    "explanation": (
                        "Vous avez 58 ans. Chez les profils de référence, la valeur habituelle est de 49 ans. "
                        "C'est le 2e facteur le plus influent dans cette estimation."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "Cet outil a été testé sur 540 profils médicaux, avec une fiabilité vérifiée sur des données distinctes. "
                "Il explique environ 62 % de la variation observée entre les profils. "
                "En moyenne, l'estimation s'écarte de la valeur réelle d'environ 8.4 mmHg, soit 6.5 %."
            ),
            "limitations": (
                "L'écart typique attendu est d'environ 8 mmHg : l'estimation reste approximative. "
                "L'outil ne tient pas compte de votre traitement, de votre activité physique ni du stress du moment."
            ),
            "next_steps": (
                "Faites confirmer cette estimation par une mesure réelle de la tension. "
                "Discutez avec un médecin si la valeur mesurée est régulièrement au-dessus de 130 mmHg."
            ),
            "what_to_change": [],
            "disclaimer": "PLACEHOLDER",
        },
    },
]
_FEWSHOT_REGRESSION_EN = [
    {
        "context": {
            "label": "estimated systolic pressure of 138 mmHg",
            "confidence_text": "moderate",
            "score_pct": None,
            "task_type": "regression",
            "class_context": {
                "raw_label": "138.0",
                "target_name": "Systolic_BP",
                "positive_class": "",
                "label_meaning": "estimated numeric value: 138",
            },
            "model_quality": [
                {
                    "label": "Share of variation explained by the tool",
                    "value": "62.0 %",
                    "interpretation": "the higher the percentage, the more the estimate tracks real variations",
                },
                {
                    "label": "Average gap between the estimate and the real value",
                    "value": "8.4",
                    "interpretation": "on average, the estimate differs from the real value by this amount",
                },
                {
                    "label": "Average percentage gap",
                    "value": "6.5 %",
                    "interpretation": "on average, the estimate differs from the real value by this percentage",
                },
            ],
            "dataset_summary": [
                "Based on the analysis of 540 medical profiles",
                "Reliability verified on a separate held-out dataset",
            ],
            "top_features": [
                {
                    "label": "Body mass index",
                    "value": "29.4 kg/m²",
                    "direction": "increase",
                    "normal_range": "18.5–24.9 kg/m²",
                    "position_vs_normal": "above",
                    "training_reference": "typical value among reference profiles: 26.1 kg/m²",
                    "global_importance": "1st most influential factor out of 9 (weight: 22.0 %)",
                    "evidence_type": "lime_contribution",
                },
                {
                    "label": "Age",
                    "value": "58 years",
                    "direction": "increase",
                    "normal_range": None,
                    "position_vs_normal": "unknown",
                    "training_reference": "typical value among reference profiles: 49 years",
                    "global_importance": "2nd most influential factor out of 9",
                    "evidence_type": "lime_contribution",
                },
            ],
            "counterfactual_changes": [],
        },
        "report": {
            "summary": (
                "The estimate is 138 mmHg for systolic pressure. "
                "With moderate reliability, this number should be confirmed by an actual measurement."
            ),
            "key_factors": [
                {
                    "label": "Body mass index",
                    "value": "29.4 kg/m²",
                    "direction": "increase",
                    "explanation": (
                        "Your BMI (29.4 kg/m²) is above the usual range (18.5–24.9 kg/m²). "
                        "Among reference profiles, the typical value is 26.1 kg/m². "
                        "This is the factor that weighed most in this estimate."
                    ),
                    "normal_range": "18.5–24.9 kg/m²",
                },
                {
                    "label": "Age",
                    "value": "58 years",
                    "direction": "increase",
                    "explanation": (
                        "You are 58 years old. Among reference profiles, the typical value is 49 years. "
                        "This is the 2nd most influential factor in this estimate."
                    ),
                    "normal_range": None,
                },
            ],
            "context": (
                "This tool was tested on 540 medical profiles, with reliability verified on a separate dataset. "
                "It explains about 62 % of the variation observed across profiles. "
                "On average, the estimate differs from the real value by about 8.4 mmHg, or 6.5 %."
            ),
            "limitations": (
                "The typical expected gap is around 8 mmHg: the estimate remains approximate. "
                "The tool does not consider your medication, physical activity or current stress."
            ),
            "next_steps": (
                "Have this estimate confirmed by an actual blood pressure measurement. "
                "Discuss with a doctor if the measured value is regularly above 130 mmHg."
            ),
            "what_to_change": [],
            "disclaimer": "PLACEHOLDER",
        },
    },
]


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


def _serialize_feature(feat: FeatureContribution, *, lang: str = "fr") -> dict[str, Any]:
    """Project a FeatureContribution to the LLM-visible subset.

    Excludes ``raw_name`` (unsafe column name, already mapped to label) and
    ``weight`` (a normalized float — the order in the list already conveys
    importance, no need to leak the number).

    Includes ``position_vs_normal`` so the LLM can write "abnormally high"
    without re-deriving it from the value + range itself. That derivation is
    deterministic and lives in the context builder.

    ``direction`` is localised (Préoccupant / Rassurant in FR) so the value
    that reaches the LLM matches the label patients see — no raw English
    token leaks through if the model copies it verbatim."""
    return {
        "label": feat.label,
        "value": feat.value,
        "direction": localize_direction(feat.direction, lang),
        "normal_range": feat.normal_range,
        "position_vs_normal": feat.position_vs_normal,
        "training_reference": feat.training_reference,
        "global_importance": feat.global_importance,
        # "lime_contribution" when the value comes from a local LIME explanation;
        # "observed_value" when no explanation was available and we fell back to
        # raw input data. The distinction comes from the builder, not direction.
        "evidence_type": "observed_value" if feat.is_fallback else "lime_contribution",
    }


def _serialize_cf_change(c: CounterfactualChange, *, lang: str = "fr") -> dict[str, Any]:
    """LLM-visible projection of a counterfactual suggestion (already string-typed)."""
    return {
        "label": c.label,
        "current_value": c.current_value,
        "suggested_value": c.suggested_value,
        "direction": localize_direction(c.direction, lang),
        "magnitude_text": c.magnitude_text,
        "normal_range": c.normal_range,
        "training_reference": c.training_reference,
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
        "top_features": [
            _serialize_feature(f, lang=context.lang) for f in context.top_features[:8]
        ],
        "counterfactual_changes": [
            _serialize_cf_change(c, lang=context.lang)
            for c in context.counterfactual_changes[:5]
        ],
    }

    # Pre-compute threshold distance so the LLM can reference it in the summary
    # without having to derive it from floating-point arithmetic.
    if context.score is not None:
        threshold = context.threshold_value if context.threshold_value is not None else 0.5
        distance_pct = round(abs(float(context.score) - float(threshold)) * 100)
        out["threshold_distance_pct"] = f"{distance_pct} %"

    return out


__all__ = ["PromptBuilder"]
