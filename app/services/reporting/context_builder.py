"""Builds a typed ReportContext from a raw prediction result.

The plan's golden rule: the LLM never sees a raw float. Every numeric value
is pre-formatted to a string here. Confidence is bucketed to a categorical
text label in Python before being passed downstream. Column names and string
values from the dataset are sanitized (caps length + strips suspicious
characters) to defuse prompt-injection vectors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

# ── Sanitization ──────────────────────────────────────────────────────────────
# Keeps letters/digits/whitespace plus a small set of medical-friendly
# punctuation. Strips backticks, braces, prompt-injection markers, etc.
_SANITIZE_PATTERN = re.compile(r"[^\w\s\.\,\-\/\(\)\:°%²³µ]", re.UNICODE)
_TRANSFORMER_PREFIX_RE = re.compile(r"^(?:num|cat|remainder)__", re.IGNORECASE)
_MAX_VALUE_LEN = 200
_MAX_LABEL_LEN = 120


def sanitize_for_prompt(value: str, *, max_len: int = _MAX_VALUE_LEN) -> str:
    """Drop suspicious characters and bound length. Idempotent on safe inputs."""
    if value is None:
        return ""
    text = str(value)
    cleaned = _SANITIZE_PATTERN.sub("", text)
    return cleaned[:max_len].strip()


def _feature_lookup_name(raw_name: str) -> str:
    """Return the original column-ish name from transformed feature labels.

    Local explanation libraries often emit names such as ``num__Glucose`` after
    a ColumnTransformer. The reporting layer needs ``Glucose`` to find the
    original value, glossary entry, and training stats.
    """
    return _TRANSFORMER_PREFIX_RE.sub("", raw_name).strip()


# Names like "f_0", "x12", "var3" carry no semantic signal — leave them as-is.
_OPAQUE_NAME_RE = re.compile(r"^[a-z]_?\d+$", re.IGNORECASE)


def _humanize_feature_name(raw_name: str) -> str:
    """Best-effort label when the glossary has no entry.

    Replaces underscores/dashes with spaces and capitalises the first letter
    so ``blood_pressure`` becomes ``Blood pressure``. Opaque names such as
    ``f_0`` are returned unchanged because there is nothing to recover.
    """
    cleaned = _feature_lookup_name(raw_name)
    if not cleaned:
        return raw_name
    if _OPAQUE_NAME_RE.fullmatch(cleaned):
        return cleaned
    spaced = re.sub(r"[_\-]+", " ", cleaned).strip()
    if not spaced:
        return raw_name
    return spaced[:1].upper() + spaced[1:]


# ── Dataclasses (the data contract) ───────────────────────────────────────────

Direction = Literal["increase", "decrease", "neutral"]
NormalPosition = Literal["above", "below", "within", "unknown"]
ConfidenceText = Literal["élevée", "modérée", "faible", "non quantifiée",
                         "high", "moderate", "low", "unquantified"]


@dataclass(frozen=True)
class FeatureContribution:
    raw_name: str
    label: str
    value: str               # already formatted (e.g. "126 mg/dL")
    direction: Direction
    weight: float            # |normalized| in [0, 1], sort key only
    normal_range: Optional[str] = None
    position_vs_normal: NormalPosition = "unknown"
    training_reference: Optional[str] = None
    global_importance: Optional[str] = None
    # True when built from raw input_data (no LIME), False for LIME-derived.
    # Drives evidence_type in prompt serialization.
    is_fallback: bool = False


@dataclass(frozen=True)
class PredictionClassContext:
    raw_label: str = ""
    target_name: str = ""
    positive_class: str = ""
    label_meaning: str = ""


@dataclass(frozen=True)
class ModelQualitySignal:
    label: str
    value: str
    interpretation: str = ""


@dataclass(frozen=True)
class CounterfactualChange:
    """One actionable change suggested by DiCE.

    Pre-formatted for the LLM: ``current_value`` and ``suggested_value`` are
    already strings with units. ``magnitude_text`` is a short human phrase
    (e.g., "−24 mg/dL", "+5 kg/m²") so the LLM never sees raw floats.
    """
    label: str
    current_value: str
    suggested_value: str
    direction: Direction
    magnitude_text: str
    normal_range: Optional[str] = None
    training_reference: Optional[str] = None


@dataclass(frozen=True)
class ReportContext:
    prediction_id: str
    lang: Literal["fr", "en"]
    label: str
    confidence_text: ConfidenceText
    score_pct: Optional[str]              # "87 %" or None for regression
    top_features: list[FeatureContribution] = field(default_factory=list)
    class_context: PredictionClassContext = field(default_factory=PredictionClassContext)
    model_quality: list[ModelQualitySignal] = field(default_factory=list)
    dataset_summary: list[str] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    task_type: str = "classification"
    # Suggestions DiCE déjà formatées pour le LLM. Vide = pas de CF disponible
    # (régression, échec DiCE, ou modèle déjà du bon côté).
    counterfactual_changes: list[CounterfactualChange] = field(default_factory=list)
    # Internal fields — never forwarded to the LLM prompt; used for server-side postprocessing.
    score: Optional[float] = None
    threshold_value: Optional[float] = None


# ── Confidence bucketing ──────────────────────────────────────────────────────

_FR_BUCKETS: tuple[tuple[float, ConfidenceText], ...] = (
    (0.6, "élevée"),
    (0.3, "modérée"),
    (0.0, "faible"),
)
_EN_BUCKETS: tuple[tuple[float, ConfidenceText], ...] = (
    (0.6, "high"),
    (0.3, "moderate"),
    (0.0, "low"),
)


def bucket_confidence(score: Optional[float], lang: str) -> ConfidenceText:
    """Map a probability score to a categorical label. Symmetric around 0.5."""
    if score is None:
        return "non quantifiée" if lang == "fr" else "unquantified"
    # For binary classification, the "confidence" is how far the score is from 0.5.
    distance = abs(float(score) - 0.5) * 2.0  # in [0, 1]
    buckets = _FR_BUCKETS if lang == "fr" else _EN_BUCKETS
    for threshold, label in buckets:
        if distance >= threshold:
            return label
    return buckets[-1][1]


def _extract_r2(model_quality: list[ModelQualitySignal]) -> Optional[float]:
    """Pull the R² value from the model_quality list.

    R² appears under a patient-friendly label (no raw key), so we match by
    pre-formatted string ("XX.X %") and convert it back. When the label or
    value cannot be parsed, returns None and the caller falls back to
    "unquantified". We never fail loudly because the report must still
    render.
    """
    for sig in model_quality:
        label_lower = (sig.label or "").lower()
        if "variation" in label_lower or "variation explained" in label_lower:
            text = (sig.value or "").strip().rstrip("%").strip()
            try:
                return float(text) / 100.0
            except (TypeError, ValueError):
                return None
    return None


def _bucket_regression_reliability(r2: Optional[float], lang: str) -> ConfidenceText:
    """Three-bucket reliability for regression — sourced from R², not score.

    Mirrors the classification confidence buckets so the rest of the prompt
    (and the frontend's badge / phrasing) can stay task-agnostic. The
    thresholds are deliberately conservative: a regression model below 40 %
    R² is rarely trustworthy enough to justify a "moderate" badge.
    """
    if r2 is None:
        return "non quantifiée" if lang == "fr" else "unquantified"
    if r2 >= 0.7:
        return "élevée" if lang == "fr" else "high"
    if r2 >= 0.4:
        return "modérée" if lang == "fr" else "moderate"
    return "faible" if lang == "fr" else "low"


# ── Value formatting ──────────────────────────────────────────────────────────

def _format_value(value: Any, unit: Optional[str] = None) -> str:
    """Render a feature value as a clean human-readable string."""
    if value is None:
        return "—"
    # Booleans before numbers (bool is a subclass of int).
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value}{(' ' + unit) if unit else ''}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        # Integer-valued floats: drop the .0
        if abs(value - round(value)) < 1e-9 and abs(value) < 1e9:
            return f"{int(round(value))}{(' ' + unit) if unit else ''}"
        return f"{value:.2f}{(' ' + unit) if unit else ''}"
    return sanitize_for_prompt(str(value), max_len=_MAX_VALUE_LEN)


def _direction_from_weight(weight: float) -> Direction:
    if weight > 1e-6:
        return "increase"
    if weight < -1e-6:
        return "decrease"
    return "neutral"


# Semantic localisation for the user-visible ``direction`` field. The dataclass
# itself keeps the Literal["increase"|"decrease"|"neutral"] contract; this map
# is applied only at serialization boundaries (LLM prompt + post-processed
# report sent to the frontend). The mapping is intentionally semantic
# (Préoccupant / Rassurant) rather than verbal (augmente / diminue), so a
# non-clinical reader can decode the badge at a glance.
_DIRECTION_DISPLAY: dict[tuple[str, str], str] = {
    ("fr", "increase"): "Préoccupant",
    ("fr", "decrease"): "Rassurant",
    ("fr", "neutral"): "Neutre",
    ("en", "increase"): "Concerning",
    ("en", "decrease"): "Reassuring",
    ("en", "neutral"): "Neutral",
}


def localize_direction(direction: str, lang: str) -> str:
    """Translate an internal Direction token to its patient-facing label.

    Returns the input unchanged when the (lang, direction) pair is not in
    the table — keeps the function safe even when an unexpected value
    slips through from an upstream module.
    """
    return _DIRECTION_DISPLAY.get((lang, direction or ""), direction or "")


# ── Sensitive-label sanitisation ──────────────────────────────────────────────
# Training pipelines often pass raw class names (DECES, MALIGNE, 1) straight
# through to the report layer. Showing these verbatim to a non-clinical reader
# is harmful (cognitive shock) and legally fragile — we rewrite them into
# neutral patient-facing phrasings BEFORE they reach the LLM payload or the
# frontend renderer.
# Each entry maps an upper-cased clinical token to the patient-safe phrasing
# for both supported languages, so a French-locale lookup with raw label
# "CANCER" returns French text and an English-locale lookup with the same
# raw label returns English text.
_SENSITIVE_LABEL_MAP: dict[str, dict[str, str]] = {
    "DECES":     {"fr": "résultat préoccupant",  "en": "concerning result"},
    "DECES_1":   {"fr": "résultat préoccupant",  "en": "concerning result"},
    "MORT":      {"fr": "résultat préoccupant",  "en": "concerning result"},
    "DEATH":     {"fr": "résultat préoccupant",  "en": "concerning result"},
    "DECEASED":  {"fr": "résultat préoccupant",  "en": "concerning result"},
    "MALIGNE":   {"fr": "résultat préoccupant",  "en": "concerning result"},
    "MALIGNANT": {"fr": "résultat préoccupant",  "en": "concerning result"},
    "CANCER":    {"fr": "résultat préoccupant",  "en": "concerning result"},
    "POSITIF":   {"fr": "résultat préoccupant",  "en": "concerning result"},
    "POSITIVE":  {"fr": "résultat préoccupant",  "en": "concerning result"},
    "NEGATIF":   {"fr": "résultat rassurant",    "en": "reassuring result"},
    "NEGATIVE":  {"fr": "résultat rassurant",    "en": "reassuring result"},
    "1":         {"fr": "résultat préoccupant",  "en": "concerning result"},
    "0":         {"fr": "résultat rassurant",    "en": "reassuring result"},
}

_FALLBACK_LABEL: dict[str, str] = {
    "fr": "résultat à analyser avec votre médecin",
    "en": "result to discuss with your doctor",
}


def sanitize_label(raw_label: str, lang: str = "fr") -> str:
    """Replace clinical raw labels with patient-safe phrasings.

    See ``_SENSITIVE_LABEL_MAP`` for the static substitutions. The lookup
    now respects ``lang`` so a French dataset containing "CANCER" returns
    French text and an English-locale call with the same raw label returns
    English text. Three fallbacks apply when the upper-cased input is not
    in the map:
    - purely digit strings or 1–2 char tokens → generic "to discuss with
      your doctor" phrasing (lang-aware),
    - empty input → empty string,
    - anything else → lower-cased pass-through (at minimum no SHOUTING).
    """
    if raw_label is None:
        return ""
    key = str(raw_label).strip().upper()
    if not key:
        return ""
    lang = lang if lang in ("fr", "en") else "fr"
    mapping = _SENSITIVE_LABEL_MAP.get(key)
    if mapping:
        return mapping[lang]
    if key.isdigit() or len(key) <= 2:
        return _FALLBACK_LABEL[lang]
    return str(raw_label).lower()


# ── Raw column-name fallback ──────────────────────────────────────────────────
# When the project's glossary has no entry for a feature, the column name
# reaches the report unchanged (e.g. "f_62", "col_3"). The LLM is instructed
# to keep these as-is, but the result is unreadable. We detect the pattern
# and substitute a positional placeholder before the label propagates.
_RAW_COLUMN_RE = re.compile(
    r"^(?:f_|col_|feature_|column_|var_|x|v)\d+$",
    re.IGNORECASE,
)


def is_raw_column_name(label: str) -> bool:
    """True when the label looks like an unconfigured technical column name."""
    if not label:
        return True
    return bool(_RAW_COLUMN_RE.match(str(label).strip()))


def get_display_label(label: str, position: int, lang: str = "fr") -> str:
    """Return ``label`` when it carries semantic signal, else a positional fallback.

    ``position`` is the 0-based index in the top-features list; the visible
    placeholder is 1-based (``Indicateur 1`` / ``Indicator 1``).
    """
    if not label or is_raw_column_name(label):
        return f"Indicateur {position + 1}" if lang == "fr" else f"Indicator {position + 1}"
    return label


def _position_vs_normal(value: Any, normal_range_raw: Any) -> NormalPosition:
    """Compare a feature value to its glossary-defined normal range.

    Returns "unknown" whenever the value is not numerically comparable to the
    bounds (categorical features, missing range, NaN, etc.). The reporter
    falls back to direction-only phrasing in that case.
    """
    if not isinstance(normal_range_raw, (list, tuple)) or len(normal_range_raw) != 2:
        return "unknown"
    try:
        v = float(value)
        low, high = float(normal_range_raw[0]), float(normal_range_raw[1])
    except (TypeError, ValueError):
        return "unknown"
    if v != v:  # NaN
        return "unknown"
    if v < low:
        return "below"
    if v > high:
        return "above"
    return "within"


def _resolve_glossary_entry(glossary: Any, raw_name: str) -> dict[str, Any]:
    """Tolerant glossary lookup.

    Accepts a plain dict or any object exposing
    ``.get(raw_name) -> dict | None``.
    """
    if glossary is None:
        return {}
    # FeatureGlossary.get returns {} on miss; dict.get returns None.
    if hasattr(glossary, "get"):
        entry = glossary.get(raw_name)
        if isinstance(entry, dict):
            return entry
    return {}


def _synthetic_entry(
    raw_name: str,
    feature_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Glossary-shaped fallback built from training column_stats.

    Used only when the canonical glossary has nothing for the feature. Carries
    a humanised label and an *observed* range derived from training min/max.
    Critically: it does NOT populate ``normal_range`` because that field
    encodes a clinical reference (it would mislabel values as
    ``abnormal_high`` against an arbitrary observed bound). The observed
    spread is exposed separately as ``observed_range_text`` so the
    presentation layer can show it without claiming clinical normality.
    """
    meta = (
        feature_metadata.get(raw_name)
        or feature_metadata.get(_feature_lookup_name(raw_name))
        or {}
    )
    label = _humanize_feature_name(raw_name)
    out: dict[str, Any] = {"label_fr": label, "label_en": label}

    min_raw = meta.get("min_raw")
    max_raw = meta.get("max_raw")
    if min_raw not in (None, "") and max_raw not in (None, ""):
        out["observed_range_text"] = f"{min_raw}–{max_raw}"
    return out


def _value_from_input_data(
    input_data: dict[str, Any] | None,
    raw_name: str,
    *,
    _canonical: dict[str, Any] | None = None,
) -> Any:
    """Look up a feature value by raw or cleaned name.

    ``_canonical`` is a pre-built ``{clean_key: value}`` dict. Pass it when
    calling in a loop to avoid rebuilding it on every iteration.
    """
    if not input_data:
        return None
    for key in (
        raw_name,
        _feature_lookup_name(raw_name),
        raw_name.replace(" ", "_"),
        _feature_lookup_name(raw_name).replace(" ", "_"),
    ):
        if key in input_data:
            return input_data[key]
        c = re.sub(r"[^a-z0-9]+", "", str(key).lower())
        canon = _canonical or {re.sub(r"[^a-z0-9]+", "", str(k).lower()): v for k, v in input_data.items()}
        if c in canon:
            return canon[c]
    return None


# ── Builder ───────────────────────────────────────────────────────────────────

class ReportContextBuilder:
    """Pure function packaged as a class for symmetry with the rest of the
    reporting module. Stateless; safe to instantiate per request."""

    def build(
        self,
        *,
        prediction_id: str,
        lang: Literal["fr", "en"],
        prediction_value: Any,
        score: Optional[float],
        lime_items: Iterable[dict[str, Any]] | None,
        model_name: str,
        model_version: str,
        input_data: dict[str, Any] | None = None,
        class_context: PredictionClassContext | None = None,
        model_quality: list[ModelQualitySignal] | None = None,
        dataset_summary: list[str] | None = None,
        feature_metadata: dict[str, dict[str, str]] | None = None,
        task_type: str = "classification",
        glossary: Any = None,
        top_n: int = 5,
        threshold_value: Optional[float] = None,
        counterfactual_items: Iterable[dict[str, Any]] | None = None,
    ) -> ReportContext:
        # First strip prompt-injection markers, then route the clean label
        # through ``sanitize_label`` so clinically-loaded tokens (DECES, 1,
        # MALIGNE…) become patient-safe phrasings. The order matters: sanitize
        # for prompt FIRST so the sensitive-map lookup operates on a clean,
        # uppercase-canonical key without backticks or braces.
        raw_label_clean = sanitize_for_prompt(str(prediction_value), max_len=_MAX_LABEL_LEN)
        label = sanitize_label(raw_label_clean, lang=lang) or "—"
        # Sanitise the class_context fields the LLM (and the frontend) will
        # read. The dataclass is frozen, so we rebuild it with safe values.
        cc_in = class_context or PredictionClassContext()
        safe_class_context = PredictionClassContext(
            raw_label=sanitize_label(cc_in.raw_label, lang=lang) if cc_in.raw_label else "",
            target_name=cc_in.target_name,
            positive_class=sanitize_label(cc_in.positive_class, lang=lang) if cc_in.positive_class else "",
            label_meaning=sanitize_label(cc_in.label_meaning, lang=lang) if cc_in.label_meaning else "",
        )
        score_pct: Optional[str] = None
        if score is not None and task_type == "classification":
            score_pct = f"{round(float(score) * 100)} %"

        # Confidence sourcing differs by task:
        # - classification: distance of probability from 0.5 (existing behaviour).
        # - regression: there is no "probability" — derive a global reliability
        #   bucket from R² so the report can still say "low/moderate/high"
        #   without inventing a per-prediction certainty (which would require
        #   a quantile model we don't have).
        if task_type == "regression":
            r2 = _extract_r2(model_quality or [])
            confidence_text = _bucket_regression_reliability(r2, lang)
        else:
            confidence_text = bucket_confidence(score, lang)

        top_features = self._top_features_from_lime(
            lime_items or [],
            lang=lang,
            glossary=glossary,
            feature_metadata=feature_metadata or {},
            input_data=input_data or {},
            top_n=top_n,
        )
        if not top_features and input_data:
            top_features = self._observed_features_from_input_data(
                input_data,
                lang=lang,
                glossary=glossary,
                feature_metadata=feature_metadata or {},
                top_n=top_n,
            )

        cf_changes = self._counterfactual_changes(
            counterfactual_items or [],
            lang=lang,
            glossary=glossary,
            feature_metadata=feature_metadata or {},
            input_data=input_data or {},
        )

        return ReportContext(
            prediction_id=sanitize_for_prompt(prediction_id, max_len=64) or "unknown",
            lang=lang,
            label=label,
            confidence_text=confidence_text,
            score_pct=score_pct,
            top_features=top_features,
            class_context=safe_class_context,
            model_quality=model_quality or [],
            dataset_summary=dataset_summary or [],
            model_name=sanitize_for_prompt(model_name, max_len=80),
            model_version=sanitize_for_prompt(model_version, max_len=40),
            task_type=task_type,
            counterfactual_changes=cf_changes,
            score=score,
            threshold_value=threshold_value,
        )

    @staticmethod
    def _observed_features_from_input_data(
        input_data: dict[str, Any],
        *,
        lang: str,
        glossary: Any,
        feature_metadata: dict[str, dict[str, str]],
        top_n: int,
    ) -> list[FeatureContribution]:
        """Fallback when no local explanation was computed yet.

        This does not claim feature importance. It exposes clinically readable
        observed values so the report can still say something concrete, while
        marking direction as neutral because there is no LIME/SHAP contribution.
        Known glossary entries are preferred, then the first usable columns.
        """
        if not input_data:
            return []

        ranked: list[tuple[int, str, Any, dict[str, Any]]] = []
        for raw_key, raw_value in input_data.items():
            raw_name = sanitize_for_prompt(str(raw_key), max_len=_MAX_LABEL_LEN)
            if not raw_name:
                continue
            entry = _resolve_glossary_entry(glossary, raw_name)
            # Prefer medically known features and non-empty values.
            priority = 0 if entry else 1
            ranked.append((priority, raw_name, raw_value, entry))

        ranked.sort(key=lambda item: (item[0], item[1].lower()))

        out: list[FeatureContribution] = []
        for position, (_, raw_name, raw_value, entry) in enumerate(ranked[: max(0, top_n)]):
            label_key = "label_fr" if lang == "fr" else "label_en"
            if not entry:
                entry = _synthetic_entry(raw_name, feature_metadata)
            label_default = _humanize_feature_name(raw_name)
            label = sanitize_for_prompt(str(entry.get(label_key, label_default)), max_len=_MAX_LABEL_LEN)
            label = get_display_label(label, position, lang=lang)
            unit = entry.get("unit")
            rng = entry.get("normal_range")
            normal_range = None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                normal_range = f"{rng[0]}–{rng[1]}{(' ' + unit) if unit else ''}"
            elif entry.get("observed_range_text"):
                normal_range = (
                    f"plage observée : {entry['observed_range_text']}"
                    if lang == "fr"
                    else f"observed range: {entry['observed_range_text']}"
                )

            out.append(
                FeatureContribution(
                    raw_name=raw_name,
                    label=label,
                    value=_format_value(raw_value, unit=unit),
                    direction="neutral",
                    weight=0.0,
                    normal_range=normal_range,
                    position_vs_normal=_position_vs_normal(raw_value, rng),
                    training_reference=feature_metadata.get(raw_name, {}).get("training_reference"),
                    global_importance=feature_metadata.get(raw_name, {}).get("global_importance"),
                    is_fallback=True,
                )
            )
        return out

    @staticmethod
    def _top_features_from_lime(
        lime_items: Iterable[dict[str, Any]],
        *,
        lang: str,
        glossary: Any,
        feature_metadata: dict[str, dict[str, str]],
        input_data: dict[str, Any],
        top_n: int,
    ) -> list[FeatureContribution]:
        items = list(lime_items)
        if not items:
            return []

        # LIME items are already shaped as {feature, contribution, data} per
        # app/services/lime/__init__.py. We tolerate either "contribution" or
        # "weight" as the score field for forward compatibility.
        normalized: list[tuple[str, float, Any]] = []
        for it in items:
            raw_name = sanitize_for_prompt(str(it.get("feature", "")), max_len=_MAX_LABEL_LEN)
            if not raw_name:
                continue
            try:
                w = float(it.get("contribution", it.get("weight", 0.0)) or 0.0)
            except (TypeError, ValueError):
                w = 0.0
            normalized.append((raw_name, w, it.get("data")))

        # Sort by |contribution| descending, take top_n
        normalized.sort(key=lambda t: abs(t[1]), reverse=True)
        normalized = normalized[: max(0, top_n)]

        # Normalize weights to [0,1] for downstream display, preserving sign
        max_abs = max((abs(w) for _, w, _ in normalized), default=0.0)

        # Pre-build canonical lookup dict once — used for every feature below.
        canonical_input: dict[str, Any] = {
            re.sub(r"[^a-z0-9]+", "", str(k).lower()): v
            for k, v in input_data.items()
        }
        label_key = "label_fr" if lang == "fr" else "label_en"

        out: list[FeatureContribution] = []
        for position, (raw_name, w, raw_value) in enumerate(normalized):
            lookup_name = _feature_lookup_name(raw_name)
            entry = (
                _resolve_glossary_entry(glossary, lookup_name)
                or _resolve_glossary_entry(glossary, raw_name)
                or _synthetic_entry(raw_name, feature_metadata)
            )
            label_default = _humanize_feature_name(lookup_name)
            label = sanitize_for_prompt(str(entry.get(label_key, label_default)), max_len=_MAX_LABEL_LEN)
            label = get_display_label(label, position, lang=lang)
            unit = entry.get("unit")
            rng = entry.get("normal_range")
            normal_range = None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                normal_range = f"{rng[0]}–{rng[1]}{(' ' + unit) if unit else ''}"
            elif entry.get("observed_range_text"):
                # Synthetic fallback — explicitly tagged as observed, not normal.
                normal_range = (
                    f"plage observée : {entry['observed_range_text']}"
                    if lang == "fr"
                    else f"observed range: {entry['observed_range_text']}"
                )

            effective_value = raw_value
            if effective_value in (None, "", "—", "-"):
                effective_value = _value_from_input_data(input_data, raw_name, _canonical=canonical_input)
            value_str = _format_value(effective_value, unit=unit)
            normalized_weight = (w / max_abs) if max_abs > 0 else 0.0

            out.append(
                FeatureContribution(
                    raw_name=raw_name,
                    label=label,
                    value=value_str,
                    direction=_direction_from_weight(w),
                    weight=normalized_weight,
                    normal_range=normal_range,
                    position_vs_normal=_position_vs_normal(effective_value, rng),
                    training_reference=(
                        feature_metadata.get(raw_name, {}).get("training_reference")
                        or feature_metadata.get(lookup_name, {}).get("training_reference")
                    ),
                    global_importance=(
                        feature_metadata.get(raw_name, {}).get("global_importance")
                        or feature_metadata.get(lookup_name, {}).get("global_importance")
                    ),
                    is_fallback=False,
                )
            )
        return out

    @staticmethod
    def _counterfactual_changes(
        cf_items: Iterable[dict[str, Any]],
        *,
        lang: str,
        glossary: Any,
        feature_metadata: dict[str, dict[str, str]],
        input_data: dict[str, Any],
    ) -> list[CounterfactualChange]:
        """Format raw DiCE output into LLM-ready change suggestions.

        Each DiCE item is ``{feature, original_value, suggested_value, delta}``
        with the ColumnTransformer prefix already stripped. We:
        - resolve a clinical label + unit from the glossary,
        - format both values with the unit (so the LLM never sees raw floats),
        - emit a short ``magnitude_text`` (e.g. ``"−24 mg/dL"``) so the prompt
          can quote the change without arithmetic,
        - cap to 5 changes (DiCE may return many; the report stays focused).
        """
        items = [it for it in cf_items if isinstance(it, dict)]
        if not items:
            return []

        label_key = "label_fr" if lang == "fr" else "label_en"
        out: list[CounterfactualChange] = []
        for it in items[:5]:
            raw_name = sanitize_for_prompt(str(it.get("feature", "")), max_len=_MAX_LABEL_LEN)
            if not raw_name:
                continue
            # Invariant: ``sugg`` must always be the ABSOLUTE target value,
            # never the delta. Upstream DiCE provides both ``suggested_value``
            # (absolute) and ``delta`` (signed); when only the delta is
            # available we derive the target as ``orig - |delta|`` per the
            # patient-facing contract (CF suggestions for risk factors
            # reduce the original value).
            try:
                orig = float(it.get("original_value"))
            except (TypeError, ValueError):
                continue
            sugg_raw = it.get("suggested_value")
            delta_raw = it.get("delta")
            try:
                if sugg_raw is not None:
                    sugg = float(sugg_raw)
                    delta = (
                        float(delta_raw) if delta_raw is not None else (sugg - orig)
                    )
                elif delta_raw is not None:
                    delta = float(delta_raw)
                    sugg = round(orig - abs(delta), 2)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            if abs(delta) < 1e-9:
                continue

            entry = (
                _resolve_glossary_entry(glossary, raw_name)
                or _resolve_glossary_entry(glossary, _feature_lookup_name(raw_name))
                or _synthetic_entry(raw_name, feature_metadata)
            )
            label_default = _humanize_feature_name(raw_name)
            label = sanitize_for_prompt(str(entry.get(label_key, label_default)), max_len=_MAX_LABEL_LEN)
            unit = entry.get("unit")
            rng = entry.get("normal_range")
            normal_range = None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                normal_range = f"{rng[0]}–{rng[1]}{(' ' + unit) if unit else ''}"

            current_value = _format_value(orig, unit=unit)
            suggested_value = _format_value(sugg, unit=unit)
            direction: Direction = "increase" if delta > 0 else "decrease"
            # Plain-language change phrase rather than a bare signed number
            # (a reader can't interpret "−209.07 µU/mL"). Localised; no Unicode
            # sign so it renders identically on screen and in the PDF.
            amount = _format_value(abs(delta), unit=unit)
            if lang == "fr":
                magnitude_text = f"augmenter de {amount}" if delta > 0 else f"réduire de {amount}"
            else:
                magnitude_text = f"increase by {amount}" if delta > 0 else f"decrease by {amount}"

            out.append(
                CounterfactualChange(
                    label=label,
                    current_value=current_value,
                    suggested_value=suggested_value,
                    direction=direction,
                    magnitude_text=magnitude_text,
                    normal_range=normal_range,
                    training_reference=(
                        feature_metadata.get(raw_name, {}).get("training_reference")
                        or feature_metadata.get(_feature_lookup_name(raw_name), {}).get("training_reference")
                    ),
                )
            )
        return out
