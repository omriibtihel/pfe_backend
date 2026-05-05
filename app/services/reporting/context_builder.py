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
    ) -> ReportContext:
        label = sanitize_for_prompt(str(prediction_value), max_len=_MAX_LABEL_LEN) or "—"
        confidence_text = bucket_confidence(score, lang)
        score_pct: Optional[str] = None
        if score is not None and task_type == "classification":
            score_pct = f"{round(float(score) * 100)} %"

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

        return ReportContext(
            prediction_id=sanitize_for_prompt(prediction_id, max_len=64) or "unknown",
            lang=lang,
            label=label,
            confidence_text=confidence_text,
            score_pct=score_pct,
            top_features=top_features,
            class_context=class_context or PredictionClassContext(),
            model_quality=model_quality or [],
            dataset_summary=dataset_summary or [],
            model_name=sanitize_for_prompt(model_name, max_len=80),
            model_version=sanitize_for_prompt(model_version, max_len=40),
            task_type=task_type,
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
        for _, raw_name, raw_value, entry in ranked[: max(0, top_n)]:
            label_key = "label_fr" if lang == "fr" else "label_en"
            label = sanitize_for_prompt(str(entry.get(label_key, raw_name)), max_len=_MAX_LABEL_LEN)
            unit = entry.get("unit")
            rng = entry.get("normal_range")
            normal_range = None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                normal_range = f"{rng[0]}–{rng[1]}{(' ' + unit) if unit else ''}"

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
        for raw_name, w, raw_value in normalized:
            lookup_name = _feature_lookup_name(raw_name)
            entry = _resolve_glossary_entry(glossary, lookup_name) or _resolve_glossary_entry(glossary, raw_name)
            label = sanitize_for_prompt(str(entry.get(label_key, lookup_name)), max_len=_MAX_LABEL_LEN)
            unit = entry.get("unit")
            rng = entry.get("normal_range")
            normal_range = None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                normal_range = f"{rng[0]}–{rng[1]}{(' ' + unit) if unit else ''}"

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
