"""Extract compact model context for patient-facing reports.

The reporting prompt should receive a curated summary, not raw training
artifacts. This module turns ``TrainedModel.metrics_json`` and
``TrainedModel.artifacts_json`` into short, already-formatted strings:

- class mapping (raw model label -> readable output)
- model quality signals (accuracy, recall, ROC AUC, threshold, ...)
- dataset summary (sample counts, split type)
- per-feature references (training mean/range + global importance)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from app.services.reporting.context_builder import (
    ModelQualitySignal,
    PredictionClassContext,
    sanitize_for_prompt,
)

if TYPE_CHECKING:
    from app.models.training import TrainedModel


@dataclass(frozen=True)
class ReportingModelContext:
    display_prediction: Any
    class_context: PredictionClassContext = field(default_factory=PredictionClassContext)
    model_quality: list[ModelQualitySignal] = field(default_factory=list)
    dataset_summary: list[str] = field(default_factory=list)
    feature_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


_PREFIX_RE = re.compile(r"^(?:num|cat|remainder)__", re.IGNORECASE)


def build_reporting_model_context(
    model: "TrainedModel",
    *,
    raw_prediction: Any,
    lang: str,
    glossary: Any = None,
) -> ReportingModelContext:
    artifacts = model.artifacts_json if isinstance(model.artifacts_json, dict) else {}
    metrics = model.metrics_json if isinstance(model.metrics_json, dict) else {}
    schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}

    class_ctx, display_prediction = _build_class_context(
        raw_prediction=raw_prediction,
        schema=schema,
        metrics=metrics,
        lang=lang,
    )
    feature_metadata = _build_feature_metadata(
        artifacts=artifacts,
        lang=lang,
        glossary=glossary,
    )

    return ReportingModelContext(
        display_prediction=display_prediction,
        class_context=class_ctx,
        model_quality=_build_model_quality(metrics, lang=lang),
        dataset_summary=_build_dataset_summary(artifacts, metrics, lang=lang),
        feature_metadata=feature_metadata,
    )


def _build_class_context(
    *,
    raw_prediction: Any,
    schema: dict[str, Any],
    metrics: dict[str, Any],
    lang: str,
) -> tuple[PredictionClassContext, Any]:
    raw = sanitize_for_prompt(str(raw_prediction), max_len=40)
    target = sanitize_for_prompt(str(schema.get("target", "")), max_len=80)

    positive_class = ""
    test_meta = _dig(metrics, "test", "meta") or _dig(metrics, "holdout_test_metrics", "meta") or {}
    if isinstance(test_meta, dict) and test_meta.get("positive_label") is not None:
        positive_class = sanitize_for_prompt(str(test_meta.get("positive_label")), max_len=40)

    # Determine whether the prediction is the positive or negative class.
    # We compare normalised forms so "1", "1.0", "True" all match a stored
    # positive_label of "1". No dataset-specific heuristics needed.
    pred_str = str(raw_prediction).strip()
    pred_norm = pred_str.lower().rstrip("0").rstrip(".")  # "1.0" → "1", "0.0" → "0"
    pos_norm = positive_class.lower().rstrip("0").rstrip(".") if positive_class else ""

    is_positive = bool(pos_norm and pred_norm == pos_norm)
    is_negative = bool(positive_class and not is_positive and pred_str not in {"", "none", "nan"})

    display_prediction: Any = raw_prediction
    label_meaning = ""
    if is_positive:
        label_meaning = "classe positive" if lang == "fr" else "positive class"
        display_prediction = (
            f"resultat positif suggere" if lang == "fr" else "positive result suggested"
        )
    elif is_negative:
        label_meaning = "classe negative" if lang == "fr" else "negative class"
        display_prediction = (
            f"resultat negatif suggere" if lang == "fr" else "negative result suggested"
        )

    return (
        PredictionClassContext(
            raw_label=raw,
            target_name=target,
            positive_class=positive_class,
            label_meaning=sanitize_for_prompt(label_meaning, max_len=120),
        ),
        display_prediction,
    )


def _build_model_quality(metrics: dict[str, Any], *, lang: str) -> list[ModelQualitySignal]:
    source = _metric_source(metrics)
    if not source:
        return []

    labels = _METRIC_LABELS.get(lang, _METRIC_LABELS["en"])
    interpretations = _METRIC_INTERPRETATIONS.get(lang, _METRIC_INTERPRETATIONS["en"])
    wanted = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision_pos",
        "recall_pos",
        "f1_pos",
    )

    out: list[ModelQualitySignal] = []
    for key in wanted:
        value = source.get(key)
        if value is None:
            continue
        out.append(
            ModelQualitySignal(
                label=labels.get(key, key),
                value=_format_metric(value),
                interpretation=interpretations.get(key, ""),
            )
        )

    threshold = _dig(metrics, "test", "meta", "threshold_used")
    if threshold is None:
        threshold = _dig(metrics, "holdout_test_metrics", "meta", "threshold_used")
    if threshold is not None:
        out.append(
            ModelQualitySignal(
                label="Seuil de decision" if lang == "fr" else "Decision threshold",
                value=_format_metric(threshold),
                interpretation=(
                    "score a partir duquel la classe positive est suggeree"
                    if lang == "fr"
                    else "score from which the positive class is suggested"
                ),
            )
        )

    return out[:7]


def _build_dataset_summary(
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    *,
    lang: str,
) -> list[str]:
    split = artifacts.get("split_info") if isinstance(artifacts.get("split_info"), dict) else {}
    summary: list[str] = []

    n_samples = split.get("n_samples")
    if n_samples is not None:
        summary.append(
            f"Jeu de donnees d'entrainement: {int(n_samples)} lignes"
            if lang == "fr"
            else f"Training dataset: {int(n_samples)} rows"
        )

    if split.get("method"):
        method = sanitize_for_prompt(str(split.get("method")), max_len=60)
        if split.get("k_folds"):
            summary.append(
                f"Validation: {method}, {int(split['k_folds'])} folds"
                if lang == "fr"
                else f"Validation: {method}, {int(split['k_folds'])} folds"
            )
        else:
            summary.append(f"Validation: {method}")

    test_rows = split.get("test_rows")
    if test_rows is not None:
        summary.append(
            f"Test final: {int(test_rows)} lignes conservees"
            if lang == "fr"
            else f"Final test set: {int(test_rows)} held-out rows"
        )

    confusion = _dig(metrics, "holdout_test_metrics", "confusion_matrix") or _dig(metrics, "test", "confusion_matrix")
    if isinstance(confusion, dict) and confusion.get("matrix"):
        summary.append(
            "Matrice de confusion disponible pour interpreter les erreurs du modele"
            if lang == "fr"
            else "Confusion matrix available to interpret model errors"
        )

    return [sanitize_for_prompt(s, max_len=180) for s in summary[:4]]


def _build_feature_metadata(
    *,
    artifacts: dict[str, Any],
    lang: str,
    glossary: Any,
) -> dict[str, dict[str, str]]:
    schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    column_stats = schema.get("column_stats") if isinstance(schema.get("column_stats"), dict) else {}
    importance = artifacts.get("feature_importance") if isinstance(artifacts.get("feature_importance"), list) else []

    importance_by_feature: dict[str, str] = {}
    total = len(importance)
    for idx, item in enumerate(importance, start=1):
        if not isinstance(item, dict):
            continue
        name = _clean_feature_name(str(item.get("feature", "")))
        if not name:
            continue
        value = item.get("importance")
        if value is None:
            text = f"rang global {idx}/{total}" if lang == "fr" else f"global rank {idx}/{total}"
        else:
            text = (
                f"rang global {idx}/{total}, importance relative {_format_metric(value)}"
                if lang == "fr"
                else f"global rank {idx}/{total}, relative importance {_format_metric(value)}"
            )
        importance_by_feature[_canonical(name)] = sanitize_for_prompt(text, max_len=160)

    out: dict[str, dict[str, str]] = {}
    for raw_name, stats in column_stats.items():
        if not isinstance(stats, dict):
            continue
        name = _clean_feature_name(str(raw_name))
        meta: dict[str, str] = {}
        reference = _format_training_reference(name, stats, lang=lang, glossary=glossary)
        if reference:
            meta["training_reference"] = reference
        imp = importance_by_feature.get(_canonical(name))
        if imp:
            meta["global_importance"] = imp
        _add_aliases(out, name, meta)

    # Importance can exist for features not present in column_stats.
    for canon_name, imp in importance_by_feature.items():
        out.setdefault(canon_name, {})["global_importance"] = imp

    return out


def _format_training_reference(
    raw_name: str,
    stats: dict[str, Any],
    *,
    lang: str,
    glossary: Any,
) -> str:
    unit = ""
    if hasattr(glossary, "get"):
        entry = glossary.get(raw_name)
        if isinstance(entry, dict) and entry.get("unit"):
            unit = f" {entry['unit']}"

    mean = stats.get("mean")
    min_v = stats.get("min")
    max_v = stats.get("max")
    parts: list[str] = []
    if mean is not None:
        parts.append(
            f"moyenne entrainement: {_format_number(mean)}{unit}"
            if lang == "fr"
            else f"training mean: {_format_number(mean)}{unit}"
        )
    if min_v is not None and max_v is not None:
        parts.append(
            f"intervalle observe: {_format_number(min_v)}-{_format_number(max_v)}{unit}"
            if lang == "fr"
            else f"observed range: {_format_number(min_v)}-{_format_number(max_v)}{unit}"
        )
    return sanitize_for_prompt("; ".join(parts), max_len=180)


def _metric_source(metrics: dict[str, Any]) -> dict[str, Any]:
    for candidate in (
        _dig(metrics, "holdout_test_metrics", "legacy_flat"),
        _dig(metrics, "test", "legacy_flat"),
        _dig(metrics, "cv_summary", "mean"),
        metrics.get("cv_mean"),
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


_METRIC_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "accuracy": "Exactitude globale",
        "balanced_accuracy": "Exactitude equilibree",
        "roc_auc": "ROC AUC",
        "precision_pos": "Precision classe positive",
        "recall_pos": "Rappel classe positive",
        "f1_pos": "F1 classe positive",
    },
    "en": {
        "accuracy": "Overall accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "roc_auc": "ROC AUC",
        "precision_pos": "Positive-class precision",
        "recall_pos": "Positive-class recall",
        "f1_pos": "Positive-class F1",
    },
}

_METRIC_INTERPRETATIONS: dict[str, dict[str, str]] = {
    "fr": {
        "accuracy": "part des predictions correctes",
        "balanced_accuracy": "tient compte du desequilibre entre classes",
        "roc_auc": "capacite a separer les classes",
        "precision_pos": "fiabilite des alertes positives",
        "recall_pos": "capacite a retrouver les cas positifs",
        "f1_pos": "compromis precision/rappel sur la classe positive",
    },
    "en": {
        "accuracy": "share of correct predictions",
        "balanced_accuracy": "accounts for class imbalance",
        "roc_auc": "ability to separate classes",
        "precision_pos": "reliability of positive alerts",
        "recall_pos": "ability to find positive cases",
        "f1_pos": "precision/recall tradeoff for the positive class",
    },
}


def _format_metric(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return sanitize_for_prompt(str(value), max_len=40)
    if 0 <= v <= 1:
        return f"{round(v * 100, 1)} %"
    return _format_number(v)


def _format_number(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return sanitize_for_prompt(str(value), max_len=40)
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e9:
        return str(int(round(v)))
    return f"{v:.2f}"


def _dig(data: dict[str, Any], *path: str) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _clean_feature_name(name: str) -> str:
    cleaned = _PREFIX_RE.sub("", name).strip()
    return sanitize_for_prompt(cleaned, max_len=120)


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _add_aliases(out: dict[str, dict[str, str]], name: str, meta: dict[str, str]) -> None:
    """Store metadata under the clean name + the three ColumnTransformer prefixes.

    LIME and SHAP may return feature names as ``num__Glucose``, ``cat__Glucose``,
    or ``remainder__Glucose`` depending on the sklearn pipeline configuration.
    Context_builder strips these prefixes before lookup, so storing clean-name
    entries is sufficient for the primary lookup path — the aliases catch
    callers that haven't stripped the prefix yet.
    """
    if not meta:
        return
    copy = dict(meta)
    for key in (name, _canonical(name), f"num__{name}", f"cat__{name}", f"remainder__{name}"):
        out[key] = copy


__all__ = ["ReportingModelContext", "build_reporting_model_context"]
