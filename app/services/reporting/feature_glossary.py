"""Loader for ``feature_glossary.yaml``.

The glossary is loaded once per process and exposed as a tolerant lookup:
- exact key match (case-sensitive)
- case-insensitive fallback
- snake_case canonicalization fallback (``BloodPressure`` → ``bloodpressure``)

Missing entries return ``{}`` so the rest of the reporting pipeline degrades
gracefully (it falls back to the raw column name, see ``ReportContextBuilder``).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_GLOSSARY_PATH = Path(__file__).parent / "feature_glossary.yaml"

# Canonicalize "BloodPressure", "blood-pressure", "Blood Pressure" → "bloodpressure".
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _canonical(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())


def _load_yaml(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        logger.info("feature_glossary.missing path=%s — running without glossary", path)
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("feature_glossary.load_failed path=%s err=%s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("feature_glossary.invalid_root path=%s — expected mapping", path)
        return {}
    # Defensive coercion — yaml.safe_load may produce non-dict entries.
    out: dict[str, dict[str, Any]] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            out[str(key)] = val
    return out


@lru_cache(maxsize=1)
def _build_lookup() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Returns (exact_map, canonical_map). Cached for the process lifetime."""
    raw = _load_yaml(_GLOSSARY_PATH)
    canonical = {_canonical(k): v for k, v in raw.items()}
    return raw, canonical


class FeatureGlossary:
    """Tolerant glossary lookup. Stateless — wraps a cached dict."""

    def get(self, raw_name: str) -> dict[str, Any]:
        if not raw_name:
            return {}
        exact, canonical = _build_lookup()
        if raw_name in exact:
            return exact[raw_name]
        return canonical.get(_canonical(raw_name), {})

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Return the full glossary as the dict shape ReportContextBuilder
        expects (keyed by raw_name, with `_canonical` aliases included)."""
        exact, canonical = _build_lookup()
        # Merge with canonical aliases so callers can pass ANY name variant.
        merged: dict[str, dict[str, Any]] = dict(exact)
        for key, val in canonical.items():
            merged.setdefault(key, val)
        return merged

    @staticmethod
    def reload() -> None:
        """Force a re-read of the YAML — useful in tests after writing
        a custom glossary file."""
        _build_lookup.cache_clear()


# Module-level singleton — the glossary is small and read-only.
feature_glossary = FeatureGlossary()


__all__ = ["FeatureGlossary", "feature_glossary"]
