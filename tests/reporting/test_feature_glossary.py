"""Sprint 3 unit tests for the feature glossary loader.

Covers:
- YAML parsing of the shipped file
- Tolerant lookup (case + canonical form)
- Graceful degradation when the file is missing or malformed
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.services.reporting import feature_glossary as glossary_mod
from app.services.reporting.feature_glossary import FeatureGlossary, _build_lookup


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a fresh cache; the loader is process-singleton."""
    _build_lookup.cache_clear()
    yield
    _build_lookup.cache_clear()


def test_shipped_glossary_loads_known_keys():
    g = FeatureGlossary()
    assert g.get("glucose")["label_fr"] == "Glycémie"
    assert g.get("bmi")["label_en"] == "Body Mass Index (BMI)"


def test_lookup_is_case_insensitive():
    g = FeatureGlossary()
    assert g.get("BMI")["label_fr"] == "Indice de masse corporelle (IMC)"
    assert g.get("Glucose_Fasting")["label_en"] == "Fasting glucose"


def test_lookup_canonicalizes_separators():
    """Underscores, dashes, spaces, and case all map to the canonical form."""
    g = FeatureGlossary()
    a = g.get("blood-pressure")
    b = g.get("Blood Pressure")
    c = g.get("BloodPressure")
    d = g.get("bloodpressure")
    assert a == b == c == d
    assert a.get("unit") == "mmHg"


def test_missing_key_returns_empty_dict():
    g = FeatureGlossary()
    assert g.get("never_seen_column") == {}


def test_missing_file_does_not_crash(tmp_path, monkeypatch):
    """A deployment without a glossary file should still boot cleanly."""
    fake_path = tmp_path / "no_such_file.yaml"
    monkeypatch.setattr(glossary_mod, "_GLOSSARY_PATH", fake_path)
    _build_lookup.cache_clear()
    assert FeatureGlossary().get("anything") == {}


def test_malformed_yaml_returns_empty(tmp_path, monkeypatch):
    bad = tmp_path / "broken.yaml"
    bad.write_text("not: valid: yaml: : :\n", encoding="utf-8")
    monkeypatch.setattr(glossary_mod, "_GLOSSARY_PATH", bad)
    _build_lookup.cache_clear()
    assert FeatureGlossary().get("anything") == {}


def test_root_must_be_mapping(tmp_path, monkeypatch):
    bad = tmp_path / "list.yaml"
    bad.write_text("- foo\n- bar\n", encoding="utf-8")
    monkeypatch.setattr(glossary_mod, "_GLOSSARY_PATH", bad)
    _build_lookup.cache_clear()
    assert FeatureGlossary().get("anything") == {}


def test_custom_yaml_overrides(tmp_path, monkeypatch):
    """Reload mechanism: the user can ship a project-specific glossary."""
    custom = tmp_path / "g.yaml"
    custom.write_text(
        textwrap.dedent(
            """
            my_feature:
              label_fr: "Ma feature"
              label_en: "My feature"
              unit: "u"
              normal_range: [0, 1]
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(glossary_mod, "_GLOSSARY_PATH", custom)
    _build_lookup.cache_clear()
    g = FeatureGlossary()
    entry = g.get("my_feature")
    assert entry["label_fr"] == "Ma feature"
    assert entry["normal_range"] == [0, 1]


def test_reload_invalidates_cache(tmp_path, monkeypatch):
    p = tmp_path / "g.yaml"
    p.write_text("a:\n  label_fr: 'A'\n  label_en: 'A'\n", encoding="utf-8")
    monkeypatch.setattr(glossary_mod, "_GLOSSARY_PATH", p)
    _build_lookup.cache_clear()
    g = FeatureGlossary()
    assert g.get("a")["label_fr"] == "A"
    # Edit on disk, then reload
    p.write_text("a:\n  label_fr: 'B'\n  label_en: 'B'\n", encoding="utf-8")
    g.reload()
    assert g.get("a")["label_fr"] == "B"
