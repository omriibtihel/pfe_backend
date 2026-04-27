from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from app.api.routes.version_schema import _build_columns_meta

STATS_FIELDS = {
    "skewness", "outlier_count", "outlier_ratio", "has_negative", "parasites",
    "min_val", "max_val", "mean_val", "median_val", "q1_val", "q3_val",
}


def _schema_row(inferred_kind: str = "numeric", override_kind=None, confidence: float = 0.9):
    row = MagicMock()
    row.inferred_kind = inferred_kind
    row.override_kind = override_kind
    row.confidence = confidence
    return row


class TestBuildColumnsMeta:
    # ── output shape ──────────────────────────────────────────────────────────

    def test_returns_total_rows(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _build_columns_meta(df, {"a": _schema_row()})
        assert result["total_rows"] == 3

    def test_returns_one_entry_per_column(self):
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        result = _build_columns_meta(df, {
            "a": _schema_row("numeric"),
            "b": _schema_row("categorical"),
        })
        assert len(result["columns"]) == 2

    # ── stats fields present ──────────────────────────────────────────────────

    def test_numeric_column_includes_all_5_stats_fields(self):
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = _build_columns_meta(df, {"score": _schema_row("numeric")})

        col = result["columns"][0]
        assert STATS_FIELDS.issubset(col.keys())

    def test_non_numeric_column_stats_are_none(self):
        df = pd.DataFrame({"city": ["Paris", "Lyon", "Nice", "Marseille"]})
        result = _build_columns_meta(df, {"city": _schema_row("categorical")})

        col = result["columns"][0]
        assert col["skewness"] is None
        assert col["outlier_count"] is None
        assert col["outlier_ratio"] is None
        assert col["has_negative"] is None
        assert col["min_val"] is None
        assert col["max_val"] is None
        assert col["mean_val"] is None
        assert col["median_val"] is None
        assert col["q1_val"] is None
        assert col["q3_val"] is None

    def test_numeric_column_stats_are_not_none_for_sufficient_data(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = _build_columns_meta(df, {"v": _schema_row("numeric")})

        col = result["columns"][0]
        assert col["skewness"] is not None
        assert col["outlier_count"] is not None
        assert col["outlier_ratio"] is not None
        assert col["has_negative"] is not None

    # ── inf handling ──────────────────────────────────────────────────────────

    def test_inf_column_does_not_crash(self):
        df = pd.DataFrame({"val": [1.0, 2.0, np.inf, -np.inf, 5.0, 6.0, 7.0, 8.0]})
        # Must not raise
        result = _build_columns_meta(df, {"val": _schema_row("numeric")})

        col = result["columns"][0]
        assert STATS_FIELDS.issubset(col.keys())

    def test_inf_replaced_before_parasite_detection(self):
        # Numeric dtype → detect_parasites returns None immediately (inf or not)
        df = pd.DataFrame({"val": [1.0, 2.0, np.inf, 4.0, 5.0]})
        result = _build_columns_meta(df, {"val": _schema_row("numeric")})

        col = result["columns"][0]
        assert col["parasites"] is None

    def test_inf_values_excluded_from_missing_count(self):
        # inf is replaced with NaN, so it counts as missing
        df = pd.DataFrame({"val": [1.0, 2.0, np.inf, 4.0, 5.0]})
        result = _build_columns_meta(df, {"val": _schema_row("numeric")})

        col = result["columns"][0]
        assert col["missing"] == 1  # one inf → one NaN → one missing

    # ── parasite detection ────────────────────────────────────────────────────

    def test_parasite_detected_in_mixed_string_column(self):
        # 4 numeric strings + 1 parasite → convertible_ratio = 0.8 >= 0.5
        df = pd.DataFrame({"amount": ["1.0", "2.0", "3.0", "4.0", "INVALID"]})
        result = _build_columns_meta(df, {"amount": _schema_row("other")})

        col = result["columns"][0]
        assert col["parasites"] is not None
        assert col["parasites"]["count"] == 1
        assert "INVALID" in col["parasites"]["distinct"]

    def test_no_parasite_for_clean_string_numeric_column(self):
        df = pd.DataFrame({"price": ["1.0", "2.0", "3.0", "4.0", "5.0"]})
        result = _build_columns_meta(df, {"price": _schema_row("other")})

        col = result["columns"][0]
        assert col["parasites"] is None

    def test_no_parasite_for_all_text_column(self):
        # All non-numeric → convertible_ratio < 0.5 → no parasites
        df = pd.DataFrame({"note": ["a", "b", "c", "d", "e"]})
        result = _build_columns_meta(df, {"note": _schema_row("text")})

        col = result["columns"][0]
        assert col["parasites"] is None

    # ── kind resolution ───────────────────────────────────────────────────────

    def test_override_kind_takes_precedence_over_inferred(self):
        df = pd.DataFrame({"label": [0, 1, 1, 0, 0]})
        result = _build_columns_meta(df, {"label": _schema_row("numeric", override_kind="binary")})

        col = result["columns"][0]
        assert col["kind"] == "binary"
        assert col["override_kind"] == "binary"
        assert col["inferred_kind"] == "numeric"

    def test_missing_schema_row_falls_back_to_other(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _build_columns_meta(df, {})  # no row for "x"

        col = result["columns"][0]
        assert col["kind"] == "other"
        assert col["confidence"] == 0.0

    # ── counts ────────────────────────────────────────────────────────────────

    def test_counts_accumulate_correctly(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        })
        result = _build_columns_meta(df, {
            "a": _schema_row("numeric"),
            "b": _schema_row("numeric"),
            "c": _schema_row("categorical"),
        })

        assert result["counts"]["numeric"] == 2
        assert result["counts"]["categorical"] == 1

    def test_invalid_effective_kind_falls_back_to_other_in_counts(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        row = _schema_row("numeric")
        row.inferred_kind = "unknown_kind"
        row.override_kind = None
        result = _build_columns_meta(df, {"x": row})

        assert result["counts"]["other"] == 1
