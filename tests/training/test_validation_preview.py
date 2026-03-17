from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.data.preview import (
    PreviewValidationError,
    _to_json_rows,
    build_validation_preview,
    clear_validation_preview_caches,
)


def _make_payload(*, val_ratio: int = 15, preview_subset: str = "train", preview_mode: str = "head", preview_n: int = 25) -> dict:
    return {
        "datasetVersionId": 1,
        "targetColumn": "Outcome",
        "taskType": "classification",
        "models": ["randomforest"],
        "metrics": ["accuracy"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": val_ratio,
        "testRatio": 30 - val_ratio,
        "kFolds": 5,
        "useGridSearch": False,
        "useSmote": False,
        "preprocessing": {
            "defaults": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "none",
                "categoricalEncoding": "none",
            },
            "columns": {},
        },
        "include": {"preview": True},
        "preview": {
            "subset": preview_subset,
            "mode": preview_mode,
            "n": preview_n,
            "seed": 42,
        },
    }


def _make_df(rows: int = 140) -> pd.DataFrame:
    idx = np.arange(rows)
    return pd.DataFrame(
        {
            "age": (18 + idx).astype(float),
            "bp": (120 + (idx % 17)).astype(float),
            "Outcome": np.where(idx % 2 == 0, 0, 1),
        }
    )


def test_validate_preview_returns_table():
    clear_validation_preview_caches()
    df = _make_df()
    out = build_validation_preview(_make_payload(preview_n=40), df, dataset_version_id=1)

    preview = out.get("previewTransformed") or {}
    columns = preview.get("columns") or []
    rows = preview.get("rows") or []
    meta = out.get("previewMeta") or {}

    assert isinstance(columns, list) and len(columns) > 0
    assert isinstance(rows, list) and len(rows) <= 40
    assert all(isinstance(row, list) and len(row) == len(columns) for row in rows)
    assert meta.get("subset") == "train"
    assert meta.get("fittedOn") == "train"


def test_validate_preview_fit_on_train_deterministic():
    clear_validation_preview_caches()
    df = _make_df()
    payload = _make_payload(preview_n=30, preview_mode="head")

    out1 = build_validation_preview(payload, df, dataset_version_id=7)
    out2 = build_validation_preview(payload, df, dataset_version_id=7)

    assert out1["previewTransformed"] == out2["previewTransformed"]
    assert out1["previewMeta"]["fromCache"] is False
    assert out2["previewMeta"]["fromCache"] is True
    assert out1["previewMeta"]["splitSeed"] == 42


def test_validate_preview_subset_unavailable():
    clear_validation_preview_caches()
    df = _make_df()
    payload = _make_payload(val_ratio=0, preview_subset="val")

    with pytest.raises(PreviewValidationError) as exc_info:
        build_validation_preview(payload, df, dataset_version_id=9)

    assert "unavailable" in str(exc_info.value).lower()
    assert exc_info.value.code == "preview_subset_unavailable"


def test_preview_json_serialization_keeps_float_domain():
    transformed = np.array([[np.int64(1), np.float64(0.4090909), np.int64(0)]], dtype=object)
    rows = _to_json_rows(transformed)

    assert rows == [[1.0, 0.4090909, 0.0]]
    assert [type(v).__name__ for v in rows[0]] == ["float", "float", "float"]
