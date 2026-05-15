from __future__ import annotations

import pandas as pd

from app.services.data.loader import (
    _coerce_blank_strings_to_nan,
    _coerce_decimal_commas,
    load_dataframe,
)
from app.services.data.preview import build_validation_preview


def _preview_payload() -> dict:
    return {
        "datasetVersionId": 1,
        "targetColumn": "target",
        "taskType": "classification",
        "models": ["svm"],
        "metrics": ["accuracy"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "kFolds": 5,
        "useGridSearch": False,
        "useSmote": False,
        "preprocessing": {
            "defaults": {
                "numericImputation": "none",
                "numericScaling": "standard",
                "categoricalImputation": "none",
                "categoricalEncoding": "onehot",
            },
            "columns": {},
        },
    }


def test_coerce_decimal_commas_handles_pandas_string_dtype():
    df = pd.DataFrame(
        {
            "feature": pd.Series(["4,5", "2,5", None], dtype="string"),
            "target": [0, 1, 0],
        }
    )

    out = _coerce_decimal_commas(df.copy())

    assert pd.api.types.is_float_dtype(out["feature"])
    assert out.loc[0, "feature"] == 4.5
    assert out.loc[1, "feature"] == 2.5
    assert pd.isna(out.loc[2, "feature"])


def test_coerce_blank_strings_replaces_empty_and_whitespace_with_nan():
    # CSVs often contain a literal space or empty string instead of true NaN.
    # These must be normalised so they don't masquerade as categorical values
    # and so SimpleImputer(strategy="median") doesn't crash downstream.
    df = pd.DataFrame(
        {
            "age": ["45", " ", "60", "", "30"],
            "sex": ["male", "female", "", "  ", "male"],
            "already_numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out = _coerce_blank_strings_to_nan(df.copy())

    # Numeric-looking column: spaces/empties become NaN, real values preserved as strings
    assert out.loc[0, "age"] == "45"
    assert pd.isna(out.loc[1, "age"])
    assert out.loc[2, "age"] == "60"
    assert pd.isna(out.loc[3, "age"])
    # Categorical column: same treatment
    assert pd.isna(out.loc[2, "sex"])
    assert pd.isna(out.loc[3, "sex"])
    assert out.loc[0, "sex"] == "male"
    # Already-numeric column is untouched
    assert (out["already_numeric"] == df["already_numeric"]).all()


def test_load_dataframe_converts_whitespace_cells_to_nan(tmp_path):
    # End-to-end: a CSV with a blank cell in a numeric column must result in
    # a float64 column with NaN, not an object column with " ".  Otherwise the
    # downstream SimpleImputer(strategy="median") would crash with
    # "could not convert string to float: ' '".
    csv_path = tmp_path / "with_blanks.csv"
    csv_path.write_text(
        "\n".join(
            [
                "age,glucose,target",
                "45, ,0",
                "50,120,1",
                "60,98,0",
                "55,110,1",
            ]
        ),
        encoding="utf-8",
    )

    df = load_dataframe(csv_path)

    # The blank glucose cell must be NaN (not the string " ")
    assert pd.isna(df.loc[0, "glucose"])
    # And the column may be coerced numerically downstream without error
    numeric = pd.to_numeric(df["glucose"], errors="raise")
    assert numeric.iloc[1] == 120


def test_load_dataframe_allows_preview_with_decimal_comma_columns(tmp_path):
    csv_path = tmp_path / "decimal_commas.csv"
    csv_path.write_text(
        "\n".join(
            [
                "feature,secondary,target",
                '"4,5",10,0',
                '"2,5",11,1',
                '"3,5",12,0',
                '"1,5",13,1',
                '"4,0",14,0',
                '"2,0",15,1',
                '"3,0",16,0',
                '"1,0",17,1',
                '"4,8",18,0',
                '"2,8",19,1',
            ]
        ),
        encoding="utf-8",
    )

    df = load_dataframe(csv_path)
    out = build_validation_preview(
        {
            **_preview_payload(),
            "include": {"preview": True},
            "preview": {"subset": "train", "mode": "head", "n": 3},
        },
        df,
        dataset_version_id=1,
    )

    assert pd.api.types.is_float_dtype(df["feature"])
    assert any(name == "num__feature" for name in out["previewTransformed"]["columns"])
    assert len(out["previewTransformed"]["rows"]) == 3
    assert all(isinstance(row[0], float) for row in out["previewTransformed"]["rows"])
