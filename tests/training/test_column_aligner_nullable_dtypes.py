"""Regression tests for pandas nullable / extension dtypes reaching sklearn.

Pandas >= 3.0 reads text CSV columns as the nullable ``StringDtype`` (carrying
``pd.NA``) instead of ``object`` + ``np.nan``, and ``ColumnAligner._coerce_dtype``
emits nullable ``boolean`` / ``string`` columns. Both feed ``pd.NA`` into the
preprocessing ``ColumnTransformer``, where sklearn's ``check_array`` runs
``float(pd.NA)`` and raises::

    TypeError: float() argument must be a string or a real number, not 'NAType'

This used to fail *every* CV fold for *every* model. ``ColumnAligner.transform``
now denullifies its output so the rest of the pipeline sees the numpy model it
was written against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.preparation_ml.preprocessing.transformers import (
    ColumnAligner,
    _denullify_for_sklearn,
)
from app.services.preparation_ml.preprocessing import build_preprocessor
from app.services.training.config.schema import TrainingConfig


def _payload(model: str = "randomforest") -> dict:
    return {
        "targetColumn": "Outcome",
        "taskType": "classification",
        "models": [model],
        "metrics": ["accuracy", "f1"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "kFolds": 5,
        "useGridSearch": False,
        "useSmote": False,
        "preprocessing": {
            "defaults": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "constant",
                "categoricalEncoding": "onehot",
            },
            "columns": {},
        },
    }


def test_float_na_reproduces_the_original_crash():
    """Guards the assumption the whole fix rests on: float(pd.NA) raises."""
    with pytest.raises(TypeError, match="NAType"):
        float(pd.NA)


def test_denullify_strips_all_extension_dtypes():
    df = pd.DataFrame(
        {
            "num_as_string": pd.Series(["1.5", "2.0", pd.NA, "4.0"], dtype="string"),
            "flag": pd.Series([True, False, pd.NA, True], dtype="boolean"),
            "count": pd.Series([1, 2, pd.NA, 4], dtype="Int64"),
            "label": pd.Series(["a", pd.NA, "b", "a"], dtype="string"),
        }
    )

    out = _denullify_for_sklearn(df.copy())

    # No pandas extension dtype survives.
    assert not any(pd.api.types.is_extension_array_dtype(out[c].dtype) for c in out.columns)
    # No pd.NA survives — every missing cell is a real np.nan.
    assert not any(
        (v is pd.NA) for col in out.columns for v in out[col].tolist()
    )
    # Nullable boolean / integer became numpy float (pd.NA → np.nan).
    assert out["flag"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(out["flag"].iloc[2])
    assert out["count"].iloc[3] == 4.0
    # The numeric-looking string column can now be cast to float by sklearn.
    np.asarray(out["num_as_string"], dtype=float)


def test_column_aligner_transform_emits_no_extension_dtypes():
    raw = pd.DataFrame(
        {
            "num_as_string": pd.Series(["1.5", "2.0", pd.NA, "4.0"], dtype="string"),
            "city": pd.Series(["A", "B", pd.NA, "A"], dtype="string"),
        }
    )
    aligner = ColumnAligner(
        feature_names=["num_as_string", "city"],
        dtypes={"num_as_string": "float64", "city": "object"},
    )
    out = aligner.fit_transform(raw)
    assert not any(pd.api.types.is_extension_array_dtype(out[c].dtype) for c in out.columns)
    # float-castable numeric column → no NAType
    np.asarray(pd.to_numeric(out["num_as_string"], errors="coerce"), dtype=float)


def test_preprocessor_fit_does_not_crash_on_nullable_string_numeric_column():
    """End-to-end: align → build_preprocessor → fit_transform must not raise.

    Before the fix this raised TypeError('... not NAType') inside the numeric
    SimpleImputer for *every* model.
    """
    n = 12
    rng = np.random.default_rng(0)
    # A genuinely-numeric feature pandas 3.0 would surface as StringDtype because
    # of a couple of blank cells (→ pd.NA).
    vals = [f"{v:.1f}" for v in rng.normal(size=n)]
    vals[3] = pd.NA
    vals[7] = pd.NA
    df = pd.DataFrame(
        {
            "weight": pd.Series(vals, dtype="string"),
            "city": pd.Series(["A", "B", "C"] * 4, dtype="string"),
            "Outcome": [0, 1] * 6,
        }
    )
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].to_numpy()

    aligner = ColumnAligner(
        feature_names=list(X.columns),
        dtypes={c: str(dt) for c, dt in X.dtypes.items()},
    )
    X_aligned = aligner.fit_transform(X)

    cfg = TrainingConfig.from_front(_payload("randomforest"))
    spec = build_preprocessor(X_aligned, cfg.preprocessing)
    transformed = spec.preprocessor.fit_transform(X_aligned, y)

    # Imputation filled the two missing weights → finite numeric matrix.
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert np.isfinite(dense).all()
    assert dense.shape[0] == n
