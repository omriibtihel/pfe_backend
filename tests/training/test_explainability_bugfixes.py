"""Regression tests for the explainability bug fixes (session 2026-06-03).

Covers four issues found while auditing SHAP / LIME / counterfactual logic:

  #1  Counterfactual inverse-scaling never applied — inv_scale_map is keyed by the
      original column ("Glucose") but looked up with the prefixed prep name
      ("num__Glucose"), so suggestions were shown in z-score space.
  #2  _get_preprocessed_feature_names ignored the post-prep `select`
      (VarianceThreshold) step, so dropping any column desynced names from the
      transformed matrix and every explainer fell back to f_0, f_1 …
  #3  LIME explained class index 1 blindly instead of the agreed positive class,
      so LIME and SHAP could show opposite directions.
  #4  get_positive_class_index didn't unwrap FLAML AutoML wrappers, so classes_
      was missing and it always returned 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.counterfactual.local_counterfactual import (
    _format_items,
    compute_counterfactual,
)
from app.services.shap._utils import get_positive_class_index
from app.services.shap.global_shap import (
    _get_preprocessed_feature_names,
    _transform_except_model,
)


# ══════════════════════════════════════════════════════════════════════════════
# #1 — Counterfactual inverse-scaling with prefixed prep names
# ══════════════════════════════════════════════════════════════════════════════

def test_format_items_inverse_scales_prefixed_prep_names():
    """inv_scale_map keyed by original name must still apply to 'num__' names."""
    # x_original = x_prep * 30 + 120   (StandardScaler with mean=120, scale=30)
    inv_scale_map = {"Glucose": (120.0, 30.0)}
    raw = [{"col": "num__Glucose", "col_idx": 0, "orig_prep": 0.0, "sugg_prep": 1.0,
            "delta_abs": 1.0}]

    items = _format_items(raw, inv_scale_map)

    assert len(items) == 1
    assert items[0]["feature"] == "Glucose"
    # Before the fix these were the raw z-scores (0.0 and 1.0).
    assert items[0]["original_value"] == pytest.approx(120.0)
    assert items[0]["suggested_value"] == pytest.approx(150.0)
    assert items[0]["delta"] == pytest.approx(30.0)


def _make_clinical_pipe(seed: int = 0):
    """Binary pipeline on clinical-scale features (NOT standard-normal)."""
    rng = np.random.default_rng(seed)
    n = 400
    glucose = rng.normal(120.0, 30.0, n)   # mg/dL
    bmi = rng.normal(28.0, 6.0, n)         # kg/m²
    # Higher glucose & BMI → positive (diabetic-like) label.
    logit = 0.06 * (glucose - 120.0) + 0.18 * (bmi - 28.0) + rng.normal(0, 0.3, n)
    y = (logit > 0).astype(int)
    df = pd.DataFrame({"Glucose": glucose, "BMI": bmi})
    prep = ColumnTransformer([("num", StandardScaler(), ["Glucose", "BMI"])])
    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=40, max_depth=5, random_state=seed)),
    ])
    pipe.fit(df, y)
    return pipe, df, y


def test_counterfactual_returns_clinical_units_not_zscores():
    """End-to-end: suggested/original values must be in original feature units.

    The counterfactual operates in StandardScaler space; without inverse-scaling
    a Glucose value would come back as a z-score (~0±3) instead of ~mg/dL (~100s).
    Since inverse-scaling is an exact reconstruction, original_value must equal
    the row's raw input value.
    """
    pipe, df, _ = _make_clinical_pipe()
    proba = pipe.predict_proba(df)[:, 1]
    row_idx = int(np.argmax(proba))            # most confidently-positive row
    row = df.iloc[[row_idx]]

    result = compute_counterfactual(
        pipe, row,
        features_to_vary=["Glucose", "BMI"],
        task_type="classification",
        feature_names=list(df.columns),
    )
    if result is None:
        pytest.skip("no flip found for this seed")

    raw_row = row.iloc[0]
    for item in result:
        feat = item["feature"]
        assert feat in ("Glucose", "BMI")
        # original_value reconstructs the raw input → clinical units, not z-score.
        assert item["original_value"] == pytest.approx(float(raw_row[feat]), rel=1e-6, abs=1e-6)
        # And it is clearly in clinical range, never a tiny z-score.
        if feat == "Glucose":
            assert abs(item["original_value"]) > 20.0


# ══════════════════════════════════════════════════════════════════════════════
# #2 — _get_preprocessed_feature_names respects the `select` step
# ══════════════════════════════════════════════════════════════════════════════

def test_preprocessed_feature_names_respects_select_step():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": np.full(n, 5.0),     # constant → zero variance after scaling
    })
    y = (df["a"] + df["b"] > 0).astype(int)

    prep = ColumnTransformer([("num", StandardScaler(), ["a", "b", "c", "d"])])
    pipe = Pipeline([
        ("prep", prep),
        ("select", VarianceThreshold(threshold=0.0)),
        ("model", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(df, y)

    names = _get_preprocessed_feature_names(pipe, list(df.columns))

    # The constant column is dropped by VarianceThreshold → must be absent, and
    # the count must match the matrix the model actually sees.
    transformed_width = _transform_except_model(pipe, df).shape[1]
    assert len(names) == transformed_width
    assert "num__d" not in names
    assert names == ["num__a", "num__b", "num__c"]


# ══════════════════════════════════════════════════════════════════════════════
# #4 — get_positive_class_index unwraps FLAML wrappers
# ══════════════════════════════════════════════════════════════════════════════

def test_get_positive_class_index_unwraps_flaml():
    class _Estimator:
        classes_ = np.array(["healthy", "diabetic"])

    class _FlamlBase:
        # flaml.AutoML shape: .model is a BaseEstimator whose .estimator is sklearn
        estimator = _Estimator()

    class _AutoML:
        model = _FlamlBase()

    automl = _AutoML()
    assert get_positive_class_index(automl, "diabetic") == 1
    assert get_positive_class_index(automl, "healthy") == 0
    # Unknown label / no positive_label → safe fallback to 1.
    assert get_positive_class_index(automl, None) == 1


# ══════════════════════════════════════════════════════════════════════════════
# #3 — LIME orients toward the agreed positive class
# ══════════════════════════════════════════════════════════════════════════════

def test_lime_orientation_follows_positive_label():
    pytest.importorskip("lime")
    from app.services.lime.local_lime import compute_local_lime

    pipe, df, _ = _make_clinical_pipe(seed=1)
    background = _transform_except_model(pipe, df)
    row = df.iloc[[int(np.argmax(pipe.predict_proba(df)[:, 1]))]]

    # classes_ == [0, 1]; positive_label=1 → class index 1, positive_label=0 → 0.
    items_pos = compute_local_lime(
        pipe, row, task_type="classification", background=background,
        feature_names=list(df.columns), random_state=0, positive_label=1,
    )
    items_neg = compute_local_lime(
        pipe, row, task_type="classification", background=background,
        feature_names=list(df.columns), random_state=0, positive_label=0,
    )
    assert items_pos is not None and items_neg is not None

    # Binary LIME: explaining class 1 vs class 0 yields exactly-negated weights.
    neg_by_feat = {it["feature"]: it["contribution"] for it in items_neg}
    top = items_pos[0]
    assert top["feature"] in neg_by_feat
    # Opposite sign (the whole point of orientation), magnitudes equal.
    assert top["contribution"] == pytest.approx(-neg_by_feat[top["feature"]], rel=1e-6, abs=1e-9)
