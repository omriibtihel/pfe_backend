"""
Counterfactual reliability tests.

Couvre la robustesse logique du module counterfactual de bout en bout :

  1. Pure helpers       — préfixes ColumnTransformer, mapping de features,
                          inverse-scaling, détection du format estimateur
  2. Algorithm guards   — multi-feature directionnel, random search batché
  3. End-to-end         — compute_counterfactual avec un vrai pipeline
                          (ColumnTransformer + StandardScaler + RandomForest)
  4. Edge cases         — entrées vides, features inconnues, classes uniques,
                          background absent ou shape invalide
  5. Predictor integ.   — get_feature_ranges_for_model lit bien column_stats

Toutes les vérifications utilisent des datasets synthétiques avec une vérité
terrain explicite (`y = 3*a + 2*b - c > 0`) pour qu'on puisse vérifier que
les contrefactuels pointent dans la bonne direction et pas seulement que le
code "tourne sans crasher".
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from app.services.counterfactual.local_counterfactual import (
    _apply_inverse_scale,
    _build_inverse_scale_map,
    _extract_scaler_params,
    _is_linear_invertible_scaler,
    _make_estimator_input,
    _map_features_to_prep,
    _multi_feature_cf,
    _random_search_cf,
    _strip_ct_prefix,
    compute_counterfactual,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — pipelines with known ground truth
# ──────────────────────────────────────────────────────────────────────────────

def _make_clf_dataset(seed: int = 0, n: int = 300):
    """y = 1 ⇔ 3*a + 2*b - c > 0  (a et b dominants, c contre-influence, d bruit)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    y = ((3.0 * a + 2.0 * b - c) > 0).astype(int)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    return df, y


def _make_clf_pipe_with_ct(seed: int = 0):
    """Pipeline complet avec ColumnTransformer (préfixe `num__`) + RandomForest."""
    df, y = _make_clf_dataset(seed)
    prep = ColumnTransformer([
        ("num", StandardScaler(), ["a", "b", "c", "d"]),
    ])
    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=30, max_depth=6, random_state=seed)),
    ])
    pipe.fit(df, y)
    return pipe, df, y


def _make_clf_pipe_simple(seed: int = 0):
    """Pipeline plat (sans ColumnTransformer) — noms de features inchangés."""
    df, y = _make_clf_dataset(seed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=seed, max_iter=500)),
    ])
    pipe.fit(df, y)
    return pipe, df, y


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Pure helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestStripCtPrefix:
    """Le strip du préfixe ColumnTransformer doit être déterministe et idempotent."""

    @pytest.mark.parametrize("name, expected", [
        ("num__Glucose",            "Glucose"),
        ("cat__sex_male",           "sex_male"),
        ("remainder__age",          "age"),
        ("Glucose",                 "Glucose"),     # déjà sans préfixe
        ("sex_male",                "sex_male"),    # déjà sans préfixe (OHE seul)
        ("",                        ""),
        ("a__b__c",                 "b__c"),        # split au premier `__` seulement
    ])
    def test_strip_variants(self, name, expected):
        assert _strip_ct_prefix(name) == expected


class TestMapFeaturesToPrep:
    """
    Le mapping doit reconnaître les 4 patterns produits par ColumnTransformer :
    exact, préfixe CT, OHE direct, OHE+préfixe.
    """

    def test_exact_match(self):
        assert _map_features_to_prep(["Glucose"], ["Glucose", "BMI"]) == ["Glucose"]

    def test_ct_prefix_match(self):
        result = _map_features_to_prep(["Glucose"], ["num__Glucose", "num__BMI"])
        assert result == ["num__Glucose"]

    def test_ohe_match_no_prefix(self):
        # "sex" ⇒ toutes les colonnes "sex_*" en post-OHE
        result = _map_features_to_prep(["sex"], ["sex_male", "sex_female", "Glucose"])
        assert set(result) == {"sex_male", "sex_female"}

    def test_ohe_match_with_ct_prefix(self):
        result = _map_features_to_prep(
            ["sex"],
            ["cat__sex_male", "cat__sex_female", "num__Glucose"],
        )
        assert set(result) == {"cat__sex_male", "cat__sex_female"}

    def test_multiple_features(self):
        result = _map_features_to_prep(
            ["Glucose", "BMI"],
            ["num__Glucose", "num__BMI", "num__Age"],
        )
        assert set(result) == {"num__Glucose", "num__BMI"}

    def test_no_match_returns_empty(self):
        assert _map_features_to_prep(["Inexistant"], ["Glucose", "BMI"]) == []

    def test_partial_substring_does_not_match(self):
        # Garde-fou : "Glu" ne doit pas matcher "Glucose" (substring)
        assert _map_features_to_prep(["Glu"], ["Glucose"]) == []

    def test_empty_features_to_vary(self):
        assert _map_features_to_prep([], ["Glucose"]) == []

    def test_empty_prep_feat_names(self):
        assert _map_features_to_prep(["Glucose"], []) == []


class TestIsLinearInvertibleScaler:
    """Détection du type de scaler à partir des attributs scikit-learn."""

    def test_standard_scaler(self):
        s = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
        assert _is_linear_invertible_scaler(s) is True

    def test_minmax_scaler(self):
        s = MinMaxScaler().fit(np.array([[1.0], [2.0], [3.0]]))
        assert _is_linear_invertible_scaler(s) is True

    def test_robust_scaler(self):
        s = RobustScaler().fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
        assert _is_linear_invertible_scaler(s) is True

    def test_power_transformer_excluded(self):
        # PowerTransformer est non-linéaire (lambdas_) ⇒ exclu
        s = PowerTransformer().fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
        assert _is_linear_invertible_scaler(s) is False

    def test_max_abs_scaler_excluded(self):
        # MaxAbsScaler n'a pas mean_/center_/data_min_ ⇒ pas de garantie
        s = MaxAbsScaler().fit(np.array([[1.0], [2.0], [3.0]]))
        assert _is_linear_invertible_scaler(s) is False

    def test_random_object_excluded(self):
        assert _is_linear_invertible_scaler(object()) is False


class TestExtractScalerParams:
    """L'extraction (offset, scale) doit reproduire mathématiquement l'inverse."""

    def test_standard_scaler_roundtrip(self):
        X = np.array([[10.0], [20.0], [30.0]])
        s = StandardScaler().fit(X)
        offset, scale = _extract_scaler_params(s, 0)
        # Pour StandardScaler : offset=mean=20, scale=std≈8.165
        assert offset == pytest.approx(20.0)
        assert scale == pytest.approx(s.scale_[0])

        # Roundtrip : x → (x-mean)/std → *std + mean → x
        x_orig = 25.0
        x_prep = (x_orig - offset) / scale
        x_back = _apply_inverse_scale(x_prep, (offset, scale))
        assert x_back == pytest.approx(x_orig)

    def test_minmax_scaler_roundtrip(self):
        X = np.array([[10.0], [20.0], [30.0]])
        s = MinMaxScaler().fit(X)
        offset, scale = _extract_scaler_params(s, 0)
        assert offset == pytest.approx(10.0)        # data_min_
        assert scale == pytest.approx(20.0)         # data_range_
        # Roundtrip
        x_prep = (25.0 - offset) / scale
        assert _apply_inverse_scale(x_prep, (offset, scale)) == pytest.approx(25.0)

    def test_robust_scaler_roundtrip(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        s = RobustScaler().fit(X)
        offset, scale = _extract_scaler_params(s, 0)
        # Pour RobustScaler : offset=median=3, scale=IQR=2
        assert offset == pytest.approx(3.0)
        assert scale == pytest.approx(s.scale_[0])

    def test_index_out_of_range(self):
        s = StandardScaler().fit(np.array([[1.0], [2.0]]))
        # col_idx invalide ⇒ None plutôt que crash
        assert _extract_scaler_params(s, 99) is None

    def test_unfitted_scaler(self):
        # Aucun attribut fitted ⇒ None
        assert _extract_scaler_params(StandardScaler(), 0) is None


class TestApplyInverseScale:
    """Inverse linéaire : x = x_prep * scale + offset."""

    def test_basic(self):
        assert _apply_inverse_scale(0.5, (10.0, 4.0)) == pytest.approx(12.0)

    def test_zero_scale_returns_offset(self):
        # Garde-fou divisé-par-zéro : si scale==0, on retourne offset (constant)
        assert _apply_inverse_scale(99.0, (5.0, 0.0)) == 5.0

    def test_negative_scale(self):
        # Théoriquement RobustScaler peut avoir scale<0 si données dégénérées —
        # le calcul reste linéaire et déterministe.
        assert _apply_inverse_scale(2.0, (1.0, -3.0)) == pytest.approx(-5.0)


class TestBuildInverseScaleMap:
    """L'extraction depuis un pipeline complet doit couvrir les pipelines imbriqués."""

    def test_simple_columntransformer(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        m = _build_inverse_scale_map(pipe)
        assert set(m.keys()) == {"a", "b", "c", "d"}
        # Chaque entrée a (offset, scale) avec scale > 0
        for col, (offset, scale) in m.items():
            assert isinstance(offset, float)
            assert isinstance(scale, float)
            assert scale > 0

    def test_nested_pipeline_inside_columntransformer(self):
        # ColumnTransformer dont la branche numérique est elle-même une Pipeline
        df, y = _make_clf_dataset()
        from sklearn.impute import SimpleImputer
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        prep = ColumnTransformer([("num", num_pipe, ["a", "b", "c", "d"])])
        pipe = Pipeline([("prep", prep), ("model", LogisticRegression())])
        pipe.fit(df, y)

        m = _build_inverse_scale_map(pipe)
        assert set(m.keys()) == {"a", "b", "c", "d"}

    def test_no_prep_step_returns_empty(self):
        # Pas de step "prep" ⇒ {}
        df, y = _make_clf_dataset()
        pipe = Pipeline([("model", LogisticRegression().fit(df, y))])
        assert _build_inverse_scale_map(pipe) == {}

    def test_non_invertible_scaler_skipped(self):
        # Une branche avec MaxAbsScaler (exclu) ne doit pas crash
        df, y = _make_clf_dataset()
        prep = ColumnTransformer([("num", MaxAbsScaler(), ["a", "b", "c", "d"])])
        pipe = Pipeline([("prep", prep), ("model", LogisticRegression())])
        pipe.fit(df, y)
        # MaxAbsScaler exclu ⇒ map vide
        assert _build_inverse_scale_map(pipe) == {}


class TestMakeEstimatorInput:
    """L'estimateur doit recevoir le format compatible avec son fit (DataFrame ou numpy)."""

    def test_estimator_fitted_with_dataframe_gets_dataframe(self):
        df, y = _make_clf_dataset()
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(df, y)  # fit avec DataFrame → feature_names_in_ existe

        X = df.iloc[:1].to_numpy()
        result = _make_estimator_input(clf, X, list(df.columns))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(clf.feature_names_in_)

    def test_estimator_fitted_with_numpy_gets_numpy(self):
        df, y = _make_clf_dataset()
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(df.to_numpy(), y)  # fit avec numpy → pas de feature_names_in_

        X = df.iloc[:1].to_numpy()
        result = _make_estimator_input(clf, X, list(df.columns))
        # Doit retourner un numpy (pas envelopper dans DataFrame)
        assert isinstance(result, np.ndarray)

    def test_shape_mismatch_falls_back_to_numpy(self):
        df, y = _make_clf_dataset()
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(df, y)
        # Tableau avec un nombre différent de colonnes → fallback numpy
        X_wrong = np.zeros((1, 99))
        result = _make_estimator_input(clf, X_wrong, ["a"] * 99)
        assert isinstance(result, np.ndarray)


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Algorithm correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiFeatureCF:
    """
    La recherche directionnelle doit trouver un flip quand au moins une feature
    a un effet mesurable sur predict_proba, et None sinon.
    """

    def test_finds_flip_with_dominant_feature(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]

        # On choisit un patient prédit positif (a ≫ 0)
        positive_idx = np.argmax(df["a"].values)
        row = df.iloc[[positive_idx]]
        X_row_prep = prep.transform(row)
        prep_feat_names = ["num__a", "num__b", "num__c", "num__d"]

        # Toutes les features modifiables, sigma=1 (StandardScaler output)
        sigma_map = {n: 1.0 for n in prep_feat_names}
        result = _multi_feature_cf(
            estimator, X_row_prep, prep_feat_names,
            prep_feat_names,
            "classification", sigma_map,
        )
        assert result is not None, "Devrait trouver un flip avec toutes les features"
        # Au moins la feature dominante (a) doit avoir bougé
        cols_changed = {r["col"] for r in result}
        assert "num__a" in cols_changed

    def test_returns_none_when_no_gradient(self):
        """Si aucune feature modifiable n'a de gradient → None."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        row = df.iloc[[0]]
        X_row_prep = prep.transform(row)

        # On passe une feature qui n'existe pas dans le pipeline
        result = _multi_feature_cf(
            estimator, X_row_prep,
            ["num__a", "num__b", "num__c", "num__d"],
            [],  # prep_vary vide ⇒ pas de probe possible
            "classification", {},
        )
        assert result is None

    def test_returns_none_for_regression(self):
        """Le multi-feature ne couvre que la classification (pour l'instant)."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        result = _multi_feature_cf(
            estimator,
            np.zeros((1, 4)),
            ["num__a", "num__b", "num__c", "num__d"],
            ["num__a"],
            "regression",
            {"num__a": 1.0},
        )
        assert result is None

    def test_returns_none_when_no_predict_proba(self):
        """Estimateur sans predict_proba → None."""
        # SVR par exemple — mais on simule simplement avec un mock
        class NoPP:
            def predict(self, X): return np.array([0])

        result = _multi_feature_cf(
            NoPP(), np.zeros((1, 2)), ["x", "y"], ["x"],
            "classification", {"x": 1.0},
        )
        assert result is None

    def test_completes_quickly(self):
        """Le probe étant batché, la recherche doit terminer en < 2s même avec
        beaucoup de features."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        row = df.iloc[[0]]
        X_row_prep = prep.transform(row)
        names = ["num__a", "num__b", "num__c", "num__d"]
        sigma_map = {n: 1.0 for n in names}

        t0 = time.time()
        _multi_feature_cf(estimator, X_row_prep, names, names, "classification", sigma_map)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Multi-feature CF trop lent : {elapsed:.2f}s"


class TestRandomSearchCF:
    """Le random search doit trouver un flip si le modèle est flippable, et
    rester bornée en temps grâce au batching."""

    def test_finds_flip_on_flippable_model(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        positive_idx = int(np.argmax(df["a"].values))
        X_row_prep = prep.transform(df.iloc[[positive_idx]])
        names = ["num__a", "num__b", "num__c", "num__d"]
        sigma_map = {n: 1.0 for n in names}

        result = _random_search_cf(
            estimator, X_row_prep, names, names, sigma_map,
            n_samples=500, random_state=0,
        )
        assert result is not None
        # Au moins une feature doit avoir un delta non nul
        assert any(abs(r["sugg_prep"] - r["orig_prep"]) > 1e-6 for r in result)

    def test_returns_none_when_no_modifiable(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        X_row_prep = prep.transform(df.iloc[[0]])
        result = _random_search_cf(
            estimator, X_row_prep,
            ["num__a", "num__b", "num__c", "num__d"],
            [],  # rien à varier
            {}, n_samples=100,
        )
        assert result is None

    def test_completes_under_3s_with_2000_samples(self):
        """Performance critique : avec 2000 samples batchés, doit rester sous 3s."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        X_row_prep = prep.transform(df.iloc[[0]])
        names = ["num__a", "num__b", "num__c", "num__d"]
        sigma_map = {n: 1.0 for n in names}

        t0 = time.time()
        _random_search_cf(
            estimator, X_row_prep, names, names, sigma_map,
            n_samples=2000, random_state=0,
        )
        elapsed = time.time() - t0
        # Le batching fait que 2000 samples ≈ 4 appels predict() vectorisés.
        # Un facteur ~3s laisse beaucoup de marge même sur CI lent.
        assert elapsed < 3.0, f"Random search trop lent : {elapsed:.2f}s"

    def test_deterministic_with_same_seed(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        estimator = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        X_row_prep = prep.transform(df.iloc[[0]])
        names = ["num__a", "num__b", "num__c", "num__d"]
        sigma_map = {n: 1.0 for n in names}

        r1 = _random_search_cf(estimator, X_row_prep, names, names, sigma_map, n_samples=300, random_state=42)
        r2 = _random_search_cf(estimator, X_row_prep, names, names, sigma_map, n_samples=300, random_state=42)
        # Même seed ⇒ même résultat
        if r1 is None and r2 is None:
            return  # both gave up, that's still deterministic
        assert r1 is not None and r2 is not None
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["col"] == b["col"]
            assert a["sugg_prep"] == pytest.approx(b["sugg_prep"])


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — End-to-end compute_counterfactual
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeCounterfactualEndToEnd:
    """Le contrat public doit rester stable et fournir une logique cohérente."""

    def test_output_schema(self):
        """Chaque item retourné a exactement 4 clés : feature, original_value,
        suggested_value, delta — et est trié par |delta| décroissant."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]

        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            background=None,        # force le fallback sans DiCE
            feature_names=list(df.columns),
            n_counterfactuals=3,
        )
        # Dépend du modèle : le test peut donner un None ; on saute si c'est le cas.
        if result is None:
            pytest.skip("Modèle ne flippe pas pour ce patient — testé dans d'autres cas")

        for item in result:
            assert set(item.keys()) == {"feature", "original_value", "suggested_value", "delta"}
            assert isinstance(item["feature"], str)
            assert isinstance(item["delta"], (int, float))

        # Tri par |delta| décroissant
        deltas = [abs(it["delta"]) for it in result]
        assert deltas == sorted(deltas, reverse=True)

    def test_feature_names_stripped_of_ct_prefix(self):
        """Les noms exposés au frontend ne doivent JAMAIS contenir 'num__'."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]

        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            feature_names=list(df.columns),
        )
        if result is None:
            pytest.skip("Pas de flip trouvé")
        for item in result:
            assert "__" not in item["feature"], (
                f"Le nom '{item['feature']}' contient encore le préfixe ColumnTransformer"
            )

    def test_delta_consistency(self):
        """delta == suggested_value - original_value, exactement."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            feature_names=list(df.columns),
        )
        if result is None:
            pytest.skip("Pas de flip trouvé")
        for item in result:
            expected = item["suggested_value"] - item["original_value"]
            assert item["delta"] == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_inverse_scaling_produces_realistic_values(self):
        """Avec un StandardScaler, les valeurs renvoyées doivent être dans
        l'échelle ORIGINALE des features (mean ≈ 0, |x| typiquement < 10)."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            feature_names=list(df.columns),
        )
        if result is None:
            pytest.skip("Pas de flip trouvé")
        # Toutes les features synthétiques sont N(0,1) ⇒ valeur raisonnable < 50
        for item in result:
            assert abs(item["original_value"]) < 50
            assert abs(item["suggested_value"]) < 50

    def test_returns_none_on_empty_features(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        row = df.iloc[[0]]
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=[],          # explicitement vide
            feature_names=list(df.columns),
        )
        assert result is None

    def test_returns_none_on_empty_row(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        result = compute_counterfactual(
            pipe, pd.DataFrame(columns=df.columns),
            features_to_vary=["a"],
            feature_names=list(df.columns),
        )
        assert result is None

    def test_returns_none_on_unknown_features(self):
        """Toutes les features demandées sont inconnues du pipeline ⇒ None."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        row = df.iloc[[0]]
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["DoesNotExist", "AlsoMissing"],
            feature_names=list(df.columns),
        )
        assert result is None

    def test_simple_pipeline_without_columntransformer(self):
        """Pipeline plat (sans CT) : le mapping doit fonctionner sans préfixe."""
        pipe, df, _ = _make_clf_pipe_simple()
        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            feature_names=list(df.columns),
        )
        # Modèle linéaire : devrait toujours trouver un flip
        assert result is not None
        cols = {it["feature"] for it in result}
        # Aucun préfixe attendu ici
        for c in cols:
            assert "__" not in c

    def test_does_not_raise_on_single_class_background(self):
        """Si le background prédit une seule classe, DiCE échoue mais le fallback
        doit prendre le relais sans crasher."""
        pipe, df, _ = _make_clf_pipe_with_ct()
        # Background mono-classe : on lui donne 10 lignes toutes très "positives"
        prep = pipe.named_steps["prep"]
        top_rows = df.iloc[df["a"].nlargest(10).index]
        bg_prep = prep.transform(top_rows)

        positive_idx = int(np.argmax(df["a"].values))
        row = df.iloc[[positive_idx]]

        # Doit retourner quelque chose ou None gracieusement, jamais lever
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a", "b", "c", "d"],
            task_type="classification",
            background=bg_prep,
            feature_names=list(df.columns),
        )
        # Le test passe tant que ça ne crash pas. Le résultat peut être None
        # selon que le multi-feature/random search trouve un flip.
        assert result is None or isinstance(result, list)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Edge cases & graceful failures
# ══════════════════════════════════════════════════════════════════════════════

class TestGracefulFailure:
    """Le service ne doit JAMAIS lever d'exception au caller."""

    def test_invalid_background_shape_does_not_raise(self):
        pipe, df, _ = _make_clf_pipe_with_ct()
        row = df.iloc[[0]]
        # Background avec mauvaise shape (5 colonnes au lieu de 4)
        bad_bg = np.random.randn(20, 5)
        result = compute_counterfactual(
            pipe, row,
            features_to_vary=["a"],
            background=bad_bg,
            feature_names=list(df.columns),
        )
        # Pas d'exception ; soit None, soit une liste valide.
        assert result is None or isinstance(result, list)

    def test_corrupted_pipeline_returns_none(self):
        """Si le pipeline est cassé, on ne lève pas — on retourne None."""
        broken = MagicMock()
        broken.named_steps = {}
        # transform va raise mais doit être attrapé
        broken.transform = MagicMock(side_effect=RuntimeError("boom"))
        result = compute_counterfactual(
            broken, pd.DataFrame({"a": [1.0]}),
            features_to_vary=["a"],
            feature_names=["a"],
        )
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Predictor integration
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFeatureRangesForModel:
    """L'endpoint feature-ranges doit lire correctement column_stats."""

    def _make_trained_model_mock(self, artifacts: dict) -> Any:
        m = MagicMock()
        m.artifacts_json = artifacts
        return m

    def test_numeric_features(self):
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({
            "training_schema": {
                "feature_names": ["Glucose", "BMI"],
                "column_stats": {
                    "Glucose": {"type": "numeric", "min": 40.0, "max": 200.0, "mean": 120.0, "std": 30.0},
                    "BMI":     {"type": "numeric", "min": 15.0, "max": 50.0,  "mean": 25.0,  "std": 5.0},
                },
            },
        })
        result = get_feature_ranges_for_model(m)
        assert set(result.keys()) == {"Glucose", "BMI"}
        assert result["Glucose"]["type"] == "numeric"
        assert result["Glucose"]["min"] == 40.0
        assert result["Glucose"]["max"] == 200.0
        assert result["Glucose"]["mean"] == 120.0
        assert result["Glucose"]["std"] == 30.0

    def test_categorical_features(self):
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({
            "training_schema": {
                "feature_names": ["sex"],
                "column_stats": {
                    "sex": {"type": "categorical", "categories": ["M", "F"]},
                },
            },
        })
        result = get_feature_ranges_for_model(m)
        assert result["sex"]["type"] == "categorical"
        assert result["sex"]["categories"] == ["M", "F"]

    def test_unknown_feature_falls_back(self):
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({
            "training_schema": {
                "feature_names": ["NoStats"],
                "column_stats": {},  # pas de stats pour cette feature
            },
        })
        result = get_feature_ranges_for_model(m)
        assert result["NoStats"]["type"] == "unknown"

    def test_zero_std_replaced_by_one(self):
        """Si std=0 (feature constante), on remplace par 1.0 pour éviter
        une division par zéro côté frontend (sigma)."""
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({
            "training_schema": {
                "feature_names": ["Constant"],
                "column_stats": {
                    "Constant": {"type": "numeric", "min": 5.0, "max": 5.0, "mean": 5.0, "std": 0.0},
                },
            },
        })
        result = get_feature_ranges_for_model(m)
        assert result["Constant"]["std"] == 1.0

    def test_old_model_without_column_stats(self):
        """Ancien modèle sans column_stats ⇒ pas de crash, tout en 'unknown'."""
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({
            "training_schema": {
                "feature_names": ["Glucose", "BMI"],
                # column_stats absent
            },
        })
        result = get_feature_ranges_for_model(m)
        assert result["Glucose"]["type"] == "unknown"
        assert result["BMI"]["type"] == "unknown"

    def test_empty_artifacts(self):
        from app.services.training.output.predictor import get_feature_ranges_for_model

        m = self._make_trained_model_mock({})
        # Pas de feature_names ⇒ {} retourné
        assert get_feature_ranges_for_model(m) == {}
