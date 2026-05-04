"""
End-to-end reliability audit of the explainability module.

Sections
--------
1. Source des données           — provenance et fidélité des inputs (background, names, raw values)
2. Cohérence SHAP               — vérité terrain, propriété d'efficacité Shapley, régression GridSearch
3. Cohérence LIME               — top-K, tri, stabilité
4. Mode both                    — co-existence SHAP + LIME, dégradation indépendante
5. Compatibilité descendante    — défaut method=shap, alias predict_with_shap

Tous les datasets utilisés ont une vérité terrain explicite (y = 3*a + ...) afin
qu'on puisse vérifier que les explications pointent vers la *vraie* feature
informative — un test pass-through (le code tourne) ne valide pas la fiabilité.
"""
from __future__ import annotations

import os
import json
import tempfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.shap import compute_global_shap, compute_local_shap
from app.services.lime import compute_local_lime
from app.services.training.output.predictor import (
    predict_with_explanations,
    predict_with_shap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic datasets with explicit ground truth
# ─────────────────────────────────────────────────────────────────────────────

def _make_regression_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """y = 3*a + 0.05*noise — a is the only informative feature; b,c,d,e are pure noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    y = 3.0 * a + 0.05 * rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "target": y})


def _make_classification_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """y = 1 if 3*a > 0 else 0  — a is the only informative feature."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    logits = 3.0 * a + 0.1 * rng.normal(0, 1, n)
    y = (logits > 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "target": y})


def _train_pipeline(df: pd.DataFrame, *, task: str = "classification"):
    """Train a small pipeline on the given DataFrame. Returns (pipeline, X, y)."""
    X = df.drop(columns=["target"])
    y = df["target"].values
    if task == "classification":
        model: Any = RandomForestClassifier(n_estimators=30, random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=30, random_state=0)
    pipe = Pipeline([("prep", StandardScaler()), ("model", model)])
    pipe.fit(X, y)
    return pipe, X, y


def _make_trained_model_mock(pipeline, artifacts: dict, *, task_type: str = "classification"):
    """
    Build a TrainedModel-like mock with the pipeline pickled to a real temp file.
    Returns (mock, pkl_path) — caller is responsible for cleaning up the pkl_path.
    """
    import joblib
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    joblib.dump(pipeline, tmp.name)
    tmp.close()

    m = MagicMock()
    m.id = 1
    m.session_id = 1
    m.model_type = "randomforest"
    m.task_type = task_type
    m.artifacts_json = {**artifacts, "model_pkl": tmp.name}
    return m, tmp.name


def _build_artifacts_with_shap_background(pipeline, X: pd.DataFrame, *, task: str = "classification") -> dict:
    """Run compute_global_shap to populate artifacts['shap']['background_data'] like the orchestrator does."""
    shap_block = compute_global_shap(pipeline, X, task_type=task, feature_names=list(X.columns))
    assert shap_block is not None, "Setup failure: SHAP global is None on a healthy synthetic pipeline"
    return {
        "shap": shap_block,
        "training_schema": {"feature_names": list(X.columns)},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. Source des données
# ═════════════════════════════════════════════════════════════════════════════

class TestDataSources:
    """
    Vérifie que les explications consomment exactement les artefacts produits à
    l'entraînement — pas une copie reconstruite à la volée. Une dérive ici
    voudrait dire qu'on explique un modèle X avec les statistiques d'un modèle Y.
    """

    def test_lime_uses_shap_background_from_artifacts(self, monkeypatch):
        """
        Pourquoi : la *même* baseline doit servir aux deux méthodes pour que la
        comparaison SHAP/LIME soit honnête. Si LIME se reconstruisait un
        background interne, les contributions seraient incomparables.

        Comment : on intercepte l'argument `background` reçu par compute_local_lime
        et on vérifie qu'il est bit-à-bit identique à artifacts["shap"]["background_data"].
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        captured: dict = {}

        def _spy_lime(*args, **kwargs):
            captured["background"] = kwargs.get("background")
            # Return a plausible result so the predictor proceeds normally
            return [{"feature": "a", "contribution": 0.1, "data": 0.0}]

        # Patch the symbol that predict_with_explanations imports lazily
        import app.services.lime as lime_pkg
        monkeypatch.setattr(lime_pkg, "compute_local_lime", _spy_lime)

        try:
            predict_with_explanations(model_mock, X.iloc[[0]], method="lime")

            expected_bg = np.asarray(artifacts["shap"]["background_data"], dtype=float)
            received_bg = captured["background"]
            assert received_bg is not None, "LIME did not receive any background"
            assert np.array_equal(np.asarray(received_bg, dtype=float), expected_bg), (
                "LIME background diverges from the SHAP background stored in artifacts"
            )
        finally:
            os.unlink(pkl)

    def test_feature_names_come_from_fitted_preprocessor(self):
        """
        Pourquoi : si feature_names était hardcodé, l'expansion OHE ('color' →
        'color_red', 'color_blue', ...) ne serait pas reflétée et les barres
        SHAP/LIME afficheraient des noms qui ne correspondent à aucune colonne
        réelle du modèle.

        Comment : on entraîne avec une colonne catégorielle qui se développe à
        l'OHE, et on vérifie que les noms exposés par SHAP global incluent les
        colonnes expansées (issues de get_feature_names_out du préprocesseur).
        """
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "age": rng.normal(40, 10, n),
            "color": rng.choice(["red", "green", "blue"], size=n),
            "target": rng.integers(0, 2, n),
        })
        X = df.drop(columns=["target"])
        y = df["target"].values

        prep = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["age"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["color"]),
            ]
        )
        pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=20, random_state=0))])
        pipe.fit(X, y)

        result = compute_global_shap(pipe, X, task_type="classification", feature_names=list(X.columns))
        assert result is not None
        feature_names = {item["feature"] for item in result["summary"]}

        # The preprocessor must have produced expanded names — not the raw "color".
        assert any("color" in fn for fn in feature_names), \
            f"OHE-expanded names absent from SHAP output: {feature_names}"
        # And not just the raw "color" alone (which would mean we used hardcoded names).
        assert "color" not in feature_names or len([f for f in feature_names if f.startswith("color")]) > 1, \
            "Only raw 'color' present — feature_names appear to bypass the fitted preprocessor"

    def test_data_field_preserves_raw_input_values(self):
        """
        Pourquoi : le champ `data` de chaque contribution doit refléter la
        valeur brute saisie par l'utilisateur, pas la valeur normalisée. Sinon
        l'utilisateur verrait "age=−1.45" au lieu de "age=42" — incompréhensible.

        Comment : on crée une ligne avec une valeur extrême (42.0) sur 'a',
        on appelle compute_local_shap, puis on vérifie que l'entrée 'a' a bien
        data=42.0 et NON la valeur standardisée (qui serait ~ (42 - mean) / std).
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        # Inject a known extreme value so we can detect normalization unambiguously
        row = X.iloc[[0]].copy()
        row.iloc[0, row.columns.get_loc("a")] = 42.0

        items = compute_local_shap(pipe, row, task_type="classification", feature_names=list(X.columns))
        assert items is not None
        a_item = next((it for it in items if it["feature"] == "a"), None)
        assert a_item is not None, "Feature 'a' missing from local SHAP output"
        assert a_item["data"] == pytest.approx(42.0), (
            f"`data` was transformed (expected raw 42.0, got {a_item['data']}). "
            "This would mean predictor passed a normalized value into the response."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Cohérence SHAP
# ═════════════════════════════════════════════════════════════════════════════

class TestShapCoherence:
    """
    Vérifie que les chiffres SHAP sont *justes*, pas seulement présents.
    Sans ces tests, un bug silencieux (ex : explainer entraîné sur la
    mauvaise classe) passerait inaperçu.
    """

    def test_shap_top_feature_matches_ground_truth(self):
        """
        Pourquoi : la feature 'a' génère 100 % du signal (y = 3*a). SHAP doit
        l'identifier comme top-1. Si non, le routage estimator → explainer est
        cassé ou les valeurs sont inversées.
        """
        df = _make_regression_df(n=400)
        pipe, X, _ = _train_pipeline(df, task="regression")
        result = compute_global_shap(pipe, X, task_type="regression", feature_names=list(X.columns))
        assert result is not None
        top_feature = result["summary"][0]["feature"]
        assert top_feature == "a", (
            f"SHAP top-1 = {top_feature!r}, expected 'a' (ground truth: y=3*a). "
            f"Full ranking: {[d['feature'] for d in result['summary']]}"
        )

    def test_shap_efficiency_property_binary(self):
        """
        Pourquoi : la propriété d'efficacité de Shapley (TreeExplainer exact) :
        sum(shap_values_for_class_1) + expected_value_class_1 ≈ predict_proba(row)[1]
        Si elle ne tient pas, les valeurs SHAP ne représentent plus une vraie
        décomposition additive — l'interprétation devient frauduleuse.
        """
        df = _make_classification_df(n=300)
        pipe, X, _ = _train_pipeline(df, task="classification")

        # Get expected_value via the same path the orchestrator uses
        global_result = compute_global_shap(pipe, X, task_type="classification", feature_names=list(X.columns))
        assert global_result is not None
        expected_value = global_result["expected_value"]
        assert expected_value is not None

        row = X.iloc[[0]]
        items = compute_local_shap(pipe, row, task_type="classification", feature_names=list(X.columns))
        assert items is not None
        sum_shap = sum(it["shap_value"] for it in items)

        # Predicted probability of the positive class for this row
        proba_pos = float(pipe.predict_proba(row)[0, 1])

        reconstructed = expected_value + sum_shap
        # TreeExplainer is exact — tolerance is tight; any gap > 1e-3 signals a routing bug
        assert reconstructed == pytest.approx(proba_pos, abs=1e-3), (
            f"Shapley efficiency violated: E[f] + Σshap = {reconstructed:.4f}, "
            f"predict_proba = {proba_pos:.4f} (gap = {abs(reconstructed - proba_pos):.4f})"
        )

    def test_shap_global_works_after_gridsearch_fix(self):
        """
        Régression Tâche 1 : SHAP global doit être calculé même quand un
        GridSearch est actif. Avant le fix, le bloc shap était None à cause de
        la garde `_shap_allowed = cfg.search_type == "none"`.
        """
        from app.services.training.config.schema import TrainingConfig
        from app.services.training.orchestrator import run_one_model

        df = _make_classification_df(n=200)

        cfg = TrainingConfig.from_front({
            "targetColumn": "target",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1"],
            "splitMethod": "holdout",
            "trainRatio": 70, "valRatio": 0, "testRatio": 30,
            "useGridSearch": True,  # ← the regression vector
            "balancing": {"strategy": "none", "apply_threshold": False},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        })

        result = run_one_model(df, cfg, "randomforest")
        artifacts = result.artifacts_json

        # The bug was: artifacts["shap"] was missing or None when search was active
        assert "shap" in artifacts, "SHAP global block missing from artifacts (Tâche 1 regression)"
        shap_block = artifacts["shap"]
        assert shap_block is not None, "SHAP global is None despite a refit pipeline (Tâche 1 regression)"
        assert isinstance(shap_block.get("summary"), list) and len(shap_block["summary"]) > 0, (
            "SHAP global summary empty — fix did not produce values"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Cohérence LIME
# ═════════════════════════════════════════════════════════════════════════════

class TestLimeCoherence:

    def test_lime_top3_includes_ground_truth_feature(self):
        """
        Pourquoi : LIME approxime localement avec un modèle linéaire. Pour un
        DGP où 'a' domine massivement, 'a' devrait apparaître dans le top-3
        de toute explication locale individuelle. Une absence systématique
        signalerait un bug dans le mapping feature_idx → feature_name.
        """
        df = _make_classification_df(n=300)
        pipe, X, _ = _train_pipeline(df, task="classification")
        bg = pipe.named_steps["prep"].transform(X.iloc[:50]).astype(float)

        # Average top-3 inclusion rate across 5 different rows for robustness
        n_rows_with_a_in_top3 = 0
        n_total = 5
        for i in range(n_total):
            items = compute_local_lime(
                pipe, X.iloc[[i]], task_type="classification",
                background=bg, feature_names=list(X.columns),
                num_samples=400, random_state=42 + i,
            )
            assert items is not None and len(items) > 0
            top3 = [it["feature"] for it in items[:3]]
            if "a" in top3:
                n_rows_with_a_in_top3 += 1

        assert n_rows_with_a_in_top3 >= 4, (
            f"'a' in LIME top-3 only {n_rows_with_a_in_top3}/{n_total} times — "
            "LIME may be misranking the informative feature"
        )

    def test_lime_sorted_by_abs_contribution_descending(self):
        """
        Pourquoi : le frontend repose sur cet ordre pour ne montrer que les
        N premières features. Un tri cassé afficherait du bruit en haut.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        bg = pipe.named_steps["prep"].transform(X.iloc[:50]).astype(float)

        items = compute_local_lime(
            pipe, X.iloc[[0]], task_type="classification",
            background=bg, feature_names=list(X.columns),
            num_samples=400,
        )
        assert items is not None
        abs_vals = [abs(it["contribution"]) for it in items]
        assert abs_vals == sorted(abs_vals, reverse=True), \
            f"LIME items not sorted by |contribution| desc: {abs_vals}"

    def test_lime_stability_across_runs(self):
        """
        Pourquoi : LIME est stochastique (échantillonne autour du point). Si la
        variance du top-1 dépasse 20 %, l'interprétation devient instable et
        un utilisateur verrait des explications différentes pour la *même*
        prédiction selon le moment où il clique. Tolérance volontairement
        souple — LIME n'est pas déterministe sans seed fixe.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        bg = pipe.named_steps["prep"].transform(X.iloc[:50]).astype(float)
        row = X.iloc[[0]]

        top1_values = []
        for seed in (1, 2, 3):
            items = compute_local_lime(
                pipe, row, task_type="classification",
                background=bg, feature_names=list(X.columns),
                num_samples=600, random_state=seed,
            )
            assert items is not None and len(items) > 0
            top1_values.append(abs(items[0]["contribution"]))

        mean_top1 = float(np.mean(top1_values))
        max_dev = max(abs(v - mean_top1) for v in top1_values)
        rel_dev = max_dev / max(mean_top1, 1e-6)
        assert rel_dev < 0.20, (
            f"LIME top-1 unstable: values={top1_values}, mean={mean_top1:.4f}, "
            f"relative max deviation={rel_dev:.2%} (>20%)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Mode both
# ═════════════════════════════════════════════════════════════════════════════

class TestBothMode:

    def test_both_returns_shap_and_lime_keys(self):
        """
        Pourquoi : c'est le contrat de l'API en mode `?method=both`. Le frontend
        en dépend pour rendre le ComparisonPanel.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        try:
            result = predict_with_explanations(model_mock, X.iloc[[0]], method="both")
            row0 = result["rows"][0]
            assert "shap" in row0 and len(row0["shap"]) > 0, "method=both missing 'shap' key"
            assert "lime" in row0 and len(row0["lime"]) > 0, "method=both missing 'lime' key"
        finally:
            os.unlink(pkl)

    def test_both_returns_lime_when_shap_fails(self, monkeypatch):
        """
        Pourquoi : les deux méthodes doivent dégrader indépendamment. Si SHAP
        crashe, on doit quand même servir LIME — sinon l'utilisateur perd toute
        explicabilité au moindre incident sur l'une des deux.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        def _broken_shap(*a, **kw):
            raise RuntimeError("simulated SHAP failure")

        import app.services.shap as shap_pkg
        monkeypatch.setattr(shap_pkg, "compute_local_shap", _broken_shap)

        try:
            result = predict_with_explanations(model_mock, X.iloc[[0]], method="both")
            row0 = result["rows"][0]
            assert "shap" not in row0, "SHAP key present despite simulated failure"
            assert "lime" in row0 and len(row0["lime"]) > 0, "LIME key absent — SHAP failure dragged LIME down"
        finally:
            os.unlink(pkl)

    def test_both_returns_shap_when_lime_fails(self, monkeypatch):
        """Symétrique du test précédent : LIME crash ne doit pas masquer SHAP."""
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        def _broken_lime(*a, **kw):
            raise RuntimeError("simulated LIME failure")

        import app.services.lime as lime_pkg
        monkeypatch.setattr(lime_pkg, "compute_local_lime", _broken_lime)

        try:
            result = predict_with_explanations(model_mock, X.iloc[[0]], method="both")
            row0 = result["rows"][0]
            assert "lime" not in row0, "LIME key present despite simulated failure"
            assert "shap" in row0 and len(row0["shap"]) > 0, "SHAP key absent — LIME failure dragged SHAP down"
        finally:
            os.unlink(pkl)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Compatibilité descendante
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_default_method_returns_shap_only(self):
        """
        Pourquoi : tous les appels existants à /predict/json/explain (sans
        ?method) doivent garder strictement le même format qu'avant la refonte
        : clé `shap` présente, clé `lime` absente.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        try:
            # Default — no method argument passed
            result = predict_with_explanations(model_mock, X.iloc[[0]])
            row0 = result["rows"][0]
            assert "shap" in row0, "Default call lost the 'shap' key (backward compat broken)"
            assert "lime" not in row0, "Default call leaked an unexpected 'lime' key"
            # Schema sanity for any consumer relying on the original SHAP shape
            for it in row0["shap"]:
                assert set(it.keys()) >= {"feature", "shap_value", "data"}
        finally:
            os.unlink(pkl)

    def test_predict_with_shap_alias_matches_explanations_shap(self):
        """
        Pourquoi : `predict_with_shap` est conservé comme alias. On veut
        garantir qu'il appelle réellement le même chemin que
        `predict_with_explanations(method='shap')` — pas une copie divergente
        qui pourrait dériver dans le temps.
        """
        df = _make_classification_df(n=200)
        pipe, X, _ = _train_pipeline(df, task="classification")
        artifacts = _build_artifacts_with_shap_background(pipe, X, task="classification")
        model_mock, pkl = _make_trained_model_mock(pipe, artifacts)

        try:
            row = X.iloc[[0]]
            via_alias = predict_with_shap(model_mock, row)
            via_explicit = predict_with_explanations(model_mock, row, method="shap")

            # Same set of top-level keys
            assert set(via_alias["rows"][0].keys()) == set(via_explicit["rows"][0].keys()), \
                "predict_with_shap diverges from predict_with_explanations on row keys"

            # Same SHAP values (deterministic on a fixed RandomForest)
            shap_alias = via_alias["rows"][0].get("shap") or []
            shap_explicit = via_explicit["rows"][0].get("shap") or []
            assert len(shap_alias) == len(shap_explicit)
            for a_it, e_it in zip(shap_alias, shap_explicit):
                assert a_it["feature"] == e_it["feature"]
                assert a_it["shap_value"] == pytest.approx(e_it["shap_value"], rel=1e-9, abs=1e-12)
        finally:
            os.unlink(pkl)
