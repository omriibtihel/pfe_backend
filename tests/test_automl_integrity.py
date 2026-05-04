"""
Audit d'intégrité du module AutoML / Mode Intelligent.

L'objectif n'est pas de vérifier que les fonctions « tournent » (c'est dans
test_recommendation_engine.py et test_dataset_profiler.py), mais de vérifier
qu'elles disent la VÉRITÉ à l'interface — qu'elles ne trompent pas l'utilisateur.

Sections
--------
1. Honnêteté des métriques      — _extract_primary_score ne ment pas sur la source
2. Recommandation → entraînement — le payload recommandé s'exécute réellement
3. Robustesse AutoML (FLAML)     — outputs fiables, score de test depuis un vrai holdout
4. Validité zero-shot            — les HPs préconfigurés ne plantent pas sklearn
5. Roundtrip MetaLearner         — écriture/lecture sans perte ni corruption
6. Précision du reasoning        — les justifications reflètent la décision réelle
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.services.data.profiler import DatasetProfiler
from app.services.training.intelligence.recommender import RecommendationEngine
from app.services.training.intelligence.meta_learner import MetaLearner, TrainingRecord
from app.services.training.config.schema import TrainingConfig
from app.services.training.config.zero_shot import get_zero_shot_hyperparams, _CONFIGS
from app.services.training.orchestrator import run_one_model


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic datasets avec signal explicite
# ─────────────────────────────────────────────────────────────────────────────

def _make_binary_df(n: int = 300, ir: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Classification binaire. feature[0] = signal pur, feature[1-3] = bruit."""
    rng = np.random.default_rng(seed)
    minority_n = max(6, int(n / (1 + ir)))
    majority_n = n - minority_n
    y = np.array([0] * majority_n + [1] * minority_n)
    X = rng.normal(0, 1, (n, 4))
    X[:, 0] += y * 2
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _make_regression_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers partagés
# ─────────────────────────────────────────────────────────────────────────────

_profiler = DatasetProfiler()
_engine = RecommendationEngine()


def _profile(df: pd.DataFrame) -> Any:
    return _profiler.profile(df, "target")


def _recommend(df: pd.DataFrame, target_column: str = "target") -> Any:
    return _engine.recommend(_profile(df), target_column=target_column)


def _minimal_cfg(task: str = "classification", model: str = "randomforest",
                 metrics: list[str] | None = None) -> dict:
    if metrics is None:
        metrics = ["roc_auc"] if task == "classification" else ["rmse"]
    return {
        "targetColumn": "target",
        "taskType": task,
        "models": [model],
        "metrics": metrics,
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 0,
        "testRatio": 30,
        "balancing": {"strategy": "none", "apply_threshold": False},
        "preprocessing": {
            "numericImputation": "median",
            "numericScaling": "standard",
            "categoricalImputation": "most_frequent",
            "categoricalEncoding": "onehot",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. Honnêteté des métriques
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricHonesty:
    """
    _extract_primary_score choisit une métrique pour comparer les modèles entre eux.
    Si elle choisit silencieusement une métrique différente de celle configurée,
    l'UI affiche « meilleur score : roc_auc = 0.92 » alors que c'est en réalité
    accuracy = 0.92 — comparaison invalide si les deux métriques ne varient pas ensemble.
    """

    def _score(self, metrics_json: dict, metrics_list: list[str]):
        from app.services.training.training_service import _extract_primary_score
        cfg = TrainingConfig.from_front({**_minimal_cfg(), "metrics": metrics_list})
        return _extract_primary_score(metrics_json, cfg)

    def test_source_is_user_defined_when_metric_present(self):
        """
        Si la métrique configurée est dans metrics_json["test"], source doit être
        "user_defined". L'UI peut alors l'afficher sans réserve.
        """
        metrics_json = {"test": {"roc_auc": 0.85, "accuracy": 0.79}}
        name, score, source = self._score(metrics_json, ["roc_auc"])
        assert source == "user_defined", (
            f"source='{source}' alors que roc_auc est présent dans metrics_json. "
            "L'UI montrerait la bonne valeur mais sous un label potentiellement trompeur."
        )
        assert name == "roc_auc"
        assert score == pytest.approx(0.85)

    def test_source_is_fallback_when_configured_metric_absent(self):
        """
        Si la métrique configurée n'est pas dans les résultats, source doit être
        "fallback" — jamais "user_defined". Un "user_defined" sur un substitut
        ferait croire à l'UI que la bonne métrique a été calculée.
        """
        metrics_json = {"test": {"accuracy": 0.80}}  # pas de roc_auc
        name, score, source = self._score(metrics_json, ["roc_auc"])
        assert source == "fallback", (
            f"source='{source}' alors que roc_auc est absent et qu'un fallback a été utilisé. "
            "L'UI afficherait cette valeur sous le label 'roc_auc' par erreur."
        )
        assert name != "roc_auc", "Le nom retourné doit refléter le vrai substitut utilisé"

    def test_alias_resolves_to_user_defined(self):
        """
        Certaines métriques ont des alias (f1 → f1_pos, f1_weighted). Si un alias
        est présent, source doit rester "user_defined" — pas un fallback camouflé.
        """
        metrics_json = {"test": {"f1_pos": 0.72}}
        name, score, source = self._score(metrics_json, ["f1"])
        assert source == "user_defined", (
            "f1_pos est un alias connu de f1 — source devrait être user_defined, pas fallback. "
            "Sinon l'UI afficherait f1_pos=0.72 comme un fallback alors que c'est la bonne métrique."
        )
        assert score == pytest.approx(0.72)

    def test_returns_none_when_metrics_json_empty(self):
        """
        Si metrics_json est vide (entraînement avorté), _extract_primary_score
        doit retourner score=None — jamais une valeur inventée.
        Le caller peut alors détecter qu'il n'y a pas de résultat valide.
        """
        from app.services.training.training_service import _extract_primary_score
        cfg = TrainingConfig.from_front(_minimal_cfg())
        name, score, source = _extract_primary_score({}, cfg)
        assert score is None, (
            f"score={score} retourné sur metrics_json vide. "
            "Le caller ne peut pas distinguer un vrai 0.0 d'un résultat manquant."
        )
        assert name is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Cohérence recommandation → entraînement
# ═════════════════════════════════════════════════════════════════════════════

class TestRecommendationToTraining:
    """
    La recommandation doit proposer des configs qui s'exécutent réellement
    et produisent les métriques qu'elle promet.
    """

    def test_payload_produces_valid_training_config(self):
        """
        Le payload renvoyé au frontend doit être parseable par TrainingConfig.from_front()
        sans exception — c'est la première chose que fait le backend quand l'utilisateur
        lance l'entraînement avec la recommandation acceptée.

        """
        df = _make_binary_df(n=300, ir=1.5)
        rec = _recommend(df)
        try:
            cfg = TrainingConfig.from_front(rec.training_config_payload)
        except Exception as e:
            pytest.fail(
                f"TrainingConfig.from_front() a échoué sur le payload recommandé : {e}\n"
                f"Payload : {rec.training_config_payload}"
            )
        assert cfg.target_column == "target"
        assert len(cfg.models) >= 1

    def test_first_recommended_model_trains_without_error(self):
        """
        Le premier modèle recommandé doit s'entraîner sans exception.
        Un crash ici signifie que l'UI recommande quelque chose qui ne fonctionne pas.
        """
        df = _make_binary_df(n=300, ir=1.0)
        rec = _recommend(df)
        cfg = TrainingConfig.from_front(rec.training_config_payload)
        model = rec.recommended_models[0]
        try:
            result = run_one_model(df, cfg, model_type=model)
        except Exception as e:
            pytest.fail(f"Le modèle recommandé '{model}' a planté à l'entraînement : {e}")
        assert result is not None
        assert isinstance(result.metrics_json, dict) and result.metrics_json

    def test_recommended_metric_present_in_training_results(self):
        """
        La métrique affichée sur la carte modèle dans l'UI doit être réellement calculée.
        Si recommended_metric n'est pas dans metrics_json["test"], l'UI montre un score
        sous un label inventé.
        """
        df = _make_binary_df(n=300, ir=1.0)
        rec = _recommend(df)
        cfg = TrainingConfig.from_front(rec.training_config_payload)
        result = run_one_model(df, cfg, model_type=rec.recommended_models[0])

        primary_metric = rec.recommended_metric
        test_metrics = result.metrics_json.get("test", {})

        # Known aliases accepted (the pipeline may store under a variant name)
        _aliases: dict[str, list[str]] = {
            "roc_auc": ["roc_auc", "roc_auc_ovr_weighted"],
            "f1": ["f1", "f1_pos", "f1_weighted"],
            "pr_auc": ["pr_auc", "average_precision"],
            "rmse": ["rmse"],
            "mae": ["mae"],
            "accuracy": ["accuracy"],
            "f1_macro": ["f1_macro", "f1"],
            "f1_weighted": ["f1_weighted", "f1"],
        }
        candidates = _aliases.get(primary_metric, [primary_metric])
        found = any(k in test_metrics for k in candidates)
        assert found, (
            f"Métrique recommandée '{primary_metric}' (ni ses alias {candidates}) "
            f"absente des résultats test : {list(test_metrics.keys())}\n"
            "L'UI afficherait un score sous un label qui ne correspond à aucune valeur calculée."
        )

    def test_reasoning_has_multiple_nonempty_entries(self):
        """
        Le dictionnaire reasoning doit couvrir plusieurs axes de décision.
        Un reasoning à une seule entrée = justification tronquée dans l'UI.
        """
        df = _make_binary_df(n=300, ir=5.0)
        rec = _recommend(df)
        reasoning = rec.reasoning
        assert len(reasoning) >= 3, (
            f"reasoning n'a que {len(reasoning)} entrée(s) : {list(reasoning.keys())}. "
            "L'UI n'a pas assez d'informations pour justifier toutes les recommandations."
        )
        empty_keys = [k for k, v in reasoning.items() if not str(v).strip()]
        assert not empty_keys, f"Valeurs vides dans reasoning : {empty_keys}"

    def test_warnings_emitted_for_tiny_dataset(self):
        """
        Un dataset de 35 lignes est trop petit pour des résultats fiables.
        L'UI doit afficher un avertissement explicite — pas une recommandation
        confiante sans réserve.
        """
        df = _make_binary_df(n=35, ir=1.0)
        rec = _recommend(df)
        assert len(rec.warnings) > 0, (
            "Aucun avertissement sur un dataset de 35 lignes. "
            "L'UI afficherait des recommandations sans réserve sur des données insuffisantes."
        )
        combined = " ".join(rec.warnings).lower()
        has_size_mention = any(word in combined for word in [
            "small", "few", "peu", "trop", "minimum", "warn", "tiny", "limit", "insuffis",
        ])
        assert has_size_mention, (
            f"Les avertissements ne mentionnent pas la taille insuffisante : {rec.warnings}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Robustesse AutoML (FLAML)
# ═════════════════════════════════════════════════════════════════════════════

pytest.importorskip("flaml", reason="FLAML non installé — section AutoML ignorée")


class TestAutoMLRunner:
    """
    run_automl doit retourner des résultats honnêtes : score de test depuis un vrai
    holdout, seuil calibré pour les datasets déséquilibrés, budget réellement adapté
    pour les petits datasets.
    """

    def _cfg(self, task: str = "classification", time_budget: int = 10,
             test_ratio: float = 0.2) -> Any:
        from app.services.training.config.automl import AutoMLConfig
        return AutoMLConfig(
            dataset_version_id=0,
            target_column="target",
            task_type=task,
            time_budget=time_budget,
            test_ratio=test_ratio,
        )

    def test_returns_exactly_one_best_result(self):
        """
        run_automl doit toujours retourner exactement un résultat is_best=True.
        Zéro = le backend crashe ; deux = le choix du « meilleur » devient ambigu.
        """
        from app.services.training.automl_runner import run_automl
        df = _make_binary_df(n=200)
        results = run_automl(df, self._cfg())
        assert len(results) >= 1, "run_automl a retourné une liste vide"
        best = [r for r in results if r.is_best]
        assert len(best) == 1, (
            f"{len(best)} résultat(s) marqué(s) is_best=True — attendu exactement 1."
        )

    def test_test_score_comes_from_real_holdout(self):
        """
        Quand test_ratio > 0, le score de test doit venir d'un vrai holdout
        (evaluation_strategy='holdout_test', has_test=True). Sinon l'UI affiche
        un score d'entraînement comme s'il s'agissait d'une évaluation indépendante.
        """
        from app.services.training.automl_runner import run_automl
        df = _make_binary_df(n=200)
        results = run_automl(df, self._cfg(test_ratio=0.2))
        best = next(r for r in results if r.is_best)
        m = best.metrics_json
        assert m.get("has_test") is True, (
            "has_test=False malgré test_ratio=0.2. "
            "Le score affiché dans l'UI serait un score d'entraînement, pas de test."
        )
        assert m.get("evaluation_strategy") == "holdout_test", (
            f"evaluation_strategy='{m.get('evaluation_strategy')}' — attendu 'holdout_test'."
        )
        test_block = m.get("test", {})
        assert isinstance(test_block, dict) and len(test_block) > 0, (
            "metrics_json['test'] est vide malgré has_test=True — score de test introuvable."
        )

    def test_test_score_reasonable_on_separable_binary(self):
        """
        Sur un dataset bien séparé (signal fort sur feature[0]), le score de test
        doit dépasser le hasard. Un score ≤ 0.55 serait le signe d'un bug grave
        (mauvais preprocessing, leakage, ou AutoML n'a rien appris).
        """
        from app.services.training.automl_runner import run_automl
        df = _make_binary_df(n=300, ir=1.0, seed=0)
        results = run_automl(df, self._cfg(time_budget=15))
        best = next(r for r in results if r.is_best)
        test_metrics = best.metrics_json.get("test", {})
        score = test_metrics.get("roc_auc") or test_metrics.get("accuracy")
        assert score is not None, f"Aucune métrique de test disponible : {test_metrics}"
        assert score > 0.55, (
            f"Score de test {score:.3f} proche du hasard sur données bien séparées. "
            "Le modèle affiché dans l'UI ne serait pas meilleur qu'un coin à lancer."
        )

    def test_threshold_calibrated_away_from_05_on_imbalanced_data(self):
        """
        Avec IR=9 et assez d'échantillons, la calibration de seuil doit produire
        un threshold ≠ 0.5. Un seuil bloqué à 0.5 sur données déséquilibrées
        signifie que la calibration n'a pas fonctionné — tous les positifs
        seraient probablement mal classifiés.
        """
        from app.services.training.automl_runner import run_automl
        df = _make_binary_df(n=500, ir=9.0)
        results = run_automl(df, self._cfg(time_budget=15))
        best = next(r for r in results if r.is_best)
        m = best.metrics_json
        threshold = m.get("threshold_used", 0.5)
        n_train = m.get("split_info", {}).get("train_rows", 0)
        # Calibration runs only when n_train >= 100 (_THRESH_MIN_SAMPLES)
        if n_train >= 100:
            assert threshold != pytest.approx(0.5, abs=0.01), (
                f"threshold_used={threshold:.3f} — calibration n'a pas bougé le seuil "
                "malgré IR=9 et {n_train} échantillons d'entraînement. "
                "L'UI afficherait 0.5 comme seuil optimal, ce qui est incorrect."
            )

    def test_budget_capped_for_tiny_dataset(self):
        """
        50 lignes × 4 features = complexity=200 → cap interne = 20s.
        artifacts_json['automl']['budget_used_s'] doit être ≤ 20 même si
        l'utilisateur a demandé 120s.
        """
        from app.services.training.automl_runner import run_automl
        df = _make_binary_df(n=50, ir=1.0)
        results = run_automl(df, self._cfg(time_budget=120))
        best = next(r for r in results if r.is_best)
        budget_used = best.artifacts_json.get("automl", {}).get("budget_used_s")
        assert budget_used is not None, "budget_used_s absent des artifacts — impossible de vérifier le budget réel"
        assert budget_used <= 20, (
            f"budget_used_s={budget_used}s pour 50 lignes (cap attendu=20s). "
            "L'adaptation de budget ne fonctionne pas — "
            "l'UI afficherait '120s de recherche' sur un dataset qui n'en bénéficie pas."
        )
        assert budget_used <= 120, "budget_used_s dépasse le budget demandé par l'utilisateur"

    def test_regression_task_type_and_rmse_in_results(self):
        """
        En mode régression, task_type doit être 'regression' et rmse (ou mae)
        doit être présent dans les métriques de test.
        Un task_type='classification' sur une target continue produirait des
        métriques absurdes (roc_auc sur des floats continus) affichées dans l'UI.
        """
        from app.services.training.automl_runner import run_automl
        df = _make_regression_df(n=200)
        results = run_automl(df, self._cfg(task="regression", time_budget=10))
        best = next(r for r in results if r.is_best)
        assert best.task_type == "regression", (
            f"task_type='{best.task_type}' — attendu 'regression'. "
            "Le modèle tournerait en mode classification sur une target continue."
        )
        test_metrics = best.metrics_json.get("test", {})
        has_regression_metric = "rmse" in test_metrics or "mae" in test_metrics or "r2" in test_metrics
        assert has_regression_metric, (
            f"Aucune métrique de régression (rmse/mae/r2) dans : {list(test_metrics.keys())}. "
            "L'UI afficherait des métriques vides ou incorrectes pour ce modèle."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Validité des configurations zero-shot
# ═════════════════════════════════════════════════════════════════════════════

class TestZeroShotValidity:
    """
    Les HPs pré-remplis dans la recommandation doivent être valides dès le départ.
    Un HP invalide fait crasher l'entraînement immédiatement après que l'utilisateur
    ait accepté la recommandation.
    """

    def test_all_registered_hp_values_are_json_serializable(self):
        """
        Les HPs sont envoyés via l'API (JSON) et stockés en base.
        Des valeurs non-sérialisables (lambdas, objets sklearn) causeraient
        des crashes silencieux au stockage ou à la désérialisation.
        """
        errors = []
        for scenario_key, models in _CONFIGS.items():
            for model_name, hp in models.items():
                try:
                    json.dumps(hp)
                except (TypeError, ValueError) as e:
                    errors.append(f"{scenario_key}/{model_name}: {e}")
        assert not errors, "HPs non-JSON-sérialisables détectés :\n" + "\n".join(errors)

    def test_unknown_model_returns_empty_dict_not_error(self):
        """
        Un modèle non enregistré doit retourner un dict vide (les defaults sklearn).
        Retourner None ou lever une exception crasherait le builder.
        """
        result = get_zero_shot_hyperparams(
            size_cat="small",
            task_type="binary_classification",
            imbalance_ratio=1.0,
            models=["mon_modele_inexistant"],
        )
        assert "mon_modele_inexistant" in result, "La clé du modèle inconnu est absente du résultat"
        assert result["mon_modele_inexistant"] == {}, (
            f"Modèle inconnu a retourné {result['mon_modele_inexistant']} au lieu de {{}}"
        )

    def test_scenario_coverage_for_all_size_categories(self):
        """
        Les 3 catégories de taille (small, medium, large) doivent toutes être couvertes
        pour binary_classification et regression. Un gap = la recommandation retournerait
        des HPs vides pour ce cas, sans le signaler.
        """
        required = [
            ("small",  "binary_classification", 1.0),
            ("small",  "binary_classification", 8.0),
            ("medium", "binary_classification", 1.0),
            ("large",  "binary_classification", 1.0),
            ("small",  "regression", None),
            ("medium", "regression", None),
            ("large",  "regression", None),
        ]
        missing = []
        for size, task, ir in required:
            hps = get_zero_shot_hyperparams(
                size_cat=size, task_type=task, imbalance_ratio=ir,
                models=["randomforest", "lightgbm", "ridge"],
            )
            non_empty = [m for m, h in hps.items() if h]
            if not non_empty:
                missing.append(f"{size}/{task}/ir={ir}")
        assert not missing, (
            f"Aucun HP zero-shot pour : {missing}. "
            "L'UI recommanderait ces cas sans HPs préconfigurés."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Roundtrip MetaLearner
# ═════════════════════════════════════════════════════════════════════════════

class TestMetaLearnerRoundtrip:
    """
    Le MetaLearner accumule l'historique pour améliorer les futures recommandations.
    Une corruption silencieuse = warm-start erroné affiché avec confiance à l'utilisateur.
    """

    def _record(self, **overrides) -> TrainingRecord:
        base: dict[str, Any] = dict(
            dataset_meta_features={"n_samples": 300, "n_features": 4},
            n_samples=300, n_features=4,
            task_type="binary_classification",
            dataset_size_category="small",
            best_algorithm="randomforest",
            best_hyperparameters={"n_estimators": 100},
            best_score=0.85,
            metric_used="roc_auc",
            resampling_method=None,
            split_method="holdout",
            search_type="none",
            config_mode="manual",
            user_overrides=None,
            training_time_seconds=12.5,
            session_id=99,
            project_id=1,
        )
        base.update(overrides)
        return TrainingRecord(**base)

    def test_write_then_read_preserves_all_fields(self):
        """
        Écrire un record puis le relire doit reproduire exactement tous les champs.
        Une perte de champs corromprait les recommandations warm-start.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetaLearner(tmpdir)
            ml.record(project_id=1, record=self._record())
            records = ml.load_records(project_id=1)

            assert len(records) == 1, f"1 record écrit, {len(records)} relus"
            r = records[0]
            assert r.best_algorithm == "randomforest"
            assert r.best_score == pytest.approx(0.85)
            assert r.metric_used == "roc_auc"
            assert r.n_samples == 300
            assert r.n_features == 4
            assert r.task_type == "binary_classification"
            assert r.resampling_method is None

    def test_multiple_records_all_accumulated(self):
        """
        Plusieurs entraînements pour le même projet doivent tous être conservés.
        Une troncature ferait perdre des données historiques essentielles.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetaLearner(tmpdir)
            for i in range(5):
                ml.record(project_id=1, record=self._record(session_id=i, best_score=0.80 + i * 0.02))
            records = ml.load_records(project_id=1)
            assert len(records) == 5, f"5 records écrits, {len(records)} relus"

    def test_warm_start_returns_best_scorer_not_first_inserted(self):
        """
        best_algorithm_for_profile doit retourner l'algorithme avec le meilleur score,
        pas le premier enregistré. Un warm-start erroné suggérerait le mauvais modèle.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetaLearner(tmpdir)
            ml.record(1, self._record(best_algorithm="logisticregression", best_score=0.70))
            ml.record(1, self._record(best_algorithm="randomforest", best_score=0.85))
            ml.record(1, self._record(best_algorithm="lightgbm", best_score=0.79))

            best = ml.best_algorithm_for_profile(
                project_id=1,
                task_type="binary_classification",
                size_category="small",
                metric="roc_auc",
            )
            assert best == "randomforest", (
                f"Warm-start retourne '{best}' — attendu 'randomforest' (score=0.85). "
                "Le système recommanderait le mauvais modèle au prochain entraînement."
            )

    def test_malformed_jsonl_line_skipped_without_crash(self):
        """
        Un enregistrement corrompu (crash précédent, écriture partielle) ne doit pas
        bloquer le chargement de tout l'historique — il doit être ignoré silencieusement.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetaLearner(tmpdir)
            ml.record(project_id=1, record=self._record(session_id=1))

            jsonl_path = os.path.join(tmpdir, "1", "meta_learning.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write("{bad json line\n")
                f.write('{"incomplete": true}\n')  # JSON valide mais champs manquants

            ml.record(project_id=1, record=self._record(session_id=2))

            records = ml.load_records(project_id=1)
            assert len(records) == 2, (
                f"{len(records)} record(s) chargé(s) — attendu 2 (les lignes malformées "
                "doivent être ignorées, pas lever une exception qui bloquerait tout)."
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Précision du reasoning
# ═════════════════════════════════════════════════════════════════════════════

class TestReasoningAccuracy:
    """
    Le reasoning affiché dans l'UI doit refléter la décision réellement prise.
    Un reasoning incorrect = l'utilisateur fait confiance à une explication fausse.
    """

    def test_high_imbalance_reasoning_mentions_imbalance(self):
        """
        Avec IR=12, le reasoning doit mentionner le déséquilibre. Sans ça,
        l'utilisateur voit « SMOTE recommandé » sans comprendre pourquoi.
        """
        df = _make_binary_df(n=400, ir=12.0)
        rec = _recommend(df)
        text = " ".join(str(v) for v in rec.reasoning.values()).lower()
        keywords = ["imbalanc", "déséquilibr", "ratio", "minorit", "class weight", "smote", "resamp"]
        assert any(kw in text for kw in keywords), (
            f"IR=12 mais le reasoning ne mentionne pas le déséquilibre.\n"
            f"Reasoning : {rec.reasoning}\n"
            "L'utilisateur verrait SMOTE recommandé sans aucune explication."
        )

    def test_payload_split_method_matches_cv_recommendation(self):
        """
        training_config_payload['splitMethod'] doit correspondre exactement à
        recommended_cv_strategy. Un mismatch = l'entraînement utilise une stratégie
        différente de ce que le reasoning explique à l'utilisateur.
        """
        df = _make_binary_df(n=300)
        rec = _recommend(df)
        payload_split = rec.training_config_payload.get("splitMethod")
        assert payload_split == rec.recommended_cv_strategy, (
            f"payload splitMethod='{payload_split}' ≠ recommended_cv_strategy='{rec.recommended_cv_strategy}'.\n"
            "L'entraînement utiliserait une stratégie différente de celle décrite à l'utilisateur."
        )

    def test_regression_reasoning_free_of_classification_artifacts(self):
        """
        En régression, le reasoning ne doit pas mentionner SMOTE ni class_weight
        ni d'autres concepts de classification. Des artefacts de classification
        dans un reasoning de régression signaleraient un bug de routage.
        """
        df = _make_regression_df(n=300)
        rec = _recommend(df)
        text = " ".join(str(v) for v in rec.reasoning.values()).lower()
        false_positives = ["smote", "class_weight", "imbalance ratio", "minority class"]
        triggered = [fp for fp in false_positives if fp in text]
        assert not triggered, (
            f"Le reasoning de régression contient des termes de classification : {triggered}.\n"
            f"Reasoning : {rec.reasoning}"
        )

    def test_large_dataset_recommended_as_holdout(self):
        """
        Pour un dataset de 60k lignes, la recommandation doit basculer vers holdout.
        Un k-fold sur 60k lignes serait très lent — et le reasoning promettrait
        quelque chose qui ne se passerait pas vraiment.
        """
        rng = np.random.default_rng(0)
        n = 60_000
        X = rng.normal(0, 1, (n, 4))
        y = (X[:, 0] > 0).astype(int)
        df = pd.DataFrame(X, columns=list("abcd"))
        df["target"] = y

        rec = _recommend(df)
        assert rec.recommended_cv_strategy == "holdout", (
            f"recommended_cv_strategy='{rec.recommended_cv_strategy}' pour 60k lignes — "
            "attendu 'holdout'. Un k-fold sur ce volume serait prohibitif."
        )
        assert rec.training_config_payload.get("splitMethod") == "holdout", (
            "Le payload enverrait un k-fold au backend malgré la recommandation holdout. "
            "L'entraînement serait différent de ce que l'UI a affiché."
        )
