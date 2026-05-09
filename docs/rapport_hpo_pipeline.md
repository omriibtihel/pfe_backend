# Rapport : Pipeline d'Optimisation des Hyperparamètres (HPO)
## MedIQ Backend — Analyse Technique Détaillée
**Date** : 2026-04-01

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Fichiers concernés](#2-fichiers-concernés)
3. [Module models.py — Registre et Espaces de Recherche](#3-module-modelspy)
4. [Module trainer.py — Logique d'Optimisation](#4-module-trainerpy)
5. [Module schema.py — Normalisation des Hyperparamètres](#5-module-schemapy)
6. [Module orchestrator.py — Intégration](#6-module-orchestratorpy)
7. [Flux de données complet](#7-flux-de-données-complet)
8. [Méthodes de recherche — Comparatif](#8-méthodes-de-recherche--comparatif)
9. [Classification des datasets par taille](#9-classification-des-datasets-par-taille)
10. [Points forts et limites](#10-points-forts-et-limites)
11. [Valeurs de référence](#11-valeurs-de-référence)

---

## 1. Vue d'ensemble

Le pipeline HPO de MedIQ est un système **adaptatif multi-couche** qui :

- Supporte 4 modes de recherche : `none`, `grid`, `random`, `halving_random`
- Adapte automatiquement les grilles et budgets à la taille du dataset (6 catégories : micro/tiny/small/medium/large)
- Garantit l'**anti-leakage** : SMOTE/undersampling appliqué *par fold* dans le CV, jamais sur le split de validation
- Intègre l'**early stopping** pour XGBoost et LightGBM en mode `none`
- Gère les données médicales déséquilibrées via injection automatique de `class_weight` et sélection adaptative de la métrique de refit

---

## 2. Fichiers concernés

| Fichier | Rôle dans le pipeline HPO |
|---------|--------------------------|
| `app/services/training/pipeline/models.py` | Registre des 12 modèles, grilles adaptatives `_ADAPTIVE_GRIDS`, distributions continues, budgets search |
| `app/services/training/pipeline/trainer.py` | Branches de recherche (none/grid/random/halving), CV splitter, early stopping, extraction des artefacts |
| `app/services/training/config/schema.py` | Normalisation/validation des hyperparamètres utilisateur, `MODEL_HP_SCHEMA`, `normalize_model_hyperparams()` |
| `app/services/training/orchestrator.py` | Construction du `param_grid`, détection de l'imbalance, embedding du resampler, appel à `Trainer.fit()` |

---

## 3. Module `models.py`

### 3.1 Classe `ModelSpec` (ligne 77)

Conteneur immutable (`frozen=True`) décrivant un modèle :

```
ModelSpec
├── name / aliases          → clé canonique + raccourcis ("rf" → "randomforest")
├── supports_*              → capacités (classification, regression, class_weight, predict_proba, sample_weight)
├── default_params          → {task_type → {param → valeur_défaut}}
├── param_grid              → {task_type → {param → [valeurs]}}   ← grille statique
├── param_distributions     → {task_type → {param → distribution}} ← pour Random/Halving
└── estimator_factory       → Callable[[task_type, params], estimator]
```

### 3.2 Registre des 12 modèles

| Modèle | Classif. | Régr. | class_weight | predict_proba | sample_weight |
|--------|:--------:|:-----:|:------------:|:-------------:|:-------------:|
| randomforest | ✓ | ✓ | ✓ | ✓ | ✓ |
| extratrees | ✓ | ✓ | ✓ | ✓ | ✓ |
| gradientboosting | ✓ | ✓ | ✗ | ✓ | ✓ |
| xgboost | ✓ | ✓ | ✗ | ✓ | ✓ |
| lightgbm | ✓ | ✓ | ✓ | ✓ | ✓ |
| catboost | ✓ | ✓ | ✗ | ✓ | ✓ |
| logisticregression | ✓ | ✗ | ✓ | ✓ | ✓ |
| svm | ✓ | ✓ | ✓ | ✓ | ✓ |
| decisiontree | ✓ | ✓ | ✓ | ✓ | ✓ |
| knn | ✓ | ✓ | ✗ | ✓ | ✗ |
| naivebayes | ✓ | ✗ | ✗ | ✓ | ✓ |
| ridge | ✗ | ✓ | ✗ | ✗ | ✓ |

### 3.3 Distribution `log_randint` (ligne 42)

Distribution personnalisée log-uniforme discrète utilisée pour `n_estimators` et d'autres paramètres entiers :

```python
log_randint(50, 500)
# → exp(uniform(log(50), log(500)))
# Favorise les petites valeurs : plus de candidats entre 50-150 qu'entre 400-500
# Pertinent car l'amélioration de 50→100 arbres > 400→450 arbres
```

### 3.4 Grilles adaptatives `_ADAPTIVE_GRIDS` (ligne 951)

Structure à 4 niveaux : `{modèle → {tâche → {catégorie_taille → {param → [valeurs]}}}`

**Principes de conception par taille :**

**micro (< 100 lignes)** — max 6 combos
- `max_depth` limité à [3, 5] — jamais `None` (mémorisation pure interdite)
- `min_samples_leaf` élevé [4, 8] — chaque feuille couvre ≥5% des données
- Exemple RF : `{"max_depth": [3, 5], "min_samples_leaf": [4, 8]}` → 4 combos

**tiny (100–500)** — max 20 combos
- Exploration prudente, quelques dimensions
- Exemple RF : `{"n_estimators": [100, 200], "max_depth": [5, None]}` → 4 combos

**small (500–2000)** — max 80 combos
- Exploration modérée, `max_features` inclus
- `colsample_bytree` pour XGBoost (anti-overfitting)

**medium (2000–50000)** — max 100 combos
- Exploration complète : régularisation L1/L2, `min_samples_split`, profondeur étendue
- `reg_alpha`/`reg_lambda` ajoutés pour XGBoost et LightGBM

**large (≥ 50000)** — max 20 combos
- Grille compacte par contrainte CPU, pas par biais
- Hyperparamètres les plus impactants uniquement

### 3.5 Distributions pour Random/Halving Search

Exemple RandomForest (classification) :
```python
param_distributions = {
    "n_estimators":      log_randint(50, 500),       # log-uniforme discret
    "max_depth":         [3, 5, 8, 12, 20, None],    # liste discrète
    "min_samples_split": randint(2, 20),              # uniforme discret
    "min_samples_leaf":  randint(1, 15),
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap":         [True, False],               # boostrap sans remise exploré
    "class_weight":      ["balanced", "balanced_subsample", None],
}
```

Exemple LightGBM (classification) :
```python
param_distributions = {
    "n_estimators":      log_randint(50, 500),
    "learning_rate":     loguniform(0.005, 0.3),      # log-uniforme continu
    "num_leaves":        randint(15, 150),
    "reg_alpha":         loguniform(1e-3, 10),
    "reg_lambda":        loguniform(1e-3, 10),
    "min_child_samples": randint(5, 50),
    "min_split_gain":    loguniform(1e-4, 1.0),       # equiv. gamma XGBoost
}
```

### 3.6 Budgets de recherche

**RandomizedSearchCV `n_iter`** :
```
micro:  10    tiny:  25    small:  40    medium:  60    large:  80
```

**HalvingRandomSearchCV `n_candidates_init`** (factor=3) :
```
micro:  15 → 5 → 1       (21 fits total)
tiny:   30 → 10 → 3      (43 fits total)
small:  60 → 20 → 7 → 2  (89 fits total)
medium: 100 → 33 → 11 → 4 (148 fits total)
large:  50 → 17 → 6      (73 fits total)
```

---

## 4. Module `trainer.py`

### 4.1 Fonction `Trainer.fit()` — Arbre de décision (ligne 443)

```
cfg.search_type
    │
    ├── "none"
    │     ├── XGBoost/LightGBM + n_samples ≥ 50 → _fit_with_early_stopping()
    │     └── Sinon → pipeline.fit() direct
    │
    ├── "halving_random"
    │     ├── get_model_distributions()
    │     ├── n_candidates = get_halving_n_candidates(n_samples)
    │     ├── min_resources = _compute_min_resources()  [SMOTE-safe]
    │     └── HalvingRandomSearchCV.fit()
    │
    ├── "random"
    │     ├── get_model_distributions()
    │     ├── n_iter = get_adaptive_n_iter(n_samples)
    │     └── RandomizedSearchCV.fit()
    │
    └── "grid"
          ├── SI model_param_grid fourni par l'orchestrateur → utilisé verbatim
          ├── SINON SI n_samples connu → get_adaptive_model_grid()
          └── SINON → get_model_grid() (grille statique fallback)
```

**Options communes à tous les modes search :**
- `return_train_score=True` → overfit gap calculable
- `error_score=np.nan` → combinaison invalide ne fait pas crasher tout le search
- `n_jobs=-1` → parallélisation maximale
- `refit=True` → best_estimator_ refitté sur tout X_train

### 4.2 Fonction `_build_cv_splitter()` (ligne 111)

Logique de construction du CV splitter :

```python
cv = max(2, min(requested_splits, n_samples))

if classification:
    cv_cls = min(cv, counts.min())   # cap = fréquence de la classe minoritaire
    if cv_cls >= 2:
        return StratifiedKFold(cv_cls)  # stratifié si possible

return KFold(cv)  # fallback non-stratifié
```

**Garanties** :
- `cv` jamais supérieur à `n_samples` (évite les folds vides)
- `cv` jamais supérieur à la taille de la classe minoritaire
- `cv` jamais < 2

### 4.3 Fonction `_fit_with_early_stopping()` (ligne 134)

Applicable uniquement : `search_type="none"` + (XGBoost ou LightGBM) + `n_samples ≥ 50`

```
Étape 1 : Split interne 80/20 (stratifié si classification)
          X_sub (80%) + X_val (20%)

Étape 2 : Prétraitement de X_val via les steps non-model du pipeline
          preproc_pipe = clone(Pipeline(steps sans "model"))
          preproc_pipe.fit(X_sub) → transform(X_val) → X_val_prep

Étape 3 : Fit du modèle avec early stopping sur X_sub
          XGBoost  : model.fit(X_sub_prep, y_sub, eval_set=[(X_val_prep, y_val)])
                     early_stopping_rounds=50
          LightGBM : callbacks=[early_stopping(50), log_evaluation(-1)]

Étape 4 : Extraction best_n
          XGBoost  : model.best_iteration
          LightGBM : model.best_iteration_

Étape 5 : Refit sur 100% des données
          pipeline.named_steps["model"].set_params(n_estimators=best_n)
          pipeline.fit(X_train, y_train)
```

**Retourne** : `{"used": bool, "best_n_estimators": int, "early_stopping_rounds": 50}`

### 4.4 Gestion SMOTE dans le CV — `_adapt_resampler_for_cv()` (ligne 317)

**Problème** : SMOTE avec `k_neighbors=5` peut crasher si la classe minoritaire a < 5 samples dans un fold d'entraînement.

**Solution** :
```python
# Calcul du k_neighbors safe
min_class_count = counts.min()
min_train_minority = min_class_count - ceil(min_class_count / cv_splits)
safe_k = max(0, min_train_minority - 1)

# Application
if safe_k < 1:
    resampler = None  # Désactivé
elif safe_k < requested_k:
    resampler = clone_with_k(resampler, safe_k)  # Réduit
```

**Calcul de `min_resources` pour Halving + SMOTE** (`_compute_min_resources()`, ligne 231) :
```python
# Sans resampler : "smallest" (défaut sklearn, OK)
# Avec resampler : garantir ≥ smote_k+1 samples minoritaires par fold
train_frac = (cv_splits - 1) / cv_splits
min_r = ceil((smote_k + 1) / (minority_ratio × train_frac))
return max(min_r, 20)  # floor absolu de 20 samples
```

### 4.5 Sélection de la métrique de refit — `_choose_refit_metric()` (ligne 88)

```
Classification :
  minority_ratio < 0.20 (binaire déséquilibré) → "average_precision"  (PR-AUC)
  multiclasse déséquilibré                      → "f1_macro"
  Sinon : cherche dans cfg.metrics → fallback "f1" (binaire) / "f1_weighted" (multi)

Régression :
  Cherche dans cfg.metrics → fallback "r2"
```

**Mapping métrique → scoring sklearn** :

| Métrique config | Scoring sklearn (binaire) | Scoring sklearn (multiclasse) |
|----------------|--------------------------|-------------------------------|
| `accuracy` | `accuracy` | `accuracy` |
| `f1` | `f1` | `f1_weighted` |
| `f1_macro` | `f1` | `f1_macro` |
| `roc_auc` | `roc_auc` | `roc_auc_ovr_weighted` |
| `pr_auc` | `average_precision` | *(non supporté)* |

### 4.6 Extraction des artefacts — `_extract_search_artifacts()` (ligne 352)

Retourne pour chaque search :
```python
{
    "enabled": True,
    "search_type": "grid" | "random" | "halving_random",
    "refit_metric": str,
    "cv_splits": int,
    "best_score": float,
    "best_params": {param: value},          # prefix "model__" retiré
    "best_params_full": {param: value},     # avec prefix
    "param_grid": dict,                     # grille utilisée
    "n_candidates": int,
    "cv_results_summary": [                 # top 20 combinaisons
        {
            "rank": int,
            "mean_test_score": float,
            "std_test_score": float,
            "mean_train_score": float,
            "overfit_gap": float,           # mean_train - mean_test
            "mean_fit_time_s": float,
            "params": dict,
            # Halving uniquement :
            "halving_iter": int,
            "n_resources": int,
        }
    ],
    "sample_weight_used": bool,
    "early_stopping": {"used": bool, ...},  # mode "none" uniquement
}
```

---

## 5. Module `schema.py`

### 5.1 `MODEL_HP_SCHEMA` (ligne 104)

Schéma de validation pour chaque modèle. Structure par paramètre :

```python
{
    "type": "int" | "float" | "int_or_none" | "float_or_enum" | "enum" | "enum_or_null" | "str",
    "default": valeur_par_défaut,
    "min": float,           # borne inférieure (inclusive)
    "max": float,           # borne supérieure (inclusive)
    "gt": float,            # borne inférieure (exclusive)
    "enum": [valeurs],      # valeurs autorisées pour les types enum
    "grid_values": [...],   # valeurs proposées dans l'UI
    "help": str,            # texte d'aide
    "supported_in": ["classification"] | ["regression"] | None,
}
```

**Types spéciaux** :
- `int_or_none` : entier ou `null` (ex: `max_depth=None` = illimité)
- `float_or_enum` : float ou string (ex: `max_features="sqrt"` ou `0.5`)
- `enum_or_null` : enum ou `null` (ex: `class_weight=None` = désactivé)

### 5.2 `normalize_model_hyperparams()` (ligne 744)

Flux de normalisation :

```
input: model_key, params_bruts, use_grid_search, task_type
    │
    ├── 1. Résolution du modèle (normalize_name)
    ├── 2. Parsing des clés (case-insensitive lookup dans schema)
    │       Clé inconnue → warning HP_UNKNOWN, ignorée
    ├── 3. Pour chaque param dans source :
    │       Si list + use_grid_search=True  → param_grid[param] = [coerce(v) for v in list]
    │       Si list + use_grid_search=False → warning HP_LIST_IGNORED, prend premier élément
    │       Si scalar                       → estimator_params[param] = coerce(value)
    ├── 4. Validations model-spécifiques :
    │       SVM kernel=linear → retire gamma (incompatible)
    │       DecisionTree + regression → retire criterion (classification only)
    │       Tout modèle + regression → retire class_weight
    └── output: {effective, estimator_params, param_grid, issues, warnings, errors}
```

---

## 6. Module `orchestrator.py`

### 6.1 Construction du `param_grid` (ligne 364)

```python
# Récupération des hyperparams bruts de l'utilisateur
requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})

# Normalisation : si use_grid_search=True, les listes → param_grid
hp_normalized = normalize_model_hyperparams(
    model_type_norm,
    requested_hyperparams_raw,
    use_grid_search=bool(cfg.use_grid_search),
    task_type=cfg.task_type,
)

estimator_hyperparams = dict(hp_normalized.get("estimator_params") or {})
param_grid = dict(hp_normalized.get("param_grid") or {})
# param_grid sera {} si l'user n'a rien envoyé en mode grid
# → trainer utilisera alors get_adaptive_model_grid()
```

### 6.2 Détection de l'imbalance (ligne 497)

```python
_active_balancing = decision.strategy not in {"none", "threshold_optimization"}
_is_imbalanced = (
    cfg.task_type == "classification"
    and profile.imbalance_ratio is not None
    and profile.imbalance_ratio > 3.0      # seuil : ratio majoritaire/minoritaire > 3
    and not _active_balancing              # pas déjà compensé par SMOTE/undersampling
)
# Si True → get_adaptive_model_grid() injectera class_weight=["balanced", None]
#            dans la grille pour les modèles supportant class_weight
```

### 6.3 Embedding du resampler (trainer.py, ligne 531)

**Anti-leakage pattern** :
```python
# Resampler DANS le search estimator (pas autour)
if effective_resampler is not None:
    from imblearn.pipeline import Pipeline as ImbPipeline
    search_estimator = ImbPipeline([
        ("resampler", effective_resampler),  # ← appliqué sur train fold uniquement
        ("model", pipeline.named_steps["model"])
    ])
else:
    search_estimator = pipeline
```

---

## 7. Flux de données complet

```
[Frontend]
  cfg.search_type = "none" | "grid" | "random" | "halving_random"
  cfg.model_hyperparams = {model: {param: scalar_or_list}}
  cfg.metrics = [metric_names]
  cfg.balancing.strategy = "smote" | "class_weight" | ...
        │
        ▼
[Orchestrator]
  1. normalize_model_hyperparams()
     ├── use_grid_search=True  → listes deviennent param_grid
     └── use_grid_search=False → listes ignorées, premier élément utilisé

  2. Détection imbalance
     profile.imbalance_ratio > 3.0 AND pas de balancing actif → _is_imbalanced=True

  3. Build resampler (SMOTE/undersampling)
     Adapt k_neighbors selon min_class_count et cv_splits

  4. Trainer.fit(model_param_grid=param_grid, n_samples, imbalanced, resampler)
        │
        ▼
[Trainer]
  _build_cv_splitter()
    → StratifiedKFold ou KFold, cv = min(requested, n_samples, min_class_count)

  _choose_refit_metric()
    → average_precision / f1_macro / f1 / r2 selon contexte

  _adapt_resampler_for_cv()
    → safe_k calculé, resampler ajusté ou désactivé

  Embed resampler dans ImbPipeline (anti-leakage)

  Branch search_type :
    "none"           → early stopping (XGB/LGBM) ou fit direct
    "halving_random" → HalvingRandomSearchCV(distributions, n_candidates adaptatif)
    "random"         → RandomizedSearchCV(distributions, n_iter adaptatif)
    "grid"           → GridSearchCV(grille adaptative ou param_grid utilisateur)
        │
        ▼
[TrainerFitResult]
  .fitted_pipeline = best_estimator_ (refit sur tout X_train)
  .tuning_artifacts = {
      enabled, search_type, best_params, cv_results_summary,
      n_candidates, overfit_gap, early_stopping, ...
  }
        │
        ▼
[Orchestrator → DB → Frontend]
  artifacts_json["hyperparams"] = {requested, effective, param_grid, best}
  metrics_json = résultats d'évaluation (accuracy, f1, roc_auc, npv, mcc, ...)
```

---

## 8. Méthodes de recherche — Comparatif

| Critère | `none` | `grid` | `random` | `halving_random` |
|---------|:------:|:------:|:--------:|:----------------:|
| **Source candidats** | Params fixes user | `_ADAPTIVE_GRIDS` ou param_grid user | Distributions continues `models.py` | Distributions continues `models.py` |
| **Nb candidats** | 1 | produit cartésien | 10–80 (adaptatif) | 15–100 initiaux (factor=3) |
| **Exploration espace** | Nulle | Exhaustive sur grille | Aléatoire continue | Successive halving |
| **Coût computationnel** | Très faible | Moyen à élevé | Modéré | Faible à modéré |
| **Qualité exploration** | Fixe | Bonne (si grille bien choisie) | Très bonne | Excellente (ratio coût/qualité) |
| **Early stopping XGB/LGB** | Oui | Non | Non | Non |
| **Recommandé pour** | Production / démo rapide | Contrôle précis | Datasets moyens | Datasets médicaux (recommandé) |
| **Adaptatif à la taille** | N/A | Oui (6 catégories) | Oui (n_iter) | Oui (n_candidates) |
| **Grille modifiable par user** | Oui (valeurs fixes) | Non (backend) | Non (backend) | Non (backend) |

### Principe du Successive Halving

```
Round 1 : 100 candidats × 25% des données  → garder top 33
Round 2 :  33 candidats × 50% des données  → garder top 11
Round 3 :  11 candidats × 75% des données  → garder top  4
Round 4 :   4 candidats × 100% des données → best 1

Total fits : 100+33+11+4 = 148 (vs 60 pour random sur même budget)
→ 2.5× plus de candidats explorés pour le même coût
```

---

## 9. Classification des datasets par taille

### 9.1 Catégories

| Catégorie | N samples | Combos GridSearch | n_iter Random | Candidats Halving |
|-----------|:---------:|:-----------------:|:-------------:|:-----------------:|
| **micro** | < 100 | ≤ 6 | 10 | 15 |
| **tiny** | 100 – 499 | ≤ 20 | 25 | 30 |
| **small** | 500 – 1999 | ≤ 80 | 40 | 60 |
| **medium** | 2000 – 49999 | ≤ 100 | 60 | 100 |
| **large** | ≥ 50000 | ≤ 20 | 80 | 50 |

### 9.2 Règles micro (< 100 lignes)

Avec CV 3-folds → ~65 samples d'entraînement par fold. Contraintes imposées :
- `max_depth` : jamais `None` — limité à [2, 3] (DT) ou [3, 5] (RF/ET/XGB)
- `min_samples_leaf` : [4, 8] — chaque feuille couvre ≥6% des données
- `n_estimators` : non exploré (valeur default utilisée)
- `num_leaves` LightGBM : [7, 15] — environ log2(65)

**Raison** : sur 65 samples, `max_depth=None` → arbre qui mémorise chaque patient → accuracy CV=100% → performance réelle médiocre (overfitting sévère).

### 9.3 Avantages de la classification

1. **Éviter l'overfitting du search** : explorer 100 HP sur 65 samples = apprendre le bruit, pas le signal
2. **Contrôler le temps de calcul** : XGBoost grille statique = 432 combos → avec micro = 4 combos
3. **Adapter la régularisation** : plus le dataset est petit, plus la régularisation doit être forte

---

## 10. Points forts et limites

### Points forts

| Point fort | Détail |
|-----------|--------|
| **Anti-leakage rigoureux** | SMOTE appliqué per-fold via imblearn.Pipeline |
| **Adaptivité multi-couche** | Taille + imbalance + task type influencent chaque décision |
| **Early stopping intégré** | XGBoost/LightGBM trouvent n_estimators optimal sans search |
| **Graceful degradation** | HalvingRandomSearch indisponible → fallback Random silencieux |
| **Métriques médicales** | NPV, MCC, Youden Index, specificity calculés |
| **Registre centralisé** | 12 modèles, factory pattern, capacités uniformes |
| **CV adaptatif** | StratifiedKFold avec cap sur min_class_count |

### Limites

| Limite | Impact | Contournement possible |
|--------|--------|----------------------|
| Pas de recherche Bayésienne (Optuna) | Moins efficace que TPE pour > 6 dimensions | Futur : ajouter `search_type="bayesian"` |
| param_grid user non validé en taille | Produit cartésien peut exploser si user passe 5+ valeurs par dim | Ajouter cap dans `normalize_model_hyperparams` |
| Early stopping uniquement en mode "none" | Perdu quand search actif | Architecture contrainte sklearn |
| SMOTE k_neighbors dégradé sur micro | Qualité du resampling réduite | Utiliser `class_weight` sur micro à la place |
| Pas de warm_start | Chaque search repart de zéro | Pas critique pour la plupart des cas |

---

## 11. Valeurs de référence

| Paramètre | Valeur | Localisation |
|-----------|--------|-------------|
| Seuil micro | < 100 samples | `models.py:_n_samples_to_size_cat()` |
| Seuil tiny | < 500 samples | idem |
| Seuil small | < 2000 samples | idem |
| Seuil medium | < 50000 samples | idem |
| Imbalance detection | ratio > 3.0 | `orchestrator.py:~497` |
| Refit metric auto (binaire déséquilibré) | `average_precision` | `trainer.py:_choose_refit_metric()` |
| Min samples pour early stopping | 50 | `trainer.py:~473` |
| ES val split size | 20% | `trainer.py:_fit_with_early_stopping()` |
| Early stopping rounds | 50 | `trainer.py:_EARLY_STOPPING_ROUNDS` |
| HalvingRandom factor | 3 | `trainer.py:HalvingRandomSearchCV(factor=3)` |
| Min resources floor (SMOTE) | 20 samples | `trainer.py:_compute_min_resources()` |
| Min CV splits | 2 | `trainer.py:_build_cv_splitter()` |
| Max cv_results_summary | 20 lignes | `trainer.py:_summarize_cv_results()` |
| VarianceThreshold | 0.01 | `preprocessing.py` |
| SMOTE k_neighbors default | 5 | `orchestrator.py` |



---------------------------------------------------------------------------------------------------

1. Anti-leakage SMOTE/undersampling
Le problème sans anti-leakage
Imagine un dataset médical : 900 sains + 100 malades.

Mauvaise approche (leakage) :


Dataset complet (1000)
    ↓ SMOTE appliqué ICI → crée 800 synthétiques → 900 sains + 900 malades
    ↓ Split CV fold 1
    ├── Train (80%) : 1440 samples (dont synthétiques)
    └── Val   (20%) : 360 samples  (dont synthétiques issus des mêmes originaux)
Le modèle valide sur des samples synthétiques interpolés depuis ses données d'entraînement → la performance CV est optimiste → les métriques ne représentent pas la réalité.

Bonne approche (anti-leakage) :


Dataset complet (1000)
    ↓ Split CV fold 1 D'ABORD
    ├── Train fold (800) : 720 sains + 80 malades
    │       ↓ SMOTE appliqué ICI SEULEMENT → 720 sains + 720 malades
    │       ↓ Modèle entraîné sur ces 1440
    └── Val fold (200) : 180 sains + 20 malades ← JAMAIS touché par SMOTE
            ↓ Évaluation sur distribution réelle
Les métriques CV reflètent la vraie performance sur des patients non vus.

Comment c'est implémenté dans le code
trainer.py lignes 531–546 :


# Le resampler est DANS le search estimator (pas autour)
search_estimator = ImbPipeline([
    ("resampler", effective_resampler),  # ← s'exécute sur train fold seulement
    ("model", pipeline.named_steps["model"])
])

# GridSearchCV/RandomizedSearchCV/HalvingRandom applique ce pipeline
# par fold → SMOTE ne voit jamais le fold de validation
gs = GridSearchCV(estimator=search_estimator, ...)
Si le resampler était autour du GridSearchCV → SMOTE sur tout le dataset avant le CV → leakage.

2. Déséquilibre : class_weight automatique + métrique adaptative
2a. Injection automatique de class_weight
Condition (orchestrator.py ligne 497) :


_is_imbalanced = (
    imbalance_ratio > 3.0          # ex: 900 sains / 100 malades = ratio 9 → True
    AND not _active_balancing       # l'utilisateur N'A PAS déjà choisi SMOTE/undersampling
)
Si _is_imbalanced = True et que le modèle supporte class_weight (RF, LogReg, SVM, DT, ExtraTrees) → la grille GridSearch injecte automatiquement :


grid["class_weight"] = ["balanced", None]
# GridSearch testera les deux et choisira le meilleur
2b. Sélection adaptative de la métrique de refit
trainer.py:_choose_refit_metric() :


if minority_ratio < 0.20:          # classe minoritaire < 20% du dataset
    return "average_precision"     # PR-AUC → pénalise les faux positifs sur la minorité

elif multiclasse déséquilibré:
    return "f1_macro"              # donne le même poids à chaque classe
Pourquoi average_precision et pas accuracy ?

Sur 900 sains + 100 malades, un modèle qui prédit toujours "sain" a :

accuracy = 90% → paraît excellent
average_precision ≈ 0.10 → révèle que les malades ne sont jamais détectés
C'est critique en médical — les faux négatifs (malades classés sains) sont les erreurs les plus coûteuses.

2c. Si l'utilisateur choisit déjà une méthode d'équilibrage
_active_balancing (orchestrator.py) :


_active_balancing = decision.strategy not in {"none", "threshold_optimization"}
Stratégie choisie	_active_balancing	Injection class_weight grille	Métriq. refit
none	False	Oui si ratio > 3	Adaptative
threshold_optimization	False	Oui si ratio > 3	Adaptative
smote	True	Non — SMOTE gère déjà l'équilibre	Adaptative
smote_tomek	True	Non	Adaptative
random_undersampling	True	Non	Adaptative
class_weight (strategy)	True	Non — déjà fixé dans estimator	Adaptative
Logique : si l'utilisateur a déjà choisi SMOTE → ajouter class_weight="balanced" en plus serait une double compensation → le modèle surpondèrerait les malades deux fois → biais dans l'autre sens.

La métrique de refit adaptative (average_precision, f1_macro) s'applique toujours, quelle que soit la stratégie d'équilibrage choisie — c'est indépendant.