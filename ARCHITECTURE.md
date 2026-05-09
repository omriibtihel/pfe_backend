# Architecture — MedIQ Backend

## Vue d'ensemble

Le projet suit une architecture **hybride couches × domaines** en deux niveaux.

**Niveau 1 — couches techniques** à la racine de `app/` :

```
app/
├── api/                # Couche HTTP : routes, dépendances FastAPI
│   ├── routes/         # Un fichier par domaine (training.py, nettoyage.py…)
│   ├── utils_shared/   # Helpers partagés propres à la couche HTTP
│   └── utils/          # Barrels de rétrocompat → utils_shared/ ou services/
├── services/           # Logique métier, organisée par domaine fonctionnel
├── schemas/            # Schémas Pydantic I/O, organisés par domaine
├── models/             # Entités SQLAlchemy
├── crud/               # Accès données (queries, mutations)
├── core/               # Config, sécurité, utilitaires transversaux
├── db/                 # Session SQLAlchemy, base déclarative
└── tasks/              # Workers Celery
```

**Niveau 2 — sous-dossiers par domaine** dans les couches concernées :

```
services/
├── training/           # Pipeline ML complet (intouchable — vagues 1-3 finalisées)
├── nettoyage/          # Nettoyage tabulaire sûr (sans leakage)
├── data/               # Chargement, preview, profiling de données brutes
├── preparation_ml/     # Préprocessing, feature engineering, balancing (intouchable)
└── shap/               # Explainabilité SHAP globale et locale

schemas/
└── training/           # Schémas training découpés (base, config, results, intelligence)
```

---

## Règles de couches

### `api/`
- Contient uniquement du code HTTP : déclaration de routes, lecture des paramètres de requête, sérialisation des réponses.
- **Interdit** : logique métier, manipulation de DataFrames, accès direct à la base de données hors injection de `Session` via `Depends`.
- Importe depuis `services/`, `schemas/`, `crud/`, `core/`. N'est jamais importé par `services/`.

### `services/`
- Contient toute la logique métier.
- **Interdit** : importer depuis `app.api.*` (violation de la hiérarchie descendante).
- Exception documentée : `services/nettoyage/rebuild.py` et `services/nettoyage/df_utils.py` importent temporairement depuis `api/utils/` — à corriger quand ces helpers auront migré vers `api/utils_shared/` ou `core/`.

### `schemas/`
- Schémas Pydantic d'entrée/sortie uniquement. Aucune logique, aucun accès DB.
- Les schémas d'un domaine peuvent importer des schémas d'un autre domaine (ex : `schemas/training/base.py` re-exporte des types depuis `schemas/preparation.py`).

### `models/`
- Déclarations SQLAlchemy pures. Aucune logique métier.
- `db/base.py` importe tous les modèles pour que les migrations Alembic les détectent.

### `crud/`
- Requêtes et mutations SQL uniquement. Reçoit une `Session`, retourne des objets ORM ou des scalaires.
- N'instancie jamais de logique métier, ne lève jamais d'`HTTPException` (lever des exceptions Python ordinaires).

### `core/`
- Configuration applicative, sécurité (JWT, hashing), utilitaires transversaux sans domaine propre.
- Peut être importé par toutes les autres couches.

### `db/`
- Session factory, base déclarative, imports de tous les modèles pour Alembic.

### `tasks/`
- Workers Celery. Importent depuis `services/` pour déléguer le travail.

### Règle générale de dépendance
```
tasks → services → crud → models
               ↘ schemas
api → services / schemas / crud / core
core → (rien dans app/)
```
Les dépendances sont toujours **descendantes**. Aucun import circulaire.

---

## Conventions de fichiers

| Convention | Règle |
|---|---|
| Taille maximale | 350 lignes. Au-delà, découper en sous-modules. |
| Responsabilité | Un fichier = une responsabilité unique. |
| Barrel obligatoire | Tout dossier exposant une API publique doit avoir un `__init__.py` qui re-exporte les symboles publics. |
| Nommage | `snake_case` pour les fichiers et dossiers. |
| Commentaires | Uniquement pour les invariants non évidents ou les contournements de bugs. Pas de docstrings narratives. |
| Imports | Absolus (`from app.services.training.pipeline.trainer import Trainer`). Pas d'imports relatifs. |
| Nouveaux fichiers | Vérifier qu'aucun fichier existant ne peut accueillir la responsabilité avant de créer un nouveau. |

---

## Carte des domaines

| Domaine | Couche | Fichiers principaux | Responsabilité |
|---|---|---|---|
| **training** | `services/training/` | `orchestrator.py`, `training_service.py`, `notifier.py`, `presenter.py` | Entraînement ML complet : orchestration, pipeline, CV, AutoML, SSE, persistence |
| **training / config** | `services/training/config/` | `schema/training_config.py`, `builder.py`, `normalization.py`, `validation.py` | Schéma de configuration, normalisation, builder intelligent/manuel |
| **training / pipeline** | `services/training/pipeline/` | `trainer.py`, `evaluator.py`, `splitters.py`, `metrics/`, `models/` | Exécution du pipeline ML, métriques, registre des modèles |
| **training / intelligence** | `services/training/intelligence/` | `recommender.py`, `meta_learner.py`, `metric_selector.py`, `imbalance_handler.py` | Mode intelligent : recommandations, méta-apprentissage |
| **training / output** | `services/training/output/` | `predictor.py`, `reporter.py`, `persistence.py`, `audit.py` | Prédiction, génération des artefacts, persistance DB |
| **nettoyage** | `services/nettoyage/` | `rebuild.py`, `df_utils.py` | Opérations de nettoyage tabulaire sans leakage statistique |
| **data** | `services/data/` | `loader.py`, `preview.py`, `profiler.py`, `column_inference.py` | Chargement de fichiers, preview, inférence de types de colonnes |
| **preparation_ml** | `services/preparation_ml/` | `preprocessing/`, `feature_engineering/`, `balancing/`, `splitters.py` | Préprocessing sklearn, feature engineering AST, balancing imbalanced |
| **shap** | `services/shap/` | `global_shap.py`, `local_shap.py`, `_explainer.py` | Explainabilité SHAP globale (feature importance) et locale (par prédiction) |
| **schemas / training** | `schemas/training/` | `base.py`, `config.py`, `results.py`, `intelligence.py`, `__init__.py` | Schémas Pydantic du domaine training, découpés par responsabilité |

---

## Dette technique connue

Aucune dette active. La suite de tests passe à **264 / 0** (0 échec).

---

## Guide du contributeur

### Où créer un nouveau fichier

| Type de fichier | Emplacement |
|---|---|
| Nouvelle route HTTP | `app/api/routes/<domaine>.py` |
| Nouveau service / logique métier | `app/services/<domaine>/<fichier>.py` |
| Nouveau schéma Pydantic | `app/schemas/<domaine>.py` ou `app/schemas/<domaine>/<fichier>.py` si le domaine a déjà un sous-dossier |
| Nouvelle entité ORM | `app/models/<entité>.py` + import dans `app/db/base.py` |
| Nouvelle query SQL | `app/crud/<domaine>.py` |
| Utilitaire transversal (pas de domaine) | `app/core/<fichier>.py` |
| Worker Celery | `app/tasks/<fichier>.py` |

### Ajouter un nouveau domaine fonctionnel

1. Créer `app/services/<domaine>/` avec un `__init__.py` barrel.
2. Créer les fichiers métier dans ce dossier.
3. Si le domaine expose une API, créer `app/api/routes/<domaine>.py` et enregistrer le router dans `app/main.py`.
4. Si le domaine a des schémas non triviaux (> 3 classes), créer `app/schemas/<domaine>/` avec `__init__.py`.
5. Si le domaine accède à la base de données, créer `app/crud/<domaine>.py`.
6. Mettre à jour la **Carte des domaines** ci-dessus.
7. Ajouter les tests dans `tests/<domaine>/`.

### Modifier un domaine intouchable

`services/training/` et `services/preparation_ml/` sont des domaines stabilisés. Toute modification doit :
- Passer la suite de tests complète sans nouvelle régression.
- Respecter les interfaces publiques exposées via leurs `__init__.py`.
- Faire l'objet d'une revue explicite avant merge.
