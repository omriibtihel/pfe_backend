# Rapport : Stockage des donnees dans le pipeline ML
## MedicalVision Backend
**Date** : 2026-04-20

---

## 1. Vue d'ensemble

Dans ce projet, les donnees ne sont pas stockees sous un seul format unique. Le pipeline utilise en pratique **4 couches de stockage** :

1. **Fichiers de donnees sur disque**
   - principalement en **CSV**
   - emplacement racine : `storage/projects/<project_id>/...`

2. **Etat de travail et historique en base SQL**
   - tables `datasets`, `dataset_versions`, `processing_operations`, `training_sessions`, `trained_models`, `balancing_audit`
   - champs complexes en **JSON** ou **JSONB**

3. **Objets ML serialises**
   - pipelines/modeles sauvegardes en **`.pkl`** via `joblib`

4. **Objets temporaires en memoire**
   - `pandas.DataFrame`, `numpy.ndarray`, matrices preprocesses parfois **sparse**
   - ces objets existent pendant la preparation et l'entrainement, mais ne sont pas exportes comme dataset prepare autonome

Le point important est le suivant :

- **apres nettoyage** : les donnees sont persistees en **CSV**
- **apres preparation ML** : les donnees restent surtout **en memoire**, et la logique de preparation est stockee comme **configuration + pipeline**
- **apres modelisation** : le resultat final est stocke comme **pipeline pickle (`.pkl`) + metadonnees JSON en base**

---

## 2. Racine de stockage

La racine physique du stockage projet est :

```text
storage/projects/<project_id>/
```

Sous cette racine, on retrouve notamment :

```text
storage/projects/<project_id>/
|- datasets/
|- dataset_versions/
|- training_models/
|- exports/
`- meta_learning.jsonl
```

---

## 3. Etat brut juste apres upload

Avant meme le nettoyage, il existe deja deux representations :

### 3.1 Dataset source

Le fichier upload par l'utilisateur est stocke dans :

```text
storage/projects/<project_id>/datasets/<uuid>.<ext>
```

Formats acceptes a l'upload :

- `.csv`
- `.xlsx`
- `.xls`

La table `datasets` stocke les metadonnees du fichier source :

- `original_name`
- `stored_name`
- `file_path`
- `content_type`
- `size_bytes`
- `target_column`
- informations de workspace

### 3.2 Version initiale exploitable

Le backend cree aussi une **version initiale normalisee en CSV** dans :

```text
storage/projects/<project_id>/dataset_versions/<uuid>.csv
```

Donc, meme si l'utilisateur charge un Excel, la version de reference pour la suite du pipeline devient un **CSV**.

La table `dataset_versions` stocke pour cette version :

- `name`
- `stored_name`
- `file_path`
- `target_column`
- `can_predict`
- `operations_json`

`operations_json` contient deja un historique minimal, par exemple :

```json
[
  {"type": "original"}
]
```

---

## 4. Apres nettoyage

### 4.1 Ce qui est applique

Le module de nettoyage est volontairement limite a des operations "safe", sans leakage statistique :

- suppression de colonnes
- suppression de doublons
- suppression de lignes vides
- renommage de colonnes
- trim des espaces
- substitution de valeurs

Les imputations, encodages, normalisations et scalings **ne sont pas** stockes ici. Ils sont reportes a la phase ML.

### 4.2 Format physique apres nettoyage

Le dataset nettoye courant est stocke sur disque comme :

```text
processed_dataset_<dataset_id>.csv
```

Ce fichier est cree dans le **meme dossier que le dataset source** (ou du workspace).

Exemples :

```text
storage/projects/12/datasets/processed_dataset_45.csv
storage/projects/12/datasets/workspaces/processed_dataset_87.csv
```

Donc apres nettoyage, le format reel est :

- **CSV**
- ecrit avec `pandas.DataFrame.to_csv(..., index=False)`

### 4.3 Structure en memoire avant sauvegarde

Le nettoyage travaille sur un :

- `pandas.DataFrame`

Puis il est reecrit en CSV. Avant l'ecriture :

- les colonnes `datetime` sont converties en chaine de caracteres au format :

```text
YYYY-MM-DD HH:MM:SS
```

Autrement dit, un `datetime64` Pandas n'est pas persiste comme type natif : il devient du texte dans le CSV.

### 4.4 Historique des operations de nettoyage

Chaque operation est egalement persistee en base dans `processing_operations` :

- `op_type` : ex. `cleaning`
- `description`
- `columns` : **JSONB**
- `params` : **JSONB**
- `created_at`

Le backend ajoute aussi un resume d'effet dans :

```text
params["__result"]
```

Exemples de contenu :

- shape avant/apres
- colonnes ajoutees/supprimees
- nombre de lignes supprimees
- mapping de renommage
- nombre de valeurs modifiees

### 4.5 Sauvegarde durable d'une version nettoyee

Quand l'utilisateur valide le nettoyage, le backend cree une nouvelle version dans :

```text
storage/projects/<project_id>/dataset_versions/<uuid>.csv
```

La version reste donc un **CSV**.

En parallele, la table `dataset_versions` recupere :

- `file_path`
- `size_bytes`
- `target_column`
- `can_predict`
- `operations_json`

Ici `operations_json` est un **texte JSON serialise** qui contient l'historique des operations appliquees.

### 4.6 Point important sur les decisions de schema

Les decisions de schema (`set_kind`, `clear_kind`, `verify_categorical`, `dismiss_alert`) sont elles aussi stockees dans `processing_operations`, mais :

- elles sont stockees **en base**
- elles ne re-ecrivent pas le contenu du CSV

Donc elles representent une **couche de metadonnees**, pas une transformation physique du dataset.

---

## 5. Apres preparation ML

### 5.1 Ce qui se passe reellement

La preparation ML ne cree pas, dans le flux principal actuel, un nouveau fichier du type :

```text
prepared_dataset.csv
preprocessed_dataset.parquet
features_ready.npy
```

Ce stade est principalement **ephemere** et **en memoire**.

Les endpoints de preparation (`validate`, `profile`, `analyze-balance`, `feature-engineering/preview`) lisent une `DatasetVersion` existante, font des calculs, puis renvoient surtout des **reponses JSON API**. Ils ne sauvegardent pas un dataset prepare autonome sur disque.

### 5.2 Formes des donnees pendant la preparation

Pendant l'entrainement, les donnees circulent successivement sous plusieurs formes :

1. **Dataset version charge**
   - `pandas.DataFrame`
   - charge depuis un **CSV** de `dataset_versions`

2. **Apres separation cible/features**
   - `X` : `pandas.DataFrame`
   - `y` : `numpy.ndarray`

3. **Apres split**
   - `X_train`, `X_val`, `X_test` : `pandas.DataFrame`
   - `y_train`, `y_val`, `y_test` : `numpy.ndarray`

4. **Apres alignement schema**
   - `ColumnAligner` reconstruit un `DataFrame` avec :
     - colonnes manquantes ajoutees en `NaN`
     - colonnes supplementaires ignorees
     - ordre de colonnes identique a l'entrainement

5. **Apres feature engineering**
   - toujours un `pandas.DataFrame`
   - nouvelles colonnes calculees a partir d'expressions

6. **Apres preprocessing sklearn**
   - sortie de `ColumnTransformer`
   - peut etre :
     - une matrice dense `numpy`
     - ou une matrice **sparse**
   - en particulier quand `OneHotEncoder` est utilise

7. **Apres selection de variables**
   - matrice transformee par `VarianceThreshold`

8. **Apres balancing**
   - donnees de train potentiellement re-echantillonnees
   - toujours **en memoire**
   - jamais reecrites comme fichier dataset final

### 5.3 Ce qui est persiste a la place du dataset prepare

Au lieu de sauver un dataset prepare sur disque, le backend persiste :

#### a. Le schema d'entrainement

Dans `artifacts_json["training_schema"]`, on retrouve :

- `feature_names`
- `dtypes`
- `target`
- `preprocessing_config`
- `column_stats`
- `created_at`

Donc le backend ne stocke pas "les valeurs preparees", mais **la description du schema attendu**.

#### b. Le resume de preprocessing

Dans `artifacts_json["preprocessing"]`, on retrouve notamment :

- `selectedMethods`
- `effectiveByColumn`
- `droppedColumns`
- `columnTypes`
- `execution`
- `applied`

Ce bloc documente :

- quelles colonnes sont numeriques/categorielles
- quelles methodes sont appliquees
- quelles colonnes sont retirees
- la forme avant/apres transformation sur echantillon

#### c. Les colonnes de travail

Dans `artifacts_json["columns"]` :

- `numeric`
- `categorical`

#### d. L'etat de balancing

Dans `artifacts_json["balancing"]` et parfois `artifacts_json["balancing_cv"]` :

- strategie appliquee
- ratio de desequilibre
- `class_counts`
- `optimal_threshold`
- warnings
- details SMOTE

### 5.4 Conclusion sur cette phase

Apres preparation ML, les donnees sont donc stockees :

- **en memoire** sous forme `DataFrame` / `ndarray` / matrices sklearn
- **en metadonnees JSON** dans les artefacts d'entrainement
- **pas** comme fichier prepare persistant autonome dans le pipeline principal

---

## 6. Apres modelisation / entrainement

### 6.1 Session d'entrainement

Chaque lancement cree un enregistrement `training_sessions` avec :

- `project_id`
- `dataset_version_id`
- `status`
- `progress`
- `current_model`
- `config_json`
- `error_message`
- timestamps

`config_json` stocke la configuration d'entrainement envoyee par le frontend :

- version de dataset
- target
- split
- preprocessing
- feature engineering
- balancing
- modeles choisis
- hyperparametres

Format :

- **JSON** en base

### 6.2 Fichier du modele entraine

Le modele final est ecrit dans :

```text
storage/projects/<project_id>/training_models/<session_id>/<model_type>.pkl
```

Format :

- **`.pkl`**
- serialisation **`joblib.dump(...)`**

Ce `.pkl` contient en general le **pipeline complet d'inference** :

```text
align -> feature engineering -> preprocessing -> selection -> dense(optional) -> model
```

Donc apres modelisation, ce n'est pas seulement "le modele" qui est stocke, mais tout le pipeline necessaire pour reappliquer la preparation aux nouvelles donnees.

### 6.3 Cas manuel classique

Dans le mode manuel, le `.pkl` contient un pipeline sklearn complet compose de :

- `ColumnAligner`
- `FeatureEngineeringTransformer` (si actif)
- `ColumnTransformer` de preprocessing
- `VarianceThreshold` (si actif)
- conversion dense optionnelle
- estimateur final

### 6.4 Cas AutoML

En AutoML, le meilleur resultat est stocke comme un objet `AutoMLPipeline` qui encapsule :

- le preprocessor sklearn
- l'objet FLAML AutoML
- le schema de colonnes d'entree attendu

Donc le meilleur modele AutoML est lui aussi stocke comme **pipeline picklable complet**.

Note utile :

- le meilleur modele AutoML est enveloppe avec son preprocessing
- les resultats secondaires par estimateur peuvent etre persistes individuellement

### 6.5 Metadonnees du modele en base

Pour chaque modele entraine, la table `trained_models` stocke :

- `session_id`
- `project_id`
- `model_type`
- `task_type`
- `is_saved`
- `metrics_json`
- `artifacts_json`

Formats :

- `metrics_json` : **JSON**
- `artifacts_json` : **JSON**

### 6.6 Contenu typique de `metrics_json`

Selon le mode (holdout, CV, LOO, AutoML), `metrics_json` peut contenir :

- `train`
- `val`
- `test`
- `split_method`
- `cv`
- `fold_results`
- `cv_summary`
- `holdout_test_metrics`
- `training_time_sec`
- `threshold_used`
- `warnings`

Donc les **resultats numeriques d'evaluation** sont stockes en **JSON**, pas dans un fichier separe.

### 6.7 Contenu typique de `artifacts_json`

`artifacts_json` contient le contexte technique du modele :

- `model_pkl`
- `dataset_version_id`
- `split_info`
- `columns`
- `preprocessing`
- `training_schema`
- `balancing`
- `thresholding`
- `model`
- `feature_importance`
- `grid_search`
- `curves`
- `confusion_matrix`
- `refit_info` (CV)

Autrement dit :

- le **binaire de modelisation** est dans le `.pkl`
- les **metadonnees de modelisation** sont dans `artifacts_json`

### 6.8 Audit de balancing

Si un balancing est utilise, une trace structuree supplementaire est stockee dans `balancing_audit`.

Formats de donnees persistees :

- colonnes scalaires SQL classiques
- `class_counts` en **JSONB**
- `audit_flags` en **JSONB**
- `warnings` en **JSONB**

Cette table stocke notamment :

- taille du dataset
- niveau de desequilibre
- strategie appliquee
- rationale
- seuil optimal
- gain F1
- details SMOTE

### 6.9 Historique meta-learning

Apres la session, le meilleur resultat peut aussi etre ajoute dans :

```text
storage/projects/<project_id>/meta_learning.jsonl
```

Format :

- **JSON Lines**
- 1 entrainement = 1 ligne JSON

On y trouve par exemple :

- caracteristiques du dataset
- meilleur algorithme
- meilleurs hyperparametres
- score
- metric utilisee
- split method
- search type
- duree d'entrainement

---

## 7. Tableau de synthese par etape

| Etape | Representation principale | Format reel | Persistance |
|------|----------------------------|-------------|-------------|
| Upload source | fichier utilisateur | `.csv`, `.xlsx`, `.xls` | disque + table `datasets` |
| Version initiale | copie normalisee | `.csv` | disque + table `dataset_versions` |
| Nettoyage courant | `DataFrame` puis export | `.csv` (`processed_dataset_<id>.csv`) | disque |
| Historique nettoyage | operations | `JSONB` / texte JSON | base SQL |
| Version nettoyee | snapshot du dataset | `.csv` | disque + table `dataset_versions` |
| Preparation ML | `DataFrame`, `ndarray`, matrices sklearn/sparse | memoire | non persistee comme dataset autonome |
| Schema de preparation | metadonnees techniques | `JSON` | `artifacts_json` / `config_json` |
| Modele entraine | pipeline complet | `.pkl` | disque |
| Metriques du modele | resultats numeriques | `JSON` | table `trained_models.metrics_json` |
| Artefacts du modele | contexte technique | `JSON` | table `trained_models.artifacts_json` |
| Audit balancing | trace structuree | SQL + `JSONB` | table `balancing_audit` |
| Memoire d'apprentissage | historique projet | `.jsonl` | disque |

---

## 8. Reponse directe a la question

### Apres un nettoyage

Les donnees sont stockees :

- physiquement en **CSV**
- dans un fichier `processed_dataset_<dataset_id>.csv`
- avec historique des operations en **base SQL** (`processing_operations`)
- et, si l'utilisateur sauvegarde la version, dans `dataset_versions/<uuid>.csv`

### Apres une preparation ML

Les donnees ne sont pas stockees comme nouveau dataset fichier dans le flux principal. Elles existent surtout :

- en **memoire** (`DataFrame`, `ndarray`, matrices sparse/denses)
- et sous forme de **metadonnees JSON** qui decrivent le schema, les transformations et la logique appliquee

### Apres une modelisation

Le resultat est stocke sous 3 formes complementaires :

1. **pipeline/model en `.pkl`**
2. **metriques en `metrics_json`**
3. **artefacts techniques en `artifacts_json`**

et eventuellement :

4. **audit de balancing** en base
5. **historique meta-learning** en `meta_learning.jsonl`

---

## 9. Conclusion

Le backend suit une logique tres claire :

- **les datasets persistants** sont surtout en **CSV**
- **les etats descriptifs** sont en **JSON / JSONB**
- **les objets ML deployables** sont en **PKL**
- **la preparation ML n'est pas materialisee comme dataset final distinct**, elle est encapsulee dans le pipeline entraine

Si tu veux, je peux maintenant te faire une **version 2 du rapport sous forme de schema visuel** (workflow avec boites et fleches), ou une **version plus academique pour memoire/PFE**.
