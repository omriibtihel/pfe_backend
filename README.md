# pfe_backend

## Training API payload (`/api/projects/{project_id}/training/*`)

Le contrat `preprocessing` est maintenant **par colonne** (compatible backward avec l'ancien format global).

### Request payload (nouveau format)

```json
{
  "datasetVersionId": 12,
  "targetColumn": "Outcome",
  "taskType": "classification",
  "models": ["randomforest"],
  "metrics": ["accuracy", "f1"],
  "splitMethod": "holdout",
  "trainRatio": 70,
  "valRatio": 15,
  "testRatio": 15,
  "kFolds": 5,
  "useGridSearch": false,
  "useSmote": false,
  "preprocessing": {
    "defaults": {
      "numericImputation": "none",
      "numericScaling": "none",
      "categoricalImputation": "none",
      "categoricalEncoding": "none"
    },
    "columns": {
      "age": {
        "use": true,
        "type": "numeric",
        "numericImputation": "median",
        "numericScaling": "standard"
      },
      "city": {
        "use": true,
        "type": "categorical",
        "categoricalImputation": "most_frequent",
        "categoricalEncoding": "onehot"
      },
      "risk_level": {
        "use": true,
        "type": "ordinal",
        "categoricalEncoding": "ordinal",
        "ordinalOrder": ["low", "medium", "high"]
      },
      "patient_id": {
        "use": false
      }
    }
  }
}
```

### Règles

- Si `preprocessing` est absent: `defaults=none`, `columns={}`.
- Si un champ colonne est absent: fallback sur `defaults`.
- `use=false` retire la colonne (drop).
- L'ancien format global (`numericImputation`, `categoricalEncoding`, etc.) reste accepté.

### Capabilities

`GET /training/capabilities` expose:

- `preprocessingCapabilities.numericImputation`
- `preprocessingCapabilities.numericScaling`
- `preprocessingCapabilities.categoricalImputation`
- `preprocessingCapabilities.categoricalEncoding`
- `preprocessingCapabilities.defaults`
- `preprocessingCapabilities.supportsPerColumn=true`

### Validate response example (`POST /training/validate`)

```json
{
  "normalized_config": {
    "datasetVersionId": 12,
    "targetColumn": "Outcome",
    "taskType": "classification",
    "models": ["randomforest"],
    "preprocessing": {
      "defaults": {
        "numericImputation": "none",
        "numericScaling": "none",
        "categoricalImputation": "none",
        "categoricalEncoding": "none"
      },
      "columns": {
        "age": {
          "type": "numeric",
          "numericImputation": "median",
          "numericScaling": "standard"
        },
        "city": {
          "type": "categorical",
          "categoricalEncoding": "onehot"
        }
      }
    }
  },
  "effective_preprocessing_by_column": {
    "age": {
      "use": true,
      "inferredType": "numeric",
      "type": "numeric",
      "numericImputation": "median",
      "numericScaling": "standard",
      "categoricalImputation": "none",
      "categoricalEncoding": "none",
      "passthrough": false
    },
    "city": {
      "use": true,
      "inferredType": "categorical",
      "type": "categorical",
      "numericImputation": "none",
      "numericScaling": "none",
      "categoricalImputation": "none",
      "categoricalEncoding": "onehot",
      "passthrough": false
    }
  },
  "warnings": [],
  "errors": []
}
```
