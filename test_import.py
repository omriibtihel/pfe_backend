import pandas as pd
from app.services.training import TrainingConfig
from app.services.training.orchestrator import run_one_model

def _demo_run() -> None:
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 35, 60, 25, 45, 55, 33, 29, 41],
            "sex": ["M", "F", "F", "M", "F", "M", "M", "F", "F", "M", "F", "M"],
            "bp": [120, 130, 125, 140, 128, 150, 118, 135, 145, 132, 129, 138],
            "Outcome": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        }
    )

    cfg = TrainingConfig.from_front(
        {
            "targetColumn": "Outcome",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1", "roc_auc", "pr_auc"],
            "splitMethod": "holdout",
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": 5,
            "useGridSearch": False,
            "useSmote": False,
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )

    res = run_one_model(df, cfg, "randomforest")
    print(res.metrics_json)
    print(res.artifacts_json["split_info"])


if __name__ == "__main__":
    _demo_run()
