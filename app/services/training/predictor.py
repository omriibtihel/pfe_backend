from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from app.models.training import TrainedModel
from .persistence import load_pipeline


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    name = (filename or "").strip().lower()
    if not content:
        raise RuntimeError("Uploaded file is empty.")

    buff = BytesIO(content)
    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(buff)
    if name.endswith(".json"):
        return pd.read_json(buff)
    if name.endswith(".parquet"):
        return pd.read_parquet(buff)
    raise RuntimeError("Unsupported file type. Use CSV, JSON or Parquet.")


def _to_builtin(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _predict_with_optional_scores(pipeline: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    y_pred = pipeline.predict(X)
    y_score = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X)
    except Exception:
        y_score = None
    return np.asarray(y_pred), y_score


def _prediction_preview(y_pred: np.ndarray, y_score: np.ndarray | None, limit: int = 20) -> list[Dict[str, Any]]:
    n = min(int(len(y_pred)), int(limit))
    out: list[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {
            "row_index": int(i),
            "prediction": _to_builtin(y_pred[i]),
        }
        if y_score is not None and getattr(y_score, "ndim", 0) == 2 and y_score.shape[0] > i:
            if y_score.shape[1] == 2:
                row["score"] = float(y_score[i, 1])
            elif y_score.shape[1] > 2:
                row["score"] = float(np.max(y_score[i]))
        out.append(row)
    return out


def predict_with_trained_model(trained_model: TrainedModel, raw_df: pd.DataFrame) -> Dict[str, Any]:
    artifacts = trained_model.artifacts_json or {}
    model_pkl = artifacts.get("model_pkl")
    if not model_pkl:
        raise RuntimeError("Model artifact path is missing (model_pkl).")

    pkl_path = Path(str(model_pkl))
    if not pkl_path.exists():
        raise RuntimeError(f"Model file not found: {pkl_path}")

    pipeline = load_pipeline(pkl_path)
    y_pred, y_score = _predict_with_optional_scores(pipeline, raw_df)

    schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names = schema.get("feature_names") if isinstance(schema.get("feature_names"), list) else []

    return {
        "model_id": int(trained_model.id),
        "model_type": str(trained_model.model_type),
        "task_type": str(trained_model.task_type),
        "dataset_version_id": artifacts.get("dataset_version_id"),
        "timestamp": _utc_now_iso(),
        "n_rows": int(len(raw_df)),
        "feature_count_received": int(raw_df.shape[1]),
        "feature_count_expected": int(len(feature_names)) if feature_names else None,
        "preview": _prediction_preview(y_pred, y_score, limit=20),
    }
