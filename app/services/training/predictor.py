from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.models.training import TrainedModel
from .persistence import load_pipeline

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# File parsing
# ──────────────────────────────────────────────────────────────────────────────

def read_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    name = (filename or "").strip().lower()
    if not content:
        raise RuntimeError("Uploaded file is empty.")

    buff = io.BytesIO(content)
    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(buff)
    if name.endswith(".json"):
        return pd.read_json(buff)
    if name.endswith(".parquet"):
        return pd.read_parquet(buff)
    raise RuntimeError("Unsupported file type. Use CSV, JSON or Parquet.")


# ──────────────────────────────────────────────────────────────────────────────
# Feature schema validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_feature_schema(raw_df: pd.DataFrame, training_schema: dict) -> pd.DataFrame:
    """
    Validate that raw_df columns match the training schema.

    - Missing columns → RuntimeError with explicit list.
    - Extra columns   → warning log, columns are ignored.
    - Returns a DataFrame with columns reordered to match training_schema order.
    """
    feature_names: List[str] = training_schema.get("feature_names") or []
    if not feature_names:
        # No schema stored — skip validation, pass through as-is
        logger.warning("predict.schema_validation_skipped: no feature_names in training_schema")
        return raw_df

    received = set(raw_df.columns.tolist())
    expected = set(feature_names)

    missing = sorted(expected - received)
    if missing:
        raise RuntimeError(
            f"Input data is missing {len(missing)} required column(s): {missing}. "
            f"Expected columns: {sorted(feature_names)}"
        )

    extra = sorted(received - expected)
    if extra:
        logger.warning(
            "predict.extra_columns_ignored: columns=%s will be dropped before inference",
            extra,
        )

    # Reorder to match training order (ColumnAligner inside pipeline will handle the rest)
    return raw_df[feature_names]


# ──────────────────────────────────────────────────────────────────────────────
# Threshold helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_optimal_threshold(artifacts_json: dict, task_type: str) -> float:
    """
    Extract the calibrated optimal_threshold from artifacts.

    Looks in artifacts_json["thresholding"]["optimal_threshold"].
    Returns 0.5 if absent, None, or task is not classification.
    """
    if str(task_type).lower() != "classification":
        return 0.5
    thresholding = artifacts_json.get("thresholding")
    if not isinstance(thresholding, dict):
        return 0.5
    raw = thresholding.get("optimal_threshold")
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_builtin(v: Any) -> Any:
    """Convert numpy scalars to Python builtins; replace nan/inf with None."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None
    return v


def _run_inference(
    pipeline: Any,
    X: pd.DataFrame,
    task_type: str,
    threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run pipeline.predict() with optional threshold application for binary classification.

    Returns (y_pred, y_score_col1_or_None).
    y_score is the probability of the positive class (column index 1 for binary).
    """
    y_score: Optional[np.ndarray] = None

    # Try to get probability scores
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                y_score = proba[:, 1].astype(float)
        except Exception:
            pass

    # Apply threshold for binary classification when we have probabilities
    if task_type == "classification" and y_score is not None and threshold != 0.5:
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            # Try to get from the model step inside the pipeline
            named = getattr(pipeline, "named_steps", {})
            model = named.get("model")
            classes = getattr(model, "classes_", None)
        is_binary = classes is not None and len(classes) == 2
        if is_binary:
            y_pred = (y_score >= threshold).astype(type(classes[1]) if classes is not None else int)
            return np.asarray(y_pred), y_score

    y_pred = pipeline.predict(X)
    return np.asarray(y_pred), y_score


def _build_summary(
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    task_type: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    if task_type == "classification":
        unique, counts = np.unique(y_pred, return_counts=True)
        summary["class_distribution"] = {
            str(_to_builtin(k)): int(v) for k, v in zip(unique, counts)
        }
        avg = float(np.nanmean(y_score)) if y_score is not None else None
        summary["avg_score"] = _to_builtin(avg)
    else:
        pred_float = y_pred.astype(float)
        summary["mean"] = _to_builtin(float(np.nanmean(pred_float)))
        summary["min"] = _to_builtin(float(np.nanmin(pred_float)))
        summary["max"] = _to_builtin(float(np.nanmax(pred_float)))
        summary["std"] = _to_builtin(float(np.nanstd(pred_float)))
        summary["avg_score"] = None

    return summary


def _build_rows(
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    raw_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []
    for i in range(len(y_pred)):
        row: Dict[str, Any] = {
            "row_index": int(i),
            "prediction": _to_builtin(y_pred[i]),
            "score": _to_builtin(float(y_score[i])) if y_score is not None else None,
            "input_data": {
                col: _to_builtin(raw_df.iloc[i][col])
                for col in raw_df.columns
            },
        }
        rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Public prediction functions
# ──────────────────────────────────────────────────────────────────────────────

def predict_with_trained_model(
    trained_model: TrainedModel,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run inference on raw_df using the fitted pipeline stored in trained_model.

    Steps:
    1. Load the pkl pipeline from artifacts_json["model_pkl"].
    2. Validate feature schema (names, order).
    3. Extract optimal_threshold from artifacts_json["thresholding"].
    4. Run inference with threshold applied for binary classification.
    5. Return full result dict with ALL rows.
    """
    artifacts = trained_model.artifacts_json or {}
    model_pkl = artifacts.get("model_pkl")
    if not model_pkl:
        raise RuntimeError("Model artifact path is missing (model_pkl).")

    pkl_path = Path(str(model_pkl))
    if not pkl_path.exists():
        raise RuntimeError(f"Model file not found: {pkl_path}")

    pipeline = load_pipeline(pkl_path)

    task_type = str(trained_model.task_type or "classification").lower()
    training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names: List[str] = training_schema.get("feature_names") or []

    # Validate and reorder columns
    validated_df = validate_feature_schema(raw_df, training_schema)

    threshold = _get_optimal_threshold(artifacts, task_type)
    y_pred, y_score = _run_inference(pipeline, validated_df, task_type, threshold)

    rows = _build_rows(y_pred, y_score, validated_df)
    summary = _build_summary(y_pred, y_score, task_type)

    return {
        "model_id": int(trained_model.id),
        "session_id": int(trained_model.session_id),
        "model_type": str(trained_model.model_type),
        "task_type": task_type,
        "timestamp": _utc_now_iso(),
        "n_rows": int(len(y_pred)),
        "feature_count_received": int(raw_df.shape[1]),
        "feature_count_expected": int(len(feature_names)) if feature_names else None,
        "feature_names_expected": feature_names,
        "threshold_used": threshold,
        "rows": rows,
        "summary": summary,
    }


def predict_rows_json(
    trained_model: TrainedModel,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run inference on a list of dict rows (manual input mode).
    Converts to DataFrame and delegates to predict_with_trained_model.
    """
    if not rows:
        raise RuntimeError("No rows provided for prediction.")
    raw_df = pd.DataFrame(rows)
    return predict_with_trained_model(trained_model, raw_df)


def predict_to_csv(
    trained_model: TrainedModel,
    raw_df: pd.DataFrame,
) -> str:
    """
    Run inference and return results as a CSV string.

    Columns: row_index, prediction, score, <all input feature columns>
    """
    result = predict_with_trained_model(trained_model, raw_df)

    output = io.StringIO()
    if not result["rows"]:
        return ""

    # Build header: fixed columns first, then all input feature columns
    first_row = result["rows"][0]
    input_cols = list(first_row.get("input_data", {}).keys())
    fieldnames = ["row_index", "prediction", "score"] + input_cols

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in result["rows"]:
        flat: Dict[str, Any] = {
            "row_index": row["row_index"],
            "prediction": row["prediction"],
            "score": row["score"] if row["score"] is not None else "",
        }
        flat.update(row.get("input_data", {}))
        writer.writerow(flat)

    return output.getvalue()
