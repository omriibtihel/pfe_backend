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
from app.services.training.output.persistence import load_pipeline

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
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(buff)
    if name.endswith(".json"):
        return pd.read_json(buff)
    if name.endswith(".parquet"):
        return pd.read_parquet(buff)
    raise RuntimeError("Unsupported file type. Use CSV, Excel, JSON or Parquet.")


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
# AutoML preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_input_dtypes(df: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """
    Normalize input dtypes so every model receives numeric (or Categorical) data.

    Steps applied in order:
    1. pandas StringDtype → object  (imblearn / FLAML require object, not StringDtype)
    2. Object columns that represent numbers ("1", "2.5") → float via pd.to_numeric.
    3. Remaining object columns (true categoricals: "yes/no", region names …)
       → pd.Categorical.  XGBoost ≥ 1.7 supports Categorical natively; we also set
       enable_categorical=True on the estimator when needed.

    This is safe regardless of whether the pipeline has a preprocessing step:
    - Columns handled by a fitted ColumnTransformer are transformed inside the
      pipeline and don't reach this code path as object dtype.
    - Passthrough columns that were numeric during training but arrive as strings
      from an Excel file are the typical case that needs fixing.
    """
    # 1. StringDtype → object
    string_cols = df.select_dtypes(include="string").columns.tolist()
    if string_cols:
        df = df.astype({col: "object" for col in string_cols})
    df.columns = df.columns.astype(object)

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        return df

    df = df.copy()
    categorical_cols_added: list[str] = []

    for col in obj_cols:
        # Normalize whitespace-only / empty strings to NaN so they don't masquerade
        # as real categorical values. Without this, a CSV cell containing " " (space)
        # is counted as non-null but fails pd.to_numeric, which both skews the 90%
        # parse-rate check and lets the bogus string reach the SimpleImputer
        # downstream, where strategy="median" crashes on "could not convert string
        # to float: ' '".
        if df[col].dtype == object:
            cleaned = df[col].where(
                ~df[col].apply(lambda v: isinstance(v, str) and v.strip() == "")
            )
            df[col] = cleaned

        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        # Accept the numeric conversion if at least 90 % of non-null values parse.
        if non_null == 0 or numeric.notna().sum() >= non_null * 0.9:
            df[col] = numeric
        else:
            df[col] = pd.Categorical(df[col])
            categorical_cols_added.append(col)

    # Enable XGBoost categorical support when Categorical columns are present.
    if categorical_cols_added:
        named = getattr(pipeline, "named_steps", {})
        model = named.get("model") if named else pipeline
        if model is not None and "XGB" in type(model).__name__:
            try:
                model.enable_categorical = True
            except Exception:
                pass

    return df


def _fix_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas 3.0 StringDtype columns to object dtype (FLAML compatibility).

    Kept for backward compatibility; new code should prefer _normalize_input_dtypes.
    """
    string_cols = df.select_dtypes(include="string").columns.tolist()
    if string_cols:
        df = df.astype({col: "object" for col in string_cols})
    df.columns = df.columns.astype(object)
    return df


def _apply_automl_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Reconstruct medical interaction features that were added during AutoML training."""
    pairs = artifacts.get("automl_feature_pairs") or []
    if not pairs:
        return df
    df = df.copy()
    for p in pairs:
        c1, c2, fname = p["col1"], p["col2"], p["name"]
        if c1 in df.columns and c2 in df.columns:
            df[fname] = df[c1] * df[c2]
    df.columns = df.columns.astype(object)
    return df


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
    positive_class_index: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run pipeline.predict() with optional threshold application for binary classification.

    Returns (y_pred, y_score_col1_or_None).
    y_score is the probability of the positive class (column index positive_class_index).
    """
    y_score: Optional[np.ndarray] = None

    # Try to get probability scores (binary only — multiclass has 3+ columns and
    # extracting a single column would be misleading, so y_score stays None).
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                pos_col = positive_class_index if positive_class_index < proba.shape[1] else 1
                y_score = proba[:, pos_col].astype(float)
        except Exception:
            pass

    # Apply threshold for binary classification when we have probabilities.
    # Note: when the pipeline is a ThresholdedClassifier the threshold has
    # already been embedded at save time, so pipeline.predict(X) is correct
    # and returns the original class labels — we don't enter this branch
    # because threshold == 0.5 is the *unwrapped* default.
    if task_type == "classification" and y_score is not None and threshold != 0.5:
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            # Try to get from the model step inside a sklearn Pipeline
            named = getattr(pipeline, "named_steps", {})
            model = named.get("model") if isinstance(named, dict) else None
            classes = getattr(model, "classes_", None)
        if classes is None:
            # Fallback for FLAML AutoML (no named_steps)
            try:
                classes = pipeline.model.estimator.classes_
            except AttributeError:
                pass
        is_binary = classes is not None and len(classes) == 2
        if is_binary:
            classes_arr = np.asarray(classes)
            pos_idx = positive_class_index if 0 <= positive_class_index < len(classes_arr) else 1
            neg_idx = 1 - pos_idx if pos_idx in (0, 1) else 0
            pos_cls = classes_arr[pos_idx]
            neg_cls = classes_arr[neg_idx]
            y_pred = np.where(y_score >= threshold, pos_cls, neg_cls)
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
# Data drift detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_drift(raw_df: pd.DataFrame, training_schema: dict) -> List[Dict[str, Any]]:
    """
    Lightweight drift detection: compare input DataFrame statistics against
    training-time statistics stored in training_schema["column_stats"].

    Returns a list of drift warnings (empty list = no drift detected or stats unavailable).
    Each entry: {"column": str, "type": "mean_shift"|"std_shift"|"new_categories",
                 "severity": "warning"|"critical", "detail": str}
    """
    col_stats: Dict[str, Any] = training_schema.get("column_stats") or {}
    if not col_stats:
        return []

    warnings: List[Dict[str, Any]] = []
    for col, stats in col_stats.items():
        if col not in raw_df.columns:
            continue
        series = raw_df[col].dropna()
        if len(series) == 0:
            continue

        col_type = stats.get("type", "numeric")

        if col_type == "numeric":
            # Skip high-cardinality columns (IDs, timestamps) flagged at training
            # time — they produce only false-positive drift alerts.
            if stats.get("high_cardinality"):
                continue
            train_mean = stats.get("mean")
            train_std = stats.get("std")
            if train_mean is None or train_std is None or train_std == 0:
                continue
            cur_mean = float(series.astype(float).mean())
            shift = abs(cur_mean - train_mean) / train_std
            if shift > 3.0:
                warnings.append({
                    "column": col, "type": "mean_shift", "severity": "critical",
                    "detail": f"Mean shifted {shift:.1f}σ (train={train_mean:.3g}, current={cur_mean:.3g})",
                })
            elif shift > 1.5:
                warnings.append({
                    "column": col, "type": "mean_shift", "severity": "warning",
                    "detail": f"Mean shifted {shift:.1f}σ (train={train_mean:.3g}, current={cur_mean:.3g})",
                })

        elif col_type in ("categorical", "text"):
            train_cats = set(stats.get("categories") or [])
            if not train_cats:
                continue
            cur_cats = set(series.astype(str).unique().tolist())
            new_cats = cur_cats - train_cats
            if new_cats:
                severity = "critical" if len(new_cats) > len(train_cats) * 0.3 else "warning"
                warnings.append({
                    "column": col, "type": "new_categories", "severity": severity,
                    "detail": f"{len(new_cats)} new categor(y/ies) not seen at training: {sorted(new_cats)[:5]}",
                })

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Public prediction functions
# ──────────────────────────────────────────────────────────────────────────────

def predict_with_trained_model(
    trained_model: TrainedModel,
    raw_df: pd.DataFrame,
    *,
    _preloaded_pipeline: Any = None,
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

    # Accept a pre-loaded pipeline so callers like predict_with_explanations can
    # avoid deserializing the pickle twice (Bug 12 fix).
    pipeline = _preloaded_pipeline if _preloaded_pipeline is not None else load_pipeline(pkl_path)

    task_type = str(trained_model.task_type or "classification").lower()
    training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names: List[str] = training_schema.get("feature_names") or []

    threshold = _get_optimal_threshold(artifacts, task_type)

    # Extract stored positive_class_index so binary score column is always correct.
    # Stored by reporter.py in artifacts["balancing"]["positive_class_index"].
    # Falls back to 1 (sklearn default) when absent (older models / regression).
    _balancing = artifacts.get("balancing") or {}
    _pos_idx = _balancing.get("positive_class_index")
    positive_class_index: int = int(_pos_idx) if _pos_idx is not None else 1

    is_automl = bool(artifacts.get("automl"))
    if is_automl:
        # Reconstruct any interaction features stored at training time.
        # "automl_feature_pairs" absent = old model without stored pairs; skip gracefully.
        raw_df = _apply_automl_features(raw_df, artifacts)

    # Capture the full DataFrame (including interaction features) for display before
    # the alignment step strips columns that the pipeline doesn't expect as input.
    display_df = raw_df

    # Pre-inference column validation: a *missing column* (entirely absent from
    # the input file) is a hard error — silently filling it with NaN can mask
    # real upload mistakes and produce predictions that look plausible but are
    # driven entirely by the imputer's training median.  *Missing values*
    # (NaN cells inside present columns) are fine: the imputer saved in the
    # pipeline handles them as it did at training time.  Older models that
    # were stored without a ``training_schema`` keep the previous lenient
    # behaviour (no schema = nothing to validate against).
    if feature_names:
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
        raw_df = raw_df[feature_names]

    # Normalize dtypes for all models (StringDtype → object, numeric strings → float,
    # categorical strings → pd.Categorical with XGBoost enable_categorical support).
    raw_df = _normalize_input_dtypes(raw_df, pipeline)

    y_pred, y_score = _run_inference(pipeline, raw_df, task_type, threshold, positive_class_index)

    rows = _build_rows(y_pred, y_score, display_df)
    summary = _build_summary(y_pred, y_score, task_type)

    drift_warnings = _detect_drift(raw_df, training_schema)
    if drift_warnings:
        logger.warning(
            "predict.drift_detected: model_id=%s, warnings=%d",
            trained_model.id,
            len(drift_warnings),
        )

    return {
        "model_id": int(trained_model.id),
        "session_id": int(trained_model.session_id),
        "dataset_version_id": artifacts.get("dataset_version_id"),
        "model_type": str(trained_model.model_type),
        "task_type": task_type,
        "timestamp": _utc_now_iso(),
        "n_rows": int(len(y_pred)),
        "feature_count_received": int(raw_df.shape[1]),
        "feature_count_expected": int(len(feature_names)) if feature_names else None,
        "feature_names_expected": feature_names,
        "top_features": _get_top_features_for_display(artifacts, feature_names, k=3),
        "threshold_used": threshold,
        "rows": rows,
        "preview": rows,
        "summary": summary,
        "drift_warnings": drift_warnings,
    }


def predict_with_explanations(
    trained_model: TrainedModel,
    raw_df: pd.DataFrame,
    *,
    method: str = "shap",
) -> Dict[str, Any]:
    """
    Run inference and attach per-row local explanations.

    Parameters
    ----------
    trained_model:
        The TrainedModel ORM object.
    raw_df:
        The input rows to predict + explain.
    method:
        "shap" → row["shap"] = [{feature, shap_value, data}]
        "lime" → row["lime"] = [{feature, contribution, data}]
        "both" → both keys are populated

    Falls back gracefully: if a method fails the corresponding key is simply
    absent. Inference itself never fails because of explainability.
    """
    method = (method or "shap").lower()
    if method not in ("shap", "lime", "both"):
        method = "shap"

    # Load the pipeline once and reuse for both prediction and explanation
    # (Bug 12 fix: avoids deserialising the pkl twice).
    artifacts = trained_model.artifacts_json or {}
    _shared_pipeline: Any = None
    _pkl_path_str = artifacts.get("model_pkl")
    if _pkl_path_str:
        _pkl_path = Path(str(_pkl_path_str))
        if _pkl_path.exists():
            try:
                _shared_pipeline = load_pipeline(_pkl_path)
            except Exception:
                pass

    # Core prediction — raises on failure (pkl missing, inference error, etc.)
    result = predict_with_trained_model(trained_model, raw_df, _preloaded_pipeline=_shared_pipeline)

    if _shared_pipeline is None:
        return result

    # Explanation enrichment — best-effort, never breaks prediction
    try:
        task_type = str(trained_model.task_type or "classification").lower()
        training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
        feature_names: List[str] = training_schema.get("feature_names") or []

        pipeline = _shared_pipeline  # already loaded above — no second deserialisation

        # Reload the background data saved at training time (in artifacts["shap"]).
        # Reused by both SHAP and LIME: same baseline distribution, no duplication.
        background: Optional[np.ndarray] = None
        shap_artifacts = artifacts.get("shap") or {}
        raw_bg = shap_artifacts.get("background_data")
        if raw_bg:
            try:
                background = np.array(raw_bg, dtype=float)
            except Exception:
                background = None

        # Recover the positive_label agreed at training time so SHAP values are
        # oriented toward the correct class (not blindly toward index 1).
        _balancing_info = artifacts.get("balancing") or {}
        _artifact_positive_label: Optional[str] = _balancing_info.get("positive_label")

        # Align inputs the same way predict_with_trained_model does
        df_aligned = _normalize_input_dtypes(raw_df.copy(), pipeline)
        if feature_names:
            for col in feature_names:
                if col not in df_aligned.columns:
                    df_aligned[col] = np.nan
            df_aligned = df_aligned[feature_names]

        want_shap = method in ("shap", "both")
        want_lime = method in ("lime", "both")

        compute_local_shap = None
        compute_local_lime = None
        if want_shap:
            from app.services.shap import compute_local_shap as _csh  # type: ignore
            compute_local_shap = _csh
        if want_lime:
            from app.services.lime import compute_local_lime as _cli  # type: ignore
            compute_local_lime = _cli

        for i, row_result in enumerate(result["rows"]):
            row_df = df_aligned.iloc[[i]]

            if want_shap and compute_local_shap is not None:
                try:
                    shap_items = compute_local_shap(
                        pipeline,
                        row_df,
                        task_type=task_type,
                        background=background,
                        feature_names=feature_names if feature_names else None,
                        positive_label=_artifact_positive_label,
                    )
                    if shap_items is not None:
                        row_result["shap"] = shap_items
                except Exception as exc:
                    logger.warning(
                        "predict_with_explanations: SHAP failed row=%d (model_id=%s): %s",
                        i, trained_model.id, exc,
                    )

            if want_lime and compute_local_lime is not None:
                try:
                    lime_items = compute_local_lime(
                        pipeline,
                        row_df,
                        task_type=task_type,
                        background=background,
                        feature_names=feature_names if feature_names else None,
                    )
                    if lime_items is not None:
                        row_result["lime"] = lime_items
                except Exception as exc:
                    logger.warning(
                        "predict_with_explanations: LIME failed row=%d (model_id=%s): %s",
                        i, trained_model.id, exc,
                    )

    except Exception as exc:
        logger.warning(
            "predict_with_explanations: enrichment failed (method=%s, model_id=%s): %s",
            method, trained_model.id, exc,
        )

    return result


def predict_with_shap(
    trained_model: TrainedModel,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around predict_with_explanations(method="shap").

    Kept so existing callers that import predict_with_shap keep working.
    """
    return predict_with_explanations(trained_model, raw_df, method="shap")


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


def _synthetic_background_from_stats(
    *,
    pipeline: Any,
    feature_names: List[str],
    column_stats: Dict[str, Any],
    n_samples: int = 100,
) -> Optional[np.ndarray]:
    """Build a preprocessed background from per-column training statistics.

    Used as a fallback when ``artifacts.shap.background_data`` is absent
    (older models, pipelines run before SHAP was wired in). Without a
    realistic background the counterfactual search uses sigma=1.0 for every
    feature, which produces uniform ±α deltas across all variables — a
    classic symptom of a flat scale map.

    Sampling strategy:
    - numeric columns: gaussian draws around ``mean`` with ``std`` spread,
      clipped to the observed [min, max].
    - categorical / unknown columns: filled with the column mean if numeric
      data leaked through, otherwise ``np.nan`` so the pipeline imputer
      handles them.

    Returns the preprocessed array (samples, n_prep_features) on success, or
    None if anything goes wrong — the caller treats None the same as
    "no background available".
    """
    if not feature_names or not column_stats:
        return None
    rng = np.random.default_rng(42)
    raw_columns: Dict[str, np.ndarray] = {}
    has_numeric = False
    for col in feature_names:
        cs = column_stats.get(col, {})
        if isinstance(cs, dict) and cs.get("type") == "numeric":
            has_numeric = True
            mean = float(cs.get("mean", 0.0))
            std = max(float(cs.get("std", 0.0)), 1e-3)
            lo = float(cs.get("min", mean - 3.0 * std))
            hi = float(cs.get("max", mean + 3.0 * std))
            samples = rng.normal(mean, std, n_samples)
            raw_columns[col] = np.clip(samples, lo, hi)
        else:
            # Categorical or missing stats — leave NaN; the pipeline imputer
            # fills with mode/mean during transform.
            raw_columns[col] = np.full(n_samples, np.nan, dtype=float)

    if not has_numeric:
        return None

    try:
        from app.services.shap.global_shap import _transform_except_model
        synthetic_df = pd.DataFrame(raw_columns)[feature_names]
        prep = _transform_except_model(pipeline, synthetic_df)
        if prep is None or prep.ndim != 2 or prep.shape[0] == 0:
            return None
        return np.asarray(prep, dtype=float)
    except Exception as exc:  # pragma: no cover — defensive guard
        logger.warning("counterfactual: synthetic background failed (%s)", exc)
        return None


def compute_counterfactual_for_row(
    trained_model: TrainedModel,
    row: Dict[str, Any],
    features_to_vary: List[str],
    *,
    n_counterfactuals: int = 3,
) -> Dict[str, Any]:
    """
    Compute DiCE counterfactuals for a single patient row from a saved model.

    Orchestrates pipeline loading, background retrieval, dtype normalisation,
    and delegates to app.services.counterfactual.compute_counterfactual.
    No business logic here — all explainability logic lives in the service.

    Returns
    -------
    Dict with keys: model_id, task_type, original_prediction, original_score,
    counterfactual (list or None), message (str or None).
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
    training_schema = (
        artifacts.get("training_schema")
        if isinstance(artifacts.get("training_schema"), dict)
        else {}
    )
    feature_names: List[str] = training_schema.get("feature_names") or []

    # Build input DataFrame — same normalisation as predict_with_trained_model.
    raw_df = pd.DataFrame([row])
    raw_df = _normalize_input_dtypes(raw_df, pipeline)
    if feature_names:
        for col in feature_names:
            if col not in raw_df.columns:
                raw_df[col] = np.nan
        raw_df = raw_df[feature_names]

    # Capture the current prediction so the response includes it alongside CF.
    threshold = _get_optimal_threshold(artifacts, task_type)
    _balancing = artifacts.get("balancing") or {}
    _pos_idx = _balancing.get("positive_class_index")
    positive_class_index: int = int(_pos_idx) if _pos_idx is not None else 1

    y_pred, y_score = _run_inference(pipeline, raw_df, task_type, threshold, positive_class_index)
    original_prediction = _to_builtin(y_pred[0])
    original_score = _to_builtin(float(y_score[0])) if y_score is not None else None

    # Load the preprocessed background (same artifact used by SHAP and LIME).
    background: Optional[np.ndarray] = None
    raw_bg = (artifacts.get("shap") or {}).get("background_data")
    if raw_bg:
        try:
            background = np.array(raw_bg, dtype=float)
        except Exception:
            pass

    # Fallback: synthesise a background from training_schema.column_stats so the
    # CF search has a per-feature scale to work with. Without this the pipeline
    # collapses to sigma=1.0 for every feature and produces constant ±α deltas.
    if background is None and feature_names:
        background = _synthetic_background_from_stats(
            pipeline=pipeline,
            feature_names=feature_names,
            column_stats=training_schema.get("column_stats") or {},
            n_samples=100,
        )

    from app.services.counterfactual import compute_counterfactual
    cf_items = compute_counterfactual(
        pipeline,
        raw_df,
        features_to_vary=features_to_vary,
        task_type=task_type,
        background=background,
        feature_names=feature_names if feature_names else None,
        n_counterfactuals=n_counterfactuals,
    )

    return {
        "model_id":            int(trained_model.id),
        "task_type":           task_type,
        "original_prediction": original_prediction,
        "original_score":      original_score,
        "counterfactual":      cf_items,
        "message":             None if cf_items else "No valid counterfactual found for this row.",
    }


def _get_top_features_for_display(
    artifacts: Dict[str, Any],
    feature_names: List[str],
    k: int = 3,
) -> List[str]:
    """
    Pick the K most informative ORIGINAL feature names to surface inline in
    the predictions table.

    Priority:
      1. artifacts["shap"]["summary"] — global SHAP, sorted by mean_abs_shap.
         The names there are post-preprocessing ("num__Glucose", "cat__sex_male");
         we strip the ColumnTransformer prefix and dedupe OHE expansions
         ("sex_male" + "sex_female" → "sex").
      2. Alphabetical fallback when no SHAP summary is available.

    The returned names are guaranteed to be present in `feature_names` so the
    frontend can read row.inputData[name] without surprises.
    """
    if not feature_names:
        return []

    feat_set = set(feature_names)
    top: List[str] = []

    shap_summary = (artifacts.get("shap") or {}).get("summary") or []
    for item in shap_summary:
        if len(top) >= k:
            break
        prep_name = str(item.get("feature", ""))
        # Strip ColumnTransformer prefix (e.g. "num__", "cat__")
        base = prep_name.split("__", 1)[-1] if "__" in prep_name else prep_name
        # Direct match (also covers non-OHE preprocessed features)
        if base in feat_set and base not in top:
            top.append(base)
            continue
        # OHE expansion: try all prefix lengths longest-first.
        # "cat__diagnosis_high_blood_pressure" → base="diagnosis_high_blood_pressure"
        # tries "diagnosis_high_blood_pressure", "diagnosis_high_blood", "diagnosis_high",
        # "diagnosis" — first match wins.  Single rsplit("_", 1) would get
        # "diagnosis_high_blood" which is wrong for multi-word category suffixes.
        if "_" in base:
            parts_base = base.split("_")
            for n in range(len(parts_base) - 1, 0, -1):
                stem = "_".join(parts_base[:n])
                if stem in feat_set and stem not in top:
                    top.append(stem)
                    break

    # Fill remaining slots with alphabetically-first features when SHAP is
    # missing or didn't surface enough unique names.
    if len(top) < k:
        for f in feature_names:
            if len(top) >= k:
                break
            if f not in top:
                top.append(f)

    return top[:k]


def get_feature_ranges_for_model(trained_model: TrainedModel) -> Dict[str, Dict[str, Any]]:
    """
    Return per-feature statistics used to bound interactive counterfactual sliders.

    Numeric features:     {"type": "numeric", "min", "max", "mean", "std"}
    Categorical features: {"type": "categorical", "categories": [str, ...]}

    Source: artifacts["training_schema"]["column_stats"] saved at training time.
    Returns {} for older models that lack column_stats.
    """
    artifacts = trained_model.artifacts_json or {}
    training_schema = (
        artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    )
    feature_names: List[str] = training_schema.get("feature_names") or []
    column_stats: Dict[str, Any] = training_schema.get("column_stats") or {}

    ranges: Dict[str, Dict[str, Any]] = {}
    for feat in feature_names:
        stats = column_stats.get(feat) or {}
        col_type = stats.get("type")

        if col_type == "numeric":
            ranges[feat] = {
                "type": "numeric",
                "min":  float(stats.get("min", 0.0)),
                "max":  float(stats.get("max", 1.0)),
                "mean": float(stats.get("mean", 0.0)),
                "std":  float(stats.get("std", 1.0)) or 1.0,
            }
        elif col_type in ("categorical", "text"):
            cats = stats.get("categories") or []
            ranges[feat] = {
                "type": "categorical",
                "categories": [str(c) for c in cats],
            }
        else:
            # No stats available — frontend will use a sensible default range
            # derived from the patient's own value.
            ranges[feat] = {"type": "unknown"}

    return ranges


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
