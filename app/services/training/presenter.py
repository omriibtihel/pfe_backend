# app/services/training/presenter.py
"""
Training presenter layer.

Transforms raw ORM objects (TrainedModel, TrainingSession) into the camelCase
dicts expected by the frontend.  No HTTP or DB dependency — pure data shaping.
"""
from __future__ import annotations

from typing import Any, Optional

from app.models.training import TrainedModel, TrainingSession


# ──────────────────────────────────────────────────────────────────────────────
# Internal extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_primary_score(
    metrics_all: dict,
    metrics: dict,
    task_type: str,
    mget: Any,
) -> tuple[Optional[str], float, float, float]:
    """Returns (primary_metric, test_score, train_score, training_time)."""
    primary = metrics_all.get("primary_score") if isinstance(metrics_all.get("primary_score"), dict) else {}
    primary_metric = primary.get("metric")

    if not primary_metric:
        candidates = (
            ["f1", "accuracy", "roc_auc", "pr_auc"]
            if task_type == "classification"
            else ["r2", "rmse", "mae", "mse"]
        )
        for key in candidates:
            if key in metrics:
                primary_metric = key
                break
        if not primary_metric and isinstance(metrics, dict) and metrics:
            primary_metric = next(iter(metrics.keys()))

    test_score = 0.0
    if isinstance(primary, dict) and primary.get("value") is not None:
        try:
            test_score = float(primary["value"])
        except Exception:
            pass
    elif primary_metric and isinstance(metrics, dict):
        try:
            test_score = float(metrics.get(primary_metric, 0.0))
        except Exception:
            pass

    train_score = 0.0
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_metric and primary_metric in train_block:
        try:
            train_score = float(train_block[primary_metric])
        except Exception:
            pass

    try:
        training_time = float(metrics_all.get("training_time_sec", 0.0))
    except Exception:
        training_time = 0.0

    return primary_metric, test_score, train_score, training_time


def extract_automl_fields(metrics_all: dict, artifacts: dict) -> Optional[dict]:
    """Returns the automl result dict when the model was trained with AutoML, else None."""
    if not bool(metrics_all.get("automl", False)):
        return None
    automl_artifacts = artifacts.get("automl") if isinstance(artifacts.get("automl"), dict) else {}
    return {
        "isAutoML": True,
        "isBest": bool(automl_artifacts.get("is_best", True)),
        "bestEstimator": metrics_all.get("best_estimator"),
        "nIterations": metrics_all.get("n_iterations"),
        "totalTimeS": metrics_all.get("total_time_s"),
        "timeBudgetS": automl_artifacts.get("time_budget_s"),
        "metricOptimized": automl_artifacts.get("metric_optimized"),
    }


def extract_cv_fields(
    metrics_all: dict,
    *,
    primary_metric: Optional[str],
    test_score: float,
) -> tuple[dict, float]:
    """Returns (cv_fields_dict, updated_test_score)."""
    is_cv = bool(metrics_all.get("cv", False))
    cv_summary = metrics_all.get("cv_summary") if isinstance(metrics_all.get("cv_summary"), dict) else None
    cv_fold_results = metrics_all.get("fold_results") if isinstance(metrics_all.get("fold_results"), list) else None
    k_folds_used = metrics_all.get("k_folds")
    has_holdout_test = bool(metrics_all.get("has_holdout_test", False))
    cv_mean_metrics = metrics_all.get("cv_mean") if isinstance(metrics_all.get("cv_mean"), dict) else None
    holdout_test_metrics = (
        metrics_all.get("holdout_test_metrics")
        if isinstance(metrics_all.get("holdout_test_metrics"), dict)
        else None
    )

    if is_cv and primary_metric:
        if has_holdout_test and isinstance(holdout_test_metrics, dict):
            raw = holdout_test_metrics.get(primary_metric)
            if raw is None:
                lf = holdout_test_metrics.get("legacy_flat") if isinstance(holdout_test_metrics.get("legacy_flat"), dict) else {}
                gl = holdout_test_metrics.get("global") if isinstance(holdout_test_metrics.get("global"), dict) else {}
                raw = lf.get(primary_metric) or gl.get(primary_metric)
            if raw is not None:
                try:
                    test_score = float(raw)
                except Exception:
                    pass
        elif isinstance(cv_summary, dict):
            cv_mean = cv_summary.get("mean", {})
            if isinstance(cv_mean, dict) and primary_metric in cv_mean:
                try:
                    test_score = float(cv_mean[primary_metric])
                except Exception:
                    pass

    cv_fields = {
        "isCV": is_cv,
        "nestedCv": bool(metrics_all.get("nested_cv", False)),
        "cvFoldResults": cv_fold_results,
        "cvSummary": cv_summary,
        "kFoldsUsed": k_folds_used,
        "hasHoldoutTest": has_holdout_test,
        "cvMeanMetrics": cv_mean_metrics,
        "cvTestMetrics": holdout_test_metrics,
    }
    return cv_fields, test_score


def extract_tuning_fields(artifacts: dict) -> tuple[dict, Any]:
    """Returns (gridSearch_response_dict, gs_raw)."""
    gs = artifacts.get("grid_search") if isinstance(artifacts.get("grid_search"), dict) else {}
    best_params = gs.get("best_params") if isinstance(gs.get("best_params"), dict) else None
    cv_best_score = gs.get("best_score")
    cv_scoring = gs.get("scoring")

    _raw = gs.get("cv_results_summary")
    cv_results_summary: Optional[list] = None
    if isinstance(_raw, list) and _raw:
        _mapped = []
        for row in _raw:
            if not isinstance(row, dict):
                continue
            entry: dict = {
                "params": dict(row.get("params") or {}),
                "mean_score": float(row.get("mean_score") or row.get("mean_test_score") or 0.0),
            }
            # Optional enrichment fields — present only when available
            if row.get("mean_train_score") is not None:
                entry["mean_train_score"] = float(row["mean_train_score"])
            if row.get("overfit_gap") is not None:
                entry["overfit_gap"] = float(row["overfit_gap"])
            if row.get("mean_fit_time_s") is not None:
                entry["mean_fit_time_s"] = float(row["mean_fit_time_s"])
            # HalvingRandomSearchCV-specific
            if row.get("halving_iter") is not None:
                entry["halving_iter"] = int(row["halving_iter"])
            if row.get("n_resources") is not None:
                entry["n_resources"] = int(row["n_resources"])
            _mapped.append(entry)
        cv_results_summary = _mapped or None

    grid_search_block = {
        "enabled": bool(gs.get("enabled", False)),
        "searchType": gs.get("search_type") or None,
        "cvBestScore": float(cv_best_score) if cv_best_score is not None else None,
        "cvScoring": str(cv_scoring) if cv_scoring else None,
        "bestParams": best_params,
        "cvSplits": int(gs.get("cv_splits", 0)) if gs.get("cv_splits") else None,
        "nCandidates": int(gs.get("n_candidates", 0)) if gs.get("n_candidates") else None,
        "cvResultsSummary": cv_results_summary,
    }
    return grid_search_block, gs


def extract_feature_names_for_prediction(artifacts: dict[str, Any]) -> list[str]:
    """
    Return the list of feature names required for manual prediction.

    Priority:
    1. artifacts["training_schema"]["feature_names"]  — most accurate
    2. artifacts["columns"]["numeric"] + ["categorical"]  — fallback
    3. Empty list

    AutoML interaction features are excluded (backend reconstructs them at inference time).
    """
    training_schema = artifacts.get("training_schema") if isinstance(artifacts.get("training_schema"), dict) else {}
    feature_names = training_schema.get("feature_names")

    if not isinstance(feature_names, list) or not feature_names:
        columns_block = artifacts.get("columns") if isinstance(artifacts.get("columns"), dict) else {}
        numeric = columns_block.get("numeric") or []
        categorical = columns_block.get("categorical") or []
        if isinstance(numeric, list) and isinstance(categorical, list):
            feature_names = list(numeric) + [c for c in categorical if c not in numeric]
        else:
            feature_names = []

    automl_pairs = artifacts.get("automl_feature_pairs")
    if isinstance(automl_pairs, list) and automl_pairs:
        engineered = {str(p["name"]) for p in automl_pairs if isinstance(p, dict) and "name" in p}
        feature_names = [f for f in feature_names if str(f) not in engineered]

    return [str(f) for f in feature_names]


def extract_threshold(artifacts: dict[str, Any]) -> float:
    """Return the optimal decision threshold from model artifacts, defaulting to 0.5."""
    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else {}
    raw_t = thresholding.get("optimal_threshold")
    if raw_t is None:
        return 0.5
    try:
        return float(raw_t)
    except (TypeError, ValueError):
        return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Public serialisers
# ──────────────────────────────────────────────────────────────────────────────

def model_to_front_result(
    m: TrainedModel,
    *,
    is_saved: bool | None = None,
    is_active: bool = False,
) -> dict[str, Any]:
    """Serialize a TrainedModel ORM object to the frontend result dict."""
    metrics_all = m.metrics_json or {}
    artifacts = m.artifacts_json or {}
    if is_saved is None:
        is_saved = bool(getattr(m, "is_saved", None) or artifacts.get("saved", False))

    test_metrics = metrics_all.get("test")
    metrics = test_metrics if isinstance(test_metrics, dict) else metrics_all
    metrics_legacy = metrics.get("legacy_flat") if isinstance(metrics.get("legacy_flat"), dict) else {}
    metrics_global = metrics.get("global") if isinstance(metrics.get("global"), dict) else {}
    metrics_binary = metrics.get("binary") if isinstance(metrics.get("binary"), dict) else {}

    def mget(k: str) -> Optional[float]:
        for d in (metrics, metrics_legacy, metrics_global, metrics_binary):
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    task_type = str(m.task_type or "").lower()
    primary_metric, test_score, train_score, training_time = extract_primary_score(
        metrics_all, metrics, task_type, mget
    )

    split_info = metrics_all.get("split_info") if isinstance(metrics_all.get("split_info"), dict) else None
    if split_info is None:
        split_info = artifacts.get("split_info") if isinstance(artifacts.get("split_info"), dict) else None

    grid_search_block, _ = extract_tuning_fields(artifacts)
    thresholding = artifacts.get("thresholding") if isinstance(artifacts.get("thresholding"), dict) else None
    balancing = artifacts.get("balancing") if isinstance(artifacts.get("balancing"), dict) else None
    automl_result = extract_automl_fields(metrics_all, artifacts)
    cv_fields, test_score = extract_cv_fields(metrics_all, primary_metric=primary_metric, test_score=test_score)

    # Sprint 1: Brier score from test metrics
    brier_score: Optional[float] = None
    for _d in (metrics_legacy, metrics_global, metrics_binary, metrics):
        if isinstance(_d, dict):
            _bs = _d.get("brier_score")
            if _bs is not None:
                try:
                    brier_score = float(_bs)
                    break
                except Exception:
                    pass

    # Sprint 1: Confidence intervals from metrics_json
    confidence_intervals = (
        metrics_all.get("confidence_intervals")
        if isinstance(metrics_all.get("confidence_intervals"), dict)
        else None
    )

    return {
        "id": str(m.id),
        "modelType": m.model_type,
        "status": "completed",
        "metrics": {
            "accuracy": mget("accuracy"),
            "precision": mget("precision"),
            "recall": mget("recall"),
            "f1": mget("f1"),
            "roc_auc": mget("roc_auc"),
            "pr_auc": mget("pr_auc"),
            "brier_score": brier_score,
            "precision_pos": mget("precision_pos"),
            "recall_pos": mget("recall_pos"),
            "f1_pos": mget("f1_pos"),
            "f1_macro": mget("f1_macro"),
            "specificity": mget("specificity"),
            "npv": mget("npv"),
            "mcc": mget("mcc"),
            "mse": mget("mse"),
            "rmse": mget("rmse"),
            "mae": mget("mae"),
            "r2": mget("r2"),
        },
        "trainScore": train_score,
        "testScore": test_score,
        "primaryMetric": primary_metric,
        "splitInfo": split_info,
        **cv_fields,
        "gridSearch": grid_search_block,
        "featureImportance": artifacts.get("feature_importance", []),
        "curves": artifacts.get("curves", None),
        "confusionMatrix": artifacts.get("confusion_matrix", []),
        "metricsDetailed": metrics,
        "metricsWarnings": metrics.get("warnings", []) if isinstance(metrics, dict) else [],
        "hyperparams": artifacts.get("hyperparams", None),
        "classDistribution": artifacts.get("class_distribution", None),
        "baselineMajority": artifacts.get("baseline_majority", None),
        "splitDebug": artifacts.get("split_debug", None),
        "preprocessing": artifacts.get("preprocessing", None),
        "balancing": balancing,
        "thresholding": thresholding,
        "trainingTime": training_time,
        "isSaved": bool(is_saved),
        "isActive": bool(is_active),
        "automl": automl_result,
        "confidenceIntervals": confidence_intervals,
        "learningCurves": artifacts.get("learning_curves", None),
        "permutationImportance": artifacts.get("permutation_importance", None),
        "residualAnalysis": artifacts.get("residual_analysis", None),
        "shapGlobal": artifacts.get("shap", None),
    }


def session_to_front(
    s: TrainingSession,
    models: list[TrainedModel],
    *,
    active_model_id: int | None = None,
) -> dict[str, Any]:
    """Serialize a TrainingSession + its models to the frontend session dict."""
    return {
        "id": str(s.id),
        "projectId": str(s.project_id),
        "datasetVersionId": str(s.dataset_version_id) if s.dataset_version_id else None,
        "status": s.status,
        "progress": int(s.progress or 0),
        "currentModel": s.current_model,
        "errorMessage": s.error_message,
        "activeModelId": str(active_model_id) if active_model_id is not None else None,
        "config": s.config_json,
        "results": [
            model_to_front_result(
                m,
                is_saved=bool((m.artifacts_json or {}).get("saved", False)),
                is_active=bool(active_model_id is not None and int(m.id) == int(active_model_id)),
            )
            for m in models
        ],
        "createdAt": s.created_at.isoformat() if s.created_at else None,
        "startedAt": s.started_at.isoformat() if s.started_at else None,
        "completedAt": s.finished_at.isoformat() if s.finished_at else None,
    }
