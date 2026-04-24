# app/services/training/presenter.py
"""
Training presenter layer.

Transforms raw ORM objects (TrainedModel, TrainingSession) into typed Pydantic
response models.  No HTTP or DB dependency — pure data shaping.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.models.training import TrainedModel, TrainingSession

logger = logging.getLogger(__name__)


class MetricNotApplicable(Exception):
    """No valid metric is available for this model (expected, not a pipeline error)."""
from app.schemas.training.results import (
    AnalysisBlock,
    AutoMLInfo,
    BalancingInfo,
    CVInfo,
    CurvesResponse,
    ExplainabilityResponse,
    FeatureImportanceItem,
    GridSearchInfo,
    MetricsSummary,
    ModelResultDetailResponse,
    ModelResultResponse,
    PrimaryMetric,
    SavedModelResponse,
    SplitSummary,
    ThresholdInfo,
    TrainingSessionResponse,
)

# ──────────────────────────────────────────────────────────────────────────────
# Metric display names
# ──────────────────────────────────────────────────────────────────────────────

_METRIC_DISPLAY: dict[str, str] = {
    "f1": "F1-score",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 pondéré",
    "f1_pos": "F1 (classe +)",
    "accuracy": "Accuracy",
    "roc_auc": "AUC-ROC",
    "pr_auc": "AUC-PR",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "mse": "MSE",
    "mcc": "MCC",
    "precision": "Précision",
    "recall": "Rappel",
}

# Ordered preference by task type — used when the backend has not stored a
# primary_score block.
_CLASSIFICATION_CANDIDATES = ["roc_auc", "f1", "f1_pos", "f1_macro", "accuracy", "pr_auc"]
_REGRESSION_CANDIDATES = ["rmse", "r2", "mae", "mse"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal metric extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_metrics(metrics_all: dict) -> tuple[dict, callable]:
    """
    Returns (flat_metrics_dict, mget_fn).

    flat_metrics_dict is the test-metrics dict with common sub-dicts merged in.
    mget_fn(key) returns the first non-None float found across all metric dicts.
    """
    test_block = metrics_all.get("test")
    metrics: dict = test_block if isinstance(test_block, dict) else metrics_all

    legacy = metrics.get("legacy_flat") if isinstance(metrics.get("legacy_flat"), dict) else {}
    global_ = metrics.get("global") if isinstance(metrics.get("global"), dict) else {}
    binary = metrics.get("binary") if isinstance(metrics.get("binary"), dict) else {}

    dicts = (metrics, legacy, global_, binary)

    def mget(k: str) -> Optional[float]:
        for d in dicts:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    return metrics, mget


def _extract_training_time(metrics_all: dict) -> float:
    try:
        return float(metrics_all.get("training_time_sec", 0.0))
    except Exception:
        return 0.0


def _extract_train_score(metrics_all: dict, primary_name: Optional[str]) -> float:
    if not primary_name:
        return 0.0
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_name in train_block:
        try:
            return float(train_block[primary_name])
        except Exception:
            pass
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DÉCISION 2 — get_primary_metric
# ──────────────────────────────────────────────────────────────────────────────

def get_primary_metric(
    task_type: str,
    metrics_all: dict,
    mget: callable,
) -> PrimaryMetric:
    """
    Select and return the single most informative metric for this model.

    Returns a PrimaryMetric whose ``status`` field encodes one of three states:
      "success"        — a valid float was found (normal case)
      "not_applicable" — no valid metric exists in metrics_json (expected gap)
      "error"          — an unexpected exception prevented extraction entirely

    Priority order when status="success":
    1. primary_score block stored by the training pipeline (explicit backend choice)
    2. Task-type ordered fallback (classification → AUC/F1, regression → RMSE/R²)
    3. First parseable float in the metrics dict
    """
    try:
        # 1. Use stored primary_score if available
        primary_score = metrics_all.get("primary_score")
        if isinstance(primary_score, dict):
            stored_name = primary_score.get("metric")
            stored_value = primary_score.get("value")
            if stored_name and stored_value is not None:
                try:
                    return PrimaryMetric(
                        name=str(stored_name),
                        value=float(stored_value),
                        displayName=_METRIC_DISPLAY.get(str(stored_name), str(stored_name).upper()),
                    )
                except Exception:
                    pass  # stored value uncastable — fall through to next priority

        # 2. Ordered fallback by task type
        candidates = (
            _REGRESSION_CANDIDATES
            if str(task_type).lower() == "regression"
            else _CLASSIFICATION_CANDIDATES
        )
        for key in candidates:
            val = mget(key)
            if val is not None:
                return PrimaryMetric(
                    name=key,
                    value=val,
                    displayName=_METRIC_DISPLAY.get(key, key.upper()),
                )

        # 3. First available metric
        metrics, _ = _resolve_metrics(metrics_all)
        for key, raw in metrics.items():
            if isinstance(key, str) and key not in ("legacy_flat", "global", "binary", "warnings"):
                try:
                    return PrimaryMetric(
                        name=key,
                        value=float(raw),
                        displayName=_METRIC_DISPLAY.get(key, key.upper()),
                    )
                except Exception:
                    continue

        raise MetricNotApplicable("no valid metric found in metrics_json")

    except MetricNotApplicable:
        return PrimaryMetric(name="unknown", value=None, displayName="—", status="not_applicable")
    except Exception:
        logger.error("get_primary_metric failed unexpectedly", exc_info=True)
        return PrimaryMetric(name="error", value=None, displayName="Erreur de calcul", status="error")


# ──────────────────────────────────────────────────────────────────────────────
# Sub-object builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_split_summary(metrics_all: dict, artifacts: dict) -> Optional[SplitSummary]:
    split = metrics_all.get("split_info")
    if not isinstance(split, dict):
        split = artifacts.get("split_info")
    if not isinstance(split, dict):
        return None
    return SplitSummary(
        method=split.get("method"),
        trainRows=split.get("train_rows"),
        valRows=split.get("val_rows"),
        testRows=split.get("test_rows"),
    )


def _build_automl_info(metrics_all: dict, artifacts: dict) -> Optional[AutoMLInfo]:
    if not bool(metrics_all.get("automl", False)):
        return None
    automl_artifacts = artifacts.get("automl") if isinstance(artifacts.get("automl"), dict) else {}
    return AutoMLInfo(
        isBest=bool(automl_artifacts.get("is_best", True)),
        bestEstimator=metrics_all.get("best_estimator"),
        nIterations=metrics_all.get("n_iterations"),
        totalTimeS=metrics_all.get("total_time_s"),
        timeBudgetS=automl_artifacts.get("time_budget_s"),
        metricOptimized=automl_artifacts.get("metric_optimized"),
    )


def _build_cv_info(metrics_all: dict, *, primary_name: Optional[str]) -> tuple[Optional[CVInfo], float, bool, bool]:
    """Returns (cv_info | None, test_score_override, is_cv, has_holdout)."""
    is_cv = bool(metrics_all.get("cv", False))
    if not is_cv:
        return None, 0.0, False, False

    has_holdout = bool(metrics_all.get("has_holdout_test", False))
    cv_summary = metrics_all.get("cv_summary") if isinstance(metrics_all.get("cv_summary"), dict) else None
    holdout_metrics = metrics_all.get("holdout_test_metrics") if isinstance(metrics_all.get("holdout_test_metrics"), dict) else None
    cv_mean = metrics_all.get("cv_mean") if isinstance(metrics_all.get("cv_mean"), dict) else None

    test_score_override = 0.0
    if primary_name:
        if has_holdout and isinstance(holdout_metrics, dict):
            raw = holdout_metrics.get(primary_name)
            if raw is None:
                lf = holdout_metrics.get("legacy_flat") if isinstance(holdout_metrics.get("legacy_flat"), dict) else {}
                gl = holdout_metrics.get("global") if isinstance(holdout_metrics.get("global"), dict) else {}
                raw = lf.get(primary_name) or gl.get(primary_name)
            if raw is not None:
                try:
                    test_score_override = float(raw)
                except Exception:
                    pass
        elif isinstance(cv_summary, dict):
            cv_mean_block = cv_summary.get("mean", {})
            if isinstance(cv_mean_block, dict) and primary_name in cv_mean_block:
                try:
                    test_score_override = float(cv_mean_block[primary_name])
                except Exception:
                    pass

    fold_results = metrics_all.get("fold_results") if isinstance(metrics_all.get("fold_results"), list) else None
    cv_info = CVInfo(
        kFoldsUsed=metrics_all.get("k_folds"),
        nestedCv=bool(metrics_all.get("nested_cv", False)),
        cvSummary=cv_summary,
        cvFoldResults=fold_results,
        cvMeanMetrics=cv_mean,
        cvTestMetrics=holdout_metrics,
    )
    return cv_info, test_score_override, True, has_holdout


def _build_threshold_info(artifacts: dict) -> Optional[ThresholdInfo]:
    thresh = artifacts.get("thresholding")
    if not isinstance(thresh, dict):
        return None
    return ThresholdInfo(
        enabled=bool(thresh.get("enabled", False)),
        strategy=thresh.get("strategy"),
        optimalThreshold=thresh.get("optimal_threshold"),
        improvementDelta=thresh.get("improvement_delta"),
        warnings=thresh.get("warnings") if isinstance(thresh.get("warnings"), list) else [],
    )


def _build_grid_search_info(artifacts: dict) -> Optional[GridSearchInfo]:
    gs = artifacts.get("grid_search")
    if not isinstance(gs, dict) or not gs.get("enabled"):
        return None

    raw_summary = gs.get("cv_results_summary")
    cv_results_summary = None
    if isinstance(raw_summary, list) and raw_summary:
        mapped = []
        for row in raw_summary:
            if not isinstance(row, dict):
                continue
            entry: dict[str, Any] = {
                "params": dict(row.get("params") or {}),
                "mean_score": float(row.get("mean_score") or row.get("mean_test_score") or 0.0),
            }
            for opt_key in ("mean_train_score", "overfit_gap", "mean_fit_time_s", "halving_iter", "n_resources"):
                if row.get(opt_key) is not None:
                    entry[opt_key] = row[opt_key]
            mapped.append(entry)
        cv_results_summary = mapped or None

    best_score = gs.get("best_score")
    return GridSearchInfo(
        enabled=True,
        searchType=gs.get("search_type") or None,
        cvBestScore=float(best_score) if best_score is not None else None,
        cvScoring=str(gs["scoring"]) if gs.get("scoring") else None,
        bestParams=gs.get("best_params") if isinstance(gs.get("best_params"), dict) else None,
        cvSplits=int(gs["cv_splits"]) if gs.get("cv_splits") else None,
        nCandidates=int(gs["n_candidates"]) if gs.get("n_candidates") else None,
        cvResultsSummary=cv_results_summary,
    )


def _build_balancing_info(artifacts: dict) -> Optional[BalancingInfo]:
    bal = artifacts.get("balancing")
    if not isinstance(bal, dict):
        return None
    ratio = bal.get("imbalance_ratio")
    return BalancingInfo(
        strategyApplied=str(bal["strategy_applied"]) if bal.get("strategy_applied") is not None else None,
        refitMetric=str(bal["refit_metric"]) if bal.get("refit_metric") is not None else None,
        imbalanceRatio=float(ratio) if ratio is not None else None,
    )


def _build_metrics_summary(mget: callable, task_type: str) -> MetricsSummary:
    return MetricsSummary(
        accuracy=mget("accuracy"),
        rocAuc=mget("roc_auc"),
        f1=mget("f1") if task_type != "regression" else None,
        r2=mget("r2") if task_type == "regression" else None,
        rmse=mget("rmse") if task_type == "regression" else None,
        mae=mget("mae") if task_type == "regression" else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public serialisers — typed Pydantic responses
# ──────────────────────────────────────────────────────────────────────────────

def model_to_list_result(
    m: TrainedModel,
    *,
    is_active: bool = False,
) -> ModelResultResponse:
    """Lightweight serialisation for session list / session summary views."""
    metrics_all: dict = m.metrics_json or {}
    artifacts: dict = m.artifacts_json or {}
    task_type = str(m.task_type or "").lower()

    flat_metrics, mget = _resolve_metrics(metrics_all)

    primary = get_primary_metric(task_type, metrics_all, mget)
    training_time = _extract_training_time(metrics_all)
    train_score = _extract_train_score(metrics_all, primary.name)

    cv_info, cv_test_score, is_cv, has_holdout = _build_cv_info(
        metrics_all, primary_name=primary.name
    )

    # Determine test_score: CV override > primary_score > 0
    ps_block = metrics_all.get("primary_score") if isinstance(metrics_all.get("primary_score"), dict) else {}
    raw_test = ps_block.get("value")
    test_score = cv_test_score if is_cv and cv_test_score else (float(raw_test) if raw_test is not None else primary.value)

    return ModelResultResponse(
        id=str(m.id),
        modelType=str(m.model_type),
        taskType=task_type,
        primaryMetric=primary,
        metrics=_build_metrics_summary(mget, task_type),
        trainScore=train_score,
        testScore=test_score,
        trainingTime=training_time,
        isSaved=bool(m.is_saved),
        isActive=bool(is_active),
        isCV=is_cv,
        hasHoldoutTest=has_holdout,
        testIsCvMean=bool(metrics_all.get("test_is_cv_mean", False)),
        testLabel=metrics_all.get("test_label") or None,
        splitInfo=_build_split_summary(metrics_all, artifacts),
        automl=_build_automl_info(metrics_all, artifacts),
    )


def model_to_detail_result(
    m: TrainedModel,
    *,
    is_active: bool = False,
) -> ModelResultDetailResponse:
    """Full serialisation for the /details endpoint."""
    metrics_all: dict = m.metrics_json or {}
    artifacts: dict = m.artifacts_json or {}
    task_type = str(m.task_type or "").lower()

    flat_metrics, mget = _resolve_metrics(metrics_all)

    primary = get_primary_metric(task_type, metrics_all, mget)
    training_time = _extract_training_time(metrics_all)
    train_score = _extract_train_score(metrics_all, primary.name)

    cv_info, cv_test_score, is_cv, has_holdout = _build_cv_info(
        metrics_all, primary_name=primary.name
    )

    ps_block = metrics_all.get("primary_score") if isinstance(metrics_all.get("primary_score"), dict) else {}
    raw_test = ps_block.get("value")
    test_score = cv_test_score if is_cv and cv_test_score else (float(raw_test) if raw_test is not None else primary.value)

    baseline = artifacts.get("baseline_majority")
    analysis = AnalysisBlock(
        crossValidation=cv_info,
        thresholding=_build_threshold_info(artifacts),
        gridSearch=_build_grid_search_info(artifacts),
        residualAnalysis=artifacts.get("residual_analysis"),
        confusionMatrix=artifacts.get("confusion_matrix"),
        classDistribution=artifacts.get("class_distribution"),
        baselineMajority=float(baseline) if baseline is not None else None,
        metricsWarnings=flat_metrics.get("warnings", []) if isinstance(flat_metrics.get("warnings"), list) else [],
    )

    return ModelResultDetailResponse(
        id=str(m.id),
        modelType=str(m.model_type),
        taskType=task_type,
        primaryMetric=primary,
        metrics=_build_metrics_summary(mget, task_type),
        trainScore=train_score,
        testScore=test_score,
        trainingTime=training_time,
        isSaved=bool(m.is_saved),
        isActive=bool(is_active),
        isCV=is_cv,
        hasHoldoutTest=has_holdout,
        testIsCvMean=bool(metrics_all.get("test_is_cv_mean", False)),
        testLabel=metrics_all.get("test_label") or None,
        splitInfo=_build_split_summary(metrics_all, artifacts),
        automl=_build_automl_info(metrics_all, artifacts),
        metricsDetailed=flat_metrics,
        analysis=analysis,
        preprocessing=artifacts.get("preprocessing"),
        balancing=_build_balancing_info(artifacts),
        hyperparams=artifacts.get("hyperparams"),
    )


def model_to_explainability(m: TrainedModel) -> ExplainabilityResponse:
    """Serialise les données d'explicabilité (SHAP, permutation, feature importance)."""
    artifacts: dict = m.artifacts_json or {}

    raw_fi = artifacts.get("feature_importance", [])
    feature_importance: list[FeatureImportanceItem] = []
    if isinstance(raw_fi, list):
        for item in raw_fi:
            if isinstance(item, dict) and "feature" in item and "importance" in item:
                try:
                    feature_importance.append(
                        FeatureImportanceItem(
                            feature=str(item["feature"]),
                            importance=float(item["importance"]),
                        )
                    )
                except Exception:
                    pass

    return ExplainabilityResponse(
        featureImportance=feature_importance,
        permutationImportance=artifacts.get("permutation_importance"),
        shapGlobal=artifacts.get("shap"),
    )


def model_to_curves(m: TrainedModel) -> CurvesResponse:
    """Serialise les courbes ROC, PR, calibration et d'apprentissage."""
    artifacts: dict = m.artifacts_json or {}
    curves = artifacts.get("curves") if isinstance(artifacts.get("curves"), dict) else {}
    return CurvesResponse(
        roc=curves.get("roc"),
        pr=curves.get("pr"),
        calibration=curves.get("calibration"),
        learningCurves=artifacts.get("learning_curves"),
    )


def session_to_response(
    s: TrainingSession,
    models: list[TrainedModel],
    *,
    active_model_id: int | None = None,
) -> TrainingSessionResponse:
    """Serialize a TrainingSession + its models to a typed TrainingSessionResponse."""
    config = s.config_json if isinstance(s.config_json, dict) else {}
    name = config.get("name") if isinstance(config.get("name"), str) else None

    results = [
        model_to_list_result(
            m,
            is_active=bool(active_model_id is not None and int(m.id) == int(active_model_id)),
        )
        for m in models
    ]

    return TrainingSessionResponse(
        id=str(s.id),
        projectId=str(s.project_id),
        datasetVersionId=str(s.dataset_version_id) if s.dataset_version_id else None,
        name=name,
        status=str(s.status),
        progress=int(s.progress or 0),
        currentModel=s.current_model,
        errorMessage=s.error_message,
        activeModelId=str(active_model_id) if active_model_id is not None else None,
        config=config,
        results=results,
        createdAt=s.created_at.isoformat() if s.created_at else None,
        startedAt=s.started_at.isoformat() if s.started_at else None,
        completedAt=s.finished_at.isoformat() if s.finished_at else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers used outside the presenter (prediction routes, etc.)
# ──────────────────────────────────────────────────────────────────────────────

def extract_feature_names_for_prediction(artifacts: dict[str, Any]) -> list[str]:
    """
    Return the list of feature names required for manual prediction.

    Priority:
    1. artifacts["training_schema"]["feature_names"]  — most accurate
    2. artifacts["columns"]["numeric"] + ["categorical"]  — fallback
    3. Empty list
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
# Backward-compat alias — kept so existing callers don't break during migration
# Remove once all callers are updated to session_to_response / model_to_list_result
# ──────────────────────────────────────────────────────────────────────────────

def model_to_front_result(
    m: TrainedModel,
    *,
    is_saved: bool | None = None,
    is_active: bool = False,
) -> dict[str, Any]:
    """Deprecated: use model_to_list_result or model_to_detail_result instead."""
    result = model_to_list_result(m, is_active=is_active)
    return result.model_dump()


def session_to_front(
    s: TrainingSession,
    models: list[TrainedModel],
    *,
    active_model_id: int | None = None,
) -> dict[str, Any]:
    """Deprecated: use session_to_response instead."""
    return session_to_response(s, models, active_model_id=active_model_id).model_dump()
