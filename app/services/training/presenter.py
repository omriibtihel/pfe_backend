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
from app.services.training.pipeline.cv_utils import _clean_scoring_name

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
# Metric direction and evaluation source helpers
# ──────────────────────────────────────────────────────────────────────────────

_LOWER_IS_BETTER = {"rmse", "mae", "mse", "log_loss", "brier_score"}


def _metric_direction(name: str) -> str:
    return "lower_is_better" if (name or "").lower() in _LOWER_IS_BETTER else "higher_is_better"


def _build_evaluation_source(metrics_json: dict) -> dict:
    has_holdout = metrics_json.get("has_holdout_test", False)
    test_is_cv_mean = metrics_json.get("test_is_cv_mean", False)
    test_label = metrics_json.get("test_label")
    test_n = metrics_json.get("test_n")

    if has_holdout:
        return {
            "type": "holdout_test",
            "label": test_label or "Holdout test set",
            "isIndependentTest": True,
            "nSamples": test_n,
        }
    if metrics_json.get("evaluation_strategy") == "loo":
        return {
            "type": "loo",
            "label": test_label or "LOO validation",
            "isIndependentTest": False,
            "nSamples": None,
        }
    if test_is_cv_mean:
        return {
            "type": "cv_mean",
            "label": test_label or "Moyenne CV",
            "isIndependentTest": False,
            "nSamples": None,
        }
    if metrics_json.get("evaluation_strategy") == "train_only":
        return {
            "type": "train_only",
            "label": "Entraînement uniquement — aucun jeu de test",
            "isIndependentTest": False,
            "nSamples": None,
        }
    return {
        "type": "unknown",
        "label": test_label or "Inconnu",
        "isIndependentTest": False,
        "nSamples": None,
    }


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

    Train-only guard: when evaluation_strategy is "train_only" or no test block
    exists, return an empty mget so primary_value resolves to None — never to a
    train metric value (which would mislabel a train score as a test score).
    """
    strategy = metrics_all.get("evaluation_strategy", "")
    test_block = metrics_all.get("test")

    if strategy == "train_only" or not (isinstance(test_block, dict) and test_block):
        empty: dict = {}

        def mget_none(_k: str) -> Optional[float]:
            return None

        return empty, mget_none

    metrics: dict = test_block

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


def _extract_train_score(metrics_all: dict, primary_name: Optional[str]) -> Optional[float]:
    """Return the training-set metric value for the given primary metric name.

    This value is populated for **all** models, including those whose
    evaluation_strategy is "train_only".  That is intentional: for train-only
    sessions, trainScore is the *only* numeric score available (testScore and
    primaryMetric.value are both null), so suppressing it would remove the sole
    data point consumers have.

    Callers that want evaluation scores should read ``testScore``/``primaryMetric``
    and check ``evaluationSource.type`` before interpreting ``trainScore``:

    * evaluationSource.type == "train_only"  → trainScore is a training metric,
      not a generalisation score.  Treat it as indicative only.
    * evaluationSource.type == "holdout_test" → both trainScore and testScore are
      meaningful; their gap indicates overfitting.
    """
    if not primary_name:
        return None
    train_block = metrics_all.get("train")
    if isinstance(train_block, dict) and primary_name in train_block:
        try:
            return float(train_block[primary_name])
        except Exception:
            pass
    return None


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
                        direction=_metric_direction(str(stored_name)),
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
                    direction=_metric_direction(key),
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
                        direction=_metric_direction(key),
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
    raw_metric = automl_artifacts.get("metric_optimized")
    return AutoMLInfo(
        isBest=bool(automl_artifacts.get("is_best", True)),
        bestEstimator=metrics_all.get("best_estimator"),
        nIterations=metrics_all.get("n_iterations"),
        totalTimeS=metrics_all.get("total_time_s"),
        timeBudgetS=automl_artifacts.get("time_budget_s"),
        metricOptimized=_clean_scoring_name(str(raw_metric)) if raw_metric is not None else None,
    )


def _build_cv_info(metrics_all: dict, *, primary_name: Optional[str]) -> tuple[Optional[CVInfo], Optional[float], bool, bool]:
    """Returns (cv_info | None, test_score_override, is_cv, has_holdout)."""
    is_cv = bool(metrics_all.get("cv", False))
    if not is_cv:
        return None, None, False, False

    has_holdout = bool(metrics_all.get("has_holdout_test", False))
    cv_summary = metrics_all.get("cv_summary") if isinstance(metrics_all.get("cv_summary"), dict) else None
    holdout_metrics = metrics_all.get("holdout_test_metrics") if isinstance(metrics_all.get("holdout_test_metrics"), dict) else None
    cv_mean = metrics_all.get("cv_mean") if isinstance(metrics_all.get("cv_mean"), dict) else None

    test_score_override: Optional[float] = None
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

    all_nan = bool(gs.get("all_nan_scores", False))
    gs_warnings: list[dict] = []
    if all_nan:
        gs_warnings.append({
            "severity": "error",
            "code": "GS_ALL_NAN",
            "message": (
                "GridSearch produced no valid scores. "
                "The model was trained without hyperparameter optimisation."
            ),
        })

    best_score = gs.get("best_score")
    raw_scoring = gs.get("scoring") or gs.get("refit_metric")
    return GridSearchInfo(
        enabled=True,
        searchType=gs.get("search_type") or None,
        cvBestScore=float(best_score) if best_score is not None else None,
        cvScoring=_clean_scoring_name(str(raw_scoring)) if raw_scoring else None,
        bestParams=gs.get("best_params") if isinstance(gs.get("best_params"), dict) else None,
        cvSplits=int(gs["cv_splits"]) if gs.get("cv_splits") else None,
        nCandidates=int(gs["n_candidates"]) if gs.get("n_candidates") else None,
        cvResultsSummary=cv_results_summary,
        gridSearchFailed=all_nan,
        gridSearchFailureReason=(
            "All candidate scores were NaN — "
            "the parameter grid may be incompatible with this dataset or model."
            if all_nan else None
        ),
        warnings=gs_warnings,
    )


def _build_balancing_info(artifacts: dict) -> Optional[BalancingInfo]:
    bal = artifacts.get("balancing")
    if not isinstance(bal, dict):
        return None
    ratio = bal.get("imbalance_ratio")
    raw_refit = bal.get("refit_metric")
    return BalancingInfo(
        strategyApplied=str(bal["strategy_applied"]) if bal.get("strategy_applied") is not None else None,
        refitMetric=_clean_scoring_name(str(raw_refit)) if raw_refit is not None else None,
        imbalanceRatio=float(ratio) if ratio is not None else None,
    )


def _build_metrics_summary(mget: callable, task_type: str) -> MetricsSummary:
    is_regression = task_type == "regression"
    return MetricsSummary(
        accuracy=None if is_regression else mget("accuracy"),
        precision=None if is_regression else mget("precision"),
        recall=None if is_regression else mget("recall"),
        f1=None if is_regression else mget("f1"),
        rocAuc=None if is_regression else mget("roc_auc"),
        prAuc=None if is_regression else mget("pr_auc"),
        balancedAccuracy=None if is_regression else mget("balanced_accuracy"),
        specificity=None if is_regression else mget("specificity"),
        f1Pos=None if is_regression else mget("f1_pos"),
        precisionPos=None if is_regression else mget("precision_pos"),
        recallPos=None if is_regression else mget("recall_pos"),
        precisionMacro=None if is_regression else mget("precision_macro"),
        recallMacro=None if is_regression else mget("recall_macro"),
        f1Macro=None if is_regression else mget("f1_macro"),
        precisionWeighted=None if is_regression else mget("precision_weighted"),
        recallWeighted=None if is_regression else mget("recall_weighted"),
        f1Weighted=None if is_regression else mget("f1_weighted"),
        precisionMicro=None if is_regression else mget("precision_micro"),
        recallMicro=None if is_regression else mget("recall_micro"),
        f1Micro=None if is_regression else mget("f1_micro"),
        r2=mget("r2") if is_regression else None,
        rmse=mget("rmse") if is_regression else None,
        mae=mget("mae") if is_regression else None,
        mse=mget("mse") if is_regression else None,
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
    test_score = cv_test_score if is_cv and cv_test_score is not None else (float(raw_test) if raw_test is not None else primary.value)

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
        evaluationSource=_build_evaluation_source(metrics_all),
        warnings=flat_metrics.get("warnings", []) if isinstance(flat_metrics.get("warnings"), list) else [],
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
    test_score = cv_test_score if is_cv and cv_test_score is not None else (float(raw_test) if raw_test is not None else primary.value)

    baseline_data = artifacts.get("baseline")
    raw_artifact_warnings = artifacts.get("artifact_warnings") or []
    analysis = AnalysisBlock(
        crossValidation=cv_info,
        thresholding=_build_threshold_info(artifacts),
        gridSearch=_build_grid_search_info(artifacts),
        residualAnalysis=artifacts.get("residual_analysis"),
        confusionMatrix=artifacts.get("confusion_matrix"),
        classDistribution=artifacts.get("class_distribution"),
        baseline=baseline_data if isinstance(baseline_data, dict) else None,
        metricsWarnings=flat_metrics.get("warnings", []) if isinstance(flat_metrics.get("warnings"), list) else [],
        artifactWarnings=[w for w in raw_artifact_warnings if isinstance(w, dict) and w.get("artifact") in {"residual_analysis"}],
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
        evaluationSource=_build_evaluation_source(metrics_all),
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

    raw_artifact_warnings = artifacts.get("artifact_warnings") or []
    explainability_artifacts = {"permutation_importance", "shap"}
    return ExplainabilityResponse(
        featureImportance=feature_importance,
        permutationImportance=artifacts.get("permutation_importance"),
        shapGlobal=artifacts.get("shap"),
        artifactWarnings=[w for w in raw_artifact_warnings if isinstance(w, dict) and w.get("artifact") in explainability_artifacts],
    )


def model_to_curves(m: TrainedModel) -> CurvesResponse:
    """Serialise les courbes ROC, PR, calibration et d'apprentissage."""
    artifacts: dict = m.artifacts_json or {}
    curves = artifacts.get("curves") if isinstance(artifacts.get("curves"), dict) else {}
    raw_artifact_warnings = artifacts.get("artifact_warnings") or []
    return CurvesResponse(
        roc=curves.get("roc"),
        pr=curves.get("pr"),
        calibration=curves.get("calibration"),
        learningCurves=artifacts.get("learning_curves"),
        artifactWarnings=[w for w in raw_artifact_warnings if isinstance(w, dict) and w.get("artifact") == "learning_curves"],
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
