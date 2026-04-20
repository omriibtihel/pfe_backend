"""CV-related helpers extracted from trainer.py.

Functions here are used by Trainer (via import) and, for a subset, by
orchestrator.py directly (_build_cv_splitter, _choose_refit_metric).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.services.training.pipeline.metrics import is_binary
from app.services.preparation_ml.balancing.profiler import minority_ratio
from app.services.training.utils import log_event, to_python_scalar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric / scoring helpers
# ---------------------------------------------------------------------------

def _metric_to_scoring(metric_name: str, *, task_type: str, binary: bool) -> str | None:
    m = str(metric_name or "").strip().lower()
    if task_type == "classification":
        if m == "accuracy":
            return "accuracy"
        if m in {"f1", "f1_pos"}:
            return "f1" if binary else "f1_weighted"
        if m == "f1_macro":
            return "f1" if binary else "f1_macro"
        if m == "f1_weighted":
            return "f1" if binary else "f1_weighted"
        if m == "f1_micro":
            return "f1" if binary else "f1_micro"
        if m == "precision":
            return "precision" if binary else "precision_weighted"
        if m == "precision_macro":
            return "precision" if binary else "precision_macro"
        if m == "precision_weighted":
            return "precision" if binary else "precision_weighted"
        if m == "precision_micro":
            return "precision" if binary else "precision_micro"
        if m == "recall":
            return "recall" if binary else "recall_weighted"
        if m == "recall_macro":
            return "recall" if binary else "recall_macro"
        if m == "recall_weighted":
            return "recall" if binary else "recall_weighted"
        if m == "recall_micro":
            return "recall" if binary else "recall_micro"
        if m == "roc_auc":
            return "roc_auc" if binary else "roc_auc_ovr_weighted"
        if m == "pr_auc":
            return "average_precision" if binary else None
        return None

    if task_type == "regression":
        if m == "mae":
            return "neg_mean_absolute_error"
        if m == "mse":
            return "neg_mean_squared_error"
        if m == "rmse":
            return "neg_root_mean_squared_error"
        if m == "r2":
            return "r2"
    return None


def _choose_refit_metric(task_type: str, y_train: np.ndarray, requested_metrics: Sequence[str]) -> str:
    binary = is_binary(y_train)
    if task_type == "classification":
        mr = minority_ratio(y_train)
        if binary and mr is not None and mr < 0.20:
            # Imbalanced binary: average_precision penalises false positives on minority class.
            return "average_precision"
        if not binary and mr is not None and mr < 0.20:
            # Imbalanced multiclass: f1_macro weights each class equally regardless of support.
            return "f1_macro"
        for m in requested_metrics:
            scoring = _metric_to_scoring(m, task_type=task_type, binary=binary)
            if scoring:
                return scoring
        return "f1" if binary else "f1_weighted"

    for m in requested_metrics:
        scoring = _metric_to_scoring(m, task_type=task_type, binary=binary)
        if scoring:
            return scoring
    return "r2"


# ---------------------------------------------------------------------------
# CV splitter
# ---------------------------------------------------------------------------

def _build_cv_splitter(task_type: str, y_train: np.ndarray, requested_splits: int, random_state: int = 42) -> Tuple[Any, int]:
    n_samples = int(len(y_train))
    if n_samples < 2:
        raise RuntimeError("Not enough training samples for CV tuning.")

    cv = max(2, int(requested_splits or 5))
    cv = min(cv, n_samples)

    if task_type == "classification":
        try:
            vals, counts = np.unique(y_train, return_counts=True)
            if len(vals) >= 2:
                cv_cls = min(cv, int(counts.min()))
                if cv_cls >= 2:
                    return StratifiedKFold(n_splits=cv_cls, shuffle=True, random_state=random_state), cv_cls
        except Exception:
            pass

    if cv < 2:
        raise RuntimeError("Not enough training samples for CV tuning.")
    return KFold(n_splits=cv, shuffle=True, random_state=random_state), cv


# ---------------------------------------------------------------------------
# Early stopping (XGBoost / LightGBM)
# ---------------------------------------------------------------------------

def _fit_with_early_stopping(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str,
    task_type: str,
    fit_params: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, Any]:
    """For XGBoost/LightGBM with search_type='none': find the optimal n_estimators
    via an internal 80/20 split + early stopping, then refit on the full training set.

    Returns a dict with early stopping metadata stored in tuning_artifacts.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    _EARLY_STOPPING_ROUNDS = 50
    _VAL_SIZE = 0.2

    try:
        # ── Step 1: internal split just for early stopping detection ──────────
        stratify = y_train if task_type == "classification" and len(np.unique(y_train)) >= 2 else None
        try:
            X_sub, X_val_es, y_sub, y_val_es = train_test_split(
                X_train, y_train, test_size=_VAL_SIZE,
                random_state=random_state, stratify=stratify,
            )
        except ValueError:
            # stratify fails when a class has < 2 samples
            X_sub, X_val_es, y_sub, y_val_es = train_test_split(
                X_train, y_train, test_size=_VAL_SIZE, random_state=random_state,
            )

        # ── Step 2: preprocess the validation slice through the non-model steps ─
        preproc_steps = [(k, v) for k, v in pipeline.named_steps.items() if k != "model"]
        if preproc_steps:
            preproc_pipe = clone(SkPipeline(preproc_steps))
            X_sub_prep = preproc_pipe.fit_transform(X_sub, y_sub)
            X_val_prep = preproc_pipe.transform(X_val_es)
        else:
            X_sub_prep = np.asarray(X_sub)
            X_val_prep = np.asarray(X_val_es)

        # ── Step 3: early-stop fit on the 80% subset ──────────────────────────
        model_es = clone(pipeline.named_steps["model"])
        sample_weight_sub = None
        sw_arr = fit_params.get("model__sample_weight")
        if sw_arr is not None:
            try:
                sub_indices = X_sub.index if hasattr(X_sub, "index") else range(len(X_sub))
                train_indices = X_train.index if hasattr(X_train, "index") else range(len(X_train))
                mask = [i for i, idx in enumerate(train_indices) if idx in set(sub_indices)]
                sample_weight_sub = np.asarray(sw_arr)[mask] if mask else None
            except Exception:
                sample_weight_sub = None

        best_n: int | None = None

        if model_type == "xgboost":
            model_es.set_params(early_stopping_rounds=_EARLY_STOPPING_ROUNDS)
            xgb_fit_kw: Dict[str, Any] = {"eval_set": [(X_val_prep, y_val_es)], "verbose": False}
            if sample_weight_sub is not None:
                xgb_fit_kw["sample_weight"] = sample_weight_sub
            model_es.fit(X_sub_prep, y_sub, **xgb_fit_kw)
            if hasattr(model_es, "best_iteration") and model_es.best_iteration is not None:
                best_n = int(model_es.best_iteration) + 1

        elif model_type == "lightgbm":
            try:
                from lightgbm import early_stopping as lgbm_es, log_evaluation
                lgbm_fit_kw: Dict[str, Any] = {
                    "eval_set": [(X_val_prep, y_val_es)],
                    "callbacks": [lgbm_es(_EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(-1)],
                }
                if sample_weight_sub is not None:
                    lgbm_fit_kw["sample_weight"] = sample_weight_sub
                model_es.fit(X_sub_prep, y_sub, **lgbm_fit_kw)
                if hasattr(model_es, "best_iteration_") and model_es.best_iteration_ > 0:
                    best_n = int(model_es.best_iteration_)
            except ImportError:
                pass

        if best_n is None or best_n < 5:
            pipeline.fit(X_train, y_train, **fit_params)
            return {"used": False, "reason": "no_valid_best_iteration"}

        # ── Step 4: refit on full data with the optimal n_estimators ──────────
        pipeline.named_steps["model"].set_params(n_estimators=best_n)
        pipeline.fit(X_train, y_train, **fit_params)
        return {"used": True, "best_n_estimators": best_n, "early_stopping_rounds": _EARLY_STOPPING_ROUNDS}

    except Exception as exc:
        logger.warning("Early stopping failed (%s), using standard fit.", exc)
        pipeline.fit(X_train, y_train, **fit_params)
        return {"used": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Resampler / SMOTE CV safety
# ---------------------------------------------------------------------------

def _extract_resampler_smote_k(resampler: Any) -> int | None:
    if resampler is None:
        return None

    direct_k = getattr(resampler, "k_neighbors", None)
    if isinstance(direct_k, (int, np.integer)):
        return int(direct_k)

    inner_smote = getattr(resampler, "smote", None)
    inner_k = getattr(inner_smote, "k_neighbors", None) if inner_smote is not None else None
    if isinstance(inner_k, (int, np.integer)):
        return int(inner_k)

    return None


def _max_safe_smote_k_for_cv(y_train: np.ndarray, cv_splits: int) -> int | None:
    try:
        _, counts = np.unique(np.asarray(y_train), return_counts=True)
    except Exception:
        return None

    if len(counts) < 2 or cv_splits < 2:
        return None

    min_class_count = int(counts.min())
    min_train_minority = min_class_count - int(np.ceil(min_class_count / cv_splits))
    return max(0, min_train_minority - 1)


def _clone_resampler_with_smote_k(resampler: Any, k_neighbors: int) -> Any:
    cloned = clone(resampler)

    try:
        return cloned.set_params(k_neighbors=int(k_neighbors))
    except Exception:
        pass

    try:
        return cloned.set_params(smote__k_neighbors=int(k_neighbors))
    except Exception:
        pass

    if hasattr(cloned, "k_neighbors"):
        cloned.k_neighbors = int(k_neighbors)
        return cloned

    inner_smote = getattr(cloned, "smote", None)
    if inner_smote is not None and hasattr(inner_smote, "k_neighbors"):
        inner_smote.k_neighbors = int(k_neighbors)

    return cloned


def _adapt_resampler_for_cv(
    resampler: Any,
    y_train: np.ndarray,
    cv_splits: int,
    model_type: str,
) -> tuple[Any, int | None]:
    requested_k = _extract_resampler_smote_k(resampler)
    if requested_k is None:
        return resampler, None

    safe_k = _max_safe_smote_k_for_cv(y_train, cv_splits)
    if safe_k is None or safe_k >= requested_k:
        return resampler, requested_k

    if safe_k < 1:
        log_event(
            "training.fit.resampler_disabled",
            model_type=model_type,
            reason="smote_not_feasible_for_requested_cv",
            requested_k=requested_k,
            cv_splits=cv_splits,
        )
        return None, None

    log_event(
        "training.fit.resampler_adjusted",
        model_type=model_type,
        reason="smote_k_reduced_for_inner_cv",
        requested_k=requested_k,
        effective_k=safe_k,
        cv_splits=cv_splits,
    )
    return _clone_resampler_with_smote_k(resampler, safe_k), safe_k


# ---------------------------------------------------------------------------
# Search artifact extraction
# ---------------------------------------------------------------------------

def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all numpy scalar values in a params dict to Python native types."""
    return {k: to_python_scalar(v) for k, v in params.items()}


def _summarize_cv_results(cv_results: Dict[str, Any], max_rows: int = 20) -> list[Dict[str, Any]]:
    params = list(cv_results.get("params", []) or [])
    ranks = list(np.asarray(cv_results.get("rank_test_score", [])))
    means = list(np.asarray(cv_results.get("mean_test_score", [])))
    stds = list(np.asarray(cv_results.get("std_test_score", [])))

    # Optional: present when return_train_score=True
    train_means_raw = cv_results.get("mean_train_score")
    train_means = list(np.asarray(train_means_raw)) if train_means_raw is not None else None

    # Optional: always present in sklearn CV estimators
    fit_times_raw = cv_results.get("mean_fit_time")
    fit_times = list(np.asarray(fit_times_raw)) if fit_times_raw is not None else None

    # Optional: HalvingRandomSearchCV-specific columns
    iter_col = cv_results.get("iter")
    n_resources_col = cv_results.get("n_resources")
    iters = list(np.asarray(iter_col)) if iter_col is not None else None
    resources = list(np.asarray(n_resources_col)) if n_resources_col is not None else None

    rows = []
    for i in range(min(len(params), len(ranks), len(means), len(stds))):
        mean_test_score = to_python_scalar(means[i])
        std_test_score = to_python_scalar(stds[i])
        row: Dict[str, Any] = {
            "rank": int(ranks[i]),
            "mean_test_score": float(mean_test_score) if mean_test_score is not None else None,
            "std_test_score": float(std_test_score) if std_test_score is not None else None,
            "params": {k: to_python_scalar(v) for k, v in dict(params[i]).items()},
        }
        if train_means is not None and i < len(train_means):
            mean_train_score = to_python_scalar(train_means[i])
            row["mean_train_score"] = float(mean_train_score) if mean_train_score is not None else None
            if mean_train_score is not None and mean_test_score is not None:
                # Positive gap = train > test = overfitting signal
                row["overfit_gap"] = float(mean_train_score) - float(mean_test_score)
        if fit_times is not None and i < len(fit_times):
            fit_time = to_python_scalar(fit_times[i])
            row["mean_fit_time_s"] = float(fit_time) if fit_time is not None else None
        if iters is not None and i < len(iters):
            row["halving_iter"] = int(iters[i])
        if resources is not None and i < len(resources):
            row["n_resources"] = int(resources[i])
        rows.append(row)
    rows.sort(key=lambda r: r["rank"])
    return rows[:max_rows]


def _extract_search_artifacts(
    search: Any,
    *,
    search_type: str,
    refit_metric: str,
    cv_splits: int,
    param_grid: Dict[str, Any],
    n_candidates: int,
    fit_sample_weight: Any,
    all_nan: bool = False,
) -> Dict[str, Any]:
    """Extract tuning artifacts from a fitted search estimator.

    Compatible with GridSearchCV, RandomizedSearchCV, and HalvingRandomSearchCV.
    """
    best_params_full = _sanitize_params(dict(search.best_params_ or {}))
    best_params_model = {
        k.replace("model__", "", 1): v
        for k, v in best_params_full.items()
        if k.startswith("model__")
    }
    best_score = to_python_scalar(getattr(search, "best_score_", None))
    return {
        "enabled": True,
        "search_type": search_type,
        "refit_metric": refit_metric,
        "scoring": refit_metric,
        "cv_splits": int(cv_splits),
        "best_score": float(best_score) if best_score is not None else None,
        "best_params": best_params_model,
        "best_params_full": best_params_full,
        "param_grid": param_grid,
        "n_candidates": n_candidates,
        "cv_results_summary": _summarize_cv_results(search.cv_results_),
        "sample_weight_used": bool(fit_sample_weight is not None),
        "all_nan_scores": all_nan,
    }
