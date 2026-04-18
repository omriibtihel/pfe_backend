from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split

# HalvingRandomSearchCV is experimental in sklearn 1.x — the enable import
# must precede the class import.  Guard so environments without experimental
# support degrade gracefully to RandomizedSearchCV.
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    _HALVING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HALVING_AVAILABLE = False

from app.services.preparation_ml.balancing.profiler import is_binary, minority_ratio
from app.services.training.pipeline.models import (
    get_adaptive_model_grid,
    get_adaptive_n_iter,
    get_halving_n_candidates,
    get_model_distributions,
    get_model_grid,
)
from app.services.training.utils import log_event, to_python_scalar

logger = logging.getLogger(__name__)


@dataclass
class TrainerFitResult:
    fitted_pipeline: Any
    tuning_artifacts: Dict[str, Any]


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


def _compute_min_resources(
    n_samples: int,
    cv_splits: int,
    has_resampler: bool,
    y_train: np.ndarray,
    smote_k_neighbors: int = 5,
) -> int | str:
    """Compute a safe ``min_resources`` for HalvingRandomSearchCV.

    Without a resampler, ``'smallest'`` is sklearn's smart default and works
    correctly.  With SMOTE-family resamplers, ``'smallest'`` can allocate so
    few samples per fold that the minority class ends up with fewer than
    ``k_neighbors`` samples, causing SMOTE to crash.

    The formula guarantees at least ``smote_k + 1`` minority-class samples in
    each training fold:

        min_fold_samples = ceil((smote_k + 1) / (minority_ratio × train_frac))

    where train_frac = (cv_splits - 1) / cv_splits accounts for the fraction
    of samples available to each training fold.
    """
    if not has_resampler:
        return "smallest"
    mr = minority_ratio(y_train)
    if mr is None or mr <= 0 or mr >= 1.0:
        return "smallest"
    train_frac = (cv_splits - 1) / max(cv_splits, 2)
    min_r = int(np.ceil((smote_k_neighbors + 1) / (float(mr) * train_frac)))
    return max(min_r, 20)  # absolute floor to avoid degenerate 1-sample halving


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


def _warn_if_all_nan_scores(search: Any, search_type: str, model_type: str) -> bool:
    """Return True and log a structured warning when every CV candidate scored NaN.

    This happens when *all* estimators in the search raised an exception that was
    silently converted to NaN by ``error_score=np.nan``.  In that case
    ``best_score_`` is NaN and the selected ``best_estimator_`` is effectively
    arbitrary — a dangerous silent failure.
    """
    test_scores = search.cv_results_.get("mean_test_score", np.array([]))
    if len(test_scores) == 0:
        return False
    if np.all(np.isnan(test_scores)):
        log_event(
            "training.search.all_nan_scores",
            model_type=model_type,
            search_type=search_type,
            n_candidates=len(test_scores),
            message=(
                "All CV candidates scored NaN — every estimator raised an exception "
                "(captured by error_score=np.nan). The selected best_estimator_ is "
                "unreliable. Check data shape, hyperparameter ranges, or resampler compatibility."
            ),
        )
        return True
    return False


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


class Trainer:
    def fit(
        self,
        *,
        pipeline: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fit_sample_weight: np.ndarray | None = None,
        cfg: Any,
        model_type: str,
        task_type: str,
        model_param_grid: Dict[str, list[Any]] | None = None,
        refit_metric_override: str | None = None,
        resampler: Any | None = None,
        n_samples: int | None = None,
        imbalanced: bool = False,
    ) -> TrainerFitResult:
        fit_params: Dict[str, Any] = {}
        if fit_sample_weight is not None:
            fit_params["model__sample_weight"] = np.asarray(fit_sample_weight)

        search_type = str(getattr(cfg, "search_type", "")).strip().lower()
        if not search_type or search_type not in {"grid", "random", "halving_random"}:
            # Backward-compat: derive from use_grid_search if search_type not set
            search_type = "grid" if bool(getattr(cfg, "use_grid_search", False)) else "none"

        if search_type == "none":
            _es_meta: Dict[str, Any] = {"used": False}
            _model_type_lower = str(model_type).lower()
            _n_samples_check = n_samples if n_samples is not None else len(y_train)
            # Early stopping for XGBoost/LightGBM when enough data for a val split
            if _model_type_lower in {"xgboost", "lightgbm"} and _n_samples_check >= 50:
                _es_meta = _fit_with_early_stopping(
                    pipeline=pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    model_type=_model_type_lower,
                    task_type=task_type,
                    fit_params=fit_params,
                    random_state=int(getattr(cfg, "random_state", 42)),
                )
            else:
                pipeline.fit(X_train, y_train, **fit_params)
            log_event("training.fit", model_type=model_type, tuning=False,
                      early_stopping=_es_meta.get("used", False),
                      best_n_estimators=_es_meta.get("best_n_estimators"))
            return TrainerFitResult(
                fitted_pipeline=pipeline,
                tuning_artifacts={
                    "enabled": False,
                    "param_grid": {},
                    "sample_weight_used": bool(fit_sample_weight is not None),
                    "early_stopping": _es_meta,
                },
            )

        if isinstance(refit_metric_override, str) and refit_metric_override.strip():
            refit_metric = refit_metric_override.strip()
        else:
            refit_metric = _choose_refit_metric(task_type, y_train, getattr(cfg, "metrics", []) or [])

        # Use inner_cv_folds for the GS inner loop when available (avoids
        # conflating outer-CV folds with hyperparameter search folds).
        _gs_cv_folds = (
            getattr(cfg, "inner_cv_folds", None)
            or getattr(cfg, "k_folds", 5)
            or 5
        )
        cv_splitter, cv_splits = _build_cv_splitter(
            task_type, y_train,
            int(_gs_cv_folds),
            random_state=int(getattr(cfg, "random_state", 42)),
        )

        effective_resampler = resampler
        effective_smote_k = _extract_resampler_smote_k(resampler)
        if effective_resampler is not None and task_type == "classification":
            effective_resampler, effective_smote_k = _adapt_resampler_for_cv(
                effective_resampler,
                np.asarray(y_train),
                cv_splits,
                model_type,
            )

        # Build the search estimator.
        # Anti-leakage: when a resampler (SMOTE / undersampling) is active,
        # embed it INSIDE the search estimator so synthetic samples are
        # generated only within each fold's training split — never on the
        # validation split.
        if effective_resampler is not None:
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                model_step = pipeline.named_steps.get("model")
                if model_step is None:
                    model_step = list(pipeline.named_steps.values())[-1]
                search_estimator = ImbPipeline([("resampler", effective_resampler), ("model", model_step)])
            except ImportError:
                log_event(
                    "training.fit.resampler_fallback",
                    model_type=model_type,
                    reason="imbalanced-learn not importable; resampler embedded in pipeline skipped",
                )
                search_estimator = pipeline
        else:
            search_estimator = pipeline

        # ── HalvingRandomSearchCV ─────────────────────────────────────────────
        if search_type == "halving_random":
            if not _HALVING_AVAILABLE:
                # Graceful degradation: fall through to RandomizedSearchCV.
                log_event(
                    "training.fit",
                    model_type=model_type,
                    tuning=False,
                    reason="halving_unavailable_fallback_random",
                )
                search_type = "random"
            else:
                model_distributions = get_model_distributions(model_type, task_type)
                if not model_distributions:
                    log_event("training.fit", model_type=model_type, tuning=False, reason="empty_distributions")
                    pipeline.fit(X_train, y_train, **fit_params)
                    return TrainerFitResult(
                        fitted_pipeline=pipeline,
                        tuning_artifacts={
                            "enabled": False,
                            "reason": "empty_distributions",
                            "param_grid": {},
                            "sample_weight_used": bool(fit_sample_weight is not None),
                        },
                    )

                param_distributions = {f"model__{k}": v for k, v in model_distributions.items()}

                _n_samples_actual = n_samples if n_samples is not None else int(len(y_train))
                _cfg_n_iter = getattr(cfg, "n_iter_random_search", None)
                # User override via nIterRandomSearch acts as the initial candidate count.
                n_candidates_init = int(_cfg_n_iter if _cfg_n_iter else get_halving_n_candidates(_n_samples_actual))

                # Compute a min_resources floor that is safe when a resampler is
                # embedded — 'smallest' can allocate too few samples for SMOTE.
                min_res = _compute_min_resources(
                    n_samples=_n_samples_actual,
                    cv_splits=cv_splits,
                    has_resampler=(effective_resampler is not None),
                    y_train=y_train,
                    smote_k_neighbors=int(effective_smote_k or 5),
                )

                log_event(
                    "training.fit",
                    model_type=model_type,
                    tuning=True,
                    search_type="halving_random",
                    cv_splits=cv_splits,
                    refit_metric=refit_metric,
                    n_candidates_init=n_candidates_init,
                    min_resources=min_res,
                    resampler_in_pipeline=effective_resampler is not None,
                )

                hs = HalvingRandomSearchCV(
                    estimator=search_estimator,
                    param_distributions=param_distributions,
                    n_candidates=n_candidates_init,
                    factor=3,
                    min_resources=min_res,
                    resource="n_samples",
                    max_resources="auto",
                    cv=cv_splitter,
                    scoring=refit_metric,
                    refit=True,
                    n_jobs=-1,
                    random_state=int(getattr(cfg, "random_state", 42)),
                    error_score=np.nan,
                    return_train_score=True,
                    verbose=0,
                )
                hs.fit(X_train, y_train, **fit_params)
                _hs_all_nan = _warn_if_all_nan_scores(hs, "halving_random", model_type)

                return TrainerFitResult(
                    fitted_pipeline=hs.best_estimator_,
                    tuning_artifacts=_extract_search_artifacts(
                        hs,
                        search_type="halving_random",
                        refit_metric=refit_metric,
                        cv_splits=cv_splits,
                        param_grid={},
                        n_candidates=int(sum(hs.n_candidates_)),
                        fit_sample_weight=fit_sample_weight,
                        all_nan=_hs_all_nan,
                    ),
                )

        # ── RandomizedSearchCV ────────────────────────────────────────────────
        if search_type == "random":
            model_distributions = get_model_distributions(model_type, task_type)
            if not model_distributions:
                log_event("training.fit", model_type=model_type, tuning=False, reason="empty_distributions")
                pipeline.fit(X_train, y_train, **fit_params)
                return TrainerFitResult(
                    fitted_pipeline=pipeline,
                    tuning_artifacts={
                        "enabled": False,
                        "reason": "empty_distributions",
                        "param_grid": {},
                        "sample_weight_used": bool(fit_sample_weight is not None),
                    },
                )

            param_distributions = {f"model__{k}": v for k, v in model_distributions.items()}

            # Adaptive n_iter: scales with dataset size.
            # User-supplied n_iter_random_search overrides when explicitly set.
            _n_samples_actual = n_samples if n_samples is not None else int(len(y_train))
            _adaptive_n_iter = get_adaptive_n_iter(_n_samples_actual)
            _cfg_n_iter = getattr(cfg, "n_iter_random_search", None)
            n_iter = int(_cfg_n_iter if _cfg_n_iter else _adaptive_n_iter)

            log_event(
                "training.fit",
                model_type=model_type,
                tuning=True,
                search_type="random",
                cv_splits=cv_splits,
                refit_metric=refit_metric,
                n_iter=n_iter,
                resampler_in_pipeline=effective_resampler is not None,
            )

            rs = RandomizedSearchCV(
                estimator=search_estimator,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=refit_metric,
                refit=True,
                cv=cv_splitter,
                n_jobs=-1,
                random_state=int(getattr(cfg, "random_state", 42)),
                error_score=np.nan,
                return_train_score=True,
                verbose=0,
            )
            rs.fit(X_train, y_train, **fit_params)
            _rs_all_nan = _warn_if_all_nan_scores(rs, "random", model_type)

            return TrainerFitResult(
                fitted_pipeline=rs.best_estimator_,
                tuning_artifacts=_extract_search_artifacts(
                    rs,
                    search_type="random",
                    refit_metric=refit_metric,
                    cv_splits=cv_splits,
                    param_grid={},
                    n_candidates=n_iter,
                    fit_sample_weight=fit_sample_weight,
                    all_nan=_rs_all_nan,
                ),
            )

        # ── GridSearchCV ──────────────────────────────────────────────────────
        if isinstance(model_param_grid, dict) and model_param_grid:
            # Caller provided an explicit grid (e.g. from user hyperparams) — use it verbatim.
            model_grid = dict(model_param_grid)
        elif n_samples is not None:
            # Use the adaptive grid that scales with dataset size and injects
            # class_weight for imbalanced datasets on supporting models.
            model_grid = get_adaptive_model_grid(
                model_type, task_type,
                n_samples=n_samples,
                imbalanced=imbalanced,
            )
        else:
            model_grid = get_model_grid(model_type, task_type)
        if not model_grid:
            log_event("training.fit", model_type=model_type, tuning=False, reason="empty_param_grid")
            pipeline.fit(X_train, y_train, **fit_params)
            return TrainerFitResult(
                fitted_pipeline=pipeline,
                tuning_artifacts={
                    "enabled": False,
                    "reason": "empty_param_grid",
                    "param_grid": {},
                    "sample_weight_used": bool(fit_sample_weight is not None),
                },
            )

        param_grid = {f"model__{k}": v for k, v in model_grid.items() if isinstance(v, list) and len(v) > 0}
        if not param_grid:
            log_event("training.fit", model_type=model_type, tuning=False, reason="invalid_param_grid")
            pipeline.fit(X_train, y_train, **fit_params)
            return TrainerFitResult(
                fitted_pipeline=pipeline,
                tuning_artifacts={
                    "enabled": False,
                    "reason": "invalid_param_grid",
                    "param_grid": {},
                    "sample_weight_used": bool(fit_sample_weight is not None),
                },
            )

        log_event(
            "training.fit",
            model_type=model_type,
            tuning=True,
            search_type="grid",
            cv_splits=cv_splits,
            refit_metric=refit_metric,
            resampler_in_pipeline=effective_resampler is not None,
            n_samples=n_samples,
            imbalanced=imbalanced,
            n_grid_combos=int(
                __import__("math").prod(len(v) for v in model_grid.values()) if model_grid else 0
            ),
        )

        gs = GridSearchCV(
            estimator=search_estimator,
            param_grid=param_grid,
            scoring=refit_metric,
            refit=True,
            cv=cv_splitter,
            n_jobs=-1,
            error_score=np.nan,
            return_train_score=True,
            verbose=0,
        )
        gs.fit(X_train, y_train, **fit_params)
        _gs_all_nan = _warn_if_all_nan_scores(gs, "grid", model_type)

        return TrainerFitResult(
            fitted_pipeline=gs.best_estimator_,
            tuning_artifacts=_extract_search_artifacts(
                gs,
                search_type="grid",
                refit_metric=refit_metric,
                cv_splits=cv_splits,
                param_grid=dict(model_grid),
                n_candidates=int(len(gs.cv_results_.get("params", []))),
                fit_sample_weight=fit_sample_weight,
                all_nan=_gs_all_nan,
            ),
        )
