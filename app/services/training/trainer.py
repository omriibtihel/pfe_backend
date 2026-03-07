from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold

from .balancing.profiler import is_binary, minority_ratio
from .models import get_model_distributions, get_model_grid

logger = logging.getLogger(__name__)


@dataclass
class TrainerFitResult:
    fitted_pipeline: Any
    tuning_artifacts: Dict[str, Any]


def _log_event(event: str, **payload: Any) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


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
            # Better default for imbalanced binary classification.
            return "average_precision"
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


def _to_python_scalar(v: Any) -> Any:
    """Convert numpy scalars to Python native types for JSON serialization."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all numpy scalar values in a params dict to Python native types."""
    return {k: _to_python_scalar(v) for k, v in params.items()}


def _summarize_cv_results(cv_results: Dict[str, Any], max_rows: int = 5) -> list[Dict[str, Any]]:
    params = list(cv_results.get("params", []) or [])
    ranks = list(np.asarray(cv_results.get("rank_test_score", [])))
    means = list(np.asarray(cv_results.get("mean_test_score", [])))
    stds = list(np.asarray(cv_results.get("std_test_score", [])))

    rows = []
    for i in range(min(len(params), len(ranks), len(means), len(stds))):
        rows.append(
            {
                "rank": int(ranks[i]),
                "mean_test_score": float(means[i]),
                "std_test_score": float(stds[i]),
                "params": {k: _to_python_scalar(v) for k, v in dict(params[i]).items()},
            }
        )
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
    ) -> TrainerFitResult:
        fit_params: Dict[str, Any] = {}
        if fit_sample_weight is not None:
            fit_params["model__sample_weight"] = np.asarray(fit_sample_weight)

        search_type = str(getattr(cfg, "search_type", "")).strip().lower()
        if not search_type or search_type not in {"grid", "random"}:
            # Backward-compat: derive from use_grid_search if search_type not set
            search_type = "grid" if bool(getattr(cfg, "use_grid_search", False)) else "none"

        if search_type == "none":
            _log_event("training.fit", model_type=model_type, tuning=False)
            pipeline.fit(X_train, y_train, **fit_params)
            return TrainerFitResult(
                fitted_pipeline=pipeline,
                tuning_artifacts={
                    "enabled": False,
                    "param_grid": {},
                    "sample_weight_used": bool(fit_sample_weight is not None),
                },
            )

        if isinstance(refit_metric_override, str) and refit_metric_override.strip():
            refit_metric = refit_metric_override.strip()
        else:
            refit_metric = _choose_refit_metric(task_type, y_train, getattr(cfg, "metrics", []) or [])
        cv_splitter, cv_splits = _build_cv_splitter(
            task_type, y_train,
            int(getattr(cfg, "k_folds", 5) or 5),
            random_state=int(getattr(cfg, "random_state", 42)),
        )

        # Build the search estimator (embed resampler inside search for anti-leakage).
        if resampler is not None:
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                model_step = pipeline.named_steps.get("model")
                if model_step is None:
                    model_step = list(pipeline.named_steps.values())[-1]
                search_estimator = ImbPipeline([("resampler", resampler), ("model", model_step)])
            except ImportError:
                _log_event(
                    "training.fit.resampler_fallback",
                    model_type=model_type,
                    reason="imbalanced-learn not importable; resampler embedded in pipeline skipped",
                )
                search_estimator = pipeline
        else:
            search_estimator = pipeline

        # ── RandomizedSearchCV ────────────────────────────────────────────────
        if search_type == "random":
            model_distributions = get_model_distributions(model_type, task_type)
            if not model_distributions:
                _log_event("training.fit", model_type=model_type, tuning=False, reason="empty_distributions")
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
            n_iter = int(getattr(cfg, "n_iter_random_search", 40) or 40)

            _log_event(
                "training.fit",
                model_type=model_type,
                tuning=True,
                search_type="random",
                cv_splits=cv_splits,
                refit_metric=refit_metric,
                n_iter=n_iter,
                resampler_in_pipeline=resampler is not None,
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
                verbose=0,
            )
            rs.fit(X_train, y_train, **fit_params)

            best_params_full = _sanitize_params(dict(rs.best_params_ or {}))
            best_params_model = {
                k.replace("model__", "", 1): v
                for k, v in best_params_full.items()
                if k.startswith("model__")
            }
            tuning_artifacts: Dict[str, Any] = {
                "enabled": True,
                "search_type": "random",
                "refit_metric": refit_metric,
                "scoring": refit_metric,
                "cv_splits": int(cv_splits),
                "best_score": float(rs.best_score_),
                "best_params": best_params_model,
                "best_params_full": best_params_full,
                "param_grid": {},
                "n_candidates": n_iter,
                "cv_results_summary": _summarize_cv_results(rs.cv_results_),
                "sample_weight_used": bool(fit_sample_weight is not None),
            }
            return TrainerFitResult(
                fitted_pipeline=rs.best_estimator_,
                tuning_artifacts=tuning_artifacts,
            )

        # ── GridSearchCV ──────────────────────────────────────────────────────
        model_grid = (
            dict(model_param_grid)
            if isinstance(model_param_grid, dict) and model_param_grid
            else get_model_grid(model_type, task_type)
        )
        if not model_grid:
            _log_event("training.fit", model_type=model_type, tuning=False, reason="empty_param_grid")
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
            _log_event("training.fit", model_type=model_type, tuning=False, reason="invalid_param_grid")
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

        _log_event(
            "training.fit",
            model_type=model_type,
            tuning=True,
            search_type="grid",
            cv_splits=cv_splits,
            refit_metric=refit_metric,
            resampler_in_pipeline=resampler is not None,
        )

        gs = GridSearchCV(
            estimator=search_estimator,
            param_grid=param_grid,
            scoring=refit_metric,
            refit=True,
            cv=cv_splitter,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train, **fit_params)

        best_params_full = _sanitize_params(dict(gs.best_params_ or {}))
        best_params_model = {
            k.replace("model__", "", 1): v
            for k, v in best_params_full.items()
            if k.startswith("model__")
        }

        tuning_artifacts = {
            "enabled": True,
            "search_type": "grid",
            "refit_metric": refit_metric,
            "scoring": refit_metric,
            "cv_splits": int(cv_splits),
            "best_score": float(gs.best_score_),
            "best_params": best_params_model,
            "best_params_full": best_params_full,
            "param_grid": dict(model_grid),
            "n_candidates": int(len(gs.cv_results_.get("params", []))),
            "cv_results_summary": _summarize_cv_results(gs.cv_results_),
            "sample_weight_used": bool(fit_sample_weight is not None),
        }

        return TrainerFitResult(
            fitted_pipeline=gs.best_estimator_,
            tuning_artifacts=tuning_artifacts,
        )
