from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# HalvingRandomSearchCV is experimental in sklearn 1.x — the enable import
# must precede the class import.  Guard so environments without experimental
# support degrade gracefully to RandomizedSearchCV.
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    _HALVING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HALVING_AVAILABLE = False

from app.services.preparation_ml.balancing.profiler import minority_ratio
from app.services.training.pipeline.models import (
    get_adaptive_model_grid,
    get_adaptive_n_iter,
    get_halving_n_candidates,
    get_model_distributions,
    get_model_grid,
)
from app.services.training.utils import log_event
from app.services.training.pipeline.cv_utils import (
    _adapt_resampler_for_cv,
    _build_cv_splitter,
    _choose_refit_metric,
    _extract_resampler_smote_k,
    _extract_search_artifacts,
    _fit_with_early_stopping,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerFitResult:
    fitted_pipeline: Any
    tuning_artifacts: Dict[str, Any]


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
