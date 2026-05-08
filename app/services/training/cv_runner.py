# [REFACTOR] Extracted from orchestrator.py to keep cross-validation runners
# in their own module. Helpers shared with the holdout path are imported back
# from orchestrator (cycle is safe — orchestrator pulls cv_runner symbols only
# at the very bottom of its module body, after all helpers are defined).
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV as _GridSearchCV
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.services.preparation_ml.balancing import (
    BalancingDecision,
    BalancingExecutor,
    DataProfile,
    class_counts,
    minority_ratio,
    profile_binary_dataset,
    resolve,
)
from app.services.preparation_ml.feature_engineering import FeatureEngineeringTransformer
from app.services.preparation_ml.feature_engineering.transformer import validate_feature_defs
from app.services.preparation_ml.preprocessing.preprocessing import build_preprocessor
from app.services.preparation_ml.preprocessing.transformers import (
    ColumnAligner,
    clear_clip_warnings,
    get_clip_warnings,
)
from app.services.preparation_ml.splitters import (
    iter_group_kfold_splits,
    iter_kfold_splits,
    iter_loo_splits,
    iter_repeated_kfold_splits,
)
from app.services.training.config.schema import TrainingConfig, normalize_model_hyperparams
from app.services.training.output.audit import build_and_persist_audit
from app.services.training.output.reporter import Reporter, build_training_schema
from app.services.training.pipeline.confidence import compute_bootstrap_cis
from app.services.training.pipeline.cv_utils import _build_cv_splitter, _choose_refit_metric
from app.services.training.pipeline.evaluator import Evaluator
from app.services.training.pipeline.models import build_model, get_model_capabilities
from app.services.training.pipeline.trainer import Trainer

# Helpers shared with the holdout path live in orchestrator.py — this import
# is resolved at module-load time, but orchestrator imports cv_runner only
# *after* it has finished defining these names, so there is no cycle.
from app.services.training.orchestrator import (
    ModelRunResult,
    _DENSE_REQUIRED_MODELS,
    _aggregate_cv_metrics,
    _build_inference_pipeline,
    _compute_oof_regression_metrics,
    _default_decision_for_non_classification,
    _ensure_dense_matrix,
    _ensure_scaling_for_model,
    _extract_scalar_metrics,
    _log_event,
    _log_variance_threshold,
    _resolve_positive_label_for_run,
    _smote_minority_guard,
)

logger = logging.getLogger(__name__)


def _collect_fold_warnings(
    fold_results: list[dict],
    *,
    collapse_threshold: int = 3,
) -> tuple[list[dict], list[str]]:
    """
    Surface per-fold warnings that were previously stored in fold_results but
    never propagated to metrics_json["warnings"].

    Each entry in fold_results["warnings"] is normalised to a dict and enriched
    with "fold" and "context" fields.  When the same (code, message) pair
    appears in more than *collapse_threshold* folds it is collapsed into a
    single FOLD_WARNING_REPEATED summary to prevent warning overload for large K.

    Returns:
        structured  — list[dict] suitable for metrics_json["fold_warnings"]
        string_list — list[str]  to append to the existing all_warnings string list
    """
    raw: list[dict] = []
    n_folds_total = len(fold_results)
    for fr in fold_results:
        fold_num = fr.get("fold", "?")
        for w in fr.get("warnings", []):
            if isinstance(w, str):
                entry: dict[str, Any] = {
                    "severity": "warning",
                    "code": "FOLD_WARNING",
                    "message": w,
                }
            elif isinstance(w, dict):
                entry = dict(w)
            else:
                continue
            entry["fold"] = fold_num
            entry["context"] = "cv_fold"
            raw.append(entry)

    if not raw:
        return [], []

    # Group by (code, message) to detect same-warning repetition across folds.
    groups: dict[tuple, list[dict]] = {}
    for entry in raw:
        key = (entry.get("code", "FOLD_WARNING"), entry.get("message", ""))
        groups.setdefault(key, []).append(entry)

    structured: list[dict] = []
    for (code, message), entries in groups.items():
        if len(entries) > collapse_threshold:
            affected = sorted(
                e["fold"] for e in entries if e.get("fold") is not None
            )
            structured.append({
                "severity": entries[0].get("severity", "warning"),
                "code": "FOLD_WARNING_REPEATED",
                "message": (
                    f"'{code}' appeared in {len(entries)}/{n_folds_total} folds "
                    f"(folds {','.join(str(f) for f in affected)}): {message}"
                ),
                "affected_folds": affected,
                "context": "cv_fold",
            })
        else:
            structured.extend(entries)

    string_list: list[str] = []
    for s in structured:
        if s.get("code") == "FOLD_WARNING_REPEATED":
            string_list.append(s["message"])
        else:
            f_num = s.get("fold", "?")
            string_list.append(f"fold {f_num}: {s.get('message', s.get('code', ''))}")

    return structured, string_list


def _run_kfold_cv(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    """
    Cross-validation pipeline (kfold / stratified_kfold / repeated_stratified_kfold /
    group_kfold / stratified_group_kfold).

    Anti-leakage guarantees
    -----------------------
    • When cfg.test_ratio > 0, a stratified holdout test set is carved out of
      the full data FIRST — before any preprocessing fit, before any CV fold.
      For group_kfold: GroupShuffleSplit ensures no patient spans cv/test boundary.
    • The CV loop (fold fit + val evaluation) runs on the non-test portion only.
    • Preprocessor is fit on train_fold indices ONLY, then transforms both
      train_fold and val_fold.
    • Resampling (SMOTE / undersampling) is applied on train_fold ONLY.
    • val_fold and the holdout test set are NEVER resampled.
    • group_column is excluded from X before any processing (prevents leakage).

    GridSearch + CV interaction
    ---------------------------
    GridSearch (use_grid_search=True) is intentionally DISABLED inside the
    per-fold training loop to avoid unintended double-CV (outer loop × inner
    GridSearchCV). GridSearch is applied once on the FINAL REFIT so the saved
    model benefits from hyperparameter tuning.

    Persistence (Option A — refit-final)
    -------------------------------------
    • test_ratio == 0: refit on ALL data → deployed model trained on 100% of samples.
    • test_ratio >  0: refit on non-test data → test set remains truly independent.
    The saved artifact declares whether refit covered the full dataset.
    """
    from sklearn.model_selection import train_test_split as _train_test_split

    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    _log_event(
        "training.cv.start",
        model_type=model_type_norm,
        split_method=cfg.split_method,
        k_folds=cfg.k_folds,
        task_type=cfg.task_type,
        test_ratio=float(cfg.test_ratio),
        n_repeats=int(getattr(cfg, "n_repeats", 3)),
        group_column=getattr(cfg, "group_column", None),
    )

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    if len(df2) < 10:
        raise RuntimeError("Not enough rows after dropping target NaNs (minimum 10).")

    # ── Group column handling (group_kfold / stratified_group_kfold) ──────────
    # Extract groups BEFORE building X so the group_column is never a feature.
    _is_group_cv = cfg.split_method in ("group_kfold", "stratified_group_kfold")
    groups_all: Optional[np.ndarray] = None
    if _is_group_cv:
        gc = getattr(cfg, "group_column", None)
        if not gc or gc not in df2.columns:
            raise RuntimeError(
                f"splitMethod='{cfg.split_method}' requires groupColumn "
                f"to be a valid column in the dataset. Got: '{gc}'."
            )
        groups_all = np.asarray(df2[gc].values)
        drop_cols = [cfg.target_column, gc]
    else:
        drop_cols = [cfg.target_column]

    X_all = df2.drop(columns=drop_cols)
    y_all = np.asarray(df2[cfg.target_column].values)

    fe_defs = [f.as_dict() for f in cfg.feature_engineering.features]
    fe_validation_errors = validate_feature_defs(fe_defs, list(X_all.columns))
    if fe_validation_errors:
        raise RuntimeError("Feature engineering config errors:\n" + "\n".join(fe_validation_errors))

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        y_all, cfg.positive_label
    )
    if positive_label_warning:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    # ── Optional holdout test split ───────────────────────────────────────────
    # When cfg.test_ratio > 0, we separate a stratified (for classification)
    # test set BEFORE any CV or preprocessing — pure holdout, never touched.
    # For group_kfold: use GroupShuffleSplit to keep patient groups intact.
    has_holdout_test = float(cfg.test_ratio) > 1e-6
    X_test_holdout: Optional[pd.DataFrame] = None
    y_test_holdout: Optional[np.ndarray] = None
    X_cv: pd.DataFrame = X_all
    y_cv: np.ndarray = y_all
    groups_cv: Optional[np.ndarray] = groups_all  # may be None for non-group methods

    if has_holdout_test:
        if _is_group_cv and groups_all is not None:
            # GroupShuffleSplit: ensures no patient spans the cv/test boundary
            from sklearn.model_selection import GroupShuffleSplit as _GSS
            gss = _GSS(n_splits=1, test_size=float(cfg.test_ratio), random_state=cfg.random_state)
            try:
                cv_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))
            except Exception as split_exc:
                _log_event("training.cv.holdout_split_fallback", model_type=model_type_norm, reason=str(split_exc))
                cv_idx = np.arange(int(len(y_all) * (1.0 - float(cfg.test_ratio))))
                test_idx = np.arange(len(cv_idx), len(y_all))
            X_cv = X_all.iloc[cv_idx].copy()
            X_test_holdout = X_all.iloc[test_idx].copy()
            y_cv = y_all[cv_idx]
            y_test_holdout = np.asarray(y_all[test_idx])
            groups_cv = groups_all[cv_idx]
        else:
            stratify_arr = y_all if cfg.task_type == "classification" else None
            try:
                X_cv_arr, X_test_holdout, y_cv_arr, y_test_holdout = _train_test_split(
                    X_all, y_all,
                    test_size=float(cfg.test_ratio),
                    random_state=cfg.random_state,
                    stratify=stratify_arr,
                )
            except Exception as split_exc:
                _log_event(
                    "training.cv.holdout_split_fallback",
                    model_type=model_type_norm,
                    reason=str(split_exc),
                )
                X_cv_arr, X_test_holdout, y_cv_arr, y_test_holdout = _train_test_split(
                    X_all, y_all,
                    test_size=float(cfg.test_ratio),
                    random_state=cfg.random_state,
                )
            X_cv = X_cv_arr
            y_cv = np.asarray(y_cv_arr)
            y_test_holdout = np.asarray(y_test_holdout)
        _log_event(
            "training.cv.holdout_test_split",
            model_type=model_type_norm,
            n_total=int(len(y_all)),
            n_cv=int(len(y_cv)),
            n_test=int(len(y_test_holdout)),
            test_ratio=float(cfg.test_ratio),
        )

    # Normalize hyperparams once — GridSearch disabled in fold loop (see docstring)
    requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})
    if not isinstance(requested_hyperparams_raw, dict):
        requested_hyperparams_raw = {}
    hp_normalized_no_gs = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=False,  # No inner GridSearch per fold (avoid double-CV)
        task_type=cfg.task_type,
    )
    hp_errors = [str(m) for m in (hp_normalized_no_gs.get("errors") or []) if str(m).strip()]
    if hp_errors:
        raise RuntimeError(f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}")
    hp_warnings: list[str] = [str(w) for w in (hp_normalized_no_gs.get("warnings") or []) if str(w).strip()]
    for hp_warn in hp_warnings:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=hp_warn)

    base_estimator_hp = dict(hp_normalized_no_gs.get("estimator_params") or {})
    capabilities = get_model_capabilities(model_type_norm)
    shuffle = bool(getattr(cfg, "shuffle", True))
    effective_preprocessing = _ensure_scaling_for_model(model_type_norm, cfg.preprocessing)

    # ── Nested CV: param_grid for inner GridSearch ────────────────────────────
    # When use_grid_search=True and the model has a param_grid, each outer fold
    # runs an inner GridSearchCV (nested CV). This prevents optimistic bias that
    # arises when the same data is used to both tune and evaluate the model.
    _hp_with_gs = normalize_model_hyperparams(
        model_type_norm, requested_hyperparams_raw,
        use_grid_search=True, task_type=cfg.task_type,
    )
    nested_param_grid = dict(_hp_with_gs.get("param_grid") or {})
    use_nested_cv = bool(cfg.use_grid_search) and bool(nested_param_grid)
    inner_k = int(getattr(cfg, "inner_cv_folds", 3))
    inner_scoring = _choose_refit_metric(cfg.task_type, y_cv, list(cfg.metrics))
    if use_nested_cv:
        _log_event(
            "training.cv.nested_cv_enabled",
            model_type=model_type_norm,
            inner_k=inner_k,
            inner_scoring=inner_scoring,
            n_param_combinations=len(nested_param_grid),
        )

    # ── Generate fold indices (on CV portion only) ────────────────────────────
    try:
        if cfg.split_method == "repeated_stratified_kfold":
            folds_list = list(
                iter_repeated_kfold_splits(
                    X_cv,
                    y_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    n_repeats=int(getattr(cfg, "n_repeats", 3)),
                    random_state=cfg.random_state,
                )
            )
        elif cfg.split_method in ("group_kfold", "stratified_group_kfold"):
            if groups_cv is None:
                raise RuntimeError(
                    f"splitMethod='{cfg.split_method}' requires groups — "
                    "groupColumn missing or not extracted."
                )
            folds_list = list(
                iter_group_kfold_splits(
                    X_cv,
                    y_cv,
                    groups_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    random_state=cfg.random_state,
                )
            )
        else:
            folds_list = list(
                iter_kfold_splits(
                    X_cv,
                    y_cv,
                    split_method=cfg.split_method,
                    k_folds=cfg.k_folds,
                    shuffle=shuffle,
                    random_state=cfg.random_state,
                )
            )
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    actual_k = len(folds_list)
    fold_results: List[Dict[str, Any]] = []
    # OOF predictions accumulated for leakage-free threshold calibration.
    oof_proba_parts: list = []
    oof_true_parts: list = []
    # OOF predicted class labels (classification) for pooled bootstrap CIs.
    oof_y_pred_class_parts: list = []
    # OOF predictions (regression only) for unbiased pooled RMSE/R² recomputation.
    oof_y_true_reg: list = []
    oof_y_pred_reg: list = []

    # ── Per-fold loop ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(folds_list):
        fold_num = fold_idx + 1
        _log_event("training.cv.fold_start", fold=fold_num, k=actual_k, model_type=model_type_norm)
        try:
            X_train_fold = X_cv.iloc[train_idx].copy()
            X_val_fold = X_cv.iloc[val_idx].copy()
            y_train_fold = y_cv[train_idx]
            y_val_fold = y_cv[val_idx]

            # ① Fit preprocessor on train_fold ONLY
            fold_schema = build_training_schema(
                X=X_train_fold,
                target=cfg.target_column,
                preprocessing_config=effective_preprocessing.as_dict(),
            )
            fold_aligner = ColumnAligner(
                feature_names=fold_schema["feature_names"],
                dtypes=fold_schema["dtypes"],
            )
            X_train_aligned = fold_aligner.fit_transform(X_train_fold)
            # Feature engineering: fit stats on train_fold only (anti-leakage)
            fold_fe = FeatureEngineeringTransformer(fe_defs)
            X_train_fe = fold_fe.fit_transform(X_train_aligned)
            fold_spec = build_preprocessor(X_train_fe, effective_preprocessing, kind_overrides=kind_overrides)
            X_train_prep = fold_spec.preprocessor.fit_transform(X_train_fe, y_train_fold)
            # ② Transform val_fold with the train-fitted preprocessor (no leakage)
            X_val_aligned = fold_aligner.transform(X_val_fold)
            X_val_fe = fold_fe.transform(X_val_aligned)
            X_val_prep = fold_spec.preprocessor.transform(X_val_fe)

            # ② b) Fit VarianceThreshold on train_fold ONLY, then apply to both (no leakage)
            # threshold=0.0: removes only perfectly constant features (see PreprocessingConfig).
            _vt_fold_w: list[str] = []
            if fold_spec.feature_selector is not None:
                _vt_n_before = X_train_prep.shape[1] if hasattr(X_train_prep, "shape") else 0
                _vt_fold_names: list = []
                try:
                    _vt_fold_names = list(fold_spec.preprocessor.get_feature_names_out())
                except Exception:
                    pass
                fold_spec.feature_selector.fit(X_train_prep)
                X_train_prep = fold_spec.feature_selector.transform(X_train_prep)
                X_val_prep = fold_spec.feature_selector.transform(X_val_prep)
                _vt_fold_w = _log_variance_threshold(
                    model_type_norm, _vt_n_before, X_train_prep.shape[1],
                    cfg.preprocessing.variance_threshold,
                    feature_selector=fold_spec.feature_selector,
                    feature_names_before=_vt_fold_names,
                )
                if X_train_prep.shape[1] == 0:
                    raise RuntimeError(
                        f"Preprocessing a supprimé toutes les features pour le modèle '{model_type_norm}'. "
                        "Vérifiez votre configuration de preprocessing (VarianceThreshold trop strict, "
                        "ou toutes les colonnes ont une variance nulle)."
                    )

            if model_type_norm in _DENSE_REQUIRED_MODELS:
                X_train_prep = _ensure_dense_matrix(X_train_prep)
                X_val_prep = _ensure_dense_matrix(X_val_prep)

            # ③ Balancing — train_fold ONLY (val never resampled)
            fold_decision: BalancingDecision = _default_decision_for_non_classification()
            fold_profile: Optional[DataProfile] = None
            if cfg.task_type == "classification":
                fold_profile = profile_binary_dataset(y_train_fold, X_train_fold.shape)
                fold_decision = resolve(
                    profile=fold_profile,
                    config=cfg.balancing,
                    model_supports_class_weight=capabilities["supports_class_weight"],
                    model_supports_predict_proba=capabilities["supports_predict_proba"],
                    model_supports_sample_weight=capabilities["supports_sample_weight"],
                    random_state=cfg.random_state,
                )

            fold_executor = BalancingExecutor()
            X_f = X_train_prep
            y_f = np.asarray(y_train_fold)
            fold_fit_params: Dict[str, Any] = {}
            fold_warnings: list[str] = list(_vt_fold_w)
            if cfg.task_type == "classification":
                # SMOTE guard: when the minority class on this fold is too
                # small for k_neighbors interpolation, fall back to "no
                # resampling" instead of letting imblearn raise. Mirrors the
                # LOO guard so behaviour is consistent across CV strategies.
                fold_decision, _smote_warn = _smote_minority_guard(fold_decision, y_train_fold)
                if _smote_warn is not None:
                    fold_warnings.append(_smote_warn)
                    _log_event(
                        "training.cv.fold_smote_skipped",
                        fold=fold_num,
                        model_type=model_type_norm,
                        reason=_smote_warn,
                    )
                # Convert sparse → dense ONLY when resampler requires it
                if fold_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
                    if hasattr(X_train_prep, "toarray"):
                        X_train_prep = X_train_prep.toarray()
                    if hasattr(X_val_prep, "toarray"):
                        X_val_prep = X_val_prep.toarray()
                X_f, y_f, fold_fit_params = fold_executor.apply_prefit(
                    X_train_prep, np.asarray(y_train_fold), fold_decision
                )

            fold_hp = dict(base_estimator_hp)
            if "class_weight" in fold_fit_params:
                fold_hp["class_weight"] = fold_fit_params["class_weight"]

            # ④ Train model — nested CV (inner GridSearch per fold) when enabled,
            #    otherwise plain fit.
            sw = fold_fit_params.get("sample_weight")
            fold_best_inner_params: Optional[Dict[str, Any]] = None
            if use_nested_cv:
                _inner_cv_splitter, _ = _build_cv_splitter(
                    cfg.task_type, y_f, inner_k, cfg.random_state
                )
                _inner_model = build_model(model_type_norm, cfg.task_type, fold_hp)
                _inner_pipe = Pipeline(steps=[("model", _inner_model)])
                _inner_gs = _GridSearchCV(
                    _inner_pipe, nested_param_grid,
                    cv=_inner_cv_splitter, scoring=inner_scoring,
                    refit=True, n_jobs=-1, error_score="raise",
                )
                if sw is not None:
                    _inner_gs.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
                else:
                    _inner_gs.fit(X_f, y_f)
                fold_pipeline = _inner_gs.best_estimator_
                fold_best_inner_params = {
                    k.replace("model__", "", 1): v
                    for k, v in (_inner_gs.best_params_ or {}).items()
                    if k.startswith("model__")
                }
            else:
                fold_model = build_model(model_type_norm, cfg.task_type, fold_hp)
                fold_pipeline = Pipeline(steps=[("model", fold_model)])
                if sw is not None:
                    fold_pipeline.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
                else:
                    fold_pipeline.fit(X_f, y_f)

            # ⑤ Evaluate on val_fold (untouched by resampling)
            evaluator = Evaluator(
                task_type=cfg.task_type,
                requested_metrics=cfg.metrics,
                positive_label=resolved_positive_label,
            )
            val_eval = evaluator.evaluate(fold_pipeline, X_val_prep, np.asarray(y_val_fold))

            # Collect OOF predictions for leakage-free threshold calibration.
            if cfg.task_type == "classification":
                _pp_fn = getattr(fold_pipeline, "predict_proba", None)
                if callable(_pp_fn):
                    try:
                        oof_proba_parts.append(np.asarray(_pp_fn(X_val_prep)))
                        oof_true_parts.append(np.asarray(y_val_fold))
                    except Exception:
                        pass
                # Pooled OOF y_pred (classes) feeds the bootstrap CI step below.
                try:
                    oof_y_pred_class_parts.append(np.asarray(val_eval.predictions))
                except Exception:
                    pass

            # Collect OOF predictions (regression) for unbiased pooled metrics.
            # Skipped for classification to avoid unnecessary memory use.
            if cfg.task_type == "regression":
                try:
                    oof_y_pred_reg.append(np.asarray(val_eval.predictions))
                    oof_y_true_reg.append(np.asarray(y_val_fold))
                except Exception:
                    pass

            smote_added: Optional[int] = None
            if cfg.task_type == "classification" and fold_decision.strategy in {"smote", "smote_tomek"}:
                smote_added = int(len(y_f) - len(y_train_fold))

            fold_class_dist: Optional[Dict[str, Any]] = None
            if cfg.task_type == "classification":
                fold_class_dist = {
                    "train": class_counts(y_train_fold),
                    "val": class_counts(y_val_fold),
                    "train_minority_ratio": minority_ratio(y_train_fold),
                    "val_minority_ratio": minority_ratio(y_val_fold),
                }

            fold_results.append({
                "fold": fold_num,
                "status": "ok",
                "train_size": int(len(y_train_fold)),
                "val_size": int(len(y_val_fold)),
                "metrics": val_eval.metrics,
                "balancing_strategy": str(fold_decision.strategy),
                "balancing_rationale": str(getattr(fold_decision, "rationale", "")),
                "smote_samples_added": smote_added,
                "class_distribution": fold_class_dist,
                "best_inner_params": fold_best_inner_params,
                "warnings": list(fold_warnings),
            })
            _log_event("training.cv.fold_ok", fold=fold_num, model_type=model_type_norm)

        except Exception as fold_exc:
            _log_event(
                "training.cv.fold_error",
                fold=fold_num,
                model_type=model_type_norm,
                reason=str(fold_exc),
            )
            fold_results.append({
                "fold": fold_num,
                "status": "failed",
                "error": str(fold_exc),
            })

    n_ok = sum(1 for fr in fold_results if fr.get("status") == "ok")
    if n_ok == 0:
        errors_summary = "; ".join(
            fr.get("error", "?") for fr in fold_results if fr.get("status") == "failed"
        )
        raise RuntimeError(
            f"All {actual_k} folds failed for model '{model_type_norm}'. "
            f"Errors: {errors_summary}"
        )

    cv_summary = _aggregate_cv_metrics(fold_results)

    # ── Pooled OOF recomputation for regression (unbiased RMSE/MSE/R²) ────────
    # _aggregate_cv_metrics averages each metric across folds — correct for MAE
    # but biased for RMSE (Jensen) and divergent for R².  We recompute these
    # three on the concatenated OOF predictions and override them in
    # cv_summary["mean"]; MAE and all other entries stay untouched.
    if cfg.task_type == "regression" and oof_y_true_reg and oof_y_pred_reg:
        try:
            _oof_y_true = np.concatenate(oof_y_true_reg)
            _oof_y_pred = np.concatenate(oof_y_pred_reg)
            _pooled = _compute_oof_regression_metrics(_oof_y_true, _oof_y_pred)
            _mean_block = cv_summary.get("mean") if isinstance(cv_summary.get("mean"), dict) else {}
            _mean_block["rmse"] = _pooled["rmse"]
            _mean_block["mse"]  = _pooled["mse"]
            _mean_block["r2"]   = _pooled["r2"]
            _mean_block["oof_sample_count"] = int(len(_oof_y_true))
            cv_summary["mean"] = _mean_block
            logger.info(
                "CV regression metrics recomputed on %d OOF predictions (pooled): "
                "RMSE=%.4f, R²=%.4f",
                len(_oof_y_true), _pooled["rmse"], _pooled["r2"],
            )
        except Exception as exc:
            _log_event(
                "training.cv.oof_regression_recompute_failed",
                model_type=model_type_norm,
                reason=str(exc),
            )

    # ── Bootstrap confidence intervals on pooled OOF predictions ──────────────
    # Mirrors the holdout path so the frontend reads the same shape under
    # metrics_json["confidence_intervals"] regardless of split method.
    bootstrap_cis_cv: Optional[Dict[str, Any]] = None
    _ci_y_true: Optional[np.ndarray] = None
    _ci_y_pred: Optional[np.ndarray] = None
    _ci_y_score: Optional[np.ndarray] = None
    if cfg.task_type == "regression" and oof_y_true_reg and oof_y_pred_reg:
        try:
            _ci_y_true = np.concatenate(oof_y_true_reg)
            _ci_y_pred = np.concatenate(oof_y_pred_reg)
        except Exception:
            _ci_y_true = None
            _ci_y_pred = None
    elif cfg.task_type == "classification" and oof_true_parts and oof_y_pred_class_parts:
        try:
            _ci_y_true = np.concatenate([np.asarray(a) for a in oof_true_parts])
            _ci_y_pred = np.concatenate([np.asarray(a) for a in oof_y_pred_class_parts])
        except Exception:
            _ci_y_true = None
            _ci_y_pred = None
        if oof_proba_parts:
            try:
                _proba_pooled = np.vstack(oof_proba_parts)
                if _proba_pooled.ndim == 2 and _proba_pooled.shape[1] == 2:
                    _ci_y_score = _proba_pooled[:, 1]
                elif _proba_pooled.ndim == 1:
                    _ci_y_score = _proba_pooled
            except Exception:
                _ci_y_score = None

    if _ci_y_true is not None and _ci_y_pred is not None:
        _n_ci = int(len(_ci_y_true))
        if _n_ci < 30:
            logger.warning(
                "Skipping bootstrap CIs in CV mode — fewer than 30 OOF samples (%d)",
                _n_ci,
            )
        else:
            try:
                bootstrap_cis_cv = compute_bootstrap_cis(
                    _ci_y_true, _ci_y_pred,
                    y_score=_ci_y_score,
                    task_type=cfg.task_type,
                )
            except Exception as exc:
                _log_event(
                    "training.cv.bootstrap_ci_failed",
                    model_type=model_type_norm,
                    reason=str(exc),
                )

    _log_event(
        "training.cv.end",
        model_type=model_type_norm,
        k=actual_k,
        n_ok=n_ok,
        cv_mean_metrics=cv_summary.get("mean", {}),
    )

    # ── Final refit ───────────────────────────────────────────────────────────
    # • test_ratio == 0 → refit on X_all (all data, same as before).
    # • test_ratio >  0 → refit on X_cv (non-test data only) to keep the
    #   holdout test set genuinely independent for the final evaluation.
    # GridSearch is applied here if cfg.use_grid_search=True (not per-fold).
    X_refit = X_cv
    y_refit = y_cv
    _log_event(
        "training.cv.refit_start",
        model_type=model_type_norm,
        n_samples=int(len(y_refit)),
        has_holdout_test=has_holdout_test,
    )

    final_training_schema = build_training_schema(
        X=X_refit,
        target=cfg.target_column,
        preprocessing_config=effective_preprocessing.as_dict(),
    )
    final_aligner = ColumnAligner(
        feature_names=final_training_schema["feature_names"],
        dtypes=final_training_schema["dtypes"],
    )
    X_refit_aligned = final_aligner.fit_transform(X_refit)
    # Feature engineering: fit on full refit data (anti-leakage: test set transforms via pipeline)
    final_fe = FeatureEngineeringTransformer(fe_defs)
    X_refit_fe = final_fe.fit_transform(X_refit_aligned)
    final_spec = build_preprocessor(X_refit_fe, effective_preprocessing, kind_overrides=kind_overrides)
    clear_clip_warnings()
    X_refit_prep = final_spec.preprocessor.fit_transform(X_refit_fe, y_refit)
    _prep_clip_warnings = get_clip_warnings()
    clear_clip_warnings()

    # VarianceThreshold after StandardScaler: threshold=0.0 removes only
    # perfectly constant features. Higher values (e.g. 0.01) would
    # incorrectly drop rare binary features (prevalence ~1%) whose
    # clinical relevance may be high.
    _vt_refit_w: list[str] = []
    if final_spec.feature_selector is not None:
        _vt_n_before = X_refit_prep.shape[1] if hasattr(X_refit_prep, "shape") else 0
        _vt_refit_names: list = []
        try:
            _vt_refit_names = list(final_spec.preprocessor.get_feature_names_out())
        except Exception:
            pass
        final_spec.feature_selector.fit(X_refit_prep)
        X_refit_prep = final_spec.feature_selector.transform(X_refit_prep)
        _vt_refit_w = _log_variance_threshold(
            model_type_norm, _vt_n_before, X_refit_prep.shape[1],
            cfg.preprocessing.variance_threshold,
            feature_selector=final_spec.feature_selector,
            feature_names_before=_vt_refit_names,
        )

    if model_type_norm in _DENSE_REQUIRED_MODELS:
        X_refit_prep = _ensure_dense_matrix(X_refit_prep)

    # Balancing for final refit
    final_decision: BalancingDecision = _default_decision_for_non_classification()
    final_profile: Optional[DataProfile] = None
    balancing_audit: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        final_profile = profile_binary_dataset(y_refit, X_refit.shape)
        final_decision = resolve(
            profile=final_profile,
            config=cfg.balancing,
            model_supports_class_weight=capabilities["supports_class_weight"],
            model_supports_predict_proba=capabilities["supports_predict_proba"],
            model_supports_sample_weight=capabilities["supports_sample_weight"],
            random_state=cfg.random_state,
        )

    final_executor = BalancingExecutor()
    X_final_f = X_refit_prep
    y_final_f = np.asarray(y_refit)
    final_fit_params: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        if final_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
            if hasattr(X_refit_prep, "toarray"):
                X_refit_prep = X_refit_prep.toarray()
                X_final_f = X_refit_prep
        X_final_f, y_final_f, final_fit_params = final_executor.apply_prefit(
            X_refit_prep, y_refit, final_decision
        )

    final_hp = dict(base_estimator_hp)
    if "class_weight" in final_fit_params:
        final_hp["class_weight"] = final_fit_params["class_weight"]

    # Hyperparams for final refit: re-normalize WITH GridSearch flag to get param_grid
    hp_normalized_final = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=bool(cfg.use_grid_search),
        task_type=cfg.task_type,
    )
    final_param_grid = dict(hp_normalized_final.get("param_grid") or {})
    final_estimator_hp = dict(hp_normalized_final.get("estimator_params") or {})
    if "class_weight" in final_fit_params:
        final_estimator_hp["class_weight"] = final_fit_params["class_weight"]

    _final_active_balancing = final_decision.strategy not in {"none", "threshold_optimization"}
    _final_is_imbalanced = (
        cfg.task_type == "classification"
        and final_profile is not None
        and final_profile.imbalance_ratio is not None
        and final_profile.imbalance_ratio > 3.0
        and not _final_active_balancing
    )
    final_model_raw = build_model(model_type_norm, cfg.task_type, final_estimator_hp)
    final_trainer = Trainer()
    sw_final = final_fit_params.get("sample_weight")
    final_fit_result = final_trainer.fit(
        pipeline=Pipeline(steps=[("model", final_model_raw)]),
        X_train=X_final_f,
        y_train=y_final_f,
        fit_sample_weight=np.asarray(sw_final) if sw_final is not None else None,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=final_param_grid if final_param_grid else None,
        refit_metric_override=(final_decision.refit_metric if cfg.task_type == "classification" else None),
        n_samples=int(len(y_final_f)),
        imbalanced=_final_is_imbalanced,
    )

    final_fitted_model = final_fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        final_aligner, final_spec.preprocessor, final_fitted_model, model_type_norm,
        feature_selector=final_spec.feature_selector,
        fe_transformer=final_fe,
    )

    # Threshold + audit — prefer OOF predictions for leakage-free calibration.
    optimal_threshold = 0.5
    threshold_f1_gain: Optional[float] = None
    threshold_calibration_source = "not_applicable"
    _oof_metrics_at_threshold: Optional[Dict[str, Any]] = None  # populated below for classification
    if cfg.task_type == "classification":
        # Threshold optimization is binary-only: disable it for multiclass with explicit warning.
        _cv_multiclass_threshold_disabled = False
        if final_decision.apply_threshold and len(np.unique(np.asarray(y_refit))) > 2:
            final_decision.apply_threshold = False
            _cv_multiclass_threshold_disabled = True

        # Build OOF arrays from per-fold predictions accumulated above.
        oof_proba_arr: Optional[np.ndarray] = None
        oof_true_arr: Optional[np.ndarray] = None
        if oof_proba_parts and oof_true_parts:
            try:
                oof_proba_arr = np.vstack(oof_proba_parts)
                oof_true_arr = np.concatenate(oof_true_parts)
            except Exception:
                oof_proba_arr = None
                oof_true_arr = None

        oof_available = (
            oof_proba_arr is not None
            and oof_true_arr is not None
            and oof_proba_arr.shape[0] > 0
        )
        if oof_available:
            optimal_threshold = final_executor.apply_postfit_from_oof(
                y_true=oof_true_arr,
                y_proba=oof_proba_arr,
                decision=final_decision,
                model_classes=getattr(final_fitted_model, "classes_", None),
                positive_label=resolved_positive_label,
            )
            threshold_calibration_source = "oof"
        else:
            optimal_threshold = final_executor.apply_postfit(
                fitted_pipe, X_refit, y_refit, final_decision,
                positive_label=resolved_positive_label,
            )
            threshold_calibration_source = "train_refit"
            final_executor.postfit_warnings.append(
                "threshold_calibrated_on_train_data_may_be_optimistic"
            )
        if final_executor.last_threshold_result is not None:
            threshold_f1_gain = float(final_executor.last_threshold_result.improvement_delta)
        if _cv_multiclass_threshold_disabled:
            final_executor.postfit_warnings.append("threshold_optimization_disabled_multiclass")
        if final_profile is not None:
            balancing_audit = build_and_persist_audit(
                profile=final_profile,
                decision=final_decision,
                session_id=session_id,
                db=db,
                smote_samples_added=(
                    int(len(y_final_f) - len(y_refit))
                    if final_decision.strategy in {"smote", "smote_tomek"}
                    else None
                ),
                optimal_threshold=optimal_threshold,
                threshold_f1_gain=threshold_f1_gain,
                postfit_warnings=list(final_executor.postfit_warnings or []),
            )

        # ── OOF metrics at the deployed threshold (Bug 4 fix) ─────────────────
        # cv_summary fold metrics are computed at threshold=0.5 (the threshold
        # is unknown during the fold loop). When optimal_threshold ≠ 0.5, we
        # re-evaluate on the pooled OOF probabilities at the deployed threshold
        # so users can compare CV summary against the model they will actually
        # use. Stored as cv_mean_at_threshold in metrics_json.
        _oof_metrics_at_threshold: Optional[Dict[str, Any]] = None
        if optimal_threshold != 0.5 and oof_proba_arr is not None and oof_true_arr is not None:
            try:
                _model_cls_thr = getattr(final_fitted_model, "classes_", None)
                _pos_col_thr = 1
                if resolved_positive_label is not None and _model_cls_thr is not None and len(_model_cls_thr) == 2:
                    _pos_str_thr = str(resolved_positive_label)
                    for _ii, _cc in enumerate(_model_cls_thr):
                        if str(_cc) == _pos_str_thr:
                            _pos_col_thr = _ii
                            break

                _oof_proba_col = (
                    oof_proba_arr[:, _pos_col_thr]
                    if oof_proba_arr.ndim == 2 and oof_proba_arr.shape[1] >= 2
                    else oof_proba_arr.ravel()
                )
                if _model_cls_thr is not None and len(_model_cls_thr) == 2:
                    _pos_cls_thr = _model_cls_thr[_pos_col_thr]
                    _neg_cls_thr = _model_cls_thr[1 - _pos_col_thr]
                    _oof_y_pred_thr = np.where(_oof_proba_col >= optimal_threshold, _pos_cls_thr, _neg_cls_thr)
                else:
                    _oof_y_pred_thr = (_oof_proba_col >= optimal_threshold).astype(int)

                from app.services.training.pipeline.metrics import classification_metrics as _cm
                _oof_metrics_at_threshold = _extract_scalar_metrics(
                    _cm(
                        oof_true_arr, _oof_y_pred_thr,
                        y_proba=oof_proba_arr,
                        labels=_model_cls_thr,
                        positive_label=resolved_positive_label,
                        requested_metrics=list(cfg.metrics),
                        task_type=cfg.task_type,
                    )
                )
            except Exception as _thr_exc:
                _log_event("training.cv.oof_threshold_metrics_failed", model_type=model_type_norm, reason=str(_thr_exc))

    _log_event("training.cv.refit_end", model_type=model_type_norm)

    # ── Holdout test evaluation ───────────────────────────────────────────────
    # Transform test set with preprocessor fitted on X_refit (no leakage).
    # Never resample the test set.
    holdout_test_metrics: Optional[Dict[str, Any]] = None
    holdout_test_eval = None
    if has_holdout_test and X_test_holdout is not None and y_test_holdout is not None:
        holdout_evaluator = Evaluator(
            task_type=cfg.task_type,
            requested_metrics=cfg.metrics,
            positive_label=resolved_positive_label,
        )
        holdout_test_eval = holdout_evaluator.evaluate(
            fitted_pipe,
            X_test_holdout,
            np.asarray(y_test_holdout),
            threshold=optimal_threshold,
        )
        holdout_test_metrics = holdout_test_eval.metrics
        _log_event(
            "training.cv.holdout_test_eval",
            model_type=model_type_norm,
            n_test=int(len(y_test_holdout)),
        )

    # ── Build CV-specific artifacts ───────────────────────────────────────────
    _cv_curves = (
        holdout_test_eval.metrics.get("curves")
        if holdout_test_eval is not None and isinstance(holdout_test_eval.metrics, dict)
        else None
    )

    reporter = Reporter()
    artifacts = reporter.build_cv_artifacts(
        cfg=cfg,
        fold_results=fold_results,
        cv_summary=cv_summary,
        actual_k=actual_k,
        n_folds_ok=n_ok,
        X_all=X_all,
        y_all=y_all,
        X_cv=X_cv,
        y_cv=y_cv,
        X_test_holdout=X_test_holdout,
        y_test_holdout=y_test_holdout,
        final_spec=final_spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        final_training_schema=final_training_schema,
        tuning_artifacts=final_fit_result.tuning_artifacts,
        resolved_positive_label=resolved_positive_label,
        curves=_cv_curves,
    )

    hyperparams_artifacts: Dict[str, Any] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized_final.get("effective") or {}),
        "note": (
            f"Nested CV active: inner GridSearch ({inner_k} folds, scoring={inner_scoring}) "
            "run per outer fold; GridSearch also applied on final refit."
            if use_nested_cv
            else "GridSearch (if any) applied only on final refit, not per-fold."
        ),
    }
    if bool(cfg.use_grid_search):
        hyperparams_artifacts["param_grid"] = dict(final_param_grid)
        best_params = final_fit_result.tuning_artifacts.get("best_params")
        if isinstance(best_params, dict):
            hyperparams_artifacts["best"] = dict(best_params)
    artifacts["hyperparams"] = hyperparams_artifacts

    # ── Build metrics_json ────────────────────────────────────────────────────
    # "test" key: holdout test metrics when available (honest external score),
    # otherwise CV mean metrics (backward compat for route/frontend).
    # "cv_mean" always holds the aggregated CV validation metrics.
    cv_mean_metrics = cv_summary.get("mean", {})
    # Bug 9: "test" key has dual semantics (holdout OR cv_mean). Preserve it for
    # backward compat but add an explicit "test_meta" dict so callers cannot
    # silently misinterpret CV metrics as an independent holdout evaluation.
    _test_source = "holdout" if (has_holdout_test and holdout_test_metrics is not None) else "cv_mean"
    metrics_json: Dict[str, Any] = {
        "split_method": cfg.split_method,
        "cv": True,
        "nested_cv": use_nested_cv,
        "k_folds": actual_k,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "has_holdout_test": has_holdout_test,
        "cv_mean": cv_mean_metrics,
        "test": holdout_test_metrics if (has_holdout_test and holdout_test_metrics is not None) else cv_mean_metrics,
        "test_is_cv_mean": not has_holdout_test,
        "test_label": "CV validation (moyenne des folds)" if not has_holdout_test else "Holdout test set",
        "test_meta": {
            "source": _test_source,
            "is_independent_holdout": has_holdout_test and holdout_test_metrics is not None,
            "interpretation": (
                "Independent holdout set — honest generalisation estimate."
                if _test_source == "holdout"
                else "CV mean across folds (threshold=0.5). "
                     "Use cv_mean_at_threshold (when present) for performance at deployed threshold. "
                     "NOT equivalent to an independent test set."
            ),
        },
        "training_time_sec": float(time.perf_counter() - t0),
    }
    if has_holdout_test and holdout_test_metrics is not None:
        metrics_json["holdout_test_metrics"] = holdout_test_metrics
    metrics_json["threshold_used"] = optimal_threshold
    metrics_json["threshold_source"] = threshold_calibration_source
    metrics_json["confidence_intervals"] = bootstrap_cis_cv
    # Bug 4: expose OOF metrics recomputed at the deployed threshold so users
    # can compare apples-to-apples (cv_mean uses 0.5; this uses optimal_threshold).
    if _oof_metrics_at_threshold is not None:
        metrics_json["cv_mean_at_threshold"] = _oof_metrics_at_threshold
        metrics_json["threshold_adjusted_note"] = (
            f"cv_mean was computed at threshold=0.5 during fold evaluation. "
            f"cv_mean_at_threshold applies the deployed threshold={optimal_threshold:.3f} "
            "to pooled OOF predictions for a like-for-like comparison."
        )
    # Bug 5: surface the OOF calibration limitation so users understand the
    # threshold may not be perfectly tuned for the refit-on-full-data model.
    if threshold_calibration_source == "oof":
        metrics_json["threshold_oof_note"] = (
            "Threshold was calibrated on out-of-fold predictions (fold-specific models). "
            "The deployed model (refit on all CV data) may have slightly different probability "
            "calibration — the optimal threshold could differ marginally."
        )
    cv_instability = cv_summary.get("instability_warnings", [])
    postfit_w = list(getattr(final_executor, "postfit_warnings", None) or [])
    _clip_strs = [
        f"CLIP_NEGATIVE in '{w['column']}': {w['n_clipped']} value(s) clipped before "
        f"{w['transform']} transform (min={w['min_observed']:.6g})"
        for w in _prep_clip_warnings
    ]
    _fold_structured, _fold_strs = _collect_fold_warnings(fold_results)
    all_warnings: list[str] = (
        list(hp_warnings) + list(cv_instability) + postfit_w + _clip_strs + _vt_refit_w + _fold_strs
    )
    if all_warnings:
        metrics_json["warnings"] = all_warnings
    if _prep_clip_warnings:
        metrics_json["clip_warnings"] = [
            {"severity": "warning", "code": "CLIP_NEGATIVE", **w} for w in _prep_clip_warnings
        ]
    if _fold_structured:
        metrics_json["fold_warnings"] = _fold_structured

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )


def _run_loo(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    model_type: str,
    *,
    db: Session | None = None,
    session_id: int | None = None,
    kind_overrides: dict[str, str] | None = None,
) -> ModelRunResult:
    """
    Leave-One-Out evaluation pipeline.

    Anti-leakage guarantees
    -----------------------
    • Preprocessor is fit on n-1 samples (train_fold) ONLY, transformed on 1 test sample.
    • Resampling applied to train_fold ONLY; SMOTE guard disables resampling when
      the minority class is too small for k_neighbors.
    • GridSearch is DISABLED per-fold (would create n × k_gs nested fits).

    Aggregation — prediction collection
    ------------------------------------
    Because each fold has 1 test sample, per-fold metrics (AUC, F1) are meaningless.
    Instead: collect y_true and y_pred across ALL n folds, then compute global
    metrics once. This is the statistically correct LOO estimate.

    Persistence (Option A — refit-final)
    -------------------------------------
    After n folds, refit on ALL data (same as kfold). This is the deployed model.
    Guard: refuses if n_samples > 500.
    """
    from app.services.training.pipeline.metrics import classification_metrics, regression_metrics

    t0 = time.perf_counter()
    model_type_norm = str(model_type or "").strip().lower()
    _log_event("training.loo.start", model_type=model_type_norm, task_type=cfg.task_type)

    if cfg.target_column not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_column}' not found in dataset")

    df2 = df[df[cfg.target_column].notna()].copy()
    n_samples = len(df2)
    if n_samples < 3:
        raise RuntimeError("LOO nécessite au moins 3 échantillons après suppression des NaN cibles.")
    if n_samples > 500:
        raise RuntimeError(
            f"LOO est limité à 500 échantillons (n={n_samples}). "
            "Utilisez kfold ou stratified_kfold à la place."
        )

    X_all = df2.drop(columns=[cfg.target_column])
    y_all = np.asarray(df2[cfg.target_column].values)

    fe_defs = [f.as_dict() for f in cfg.feature_engineering.features]
    fe_validation_errors = validate_feature_defs(fe_defs, list(X_all.columns))
    if fe_validation_errors:
        raise RuntimeError("Feature engineering config errors:\n" + "\n".join(fe_validation_errors))

    resolved_positive_label, positive_label_warning = _resolve_positive_label_for_run(
        y_all, cfg.positive_label
    )
    if positive_label_warning:
        _log_event("training.positive_label.warning", model_type=model_type_norm, message=positive_label_warning)

    # Normalize hyperparams — GridSearch disabled per fold
    requested_hyperparams_raw = cfg.model_hyperparams.get(model_type_norm, {})
    if not isinstance(requested_hyperparams_raw, dict):
        requested_hyperparams_raw = {}
    hp_normalized_no_gs = normalize_model_hyperparams(
        model_type_norm,
        requested_hyperparams_raw,
        use_grid_search=False,
        task_type=cfg.task_type,
    )
    hp_errors = [str(m) for m in (hp_normalized_no_gs.get("errors") or []) if str(m).strip()]
    if hp_errors:
        raise RuntimeError(f"Invalid hyperparameters for '{model_type_norm}': {' | '.join(hp_errors)}")
    hp_warnings: list[str] = [str(w) for w in (hp_normalized_no_gs.get("warnings") or []) if str(w).strip()]
    for hp_warn in hp_warnings:
        _log_event("training.hyperparams.warning", model_type=model_type_norm, message=hp_warn)

    base_estimator_hp = dict(hp_normalized_no_gs.get("estimator_params") or {})
    capabilities = get_model_capabilities(model_type_norm)
    effective_preprocessing = _ensure_scaling_for_model(model_type_norm, cfg.preprocessing)

    # ── LOO prediction collection ─────────────────────────────────────────────
    y_true_loo: list[Any] = []
    y_pred_loo: list[Any] = []
    y_proba_loo: list[Any] = []
    fold_results: List[Dict[str, Any]] = []
    n_ok = 0
    # Track the proba column index for the positive class. Captured once from
    # the first successful fold's model.classes_ so y_score_loo is oriented
    # toward resolved_positive_label instead of always defaulting to index 1.
    _loo_pos_col: int = 1
    _loo_pos_col_set: bool = False

    try:
        loo_splits = list(iter_loo_splits(X_all, y_all))
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    actual_n = len(loo_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(loo_splits):
        fold_num = fold_idx + 1
        try:
            X_train_fold = X_all.iloc[train_idx].copy()
            X_val_fold = X_all.iloc[val_idx].copy()
            y_train_fold = y_all[train_idx]
            y_val_sample = y_all[val_idx]

            # ① Fit preprocessor on train_fold ONLY (n-1 samples)
            fold_schema = build_training_schema(
                X=X_train_fold,
                target=cfg.target_column,
                preprocessing_config=effective_preprocessing.as_dict(),
            )
            fold_aligner = ColumnAligner(
                feature_names=fold_schema["feature_names"],
                dtypes=fold_schema["dtypes"],
            )
            X_train_aligned = fold_aligner.fit_transform(X_train_fold)
            # Feature engineering: fit stats on train_fold only (anti-leakage)
            fold_fe = FeatureEngineeringTransformer(fe_defs)
            X_train_fe = fold_fe.fit_transform(X_train_aligned)
            fold_spec = build_preprocessor(X_train_fe, effective_preprocessing, kind_overrides=kind_overrides)
            X_train_prep = fold_spec.preprocessor.fit_transform(X_train_fe, y_train_fold)
            X_val_aligned = fold_aligner.transform(X_val_fold)
            X_val_fe = fold_fe.transform(X_val_aligned)
            X_val_prep = fold_spec.preprocessor.transform(X_val_fe)

            # VarianceThreshold after StandardScaler: threshold=0.0 removes only
            # perfectly constant features. Higher values (e.g. 0.01) would
            # incorrectly drop rare binary features (prevalence ~1%) whose
            # clinical relevance may be high.
            _vt_loo_fold_w: list[str] = []
            if fold_spec.feature_selector is not None:
                _vt_n_before = X_train_prep.shape[1] if hasattr(X_train_prep, "shape") else 0
                _vt_loo_fold_names: list = []
                try:
                    _vt_loo_fold_names = list(fold_spec.preprocessor.get_feature_names_out())
                except Exception:
                    pass
                fold_spec.feature_selector.fit(X_train_prep)
                X_train_prep = fold_spec.feature_selector.transform(X_train_prep)
                X_val_prep = fold_spec.feature_selector.transform(X_val_prep)
                _vt_loo_fold_w = _log_variance_threshold(
                    model_type_norm, _vt_n_before, X_train_prep.shape[1],
                    cfg.preprocessing.variance_threshold,
                    feature_selector=fold_spec.feature_selector,
                    feature_names_before=_vt_loo_fold_names,
                )

            if model_type_norm in _DENSE_REQUIRED_MODELS:
                X_train_prep = _ensure_dense_matrix(X_train_prep)
                X_val_prep = _ensure_dense_matrix(X_val_prep)

            # ② Balancing — train_fold ONLY; SMOTE guard for small folds
            fold_decision: BalancingDecision = _default_decision_for_non_classification()
            fold_executor = BalancingExecutor()
            X_f = X_train_prep
            y_f = np.asarray(y_train_fold)
            fold_fit_params: Dict[str, Any] = {}

            fold_warnings: list[str] = list(_vt_loo_fold_w)
            if cfg.task_type == "classification":
                fold_profile = profile_binary_dataset(y_train_fold, X_train_fold.shape)
                fold_decision = resolve(
                    profile=fold_profile,
                    config=cfg.balancing,
                    model_supports_class_weight=capabilities["supports_class_weight"],
                    model_supports_predict_proba=capabilities["supports_predict_proba"],
                    model_supports_sample_weight=capabilities["supports_sample_weight"],
                    random_state=cfg.random_state,
                )
                # SMOTE guard: shared with the K-Fold path so behaviour is
                # identical when the minority class is too small for SMOTE.
                fold_decision, _smote_warn = _smote_minority_guard(fold_decision, y_train_fold)
                if _smote_warn is not None:
                    fold_warnings.append(_smote_warn)
                    _log_event(
                        "training.loo.fold_smote_skipped",
                        fold=fold_num,
                        model_type=model_type_norm,
                        reason=_smote_warn,
                    )
                if fold_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
                    if hasattr(X_train_prep, "toarray"):
                        X_train_prep = X_train_prep.toarray()
                    if hasattr(X_val_prep, "toarray"):
                        X_val_prep = X_val_prep.toarray()
                X_f, y_f, fold_fit_params = fold_executor.apply_prefit(
                    X_train_prep, np.asarray(y_train_fold), fold_decision
                )

            fold_hp = dict(base_estimator_hp)
            if "class_weight" in fold_fit_params:
                fold_hp["class_weight"] = fold_fit_params["class_weight"]

            # ③ Train model — no GridSearch per fold
            fold_model = build_model(model_type_norm, cfg.task_type, fold_hp)
            fold_pipeline = Pipeline(steps=[("model", fold_model)])
            sw = fold_fit_params.get("sample_weight")
            if sw is not None:
                fold_pipeline.fit(X_f, y_f, model__sample_weight=np.asarray(sw))
            else:
                fold_pipeline.fit(X_f, y_f)

            # ④ Predict on the single val sample — collect predictions
            y_pred_sample = fold_pipeline.predict(X_val_prep)
            y_true_loo.append(y_val_sample[0])
            y_pred_loo.append(y_pred_sample[0])
            if hasattr(fold_pipeline, "predict_proba"):
                proba = fold_pipeline.predict_proba(X_val_prep)
                y_proba_loo.append(proba[0])
                # Capture the positive-class column index from classes_ once.
                # All LOO folds use the same label encoding, so one capture suffices.
                if not _loo_pos_col_set and resolved_positive_label is not None:
                    _fold_classes = getattr(fold_pipeline, "classes_", None)
                    if _fold_classes is not None and len(_fold_classes) == 2:
                        pos_str = str(resolved_positive_label)
                        for _i, _cls in enumerate(_fold_classes):
                            if str(_cls) == pos_str:
                                _loo_pos_col = _i
                                _loo_pos_col_set = True
                                break

            fold_results.append({
                "fold": fold_num,
                "status": "ok",
                "train_size": int(len(y_train_fold)),
                "val_size": 1,
                "warnings": list(fold_warnings),
            })
            n_ok += 1

        except Exception as fold_exc:
            _log_event("training.loo.fold_error", fold=fold_num, model_type=model_type_norm, reason=str(fold_exc))
            fold_results.append({"fold": fold_num, "status": "failed", "error": str(fold_exc)})

    if n_ok == 0:
        errors_summary = "; ".join(
            fr.get("error", "?") for fr in fold_results if fr.get("status") == "failed"
        )
        raise RuntimeError(f"All {actual_n} LOO folds failed for '{model_type_norm}'. Errors: {errors_summary}")

    # ── Global LOO metrics from collected predictions ─────────────────────────
    y_true_arr = np.asarray(y_true_loo)
    y_pred_arr = np.asarray(y_pred_loo)
    y_proba_arr = np.asarray(y_proba_loo) if y_proba_loo else None

    # Initialise outside the if/else so it remains in scope for the
    # bootstrap CI block below regardless of task type.
    y_score_loo: Optional[np.ndarray] = None
    if cfg.task_type == "classification":
        _labels = np.unique(y_all)

        if y_proba_arr is not None:
            if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == 2:
                # Use the tracked positive-class column, not hardcoded index 1.
                y_score_loo = y_proba_arr[:, _loo_pos_col]
            else:
                y_score_loo = y_proba_arr

        loo_metrics = classification_metrics(
            y_true_arr, y_pred_arr,
            y_proba=y_proba_arr,
            y_score=y_score_loo,
            labels=_labels,
            estimator=None,
            positive_label=resolved_positive_label,
            requested_metrics=list(cfg.metrics),
            task_type=cfg.task_type,
        )
    else:
        loo_metrics = regression_metrics(y_true_arr, y_pred_arr)

    # ── Bootstrap confidence intervals on pooled LOO predictions ──────────────
    bootstrap_cis_loo: Optional[Dict[str, Any]] = None
    _n_loo_ci = int(len(y_true_arr))
    if _n_loo_ci < 30:
        logger.warning(
            "Skipping bootstrap CIs in CV mode — fewer than 30 OOF samples (%d)",
            _n_loo_ci,
        )
    else:
        try:
            bootstrap_cis_loo = compute_bootstrap_cis(
                y_true_arr, y_pred_arr,
                y_score=(y_score_loo if cfg.task_type == "classification" else None),
                task_type=cfg.task_type,
            )
        except Exception as exc:
            _log_event(
                "training.loo.bootstrap_ci_failed",
                model_type=model_type_norm,
                reason=str(exc),
            )

    # cv_summary-compatible structure for reporter reuse
    loo_flat = _extract_scalar_metrics(loo_metrics)
    cv_summary = {
        "mean": loo_flat,
        "std": {},
        "min": loo_flat,
        "max": loo_flat,
        "n_folds_ok": n_ok,
        "n_folds_per_metric": {k: n_ok for k in loo_flat},
    }

    _log_event(
        "training.loo.end",
        model_type=model_type_norm,
        n_iterations=actual_n,
        n_ok=n_ok,
        loo_metrics=loo_flat,
    )

    # ── Final refit on ALL data (Option A — same as kfold) ────────────────────
    _log_event("training.loo.refit_start", model_type=model_type_norm, n_samples=n_samples)

    final_training_schema = build_training_schema(
        X=X_all, target=cfg.target_column,
        preprocessing_config=effective_preprocessing.as_dict(),
    )
    final_aligner = ColumnAligner(
        feature_names=final_training_schema["feature_names"],
        dtypes=final_training_schema["dtypes"],
    )
    X_refit_aligned = final_aligner.fit_transform(X_all)
    final_fe = FeatureEngineeringTransformer(fe_defs)
    X_refit_fe = final_fe.fit_transform(X_refit_aligned)
    final_spec = build_preprocessor(X_refit_fe, effective_preprocessing, kind_overrides=kind_overrides)
    clear_clip_warnings()
    X_refit_prep = final_spec.preprocessor.fit_transform(X_refit_fe, y_all)
    _prep_clip_warnings = get_clip_warnings()
    clear_clip_warnings()
    # VarianceThreshold after StandardScaler: threshold=0.0 removes only
    # perfectly constant features. Higher values (e.g. 0.01) would
    # incorrectly drop rare binary features (prevalence ~1%) whose
    # clinical relevance may be high.
    _vt_loo_refit_w: list[str] = []
    if final_spec.feature_selector is not None:
        _vt_n_before = X_refit_prep.shape[1] if hasattr(X_refit_prep, "shape") else 0
        _vt_loo_refit_names: list = []
        try:
            _vt_loo_refit_names = list(final_spec.preprocessor.get_feature_names_out())
        except Exception:
            pass
        final_spec.feature_selector.fit(X_refit_prep)
        X_refit_prep = final_spec.feature_selector.transform(X_refit_prep)
        _vt_loo_refit_w = _log_variance_threshold(
            model_type_norm, _vt_n_before, X_refit_prep.shape[1],
            cfg.preprocessing.variance_threshold,
            feature_selector=final_spec.feature_selector,
            feature_names_before=_vt_loo_refit_names,
        )
    if model_type_norm in _DENSE_REQUIRED_MODELS:
        X_refit_prep = _ensure_dense_matrix(X_refit_prep)

    final_decision: BalancingDecision = _default_decision_for_non_classification()
    final_profile = None
    balancing_audit: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        final_profile = profile_binary_dataset(y_all, X_all.shape)
        final_decision = resolve(
            profile=final_profile,
            config=cfg.balancing,
            model_supports_class_weight=capabilities["supports_class_weight"],
            model_supports_predict_proba=capabilities["supports_predict_proba"],
            model_supports_sample_weight=capabilities["supports_sample_weight"],
            random_state=cfg.random_state,
        )

    final_executor = BalancingExecutor()
    X_final_f = X_refit_prep
    y_final_f = np.asarray(y_all)
    final_fit_params: Dict[str, Any] = {}
    if cfg.task_type == "classification":
        if final_decision.strategy in {"smote", "smote_tomek", "random_undersampling"}:
            if hasattr(X_refit_prep, "toarray"):
                X_refit_prep = X_refit_prep.toarray()
                X_final_f = X_refit_prep
        X_final_f, y_final_f, final_fit_params = final_executor.apply_prefit(
            X_refit_prep, y_all, final_decision
        )

    final_hp = dict(base_estimator_hp)
    if "class_weight" in final_fit_params:
        final_hp["class_weight"] = final_fit_params["class_weight"]

    hp_normalized_final = normalize_model_hyperparams(
        model_type_norm, requested_hyperparams_raw,
        use_grid_search=bool(cfg.use_grid_search), task_type=cfg.task_type,
    )
    final_param_grid = dict(hp_normalized_final.get("param_grid") or {})
    final_estimator_hp = dict(hp_normalized_final.get("estimator_params") or {})
    if "class_weight" in final_fit_params:
        final_estimator_hp["class_weight"] = final_fit_params["class_weight"]

    final_model_raw = build_model(model_type_norm, cfg.task_type, final_estimator_hp)
    final_trainer = Trainer()
    sw_final = final_fit_params.get("sample_weight")
    final_fit_result = final_trainer.fit(
        pipeline=Pipeline(steps=[("model", final_model_raw)]),
        X_train=X_final_f,
        y_train=y_final_f,
        fit_sample_weight=np.asarray(sw_final) if sw_final is not None else None,
        cfg=cfg,
        model_type=model_type_norm,
        task_type=cfg.task_type,
        model_param_grid=final_param_grid if final_param_grid else None,
        refit_metric_override=(final_decision.refit_metric if cfg.task_type == "classification" else None),
        n_samples=int(len(y_final_f)),
        imbalanced=False,
    )

    final_fitted_model = final_fit_result.fitted_pipeline.named_steps.get("model")
    fitted_pipe = _build_inference_pipeline(
        final_aligner, final_spec.preprocessor, final_fitted_model, model_type_norm,
        feature_selector=final_spec.feature_selector,
        fe_transformer=final_fe,
    )

    optimal_threshold = 0.5
    threshold_f1_gain: Optional[float] = None
    threshold_calibration_source = "disabled"
    if cfg.task_type == "classification":
        if final_decision.apply_threshold and len(np.unique(y_all)) > 2:
            final_decision.apply_threshold = False

        # Prefer leakage-free OOF threshold calibration: y_true_loo and
        # y_proba_loo were collected per fold and contain only out-of-sample
        # predictions — the closest analogue of a validation set available in
        # the LOO regime. Fall back to (X_all, y_all) only when those arrays
        # are unavailable, and surface a warning when the fallback fires
        # (mirrors the holdout pattern at orchestrator.py:673).
        oof_proba_arr = np.asarray(y_proba_loo) if y_proba_loo else None
        oof_true_arr = np.asarray(y_true_loo) if y_true_loo else None
        oof_available = (
            final_decision.apply_threshold
            and oof_proba_arr is not None
            and oof_true_arr is not None
            and oof_true_arr.size > 0
            and oof_proba_arr.size > 0
        )

        if oof_available:
            optimal_threshold = final_executor.apply_postfit_from_oof(
                y_true=oof_true_arr,
                y_proba=oof_proba_arr,
                decision=final_decision,
                model_classes=getattr(final_fitted_model, "classes_", None),
                positive_label=resolved_positive_label,
            )
            threshold_calibration_source = "loo_oof"
        else:
            optimal_threshold = final_executor.apply_postfit(
                fitted_pipe, X_all, y_all, final_decision,
                positive_label=resolved_positive_label,
            )
            if final_decision.apply_threshold:
                threshold_calibration_source = "train_fallback"
                if optimal_threshold != 0.5:
                    final_executor.postfit_warnings.append(
                        "threshold_calibrated_on_train_data_may_be_optimistic"
                    )
        if final_executor.last_threshold_result is not None:
            threshold_f1_gain = float(final_executor.last_threshold_result.improvement_delta)
        if final_profile is not None:
            balancing_audit = build_and_persist_audit(
                profile=final_profile,
                decision=final_decision,
                session_id=session_id,
                db=db,
                smote_samples_added=(
                    int(len(y_final_f) - len(y_all))
                    if final_decision.strategy in {"smote", "smote_tomek"} else None
                ),
                optimal_threshold=optimal_threshold,
                threshold_f1_gain=threshold_f1_gain,
                postfit_warnings=list(final_executor.postfit_warnings or []),
            )

    _log_event(
        "training.loo.refit_end",
        model_type=model_type_norm,
        threshold_calibration_source=threshold_calibration_source,
    )

    # ── Artifacts and metrics_json ────────────────────────────────────────────
    reporter = Reporter()
    artifacts = reporter.build_cv_artifacts(
        cfg=cfg,
        fold_results=fold_results,
        cv_summary=cv_summary,
        actual_k=actual_n,
        n_folds_ok=n_ok,
        X_all=X_all,
        y_all=y_all,
        X_cv=X_all,
        y_cv=y_all,
        X_test_holdout=None,
        y_test_holdout=None,
        final_spec=final_spec,
        fitted_pipeline=fitted_pipe,
        balancing_audit=balancing_audit,
        final_training_schema=final_training_schema,
        tuning_artifacts=final_fit_result.tuning_artifacts,
        resolved_positive_label=resolved_positive_label,
        curves=None,
    )
    artifacts["hyperparams"] = {
        "requested": dict(requested_hyperparams_raw),
        "effective": dict(hp_normalized_final.get("effective") or {}),
        "note": "GridSearch disabled per LOO fold; applied only on final refit.",
    }

    metrics_json: Dict[str, Any] = {
        "split_method": "loo",
        "cv": True,
        "loo": True,
        "n_loo_iterations": actual_n,
        "n_loo_ok": n_ok,
        "loo_metrics": loo_metrics,
        "cv_mean": loo_flat,
        "test": loo_metrics,
        "test_is_cv_mean": True,
        "test_label": "LOO validation",
        "evaluation_strategy": "loo",
        "has_holdout_test": False,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "training_time_sec": float(time.perf_counter() - t0),
        "threshold_used": optimal_threshold,
        "threshold_source": threshold_calibration_source,
        "confidence_intervals": bootstrap_cis_loo,
    }
    loo_instability = cv_summary.get("instability_warnings", [])
    # Include postfit_warnings (e.g. threshold_calibrated_on_train_data_may_be_optimistic)
    # so the frontend can surface threshold-related issues alongside the others.
    loo_postfit_warnings = list(final_executor.postfit_warnings or [])
    _loo_clip_strs = [
        f"CLIP_NEGATIVE in '{w['column']}': {w['n_clipped']} value(s) clipped before "
        f"{w['transform']} transform (min={w['min_observed']:.6g})"
        for w in _prep_clip_warnings
    ]
    _loo_fold_structured, _loo_fold_strs = _collect_fold_warnings(fold_results)
    loo_all_warnings: list[str] = (
        list(hp_warnings) + list(loo_instability) + loo_postfit_warnings
        + _loo_clip_strs + _vt_loo_refit_w + _loo_fold_strs
    )
    if loo_all_warnings:
        metrics_json["warnings"] = loo_all_warnings
    if _prep_clip_warnings:
        metrics_json["clip_warnings"] = [
            {"severity": "warning", "code": "CLIP_NEGATIVE", **w} for w in _prep_clip_warnings
        ]
    if _loo_fold_structured:
        metrics_json["fold_warnings"] = _loo_fold_structured

    return ModelRunResult(
        model_type=model_type_norm,
        task_type=cfg.task_type,
        metrics_json=metrics_json,
        artifacts_json=artifacts,
        fitted_pipeline=fitted_pipe,
    )
