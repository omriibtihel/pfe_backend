from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core.config import PROJECTS_PATH
from app.models.training import TrainingSession

from app.services.training.config.schema import TrainingConfig
from app.services.data.loader import load_dataframe
from app.services.training.orchestrator import run_one_model
from app.crud import version_column_schema as crud_vcs
from app.services.training.data_source import resolve_training_data_path
from app.services.training.output.persistence import save_pipeline, persist_trained_model
from app.services.training.notifier import training_notifier, EventType
from app.services.training.intelligence.meta_learner import MetaLearner, build_training_record
from app.services.training.config.automl import AutoMLConfig
from app.services.training.automl_runner import run_automl

logger = logging.getLogger(__name__)

_meta_learner = MetaLearner(PROJECTS_PATH)

_LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({"rmse", "mae", "mse", "log_loss", "brier_score"})


def _now():
    return datetime.now(timezone.utc)


def _log_event(event: str, **payload: Any) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


def _update_session(db: Session, s: TrainingSession, **fields):
    for k, v in fields.items():
        setattr(s, k, v)
    db.add(s)
    db.commit()
    db.refresh(s)


def _rollback_quietly(db: Session) -> None:
    try:
        db.rollback()
    except Exception:
        pass


def _append_session_message(db: Session, s: TrainingSession, msg: str):
    current = (s.error_message or "").strip()
    msg = msg.strip()
    if not current:
        new_msg = msg
    else:
        new_msg = current if msg in current else (current + "\n" + msg)
    _update_session(db, s, error_message=new_msg)


def run_training_session(session_id: int) -> None:
    """
    Worker appelé par la route (background task).
    split->preprocess->smote(train only)->fit->eval->persist.
    Pour l’instant: orchestrator = HOLDOUT (kfold arrive juste après).
    """
    t_start = time.monotonic()
    db = SessionLocal()
    try:
        s: TrainingSession | None = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not s:
            return

        _log_event("training.session.start", session_id=session_id)
        training_notifier.emit_sync(session_id, EventType.STARTED, {"sessionId": session_id})
        _update_session(db, s, status="running", progress=3, started_at=_now())

        cfg_raw: Dict[str, Any] = s.config_json or {}
        cfg = TrainingConfig.from_front(cfg_raw)
        config_mode = str(cfg_raw.get("configMode", "manual"))

        if not cfg.target_column:
            raise RuntimeError("targetColumn is required")
        if not cfg.models:
            raise RuntimeError("No model selected")

        path_to_load, dv_id, _data_source = resolve_training_data_path(
            db, s.project_id, s.dataset_version_id)
        df = load_dataframe(path_to_load)

        if _data_source == "raw_original":
            _append_session_message(
                db, s,
                "WARNING: Training used the original unprocessed dataset. "
                "Apply and save cleaning operations before training to use cleaned data.",
            )

        out_dir = PROJECTS_PATH / str(s.project_id) / "training_models" / str(session_id)

        total = max(1, len(cfg.models))
        success_count = 0
        best_result: Any = None
        best_metric_name: str = "accuracy"
        best_score: float = -1.0

        _update_session(db, s, dataset_version_id=dv_id, progress=10, current_model=None)
        training_notifier.emit_sync(session_id, EventType.PROGRESS, {"progress": 10, "phase": "data_loaded"})

        # Fetch kind_overrides set on the cleaning page to inform preprocessing type decisions.
        kind_overrides: dict[str, str] | None = None
        if dv_id:
            schema_map = crud_vcs.get_map(db, int(dv_id))
            _raw = {col: row.override_kind for col, row in schema_map.items() if row.override_kind}
            kind_overrides = _raw or None

        for i, model_type in enumerate(cfg.models, start=1):
            # Progress range for this model: divide 10–95 equally among all models.
            progress_start = min(95, int(10 + (i - 1) * (85 / total)))
            progress_end   = min(95, int(10 + i * (85 / total)))
            current_label  = f"{model_type} ({i}/{total})"

            _update_session(db, s, progress=progress_start, current_model=current_label)
            training_notifier.emit_sync(
                session_id,
                EventType.PROGRESS,
                {"progress": progress_start, "currentModel": model_type, "index": i, "total": total},
            )
            _log_event("training.session.model.start", session_id=session_id, model_type=model_type, index=i, total=total)

            try:
                res = run_one_model(
                    df,
                    cfg,
                    model_type=model_type,
                    db=db,
                    session_id=s.id,
                    kind_overrides=kind_overrides,
                )

                # save artifact then persist to DB atomically:
                # if DB write fails, remove the orphaned pkl so disk stays clean.
                _threshold_to_embed = (
                    res.metrics_json.get("threshold_used")
                    if res.task_type == "classification"
                    else None
                )
                _balancing_arts = res.artifacts_json.get("balancing") or {}
                pkl_path = save_pipeline(
                    res.fitted_pipeline, out_dir, model_type,
                    threshold=_threshold_to_embed,
                    positive_class_index=_balancing_arts.get("positive_class_index"),
                    positive_label=_balancing_arts.get("positive_label"),
                )
                res.artifacts_json["model_pkl"] = str(pkl_path)
                res.artifacts_json["dataset_version_id"] = dv_id

                try:
                    persist_trained_model(
                        db,
                        session_id=session_id,
                        project_id=s.project_id,
                        model_type=str(model_type),
                        task_type=str(res.task_type),
                        metrics_json=res.metrics_json,
                        artifacts_json=res.artifacts_json,
                    )
                except Exception:
                    try:
                        pkl_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise

                success_count += 1

                # Track best result for meta-learning.
                # Skip train-only results: no real evaluation, must not be recorded as best.
                evaluation_strategy = res.metrics_json.get("evaluation_strategy", "")
                primary_metric_name, primary_score, _primary_source = _extract_primary_score(
                    res.metrics_json, cfg
                )
                if (
                    evaluation_strategy != "train_only"
                    and primary_metric_name is not None
                    and primary_score is not None
                ):
                    if _primary_source == "fallback":
                        _configured = next(iter(cfg.metrics or []), "(none)")
                        logger.warning(
                            "Model %s: primary score derived from fallback metric '%s', "
                            "not from configured '%s'. Best-model comparison may be inconsistent.",
                            model_type, primary_metric_name, _configured,
                        )
                    is_lower = primary_metric_name in _LOWER_IS_BETTER_METRICS
                    is_first = best_result is None
                    is_new_best = is_lower and primary_score < best_score
                    is_new_best = is_new_best or (not is_lower and primary_score > best_score)
                    if is_first or is_new_best:
                        best_score = primary_score
                        best_metric_name = primary_metric_name
                        best_result = (model_type, res)

                # Advance progress to end of this model’s range after success.
                _update_session(db, s, progress=progress_end)
                training_notifier.emit_sync(
                    session_id,
                    EventType.MODEL_COMPLETE,
                    {
                        "modelType": model_type,
                        "progress": progress_end,
                        "score": primary_score,
                        "index": i,
                        "total": total,
                    },
                )
                _log_event("training.session.model.success", session_id=session_id, model_type=model_type)

            except Exception as e:
                _rollback_quietly(db)
                _append_session_message(db, s, f"[{model_type}] {str(e)}")
                _log_event(
                    "training.session.model.error",
                    session_id=session_id,
                    model_type=model_type,
                    reason=str(e),
                )
                training_notifier.emit_sync(
                    session_id,
                    EventType.ERROR,
                    {"modelType": model_type, "error": str(e), "fatal": False},
                )
                # Still advance progress so the bar moves even on error.
                _update_session(db, s, progress=progress_end)

        training_time = time.monotonic() - t_start
        final_status = "succeeded" if success_count > 0 else "failed"
        _update_session(db, s, status=final_status, progress=100, finished_at=_now(), current_model=None)

        # Record meta-learning outcome
        if best_result is not None:
            _record_meta_learning(
                session_id=session_id,
                project_id=s.project_id,
                cfg=cfg,
                cfg_raw=cfg_raw,
                config_mode=config_mode,
                best_model_type=best_result[0],
                best_res=best_result[1],
                best_score=best_score,
                training_time=training_time,
            )

        training_notifier.emit_sync(
            session_id,
            EventType.FINAL_COMPLETE,
            {"status": final_status, "successCount": success_count, "trainingTimeS": training_time},
        )
        _log_event("training.session.end", session_id=session_id, status=final_status, success_count=success_count)

    except Exception as e:
        _rollback_quietly(db)
        s2 = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if s2:
            _update_session(
                db,
                s2,
                status="failed",
                progress=min(int(getattr(s2, "progress", 0) or 0), 99),
                error_message=str(e),
                finished_at=_now(),
            )
        training_notifier.emit_sync(session_id, EventType.ERROR, {"error": str(e), "fatal": True})
        _log_event("training.session.fatal_error", session_id=session_id, reason=str(e))
    finally:
        db.close()


# Aliases map a user-facing metric name to keys that may actually appear in
# the stored metrics dict (e.g. "roc_auc" is stored as "roc_auc_ovr_weighted"
# for multiclass models).
_PRIMARY_SCORE_ALIASES: Dict[str, tuple[str, ...]] = {
    "roc_auc":           ("roc_auc", "roc_auc_ovr_weighted"),
    "f1":                ("f1", "f1_pos", "f1_weighted"),
    "f1_macro":          ("f1_macro", "f1"),
    "f1_weighted":       ("f1_weighted", "f1"),
    "f1_micro":          ("f1_micro", "f1"),
    "precision":         ("precision", "precision_pos", "precision_weighted"),
    "recall":            ("recall", "recall_pos", "recall_weighted"),
    "pr_auc":            ("pr_auc",),
    "average_precision": ("pr_auc", "average_precision"),
    "accuracy":          ("accuracy",),
    "balanced_accuracy": ("balanced_accuracy", "accuracy"),
    "r2":                ("r2",),
    "rmse":              ("rmse",),
    "mae":               ("mae",),
    "mse":               ("mse",),
    "mcc":               ("mcc",),
}


def _score_from_metrics_json(key: str, metrics_json: Dict[str, Any]) -> Optional[float]:
    """Look up ``key`` in the test block then in the top-level of ``metrics_json``."""
    test = metrics_json.get("test")
    val = test.get(key) if isinstance(test, dict) else None
    if val is None:
        val = metrics_json.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _extract_primary_score(
    metrics_json: Dict[str, Any],
    cfg: TrainingConfig,
) -> tuple[Optional[str], Optional[float], str]:
    """Return (metric_name, score, source) for direction-aware best-model selection.

    ``source`` is ``"user_defined"`` when the score matches ``cfg.metrics[0]``
    (or one of its known aliases), or ``"fallback"`` when the hardcoded priority
    list was used because the configured metric was not found.

    Returns ``(None, None, "fallback")`` when no real evaluation metric is
    available — caller must skip such results (do not record them as 'best').
    """
    user_metric = next(iter(cfg.metrics or []), None)
    if user_metric:
        user_key = str(user_metric).strip().lower()
        for alias in _PRIMARY_SCORE_ALIASES.get(user_key, (user_key,)):
            score = _score_from_metrics_json(alias, metrics_json)
            if score is not None:
                return alias, score, "user_defined"

    # Fallback: original hardcoded priority list (unchanged)
    for key in ("roc_auc", "f1", "pr_auc", "accuracy", "r2", "f1_weighted", "f1_macro"):
        val = metrics_json.get("test", {}).get(key) or metrics_json.get(key)
        if val is not None:
            try:
                return key, float(val), "fallback"
            except Exception:
                pass
    for key in ("rmse", "mae", "mse"):
        val = metrics_json.get("test", {}).get(key) or metrics_json.get(key)
        if val is not None:
            try:
                return key, float(val), "fallback"
            except Exception:
                pass
    return None, None, "fallback"


def _record_meta_learning(
    *,
    session_id: int,
    project_id: int,
    cfg: TrainingConfig,
    cfg_raw: Dict[str, Any],
    config_mode: str,
    best_model_type: str,
    best_res: Any,
    best_score: float,
    training_time: float,
) -> None:
    try:
        primary_metric = (list(cfg.metrics) or ["accuracy"])[0]
        resampling = str(cfg.balancing.strategy) if cfg.balancing.strategy != "none" else None
        record = build_training_record(
            session_id=session_id,
            project_id=project_id,
            profile_meta={
                "n_samples": int(getattr(best_res, "n_train", 0) or 0),
                "n_features": 0,
                "task_type": str(cfg.task_type),
                "dataset_size_category": "unknown",
            },
            best_model_type=best_model_type,
            best_hyperparams=cfg.model_hyperparams.get(best_model_type, {}),
            best_score=best_score,
            metric_used=primary_metric,
            resampling_method=resampling,
            split_method=str(cfg.split_method),
            search_type=str(cfg.search_type),
            config_mode=config_mode,
            user_overrides=cfg_raw.get("userOverrides"),
            training_time_seconds=training_time,
        )
        _meta_learner.record(project_id, record)
    except Exception as exc:
        logger.debug("meta_learner record failed (non-fatal): %s", exc)


def run_automl_session(session_id: int) -> None:
    """Background task for AutoML (FLAML) training sessions."""
    t_start = time.monotonic()
    db = SessionLocal()
    try:
        s: TrainingSession | None = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not s:
            return

        _log_event("training.automl.start", session_id=session_id)
        training_notifier.emit_sync(session_id, EventType.STARTED, {"sessionId": session_id})
        _update_session(db, s, status="running", progress=5, started_at=_now())

        cfg_raw: Dict[str, Any] = s.config_json or {}
        cfg = AutoMLConfig.from_front(cfg_raw)

        if not cfg.target_column:
            raise RuntimeError("targetColumn is required")

        path_to_load, dv_id, _ = resolve_training_data_path(
            db, s.project_id, s.dataset_version_id)
        df = load_dataframe(path_to_load)

        _update_session(db, s, dataset_version_id=dv_id, progress=10, current_model="AutoML en cours…")
        training_notifier.emit_sync(session_id, EventType.PROGRESS, {"progress": 10, "phase": "data_loaded"})

        out_dir = PROJECTS_PATH / str(s.project_id) / "training_models" / str(session_id)

        # Progress callback: SSE only (thread-safe). DB updated from main thread only.
        def progress_cb(pct: int, msg: str) -> None:
            training_notifier.emit_sync(
                session_id,
                EventType.PROGRESS,
                {"progress": max(10, min(95, pct)), "phase": msg},
            )

        all_res = run_automl(df, cfg, progress_cb=progress_cb)

        best_estimator = "automl"
        success_count = 0
        for res in all_res:
            est_name = res.metrics_json.get("best_estimator") or ("automl" if res.is_best else "unknown")
            pkl_name = "automl" if res.is_best else f"automl_{est_name}"
            try:
                _automl_threshold = (
                    res.metrics_json.get("threshold_used")
                    if res.task_type == "classification"
                    else None
                )
                _automl_balancing = res.artifacts_json.get("balancing") or {}
                pkl_path = save_pipeline(
                    res.fitted_model, out_dir, pkl_name,
                    threshold=_automl_threshold,
                    positive_class_index=_automl_balancing.get("positive_class_index"),
                    positive_label=_automl_balancing.get("positive_label") or cfg.positive_label,
                )
                res.artifacts_json["model_pkl"] = str(pkl_path)
            except Exception as exc:
                logger.warning("Could not save pipeline for %s: %s", est_name, exc)
                res.artifacts_json["model_pkl"] = None
            res.artifacts_json["dataset_version_id"] = dv_id

            model_type_str = "automl" if res.is_best else est_name
            persist_trained_model(
                db,
                session_id=session_id,
                project_id=s.project_id,
                model_type=model_type_str,
                task_type=str(res.task_type),
                metrics_json=res.metrics_json,
                artifacts_json=res.artifacts_json,
            )
            success_count += 1
            if res.is_best:
                best_estimator = est_name

        training_time = time.monotonic() - t_start
        _update_session(db, s, status="succeeded", progress=100, finished_at=_now(), current_model=None)

        training_notifier.emit_sync(
            session_id,
            EventType.FINAL_COMPLETE,
            {
                "status": "succeeded",
                "successCount": success_count,
                "trainingTimeS": training_time,
                "bestEstimator": best_estimator,
            },
        )
        _log_event("training.automl.end", session_id=session_id, best_estimator=best_estimator, n_estimators=success_count)

    except Exception as e:
        _rollback_quietly(db)
        s2 = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if s2:
            _update_session(
                db,
                s2,
                status="failed",
                progress=min(int(getattr(s2, "progress", 0) or 0), 99),
                error_message=str(e),
                finished_at=_now(),
            )
        training_notifier.emit_sync(session_id, EventType.ERROR, {"error": str(e), "fatal": True})
        _log_event("training.automl.fatal_error", session_id=session_id, reason=str(e))
    finally:
        db.close()
