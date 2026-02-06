from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List
import time

import joblib
import pandas as pd
import numpy as np

from sqlalchemy.orm import Session

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# imbalanced-learn (SMOTE + Pipeline)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTE = None
    ImbPipeline = None

# xgboost / lightgbm
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

from app.db.session import SessionLocal
from app.models.training import TrainingSession, TrainedModel
from app.models.dataset_version import DatasetVersion
from app.core.config import PROJECTS_PATH


def _now():
    return datetime.now(timezone.utc)


def _update_session(db: Session, s: TrainingSession, **fields):
    for k, v in fields.items():
        setattr(s, k, v)
    db.add(s)
    db.commit()
    db.refresh(s)


def _append_session_message(db: Session, s: TrainingSession, msg: str):
    current = (s.error_message or "").strip()
    msg = msg.strip()
    if not current:
        new_msg = msg
    else:
        new_msg = current if msg in current else (current + "\n" + msg)
    _update_session(db, s, error_message=new_msg)


def _resolve_training_dataset(db: Session, project_id: int, dataset_version_id: int | None) -> Tuple[Path, int]:
    if dataset_version_id is None:
        raise RuntimeError("dataset_version_id is required for training")

    dv = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.project_id == project_id, DatasetVersion.id == dataset_version_id)
        .first()
    )
    if not dv:
        raise RuntimeError("Dataset version not found for this project")

    p = Path(dv.file_path)
    if not p.exists():
        raise RuntimeError(f"Dataset file not found: {p}")

    return p, dv.id


def _infer_numeric_columns(X: pd.DataFrame, threshold: float = 0.85) -> Tuple[List[str], List[str]]:
    num_cols: list[str] = []
    cat_cols: list[str] = []

    for c in X.columns:
        s = X[c]

        if pd.api.types.is_bool_dtype(s):
            num_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        non_na = s.notna().sum()
        if non_na == 0:
            cat_cols.append(c)
            continue

        ratio = coerced.notna().sum() / float(non_na)
        if ratio >= threshold:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return num_cols, cat_cols


def _coerce_numeric_like_columns(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    num_cols, _ = _infer_numeric_columns(X, threshold=0.85)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def _build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    X = df.drop(columns=[target])
    num_cols, cat_cols = _infer_numeric_columns(X, threshold=0.85)

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _make_estimator(model_type: str, task_type: str):
    is_clf = task_type == "classification"

    if model_type == "randomforest":
        return RandomForestClassifier(n_estimators=300, random_state=42) if is_clf else RandomForestRegressor(n_estimators=300, random_state=42)

    if model_type == "logisticregression":
        if not is_clf:
            raise ValueError("logisticregression only for classification")
        return LogisticRegression(max_iter=2000)

    if model_type == "svm":
        return SVC(probability=True) if is_clf else SVR()

    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=7) if is_clf else KNeighborsRegressor(n_neighbors=7)

    if model_type == "decisiontree":
        return DecisionTreeClassifier(random_state=42) if is_clf else DecisionTreeRegressor(random_state=42)

    if model_type == "xgboost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ValueError("Model 'xgboost' not installed. Install: pip install xgboost")
        if is_clf:
            return XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "lightgbm":
        if LGBMClassifier is None or LGBMRegressor is None:
            raise ValueError("Model 'lightgbm' not installed. Install: pip install lightgbm")
        if is_clf:
            return LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)
        return LGBMRegressor(n_estimators=1200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1)

    raise ValueError(f"Unknown model: {model_type}")


def _compute_metrics(task_type: str, metrics: list[str], y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if task_type == "classification":
        for m in metrics:
            if m == "accuracy":
                out[m] = float(accuracy_score(y_true, y_pred))
            elif m == "f1":
                out[m] = float(f1_score(y_true, y_pred, average="weighted"))
            elif m == "precision":
                out[m] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            elif m == "recall":
                out[m] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            elif m == "roc_auc":
                if y_proba is None:
                    continue
                try:
                    classes = np.unique(y_true)
                    if len(classes) == 2 and hasattr(y_proba, "shape") and y_proba.shape[1] >= 2:
                        out[m] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        out[m] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                except Exception:
                    continue
    else:
        for m in metrics:
            if m == "mae":
                out[m] = float(mean_absolute_error(y_true, y_pred))
            elif m == "mse":
                out[m] = float(mean_squared_error(y_true, y_pred))
            elif m == "rmse":
                out[m] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif m == "r2":
                out[m] = float(r2_score(y_true, y_pred))

    return out


def _primary_score(task_type: str, metrics_dict: Dict[str, Any]) -> tuple[str, float, bool]:
    if task_type == "classification":
        for k in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
            if k in metrics_dict:
                return k, float(metrics_dict[k]), True
        return "accuracy", 0.0, True
    else:
        if "r2" in metrics_dict:
            return "r2", float(metrics_dict["r2"]), True
        for k in ["rmse", "mae", "mse"]:
            if k in metrics_dict:
                return k, float(metrics_dict[k]), False
        return "r2", 0.0, True


def _safe_stratify(y: np.ndarray, task_type: str):
    if task_type != "classification":
        return None
    try:
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) < 2 or counts.min() < 2:
            return None
        return y
    except Exception:
        return None


def _validate_target_rows(df: pd.DataFrame, target: str, min_rows: int = 10) -> pd.DataFrame:
    total = int(df.shape[0])
    missing = int(df[target].isna().sum())
    kept = total - missing

    df2 = df.dropna(subset=[target])
    if kept < min_rows:
        raise RuntimeError(
            f"Not enough rows after dropping missing target. "
            f"target='{target}' total_rows={total}, missing_target={missing}, kept={kept}."
        )
    return df2


def _validate_classification_labels(y: np.ndarray):
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        raise RuntimeError("Classification requires at least 2 classes in target.")
    if counts.min() < 2:
        raise RuntimeError(
            f"Some classes have too few samples (min_class_count={int(counts.min())}). "
            f"Please ensure each class has >= 2 samples."
        )


def _build_pipeline(preprocessor, estimator, smote_enabled: bool):
    if smote_enabled:
        if SMOTE is None or ImbPipeline is None:
            raise RuntimeError("SMOTE requested but imbalanced-learn is not installed.")
        return ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", estimator),
        ])
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])


def _try_confusion_matrix(y_true, y_pred) -> list[list[int]]:
    try:
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.astype(int).tolist()
    except Exception:
        return []


def _try_feature_importance(fitted_pipe) -> list[dict]:
    """
    Retourne liste [{feature: str, importance: float}] ou [].
    Pour pipeline: prep -> model
    On tente get_feature_names_out + feature_importances_ / coef_
    """
    try:
        model = fitted_pipe.named_steps.get("model")
        prep = fitted_pipe.named_steps.get("prep")
    except Exception:
        return []

    if model is None or prep is None:
        return []

    importances = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    except Exception:
        importances = None

    if importances is None:
        return []

    try:
        names = prep.get_feature_names_out()
        names = [str(n) for n in names]
    except Exception:
        names = [f"f{i}" for i in range(len(importances))]

    if len(names) != len(importances):
        m = min(len(names), len(importances))
        names = names[:m]
        importances = importances[:m]

    idx = np.argsort(importances)[::-1][:30]
    out = [{"feature": names[i], "importance": float(importances[i])} for i in idx if float(importances[i]) > 0]
    return out


def run_training_session(session_id: int) -> None:
    db = SessionLocal()
    try:
        s: TrainingSession | None = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not s:
            return

        cfg: Dict[str, Any] = s.config_json
        project_id = s.project_id

        _update_session(db, s, status="running", progress=3, started_at=_now())

        dataset_path, dv_id = _resolve_training_dataset(db, project_id, s.dataset_version_id)
        _update_session(db, s, dataset_version_id=dv_id, progress=7)

        df = pd.read_csv(dataset_path)
        target = str(cfg.get("targetColumn") or "").strip()
        if not target:
            raise RuntimeError("targetColumn is required")
        if target not in df.columns:
            raise RuntimeError(f"Target column '{target}' not found in dataset")

        df = _validate_target_rows(df, target, min_rows=10)

        task_type = str(cfg.get("taskType") or "classification")
        models = list(cfg.get("models") or [])
        metrics = list(cfg.get("metrics") or [])
        if not models:
            raise RuntimeError("No model selected")
        if not metrics:
            raise RuntimeError("No metric selected")

        _coerce_numeric_like_columns(df, target)

        X = df.drop(columns=[target])
        y = df[target].values

        if task_type == "classification":
            _validate_classification_labels(y)

        out_dir = PROJECTS_PATH / str(project_id) / "trained_models" / f"session_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        split_method = str(cfg.get("splitMethod") or "holdout")
        use_smote = bool(cfg.get("useSmote", False))

        smote_enabled = bool(use_smote and task_type == "classification" and SMOTE is not None and ImbPipeline is not None)
        if use_smote and not smote_enabled:
            _append_session_message(db, s, "SMOTE requested but imbalanced-learn is not available → disabled.")

        total = max(len(models), 1)

        for i, model_type in enumerate(models, start=1):
            _update_session(db, s, progress=int(10 + (i - 1) * (80 / total)))

            t0 = time.perf_counter()
            estimator = _make_estimator(str(model_type), task_type)
            preprocessor = _build_preprocessor(df, target)

            final_metrics: Dict[str, Any] = {}
            artifacts: Dict[str, Any] = {"dataset_version_id": dv_id}

            if split_method == "holdout":
                train_ratio = float(cfg.get("trainRatio", 70)) / 100.0
                val_ratio = float(cfg.get("valRatio", 15)) / 100.0
                test_ratio = float(cfg.get("testRatio", 15)) / 100.0

                n = int(X.shape[0])
                if n < 5:
                    raise RuntimeError(f"Not enough rows for holdout split (rows={n}).")

                stratify = _safe_stratify(y, task_type)

                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y,
                    test_size=max(0.01, (1.0 - train_ratio)),
                    random_state=42,
                    stratify=stratify,
                )

                if val_ratio <= 0:
                    X_val, y_val = None, None
                    X_test, y_test = X_temp, y_temp
                else:
                    denom = (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 1.0
                    val_prop_in_temp = val_ratio / denom
                    stratify_temp = _safe_stratify(y_temp, task_type)

                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp,
                        test_size=max(0.01, (1.0 - val_prop_in_temp)),
                        random_state=42,
                        stratify=stratify_temp,
                    )

                pipe = _build_pipeline(preprocessor, estimator, smote_enabled=smote_enabled)
                pipe.fit(X_train, y_train)

                # ---- train score (sur train) + test metrics ----
                y_pred_train = pipe.predict(X_train)
                train_metrics = _compute_metrics(task_type, metrics, y_train, y_pred_train)

                y_pred_test = pipe.predict(X_test)
                y_proba_test = None
                if task_type == "classification" and hasattr(pipe, "predict_proba"):
                    try:
                        y_proba_test = pipe.predict_proba(X_test)
                    except Exception:
                        y_proba_test = None

                test_metrics = _compute_metrics(task_type, metrics, y_test, y_pred_test, y_proba=y_proba_test)

                final_metrics = {"train": train_metrics, "test": test_metrics}

                if X_val is not None:
                    y_pred_val = pipe.predict(X_val)
                    y_proba_val = None
                    if task_type == "classification" and hasattr(pipe, "predict_proba"):
                        try:
                            y_proba_val = pipe.predict_proba(X_val)
                        except Exception:
                            y_proba_val = None
                    val_metrics = _compute_metrics(task_type, metrics, y_val, y_pred_val, y_proba=y_proba_val)
                    final_metrics["val"] = val_metrics

                final_metrics["split_info"] = {
                    "method": "holdout",
                    "train_rows": int(len(X_train)),
                    "val_rows": int(len(X_val)) if X_val is not None else 0,
                    "test_rows": int(len(X_test)),
                }

                best_k, best_v, higher_is_better = _primary_score(task_type, test_metrics)
                final_metrics["primary_score"] = {"metric": best_k, "value": best_v, "higher_is_better": higher_is_better}

                # confusion matrix + importance
                if task_type == "classification":
                    artifacts["confusion_matrix"] = _try_confusion_matrix(y_test, y_pred_test)
                artifacts["feature_importance"] = _try_feature_importance(pipe)

            else:
                # kfold
                k = int(cfg.get("kFolds", 5))
                n = int(X.shape[0])
                if k < 2:
                    raise RuntimeError("kFolds must be >= 2 for kfold split")

                if task_type == "classification":
                    vals, counts = np.unique(y, return_counts=True)
                    max_k = int(counts.min())
                    if max_k < 2:
                        raise RuntimeError("Not enough samples per class for StratifiedKFold.")
                    if k > max_k:
                        _append_session_message(db, s, f"kFolds reduced from {k} to {max_k} (min class count).")
                        k = max_k
                    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                else:
                    if k > n:
                        k = n
                    splitter = KFold(n_splits=k, shuffle=True, random_state=42)

                fold_metrics: list[dict] = []
                fold_cms: list[np.ndarray] = []

                for tr_idx, te_idx in splitter.split(X, y):
                    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                    y_tr, y_te = y[tr_idx], y[te_idx]

                    fold_preprocessor = _build_preprocessor(df, target)
                    fold_est = clone(estimator)

                    fold_pipe = _build_pipeline(fold_preprocessor, fold_est, smote_enabled=smote_enabled)
                    fold_pipe.fit(X_tr, y_tr)

                    y_pred = fold_pipe.predict(X_te)
                    y_proba = None
                    if task_type == "classification" and hasattr(fold_pipe, "predict_proba"):
                        try:
                            y_proba = fold_pipe.predict_proba(X_te)
                        except Exception:
                            y_proba = None

                    fold_metrics.append(_compute_metrics(task_type, metrics, y_te, y_pred, y_proba=y_proba))

                    if task_type == "classification":
                        try:
                            labels = np.unique(y)
                            fold_cms.append(confusion_matrix(y_te, y_pred, labels=labels))
                        except Exception:
                            pass

                avg_test: Dict[str, Any] = {}
                for key in metrics:
                    vals_ = [fm.get(key) for fm in fold_metrics if fm.get(key) is not None]
                    if vals_:
                        avg_test[key] = float(np.mean(vals_))

                final_metrics = {
                    "kfold_folds": int(k),
                    "folds": fold_metrics,
                    "test": avg_test,
                    "split_info": {"method": "kfold", "folds": int(k), "rows": int(len(X))},
                }

                best_k, best_v, higher_is_better = _primary_score(task_type, avg_test)
                final_metrics["primary_score"] = {"metric": best_k, "value": best_v, "higher_is_better": higher_is_better}

                # fit final model on ALL data (avec SMOTE si actif)
                pipe = _build_pipeline(preprocessor, estimator, smote_enabled=smote_enabled)
                pipe.fit(X, y)

                # confusion matrix moyenne (somme / folds) si possible
                if task_type == "classification" and fold_cms:
                    cm_sum = np.sum(fold_cms, axis=0)
                    artifacts["confusion_matrix"] = cm_sum.astype(int).tolist()
                artifacts["feature_importance"] = _try_feature_importance(pipe)

            training_time = float(time.perf_counter() - t0)
            final_metrics["training_time_sec"] = training_time

            # save artifacts + model
            pkl_path = out_dir / f"{model_type}.pkl"
            joblib.dump(pipe, pkl_path)
            artifacts["model_pkl"] = str(pkl_path)

            tm = TrainedModel(
                session_id=session_id,
                project_id=project_id,
                model_type=str(model_type),
                task_type=str(task_type),
                metrics_json=final_metrics,
                artifacts_json=artifacts,
            )
            db.add(tm)
            db.commit()

        _update_session(db, s, status="succeeded", progress=100, finished_at=_now())

    except Exception as e:
        s = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if s:
            _update_session(
                db,
                s,
                status="failed",
                progress=min(int(getattr(s, "progress", 0) or 0), 99),
                error_message=str(e),
                finished_at=_now(),
            )
    finally:
        db.close()
