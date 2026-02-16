# app/services/training_service.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time

import joblib
import pandas as pd
import numpy as np

from sqlalchemy.orm import Session

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
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

IMBALANCE_THRESHOLD = 0.20  # minority ratio threshold


# ------------------------------
# Utilities
# ------------------------------

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


def _class_dist(y_arr: np.ndarray) -> Dict[str, int]:
    try:
        vals, counts = np.unique(y_arr, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}
    except Exception:
        return {}


def _ratio(v: Any, default: float) -> float:
    """Accepts 0-1 ratios OR 0-100 percentages."""
    try:
        x = float(v)
    except Exception:
        return default
    if x > 1.0:
        x = x / 100.0
    if x <= 0:
        return default
    if x >= 1:
        return 0.99
    return x


def _is_binary(y: np.ndarray) -> bool:
    try:
        return len(np.unique(y)) == 2
    except Exception:
        return False


def _positive_label(y: np.ndarray) -> Any:
    vals = np.unique(y)
    # common case 0/1
    try:
        s = set([float(v) for v in vals.tolist()])
        if s == {0.0, 1.0}:
            return 1.0
    except Exception:
        pass
    return vals.max()


def _binary_labels(y: np.ndarray) -> tuple[Any, Any]:
    """Return (neg_label, pos_label) for binary target."""
    vals = np.unique(y)
    if len(vals) != 2:
        return (None, None)
    pos = _positive_label(y)
    neg = vals[0] if vals[1] == pos else vals[1]
    return neg, pos


def _imbalance_info(y: np.ndarray) -> dict:
    try:
        yv = np.asarray(y)
        vals, counts = np.unique(yv, return_counts=True)
        total = int(counts.sum())
        if total <= 0:
            return {}
        info: dict[str, Any] = {
            "n_classes": int(len(vals)),
            "total": total,
            "counts": {str(v): int(c) for v, c in zip(vals, counts)},
        }
        if len(vals) >= 2:
            min_c = int(counts.min())
            max_c = int(counts.max())
            info["minority_count"] = min_c
            info["majority_count"] = max_c
            info["minority_ratio"] = float(min_c / total) if total else None
            info["imbalance_ratio"] = float(max_c / min_c) if min_c > 0 else None
        if len(vals) == 2:
            info["positive_label"] = str(_positive_label(yv))
        return info
    except Exception:
        return {}


def _majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    """Baseline: always predict majority class (from y_train) and score on y_eval."""
    try:
        y_tr = np.asarray(y_train)
        y_te = np.asarray(y_eval)
        if y_tr.size == 0 or y_te.size == 0:
            return {}

        vals, counts = np.unique(y_tr, return_counts=True)
        maj = vals[int(np.argmax(counts))]
        y_pred = np.full_like(y_te, fill_value=maj)

        out: dict[str, Any] = {
            "majority_label": str(maj),
            "majority_support_train": int(counts.max()),
            "train_size": int(y_tr.size),
            "eval_size": int(y_te.size),
            "metrics": {
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            },
        }

        if _is_binary(y_te):
            pos_label = _positive_label(y_te)
            out["metrics"]["precision_pos"] = float(
                precision_score(y_te, y_pred, average="binary", pos_label=pos_label, zero_division=0)
            )
            out["metrics"]["recall_pos"] = float(
                recall_score(y_te, y_pred, average="binary", pos_label=pos_label, zero_division=0)
            )
            out["metrics"]["f1_pos"] = float(
                f1_score(y_te, y_pred, average="binary", pos_label=pos_label, zero_division=0)
            )
        return out
    except Exception:
        return {}


def _infer_numeric_columns(X: pd.DataFrame, threshold: float = 0.85) -> Tuple[List[str], List[str]]:
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in X.columns:
        s = X[c]
        try:
            frac_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
            if frac_numeric >= threshold:
                num_cols.append(c)
            else:
                cat_cols.append(c)
        except Exception:
            cat_cols.append(c)
    return num_cols, cat_cols


def _detect_binary_encoded_cols(X: pd.DataFrame, max_unique: int = 2, sample_max: int = 5000) -> List[str]:
    """
    Detect columns that look like already-one-hot / boolean encoded (0/1, True/False).
    SMOTE on these columns can create fractional values (0.2, 0.7) -> less trustworthy.
    """
    cols: List[str] = []
    if X is None or X.shape[1] == 0:
        return cols

    Xs = X
    if len(X) > sample_max:
        Xs = X.sample(sample_max, random_state=42)

    for c in Xs.columns:
        try:
            s = Xs[c].dropna()
            if s.empty:
                continue
            # normalize booleans
            if s.dtype == bool:
                cols.append(c)
                continue
            # numeric check
            sn = pd.to_numeric(s, errors="coerce").dropna()
            if sn.empty:
                continue
            uniq = np.unique(sn.values)
            if len(uniq) <= max_unique:
                # accept {0,1} or {0} or {1}
                if set(uniq.tolist()).issubset({0.0, 1.0}):
                    cols.append(c)
        except Exception:
            continue

    return cols


def _build_preprocessor(df: pd.DataFrame, target: str):
    """
    ✅ Generic preprocessing:
    - numeric: median impute + scaler(with_mean=False) => works with sparse
    - categorical: most_frequent + OneHot sparse + min_frequency (groups rare categories)
    """
    X = df.drop(columns=[target])
    num_cols, cat_cols = _infer_numeric_columns(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            min_frequency=10,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # for robust SMOTE decision
    binary_encoded_cols = _detect_binary_encoded_cols(X[num_cols]) if num_cols else []
    return preprocessor, num_cols, cat_cols, binary_encoded_cols


def _validate_target_rows(df: pd.DataFrame, target: str, min_rows: int = 10) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2[df2[target].notna()]
    if len(df2) < min_rows:
        raise RuntimeError(f"Not enough rows after cleaning target NaNs (rows={len(df2)}).")
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


def _apply_class_weight_if_supported(estimator, enable: bool, model_type: str):
    if not enable:
        return estimator
    try:
        params = estimator.get_params()
        if "class_weight" in params:
            if model_type == "randomforest":
                estimator.set_params(class_weight="balanced_subsample")
            else:
                estimator.set_params(class_weight="balanced")
    except Exception:
        pass
    return estimator


def _compute_scale_pos_weight(y_train: np.ndarray) -> float | None:
    """neg/pos for XGB/LGBM on binary classification."""
    try:
        vals, counts = np.unique(y_train, return_counts=True)
        if len(vals) != 2:
            return None
        pos = _positive_label(y_train)
        pos_count = counts[list(vals).index(pos)] if pos in vals else counts.min()
        neg_count = counts.sum() - pos_count
        if pos_count <= 0:
            return None
        return float(neg_count) / float(pos_count)
    except Exception:
        return None


def _make_estimator(model_type: str, task_type: str, *, class_weight_balanced: bool = False):
    is_clf = task_type == "classification"

    if model_type == "randomforest":
        est = (
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            if is_clf else
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        )
        return _apply_class_weight_if_supported(est, class_weight_balanced, model_type)

    if model_type == "logreg":
        if not is_clf:
            raise RuntimeError("logreg is classification-only")
        est = LogisticRegression(max_iter=4000, n_jobs=-1)
        return _apply_class_weight_if_supported(est, class_weight_balanced, model_type)

    if model_type == "svm":
        est = SVC(probability=True) if is_clf else SVR()
        return _apply_class_weight_if_supported(est, class_weight_balanced, model_type)

    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=7) if is_clf else KNeighborsRegressor(n_neighbors=7)

    if model_type == "decisiontree":
        est = DecisionTreeClassifier(random_state=42) if is_clf else DecisionTreeRegressor(random_state=42)
        return _apply_class_weight_if_supported(est, class_weight_balanced, model_type)

    if model_type == "xgboost":
        if XGBClassifier is None and XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return (
            XGBClassifier(
                n_estimators=350,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )
            if is_clf else
            XGBRegressor(
                n_estimators=350,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )
        )

    if model_type == "lightgbm":
        if LGBMClassifier is None and LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed")
        return (
            LGBMClassifier(
                n_estimators=450,
                learning_rate=0.06,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
            )
            if is_clf else
            LGBMRegressor(
                n_estimators=450,
                learning_rate=0.06,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
            )
        )

    raise RuntimeError(f"Unknown model type: {model_type}")


def _get_scores_for_auc(fitted_pipe, X: pd.DataFrame):
    y_proba = None
    y_score = None
    try:
        if hasattr(fitted_pipe, "predict_proba"):
            y_proba = fitted_pipe.predict_proba(X)
    except Exception:
        y_proba = None
    try:
        if hasattr(fitted_pipe, "decision_function"):
            y_score = fitted_pipe.decision_function(X)
    except Exception:
        y_score = None
    return y_proba, y_score


def _predict_scores_binary(fitted_pipe, X: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    Return (scores, kind) for binary classification:
    - kind 'proba' => scores in [0,1] positive proba
    - kind 'score' => unbounded decision_function score
    """
    y_proba, y_score = _get_scores_for_auc(fitted_pipe, X)
    if y_proba is not None and getattr(y_proba, "ndim", 0) == 2 and y_proba.shape[1] >= 2:
        return np.asarray(y_proba[:, 1]), "proba"
    if y_score is not None:
        return np.asarray(y_score), "score"
    return np.asarray(fitted_pipe.predict(X)).astype(float), "hard"


def _best_threshold_by_f1_pos(y_true: np.ndarray, scores: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y_true, scores)
    if thr.size == 0:
        return {"enabled": False, "reason": "no_thresholds"}

    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return {
        "enabled": True,
        "threshold": float(thr[i]),
        "val_precision_pos": float(prec[i]),
        "val_recall_pos": float(rec[i]),
        "val_f1_pos": float(f1[i]),
    }


def _predict_with_threshold_binary(scores: np.ndarray, threshold: float, *, pos_label: Any, neg_label: Any) -> np.ndarray:
    scores = np.asarray(scores)
    mask = scores >= float(threshold)
    return np.where(mask, pos_label, neg_label)


def _compute_classification_metrics(y_true, y_pred, *, y_proba=None, y_score=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if _is_binary(y_true):
        neg_label, pos_label = _binary_labels(y_true)
        if pos_label is None:
            pos_label = _positive_label(y_true)

        out["precision_pos"] = float(precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))
        out["recall_pos"] = float(recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))
        out["f1_pos"] = float(f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0))

        try:
            if y_proba is not None and getattr(y_proba, "ndim", 0) == 2 and y_proba.shape[1] >= 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            elif y_score is not None:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
                out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass
    else:
        try:
            if y_proba is not None:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            pass
        try:
            out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except Exception:
            pass

    return out


def _compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _choose_primary_metric(task_type: str, y_ref: np.ndarray, metrics_dict: Dict[str, Any]) -> tuple[str, float, bool]:
    if task_type != "classification":
        if "r2" in metrics_dict:
            return "r2", float(metrics_dict["r2"]), True
        for k in ["rmse", "mae", "mse"]:
            if k in metrics_dict:
                return k, float(metrics_dict[k]), False
        return "r2", 0.0, True

    if _is_binary(y_ref):
        info = _imbalance_info(y_ref)
        mr = info.get("minority_ratio")
        if isinstance(mr, (int, float)) and mr < IMBALANCE_THRESHOLD:
            if "pr_auc" in metrics_dict:
                return "pr_auc", float(metrics_dict["pr_auc"]), True
            if "f1_pos" in metrics_dict:
                return "f1_pos", float(metrics_dict["f1_pos"]), True

    for k in ["roc_auc", "pr_auc", "f1_pos", "f1_macro", "f1", "accuracy", "precision", "recall"]:
        if k in metrics_dict:
            return k, float(metrics_dict[k]), True
    return "accuracy", 0.0, True


def _normalize_scoring(task_type: str, y: np.ndarray, scoring: Optional[str]) -> str:
    if task_type != "classification":
        return scoring or "r2"

    if not scoring:
        if _is_binary(y):
            info = _imbalance_info(y)
            mr = info.get("minority_ratio")
            if isinstance(mr, (int, float)) and mr < IMBALANCE_THRESHOLD:
                return "average_precision"
            return "roc_auc"
        return "f1_weighted"

    sc = str(scoring)
    if sc == "roc_auc" and (not _is_binary(y)):
        return "roc_auc_ovr"
    return sc


def _safe_smote_enabled(
    requested: bool,
    task_type: str,
    y_train: np.ndarray,
    *,
    cat_cols: List[str],
    binary_encoded_cols: List[str],
) -> tuple[bool, dict]:
    """
    Robust SMOTE rules:
    - If categorical columns exist => disable (we do OHE; SMOTENC would be another design)
    - If many binary-encoded columns (0/1) => disable (SMOTE creates fractional values)
    - If minority too small => disable
    """
    if not requested:
        return False, {"enabled": False, "reason": "disabled_by_user"}

    if task_type != "classification":
        return False, {"enabled": False, "reason": "not_classification"}

    if SMOTE is None or ImbPipeline is None:
        return False, {"enabled": False, "reason": "imblearn_not_installed"}

    if cat_cols:
        return False, {
            "enabled": False,
            "reason": "categorical_present_disable_smote",
            "cat_cols_count": int(len(cat_cols)),
        }

    if binary_encoded_cols and len(binary_encoded_cols) >= 2:
        return False, {
            "enabled": False,
            "reason": "binary_encoded_features_disable_smote",
            "binary_cols_count": int(len(binary_encoded_cols)),
        }

    if not _is_binary(y_train):
        return False, {"enabled": False, "reason": "multiclass_disable_smote"}

    info = _imbalance_info(y_train)
    minority = info.get("minority_count")
    if not isinstance(minority, int) or minority < 6:
        return False, {"enabled": False, "reason": "minority_too_small", "minority_count": minority}

    return True, {"enabled": True, "minority_count": int(minority)}


def _build_pipeline(preprocessor, estimator, smote_enabled: bool, y_train: Optional[np.ndarray] = None):
    if smote_enabled:
        if SMOTE is None or ImbPipeline is None:
            raise RuntimeError("SMOTE requested but imbalanced-learn is not installed.")
        k = 5
        if y_train is not None:
            info = _imbalance_info(y_train)
            mc = info.get("minority_count")
            if isinstance(mc, int):
                k = max(1, min(5, mc - 1))
        return ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=k)),
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
    try:
        model = fitted_pipe.named_steps.get("model")
        prep = fitted_pipe.named_steps.get("prep")
    except Exception:
        return []

    if model is None or prep is None:
        return []
    if not hasattr(model, "feature_importances_"):
        return []

    try:
        try:
            feature_names = list(prep.get_feature_names_out())
        except Exception:
            feature_names = []

        imps = model.feature_importances_
        items = []
        for i, imp in enumerate(imps):
            feat = feature_names[i] if i < len(feature_names) else f"f_{i}"
            items.append({"feature": str(feat), "importance": float(imp)})
        items = sorted(items, key=lambda d: d["importance"], reverse=True)[:50]
        return items
    except Exception:
        return []


def _grid_for_model(model_type: str) -> dict:
    if model_type == "randomforest":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 6, 12],
            "model__min_samples_split": [2, 5, 10],
        }
    if model_type == "logreg":
        return {"model__C": [0.3, 1.0, 3.0]}
    if model_type == "svm":
        return {"model__C": [0.5, 1.0, 3.0], "model__kernel": ["rbf", "linear"]}
    if model_type == "knn":
        return {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]}
    if model_type == "decisiontree":
        return {"model__max_depth": [None, 4, 8, 12], "model__min_samples_split": [2, 5, 10]}
    if model_type == "xgboost":
        return {
            "model__n_estimators": [250, 400],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.03, 0.08],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
    if model_type == "lightgbm":
        return {
            "model__n_estimators": [300, 600],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.03, 0.06],
        }
    return {}


def _fit_gridsearch_by_model(pipe, model_type: str, X, y, task_type: str, scoring: str, cv_splits: int):
    grid = _grid_for_model(model_type)
    if not grid:
        pipe.fit(X, y)
        return pipe, {"grid_search": {"enabled": False, "reason": "no_grid"}}

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)

    info = {
        "grid_search": {
            "enabled": True,
            "scoring": scoring,
            "cv_splits": int(cv_splits),
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_) if gs.best_score_ is not None else None,
        }
    }
    return gs.best_estimator_, info


def _make_inner_val_split(X_train: pd.DataFrame, y_train: np.ndarray, val_frac: float = 0.15):
    y_train = np.asarray(y_train)
    if len(y_train) < 50:
        return X_train, None, y_train, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    (tr_idx, va_idx), = sss.split(X_train, y_train)
    X_fit = X_train.iloc[tr_idx]
    X_val = X_train.iloc[va_idx]
    y_fit = y_train[tr_idx]
    y_val = y_train[va_idx]
    return X_fit, X_val, y_fit, y_val


def _is_imbalanced_binary(y_train: np.ndarray) -> bool:
    if not _is_binary(y_train):
        return False
    info = _imbalance_info(y_train)
    mr = info.get("minority_ratio")
    return isinstance(mr, (int, float)) and mr < IMBALANCE_THRESHOLD


def _auto_balance_estimator(
    estimator,
    model_type: str,
    task_type: str,
    y_train: np.ndarray,
    *,
    smote_enabled: bool,
    use_class_weight_flag: bool,
) -> tuple[Any, dict]:
    """
    Coherent balancing strategy:
    - If SMOTE is enabled: do not also reweight
    - Else (binary imbalanced):
        - use class_weight for models that support it
        - use scale_pos_weight for xgboost/lightgbm
    """
    balancing = {
        "enabled": False,
        "auto": True,
        "class_weight": False,
        "scale_pos_weight": False,
        "threshold": float(IMBALANCE_THRESHOLD),
    }

    if task_type != "classification":
        return estimator, balancing

    if not _is_imbalanced_binary(y_train):
        return estimator, balancing

    if smote_enabled:
        balancing["enabled"] = True
        return estimator, balancing

    balancing["enabled"] = True

    # scale_pos_weight for boosting models
    if model_type in ("xgboost", "lightgbm"):
        spw = _compute_scale_pos_weight(y_train)
        if spw is not None:
            try:
                estimator.set_params(scale_pos_weight=float(spw))
                balancing["scale_pos_weight"] = True
                balancing["scale_pos_weight_value"] = float(spw)
            except Exception:
                pass
        return estimator, balancing

    # class_weight for others (if user asked OR automatic)
    # Here we auto-enable if imbalanced.
    estimator = _apply_class_weight_if_supported(estimator, True or bool(use_class_weight_flag), model_type)
    try:
        if "class_weight" in estimator.get_params():
            balancing["class_weight"] = True
    except Exception:
        pass

    return estimator, balancing


# ------------------------------
# Main worker
# ------------------------------

def run_training_session(session_id: int) -> None:
    db = SessionLocal()
    try:
        s: TrainingSession | None = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not s:
            return

        cfg: Dict[str, Any] = s.config_json or {}
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

        task_type = str(cfg.get("taskType") or "classification").strip().lower()
        models = [str(m).strip().lower() for m in (cfg.get("models") or [])]
        if not models:
            raise RuntimeError("No model selected")

        use_smote = bool(cfg.get("useSmote", False))
        use_class_weight = bool(cfg.get("useClassWeight", False))
        use_grid = bool(cfg.get("useGridSearch", False))
        split_method = str(cfg.get("splitMethod") or "holdout").strip().lower()
        cv_splits_grid = int(cfg.get("gridCvFolds", 3) or 3)

        X = df.drop(columns=[target])
        y = df[target].values

        if task_type == "classification":
            _validate_classification_labels(y)

        out_dir = PROJECTS_PATH / str(project_id) / "training_models" / str(session_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        total = max(1, len(models))
        success_count = 0

        for i, model_type in enumerate(models, start=1):
            _update_session(db, s, progress=min(95, int(10 + (i - 1) * (80 / total))))

            t0 = time.perf_counter()
            artifacts: Dict[str, Any] = {"dataset_version_id": dv_id}
            final_metrics: Dict[str, Any] = {}

            try:
                preprocessor, num_cols, cat_cols, binary_encoded_cols = _build_preprocessor(df, target)
                estimator = _make_estimator(
                    model_type,
                    task_type,
                    class_weight_balanced=False,
                )

                # -------------------------
                # HOLDOUT
                # -------------------------
                if split_method == "holdout":
                    train_ratio = _ratio(cfg.get("trainRatio", 0.8), 0.8)
                    val_ratio = _ratio(cfg.get("valRatio", 0.0), 0.0)
                    test_ratio = _ratio(cfg.get("testRatio", 0.2), 0.2)

                    artifacts["split_debug"] = {
                        "trainRatio_input": cfg.get("trainRatio"),
                        "valRatio_input": cfg.get("valRatio"),
                        "testRatio_input": cfg.get("testRatio"),
                        "trainRatio_used": float(train_ratio),
                        "valRatio_used": float(val_ratio),
                        "testRatio_used": float(test_ratio),
                    }

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

                    artifacts["split_debug"].update({
                        "rows_all": int(X.shape[0]),
                        "rows_train": int(len(X_train)),
                        "rows_val": int(len(X_val)) if X_val is not None else 0,
                        "rows_test": int(len(X_test)),
                    })

                    artifacts["class_distribution"] = {
                        "all": _class_dist(y),
                        "train": _class_dist(y_train),
                        "val": _class_dist(y_val) if y_val is not None else {},
                        "test": _class_dist(y_test),
                    }
                    artifacts["imbalance_info"] = {
                        "all": _imbalance_info(y),
                        "train": _imbalance_info(y_train),
                        "test": _imbalance_info(y_test),
                    }
                    if task_type == "classification":
                        artifacts["baseline_majority"] = _majority_baseline(y_train, y_test)

                    # Robust SMOTE decision
                    smote_enabled, smote_meta = _safe_smote_enabled(
                        requested=use_smote,
                        task_type=task_type,
                        y_train=y_train,
                        cat_cols=cat_cols,
                        binary_encoded_cols=binary_encoded_cols,
                    )
                    artifacts["smote"] = smote_meta

                    # Auto balancing if needed (coherent strategy)
                    estimator, balancing_meta = _auto_balance_estimator(
                        estimator,
                        model_type,
                        task_type,
                        y_train,
                        smote_enabled=smote_enabled,
                        use_class_weight_flag=use_class_weight,
                    )
                    artifacts["balancing"] = balancing_meta

                    pipe = _build_pipeline(
                        preprocessor,
                        estimator,
                        smote_enabled=smote_enabled,
                        y_train=y_train if smote_enabled else None,
                    )

                    scoring = _normalize_scoring(task_type, y_train, cfg.get("gridScoring"))
                    if use_grid:
                        pipe, gs_info = _fit_gridsearch_by_model(
                            pipe, model_type, X_train, y_train, task_type, scoring, cv_splits_grid
                        )
                        artifacts.update(gs_info)
                    else:
                        pipe.fit(X_train, y_train)

                    # -------------------------
                    # Threshold tuning: binary + imbalanced
                    # - tune on val or inner-val (from train)
                    # - apply on test
                    # -------------------------
                    threshold_info: dict[str, Any] = {
                        "enabled": False,
                        "reason": "not_applied",
                        "applied_on_test": False,
                    }
                    tuned_threshold: Optional[float] = None

                    if task_type == "classification" and _is_imbalanced_binary(y_train):
                        neg_label, pos_label = _binary_labels(y_train)
                        threshold_info["enabled"] = True
                        threshold_info["pos_label"] = str(pos_label)
                        threshold_info["neg_label"] = str(neg_label)

                        if X_val is not None and y_val is not None:
                            X_fit, y_fit = X_train, y_train
                            X_thr, y_thr = X_val, y_val
                            threshold_info["val_source"] = "user_val"
                        else:
                            X_fit, X_thr, y_fit, y_thr = _make_inner_val_split(X_train, y_train, val_frac=0.15)
                            threshold_info["val_source"] = "inner_val_from_train"

                            # refit on X_fit only (avoid tuning threshold on same data)
                            if X_thr is not None and y_thr is not None and len(y_thr) > 0:
                                estimator2 = clone(estimator)
                                # keep balancing on subset
                                estimator2, _ = _auto_balance_estimator(
                                    estimator2, model_type, task_type, y_fit,
                                    smote_enabled=smote_enabled,
                                    use_class_weight_flag=use_class_weight,
                                )
                                pipe = _build_pipeline(
                                    preprocessor,
                                    estimator2,
                                    smote_enabled=smote_enabled,
                                    y_train=y_fit if smote_enabled else None,
                                )
                                if use_grid:
                                    pipe, gs_info2 = _fit_gridsearch_by_model(
                                        pipe, model_type, X_fit, y_fit, task_type, scoring, cv_splits_grid
                                    )
                                    artifacts.update(gs_info2)
                                else:
                                    pipe.fit(X_fit, y_fit)

                        if X_thr is not None and y_thr is not None and len(y_thr) > 0:
                            scores_thr, kind = _predict_scores_binary(pipe, X_thr)
                            threshold_info["score_kind"] = kind
                            tinfo = _best_threshold_by_f1_pos(y_thr, scores_thr)
                            threshold_info.update(tinfo)

                            if tinfo.get("enabled"):
                                tuned_threshold = float(tinfo["threshold"])

                    # apply threshold on test if available
                    if task_type == "classification" and tuned_threshold is not None and _is_binary(y_test):
                        neg_label_test, pos_label_test = _binary_labels(y_test)
                        test_scores, _ = _predict_scores_binary(pipe, X_test)
                        y_pred_test = _predict_with_threshold_binary(
                            test_scores,
                            tuned_threshold,
                            pos_label=pos_label_test,
                            neg_label=neg_label_test,
                        )
                        threshold_info["applied_on_test"] = True
                        threshold_info["reason"] = "applied"
                    else:
                        y_pred_test = pipe.predict(X_test)

                    artifacts["thresholding"] = threshold_info

                    # Train metrics (default predictions)
                    y_pred_train = pipe.predict(X_train)
                    y_proba_train, y_score_train = _get_scores_for_auc(pipe, X_train)
                    if task_type == "classification":
                        train_metrics = _compute_classification_metrics(
                            y_train, y_pred_train, y_proba=y_proba_train, y_score=y_score_train
                        )
                    else:
                        train_metrics = _compute_regression_metrics(y_train, y_pred_train)

                    # Test metrics
                    y_proba_test, y_score_test = _get_scores_for_auc(pipe, X_test)
                    if task_type == "classification":
                        test_metrics = _compute_classification_metrics(
                            y_test, y_pred_test, y_proba=y_proba_test, y_score=y_score_test
                        )
                    else:
                        test_metrics = _compute_regression_metrics(y_test, y_pred_test)

                    final_metrics = {"train": train_metrics, "test": test_metrics}
                    final_metrics["split_info"] = {
                        "method": "holdout",
                        "train_rows": int(len(X_train)),
                        "val_rows": int(len(X_val)) if X_val is not None else 0,
                        "test_rows": int(len(X_test)),
                        "train_ratio": float(train_ratio),
                        "val_ratio": float(val_ratio),
                        "test_ratio": float(test_ratio),
                    }

                    best_k, best_v, higher_is_better = _choose_primary_metric(task_type, y_test, test_metrics)
                    final_metrics["primary_score"] = {"metric": best_k, "value": best_v, "higher_is_better": higher_is_better}

                    if task_type == "classification":
                        artifacts["confusion_matrix"] = _try_confusion_matrix(y_test, y_pred_test)
                    artifacts["feature_importance"] = _try_feature_importance(pipe)

                # -------------------------
                # KFOLD
                # -------------------------
                elif split_method == "kfold":
                    k = int(cfg.get("kFolds", 5) or 5)
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
                        n = int(X.shape[0])
                        if k > n:
                            k = n
                        splitter = KFold(n_splits=k, shuffle=True, random_state=42)

                    fold_metrics: list[dict] = []
                    fold_cms: list[np.ndarray] = []

                    if task_type == "classification":
                        artifacts["class_distribution"] = {"all": _class_dist(y), "train": {}, "val": {}, "test": {}}
                        artifacts["imbalance_info"] = {"all": _imbalance_info(y)}
                        artifacts["baseline_majority"] = _majority_baseline(y, y)

                    # SMOTE safe (kfold global decision)
                    smote_enabled, smote_meta = _safe_smote_enabled(
                        requested=use_smote,
                        task_type=task_type,
                        y_train=y,
                        cat_cols=cat_cols,
                        binary_encoded_cols=binary_encoded_cols,
                    )
                    artifacts["smote"] = smote_meta

                    for tr_idx, te_idx in splitter.split(X, y):
                        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                        y_tr, y_te = y[tr_idx], y[te_idx]

                        fold_preprocessor, _, _, _ = _build_preprocessor(df, target)
                        fold_est = clone(estimator)
                        fold_est, _ = _auto_balance_estimator(
                            fold_est, model_type, task_type, y_tr,
                            smote_enabled=smote_enabled,
                            use_class_weight_flag=use_class_weight,
                        )

                        fold_pipe = _build_pipeline(
                            fold_preprocessor,
                            fold_est,
                            smote_enabled=smote_enabled,
                            y_train=y_tr if smote_enabled else None,
                        )
                        fold_pipe.fit(X_tr, y_tr)

                        y_pred = fold_pipe.predict(X_te)
                        y_proba, y_score = _get_scores_for_auc(fold_pipe, X_te)

                        if task_type == "classification":
                            fm = _compute_classification_metrics(y_te, y_pred, y_proba=y_proba, y_score=y_score)
                        else:
                            fm = _compute_regression_metrics(y_te, y_pred)

                        fold_metrics.append(fm)

                        if task_type == "classification":
                            try:
                                labels = np.unique(np.concatenate([np.unique(y), np.unique(y_pred)]))
                                fold_cms.append(confusion_matrix(y_te, y_pred, labels=labels))
                            except Exception:
                                pass

                    avg_test: Dict[str, Any] = {}
                    all_keys = set()
                    for fm in fold_metrics:
                        all_keys |= set(fm.keys())
                    for key in all_keys:
                        vals_ = [fm.get(key) for fm in fold_metrics if fm.get(key) is not None]
                        if vals_:
                            avg_test[key] = float(np.mean(vals_))

                    final_metrics = {
                        "kfold_folds": int(k),
                        "folds": fold_metrics,
                        "test": avg_test,
                        "split_info": {"method": "kfold", "folds": int(k), "rows": int(len(X))},
                    }

                    best_k, best_v, higher_is_better = _choose_primary_metric(task_type, y, avg_test)
                    final_metrics["primary_score"] = {"metric": best_k, "value": best_v, "higher_is_better": higher_is_better}

                    # fit final model on full data (for saving)
                    estimator_final = clone(estimator)
                    estimator_final, balancing_meta = _auto_balance_estimator(
                        estimator_final, model_type, task_type, y,
                        smote_enabled=smote_enabled,
                        use_class_weight_flag=use_class_weight,
                    )
                    artifacts["balancing"] = balancing_meta

                    pipe = _build_pipeline(preprocessor, estimator_final, smote_enabled=smote_enabled, y_train=y if smote_enabled else None)
                    pipe.fit(X, y)

                    if task_type == "classification" and fold_cms:
                        cm_sum = np.sum(fold_cms, axis=0)
                        artifacts["confusion_matrix"] = cm_sum.astype(int).tolist()
                    artifacts["feature_importance"] = _try_feature_importance(pipe)

                else:
                    raise RuntimeError(f"Unknown splitMethod: {split_method}")

                training_time = float(time.perf_counter() - t0)
                final_metrics["training_time_sec"] = training_time

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

                success_count += 1

            except Exception as e:
                _append_session_message(db, s, f"[{model_type}] {str(e)}")
                continue

        final_status = "succeeded" if success_count > 0 else "failed"
        _update_session(db, s, status=final_status, progress=100, finished_at=_now())

    except Exception as e:
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
    finally:
        db.close()
