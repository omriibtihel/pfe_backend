"""
Learning curve computation for the training pipeline.

Computes train-size vs. cross-validated score curves so the frontend can
visualise whether adding more data would improve model performance.

Anti-leakage design
-------------------
Learning curves are computed on the PREPROCESSED training matrix (the model
only, without the full pipeline), after the main training is complete.
The preprocessing step has already been fit on the training split only —
no leakage is introduced by this step.

We deliberately skip SMOTE/resampling inside the learning-curve sub-folds to
keep computation fast and avoid amplified variance from synthetic data.  The
curve reflects the real-data learning dynamics, which is what users care about.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold


# Minimum training-set size we use inside the curve (absolute samples).
_MIN_TRAIN_SAMPLES = 10
# Maximum number of points along the curve.
_MAX_POINTS = 8


def _select_scoring(task_type: str, y: np.ndarray) -> str:
    """Pick a fast sklearn scoring string appropriate for the task."""
    if task_type != "classification":
        return "r2"
    # For binary, use f1; for multiclass, use weighted f1
    n_classes = len(np.unique(y))
    return "f1" if n_classes == 2 else "f1_weighted"


def _make_cv_splitter(task_type: str, y: np.ndarray, n_splits: int, random_state: int = 42):
    """Stratified for classification when feasible, plain KFold otherwise."""
    if task_type != "classification":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _, counts = np.unique(y, return_counts=True)
    safe_splits = min(n_splits, int(counts.min()))
    if safe_splits < 2:
        return KFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)


def compute_learning_curve(
    estimator: Any,
    X_prep: np.ndarray,
    y_train: np.ndarray,
    *,
    task_type: str = "classification",
    n_points: int = _MAX_POINTS,
    cv_splits: int = 3,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Compute a learning curve (train size → cross-validated score).

    Parameters
    ----------
    estimator:
        An UNFITTED sklearn-compatible estimator.
    X_prep:
        Preprocessed training matrix (numpy array).
    y_train:
        Training labels / targets.
    task_type:
        "classification" | "regression"
    n_points:
        Number of train-size checkpoints (max `_MAX_POINTS`).
    cv_splits:
        Number of cross-validation folds inside each checkpoint (default 3
        for speed).
    random_state:
        Seed propagated to the internal CV splitter for reproducibility.

    Returns
    -------
    dict with shape::

        {
            "train_sizes":  [int, ...],   # absolute sample counts
            "train_mean":   [float, ...],
            "train_std":    [float, ...],
            "val_mean":     [float, ...],
            "val_std":      [float, ...],
            "scoring":      str,
            "n_samples":    int,
        }
    or None if computation fails or dataset is too small.
    """
    X_arr = np.asarray(X_prep)
    y_arr = np.asarray(y_train)
    n = len(y_arr)

    # Need enough samples: at least 3 × cv_splits to have meaningful splits
    min_needed = cv_splits * 3 + _MIN_TRAIN_SAMPLES
    if n < min_needed:
        return None

    scoring = _select_scoring(task_type, y_arr)
    cv = _make_cv_splitter(task_type, y_arr, cv_splits, random_state=random_state)

    # Use fractions so sklearn handles the per-fold train-size math automatically.
    # sklearn's learning_curve trains on `fraction * max_train_size` samples where
    # max_train_size = n * (cv_splits-1)/cv_splits.  Fractions stay in (0, 1].
    n_points_clamped = min(n_points, _MAX_POINTS)
    train_sizes_fractions = np.linspace(0.1, 1.0, n_points_clamped)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X_arr,
            y_arr,
            train_sizes=train_sizes_fractions,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score=np.nan,  # Degenerate folds (single class) return NaN instead of raising
        )
    except Exception:
        return None

    # For regression with negative scoring (neg_*), negate to get positive metric values.
    if scoring.startswith("neg_"):
        train_scores = -train_scores
        val_scores = -val_scores

    def _safe_list(arr: np.ndarray) -> list:
        """Convert 1-D array to list, replacing NaN → None for JSON safety."""
        return [None if np.isnan(v) else float(v) for v in arr]

    with np.errstate(all="ignore"):
        tr_mean = np.nanmean(train_scores, axis=1)
        tr_std = np.nanstd(train_scores, axis=1)
        vl_mean = np.nanmean(val_scores, axis=1)
        vl_std = np.nanstd(val_scores, axis=1)

    return {
        "train_sizes": [int(s) for s in train_sizes_abs.tolist()],
        "train_mean": _safe_list(tr_mean),
        "train_std": _safe_list(tr_std),
        "val_mean": _safe_list(vl_mean),
        "val_std": _safe_list(vl_std),
        "scoring": scoring,
        "n_samples": int(n),
    }
