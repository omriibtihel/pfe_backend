"""
Permutation feature importance for the training pipeline.

Unlike impurity-based feature_importances_ (tree models only), permutation
importance is model-agnostic: it works for every estimator that exposes
predict() or predict_proba().  It measures how much the model score drops
when a single feature's values are randomly shuffled on the TEST set —
the larger the drop, the more the model relied on that feature.

Key design choices
------------------
- Runs on the full fitted inference pipeline + raw test DataFrame (no separate
  preprocessing step needed — the pipeline handles it internally).
- Scoring string is selected the same way as learning_curves._select_scoring.
- Returns at most `max_features` items sorted by mean importance descending.
- Returns None on any failure so the orchestrator can safely ignore it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.services.training.pipeline.learning_curves import _select_scoring


_MAX_FEATURES = 30
# Cap the test set fed into permutation importance so that on large datasets the
# computation does not dominate total training time.  2 000 samples gives a
# statistically robust importance estimate while bounding runtime to O(seconds).
_MAX_IMPORTANCE_SAMPLES = 2_000


def compute_permutation_importance(
    fitted_pipe: Any,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    *,
    task_type: str = "classification",
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Optional[List[Dict[str, Any]]]:
    """
    Compute permutation importance on the held-out test set.

    Parameters
    ----------
    fitted_pipe:
        Full fitted sklearn Pipeline (align → prep → model).
    X_test_raw:
        Raw (un-preprocessed) test DataFrame — same format as training input.
    y_test:
        Test labels / targets.
    task_type:
        "classification" | "regression"
    feature_names:
        Column names for display.  Defaults to X_test_raw.columns if None.
    n_repeats:
        Number of shuffle repetitions per feature (default 5).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    List of dicts sorted by descending mean importance::

        [{"feature": str, "mean": float, "std": float}, ...]

    or None if computation fails or test set is too small.
    """
    X_arr = X_test_raw
    y_arr = np.asarray(y_test)

    if len(y_arr) < 20:
        return None

    # Cap test set size to bound computation time on large datasets.
    if len(y_arr) > _MAX_IMPORTANCE_SAMPLES:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(y_arr), size=_MAX_IMPORTANCE_SAMPLES, replace=False)
        X_arr = X_arr.iloc[idx] if isinstance(X_arr, pd.DataFrame) else X_arr[idx]
        y_arr = y_arr[idx]

    names = feature_names if feature_names is not None else list(X_test_raw.columns)
    scoring = _select_scoring(task_type, y_arr)

    try:
        result = permutation_importance(
            fitted_pipe,
            X_arr,
            y_arr,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        return None

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    items: List[Dict[str, Any]] = []
    for idx, (mean_val, std_val) in enumerate(zip(importances_mean, importances_std)):
        feat = names[idx] if idx < len(names) else f"f_{idx}"
        items.append({
            "feature": str(feat),
            "mean": float(mean_val),
            "std": float(std_val),
        })

    items.sort(key=lambda d: float(d["mean"]), reverse=True)
    return items[:_MAX_FEATURES]
