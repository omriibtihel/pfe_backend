from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level clip-warning buffer
# ---------------------------------------------------------------------------

_clip_warnings: List[Dict] = []


def get_clip_warnings() -> List[Dict]:
    """Return a copy of all clip events recorded since the last clear."""
    return list(_clip_warnings)


def clear_clip_warnings() -> None:
    """Reset the clip-warning buffer."""
    _clip_warnings.clear()


def _record_clip_event(
    transformer_name: str,
    col_name: str,
    n_clipped: int,
    min_observed: float,
    clip_floor: float,
    transform_name: str,
) -> None:
    logger.warning(
        "[%s] Clipping %d negative/zero value(s) in column '%s' before %s transform. "
        "Min value observed: %.6g. These values will be treated as %g.",
        transformer_name, n_clipped, col_name, transform_name, min_observed, clip_floor,
    )
    _clip_warnings.append({
        "transformer": transformer_name,
        "column": col_name,
        "n_clipped": n_clipped,
        "min_observed": min_observed,
        "clip_floor": clip_floor,
        "transform": transform_name,
    })


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

class LogTransformer(BaseEstimator, TransformerMixin):
    """log(x) transform — clips négatifs à ε pour robustesse à l'inférence."""

    def fit(self, X: Any, y: Any = None) -> "LogTransformer":
        if isinstance(X, pd.DataFrame):
            self.column_names_: Optional[List[str]] = [str(c) for c in X.columns]
        else:
            self.column_names_ = None
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        clip_min = 1e-10
        col_names = getattr(self, "column_names_", None)

        if arr.ndim == 1:
            n_neg = int(np.sum(arr < clip_min))
            if n_neg > 0:
                name = col_names[0] if col_names else "col_0"
                _record_clip_event("LogTransformer", name, n_neg, float(arr.min()), clip_min, "log")
        else:
            for i in range(arr.shape[1]):
                col = arr[:, i]
                n_neg = int(np.sum(col < clip_min))
                if n_neg > 0:
                    name = (
                        col_names[i]
                        if col_names and i < len(col_names)
                        else f"col_{i}"
                    )
                    _record_clip_event("LogTransformer", name, n_neg, float(col.min()), clip_min, "log")

        return np.log(np.clip(arr, clip_min, None))

    def inverse_transform(self, X: Any) -> np.ndarray:
        return np.exp(np.asarray(X, dtype=float))


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """√x transform — clips négatifs à 0 pour robustesse à l'inférence."""

    def fit(self, X: Any, y: Any = None) -> "SqrtTransformer":
        if isinstance(X, pd.DataFrame):
            self.column_names_: Optional[List[str]] = [str(c) for c in X.columns]
        else:
            self.column_names_ = None
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        clip_min = 0.0
        col_names = getattr(self, "column_names_", None)

        if arr.ndim == 1:
            n_neg = int(np.sum(arr < clip_min))
            if n_neg > 0:
                name = col_names[0] if col_names else "col_0"
                _record_clip_event("SqrtTransformer", name, n_neg, float(arr.min()), clip_min, "sqrt")
        else:
            for i in range(arr.shape[1]):
                col = arr[:, i]
                n_neg = int(np.sum(col < clip_min))
                if n_neg > 0:
                    name = (
                        col_names[i]
                        if col_names and i < len(col_names)
                        else f"col_{i}"
                    )
                    _record_clip_event("SqrtTransformer", name, n_neg, float(col.min()), clip_min, "sqrt")

        return np.sqrt(np.clip(arr, clip_min, None))

    def inverse_transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) ** 2


def _denullify_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas nullable / extension dtypes with numpy-backed dtypes.

    scikit-learn casts feature blocks to float via ``float(x)`` inside
    ``check_array``. ``float(np.nan)`` is fine but ``float(pd.NA)`` raises
    ``TypeError: float() argument must be a string or a real number, not
    'NAType'``. Pandas >= 3.0 reads CSV/Excel text columns as the nullable
    ``StringDtype`` (carrying ``pd.NA``), and :meth:`ColumnAligner._coerce_dtype`
    emits nullable ``boolean`` / ``string`` columns — both feed ``pd.NA`` into
    the preprocessing ColumnTransformer and crash every fold.

    This converts each column back to the numpy model the rest of the pipeline
    was written against:
      - nullable boolean / Int64 / Float64 → ``float64`` (``pd.NA`` → ``np.nan``)
      - StringDtype / other extension dtypes → ``object`` with ``np.nan``
      - plain ``object`` columns still holding ``pd.NA`` → ``np.nan``
    """
    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        if pd.api.types.is_extension_array_dtype(dtype):
            if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                df[col] = s.astype("float64")
            else:
                df[col] = s.astype(object).where(s.notna(), np.nan)
        elif dtype == object and s.isna().any():
            df[col] = s.where(s.notna(), np.nan)
    return df


class ColumnAligner(BaseEstimator, TransformerMixin):
    """
    Align raw inference/train data to the training feature schema.
    - Missing columns are created with NaN.
    - Extra columns are ignored.
    - Output columns keep the exact training order.
    """

    def __init__(self, feature_names: Iterable[str], dtypes: Optional[Dict[str, str]] = None):
        # Keep constructor parameters untouched for sklearn clone compatibility.
        self.feature_names = feature_names
        self.dtypes = dtypes

    def fit(self, X: Any, y: Any = None) -> "ColumnAligner":
        feature_names = list(self.feature_names or [])
        if not feature_names and isinstance(X, pd.DataFrame):
            feature_names = [str(c) for c in X.columns]
        self.feature_names_ = feature_names
        self.dtypes_ = dict(self.dtypes or {})
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = np.nan

        aligned = df.loc[:, self.feature_names_].copy()

        # Best-effort coercion based on training dtypes for more robust prediction.
        for col, dtype_str in self.dtypes_.items():
            if col not in aligned.columns:
                continue
            self._coerce_dtype(aligned, col, dtype_str)

        # Strip pandas nullable / extension dtypes (pd.NA) before sklearn —
        # otherwise the downstream ColumnTransformer crashes on float(pd.NA).
        aligned = _denullify_for_sklearn(aligned)

        return aligned

    @staticmethod
    def _coerce_dtype(df: pd.DataFrame, col: str, dtype_str: str) -> None:
        dtype_lower = str(dtype_str).lower()
        try:
            if any(token in dtype_lower for token in ("int", "float", "double", "number", "numeric")):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                return
            if "bool" in dtype_lower:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    mapped = df[col].astype("string").str.strip().str.lower().map(
                        {
                            "1": True,
                            "true": True,
                            "t": True,
                            "yes": True,
                            "y": True,
                            "on": True,
                            "0": False,
                            "false": False,
                            "f": False,
                            "no": False,
                            "n": False,
                            "off": False,
                        }
                    )
                    df[col] = mapped
                df[col] = df[col].astype("boolean")
                return

            # string/category-like
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype("string")
        except Exception:
            # Never fail alignment because a coercion could not be applied.
            return
