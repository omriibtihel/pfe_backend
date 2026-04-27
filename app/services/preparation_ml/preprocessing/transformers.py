from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """log(x) transform — clips négatifs à ε pour robustesse à l'inférence."""

    def fit(self, X: Any, y: Any = None) -> "LogTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.log(np.clip(arr, 1e-10, None))

    def inverse_transform(self, X: Any) -> np.ndarray:
        return np.exp(np.asarray(X, dtype=float))


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """√x transform — clips négatifs à 0 pour robustesse à l'inférence."""

    def fit(self, X: Any, y: Any = None) -> "SqrtTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.sqrt(np.clip(arr, 0.0, None))

    def inverse_transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float) ** 2


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
