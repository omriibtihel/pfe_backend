from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import threading
from typing import Any, Generic, Hashable, TypeVar

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .config import TrainingConfig
from .preprocessing import build_preprocessor
from .splitters import make_holdout_split

_CACHE_SPLITS_MAX = 64
_CACHE_PREP_MAX = 32
_CACHE_PREVIEW_MAX = 128
_SPLIT_RANDOM_STATE = 42


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _LRUCache(Generic[K, V]):
    def __init__(self, max_size: int):
        self._max_size = max(1, int(max_size))
        self._values: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> tuple[V | None, bool]:
        with self._lock:
            if key not in self._values:
                return None, False
            value = self._values.pop(key)
            self._values[key] = value
            return value, True

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._values:
                self._values.pop(key)
            self._values[key] = value
            while len(self._values) > self._max_size:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


_SPLIT_CACHE: _LRUCache[tuple[Any, ...], "SplitIndices"] = _LRUCache(_CACHE_SPLITS_MAX)
_PREPROCESSOR_CACHE: _LRUCache[tuple[Any, ...], Any] = _LRUCache(_CACHE_PREP_MAX)
_PREVIEW_CACHE: _LRUCache[tuple[Any, ...], "PreviewTable"] = _LRUCache(_CACHE_PREVIEW_MAX)


@dataclass(frozen=True)
class SplitIndices:
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    test_idx: tuple[int, ...]


@dataclass(frozen=True)
class PreviewOptions:
    subset: str
    mode: str
    n: int
    seed: int


@dataclass(frozen=True)
class PreviewTable:
    columns: list[str]
    rows: list[list[Any]]


class PreviewValidationError(RuntimeError):
    def __init__(self, message: str, *, code: str, column: str | None = None):
        super().__init__(message)
        self.code = str(code)
        self.column = str(column) if column else None
        self.error_details = [
            {
                "column": self.column,
                "severity": "error",
                "code": self.code,
                "message": str(message),
            }
        ]


def clear_validation_preview_caches() -> None:
    _SPLIT_CACHE.clear()
    _PREPROCESSOR_CACHE.clear()
    _PREVIEW_CACHE.clear()


def build_validation_preview(
    payload: dict[str, Any],
    df: pd.DataFrame,
    *,
    dataset_version_id: int,
) -> dict[str, Any]:
    cfg = TrainingConfig.from_front(payload)
    options = _normalize_preview_options(payload.get("preview"))

    if cfg.target_column not in df.columns:
        raise PreviewValidationError(
            f"targetColumn '{cfg.target_column}' was not found in dataset.",
            code="preview_target_missing",
            column=cfg.target_column,
        )

    df2 = df[df[cfg.target_column].notna()].copy()
    if df2.empty:
        raise PreviewValidationError(
            "Preview unavailable: target column is empty after dropping NaN rows.",
            code="preview_target_empty",
            column=cfg.target_column,
        )

    X = df2.drop(columns=[cfg.target_column])
    y = df2[cfg.target_column].to_numpy()

    split_key = _build_split_key(cfg, dataset_version_id=dataset_version_id)
    split_indices, _ = _get_or_build_split(split_key=split_key, cfg=cfg, X=X, y=y)

    train_size = len(split_indices.train_idx)
    val_size = len(split_indices.val_idx)
    test_size = len(split_indices.test_idx)

    subset_idx = _select_subset_indices(split_indices, options.subset)
    if len(subset_idx) == 0:
        raise PreviewValidationError(
            f"Requested subset '{options.subset}' is unavailable for the current split ratios.",
            code="preview_subset_unavailable",
        )

    prep_key = _build_preprocessor_key(split_key=split_key, cfg=cfg)
    fitted_preprocessor, _ = _get_or_build_preprocessor(prep_key=prep_key, cfg=cfg, X=X, split_indices=split_indices)

    preview_key = _build_preview_key(prep_key=prep_key, options=options)
    preview_table, from_cache = _PREVIEW_CACHE.get(preview_key)
    if not from_cache or preview_table is None:
        X_subset = X.loc[list(subset_idx)]
        X_sample = _sample_subset(X_subset, options)
        transformed = fitted_preprocessor.transform(X_sample)
        rows = _to_json_rows(transformed)
        columns = _resolve_feature_names(fitted_preprocessor, expected_width=len(rows[0]) if rows else None)
        if rows and len(columns) != len(rows[0]):
            columns = [f"feature_{idx}" for idx in range(len(rows[0]))]
        preview_table = PreviewTable(columns=columns, rows=rows)
        _PREVIEW_CACHE.set(preview_key, preview_table)

    return {
        "previewTransformed": {
            "columns": preview_table.columns,
            "rows": preview_table.rows,
        },
        "previewMeta": {
            "fittedOn": "train",
            "subset": options.subset,
            "mode": options.mode,
            "n": min(int(options.n), int(len(subset_idx))),
            "splitSeed": _SPLIT_RANDOM_STATE,
            "fromCache": bool(from_cache),
            "trainSize": int(train_size),
            "valSize": int(val_size),
            "testSize": int(test_size),
        },
    }


def _normalize_preview_options(raw: Any) -> PreviewOptions:
    source = raw if isinstance(raw, dict) else {}
    subset = str(source.get("subset", "train")).strip().lower()
    if subset not in {"train", "val", "test"}:
        subset = "train"

    mode = str(source.get("mode", "head")).strip().lower()
    if mode not in {"head", "random"}:
        mode = "head"

    n = _safe_int(source.get("n"), default=100)
    n = max(1, min(500, n))

    seed = _safe_int(source.get("seed"), default=42)
    return PreviewOptions(subset=subset, mode=mode, n=n, seed=seed)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _build_split_key(cfg: TrainingConfig, *, dataset_version_id: int) -> tuple[Any, ...]:
    return (
        int(dataset_version_id),
        str(cfg.split_method),
        round(float(cfg.train_ratio), 8),
        round(float(cfg.val_ratio), 8),
        round(float(cfg.test_ratio), 8),
        int(cfg.k_folds),
        int(_SPLIT_RANDOM_STATE),
        str(cfg.target_column),
        str(cfg.task_type),
    )


def _build_preprocessor_key(*, split_key: tuple[Any, ...], cfg: TrainingConfig) -> tuple[Any, ...]:
    payload = cfg.preprocessing.as_dict()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return split_key + (payload_hash,)


def _build_preview_key(*, prep_key: tuple[Any, ...], options: PreviewOptions) -> tuple[Any, ...]:
    return prep_key + (options.subset, options.mode, int(options.n), int(options.seed))


def _build_cv_preview_split(
    *,
    cfg: TrainingConfig,
    X: pd.DataFrame,
    y: np.ndarray,
) -> SplitIndices:
    """
    Build a preview-only split for kfold / stratified_kfold.

    If testRatio > 0, split off a holdout test set so the user can preview
    the test subset.  Otherwise the entire dataset is treated as train.
    There is never a val subset in CV preview mode.
    """
    idx_array = X.index.to_numpy()
    test_ratio = float(cfg.test_ratio)

    if test_ratio > 0:
        stratify_y: np.ndarray | None = None
        if cfg.task_type == "classification":
            vals, counts = np.unique(y, return_counts=True)
            if len(vals) >= 2 and int(counts.min()) >= 2:
                stratify_y = y

        try:
            train_arr, test_arr = train_test_split(
                idx_array,
                test_size=test_ratio,
                random_state=_SPLIT_RANDOM_STATE,
                stratify=stratify_y,
            )
        except Exception:
            # If stratified split fails (e.g. too few samples), fall back to plain split.
            train_arr, test_arr = train_test_split(
                idx_array,
                test_size=test_ratio,
                random_state=_SPLIT_RANDOM_STATE,
            )

        return SplitIndices(
            train_idx=tuple(int(v) for v in train_arr),
            val_idx=tuple(),
            test_idx=tuple(int(v) for v in test_arr),
        )

    # testRatio == 0 → use all data as train for preprocessing preview
    return SplitIndices(
        train_idx=tuple(int(v) for v in idx_array),
        val_idx=tuple(),
        test_idx=tuple(),
    )


def _get_or_build_split(
    *,
    split_key: tuple[Any, ...],
    cfg: TrainingConfig,
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[SplitIndices, bool]:
    cached, hit = _SPLIT_CACHE.get(split_key)
    if hit and cached is not None:
        return cached, True

    if str(cfg.split_method) in ("kfold", "stratified_kfold"):
        built = _build_cv_preview_split(cfg=cfg, X=X, y=y)
        _SPLIT_CACHE.set(split_key, built)
        return built, False

    split = make_holdout_split(
        X,
        y,
        task_type=cfg.task_type,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_state=_SPLIT_RANDOM_STATE,
    )
    built = SplitIndices(
        train_idx=tuple(int(v) for v in split.X_train.index.to_numpy()),
        val_idx=tuple(int(v) for v in split.X_val.index.to_numpy()) if split.X_val is not None else tuple(),
        test_idx=tuple(int(v) for v in split.X_test.index.to_numpy()),
    )
    _SPLIT_CACHE.set(split_key, built)
    return built, False


def _select_subset_indices(split_indices: SplitIndices, subset: str) -> tuple[int, ...]:
    if subset == "train":
        return split_indices.train_idx
    if subset == "val":
        return split_indices.val_idx
    return split_indices.test_idx


def _get_or_build_preprocessor(
    *,
    prep_key: tuple[Any, ...],
    cfg: TrainingConfig,
    X: pd.DataFrame,
    split_indices: SplitIndices,
) -> tuple[Any, bool]:
    cached, hit = _PREPROCESSOR_CACHE.get(prep_key)
    if hit and cached is not None:
        return cached, True

    train_idx = list(split_indices.train_idx)
    if not train_idx:
        raise PreviewValidationError(
            "Preview unavailable: train subset is empty.",
            code="preview_train_empty",
        )

    X_train = X.loc[train_idx]
    try:
        spec = build_preprocessor(X_train, cfg.preprocessing)
        fitted = spec.preprocessor.fit(X_train)
    except Exception as exc:
        raise PreviewValidationError(
            f"Preprocessing is invalid for preview: {str(exc)}",
            code="preview_preprocessing_invalid",
        ) from exc

    _PREPROCESSOR_CACHE.set(prep_key, fitted)
    return fitted, False


def _sample_subset(X_subset: pd.DataFrame, options: PreviewOptions) -> pd.DataFrame:
    if X_subset is None or len(X_subset) == 0:
        return X_subset.iloc[0:0]

    sample_n = min(int(options.n), int(len(X_subset)))
    if options.mode == "random":
        return X_subset.sample(n=sample_n, random_state=int(options.seed))
    return X_subset.head(sample_n)


def _resolve_feature_names(fitted_preprocessor: Any, *, expected_width: int | None = None) -> list[str]:
    names: list[str] = []

    if hasattr(fitted_preprocessor, "get_feature_names_out"):
        try:
            raw = fitted_preprocessor.get_feature_names_out()
            names = [str(item) for item in raw]
        except Exception:
            names = []

    if not names:
        names = _fallback_feature_names(fitted_preprocessor)

    if expected_width is not None and expected_width >= 0 and len(names) != expected_width:
        names = [f"feature_{idx}" for idx in range(expected_width)]

    return names


def _fallback_feature_names(fitted_preprocessor: Any) -> list[str]:
    out: list[str] = []
    transformers = list(getattr(fitted_preprocessor, "transformers_", []))

    for transformer_item in transformers:
        if not isinstance(transformer_item, tuple) or len(transformer_item) < 3:
            continue
        name, transformer, cols = transformer_item[0], transformer_item[1], transformer_item[2]
        if name == "remainder" or transformer == "drop":
            continue

        cols_list: list[str]
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            cols_list = [str(col) for col in cols]
        else:
            cols_list = [str(cols)]

        if transformer == "passthrough":
            out.extend([f"{name}__{col}" for col in cols_list])
            continue

        candidate_names: list[str] = []
        if hasattr(transformer, "get_feature_names_out"):
            try:
                raw_names = transformer.get_feature_names_out(cols_list)
                candidate_names = [str(v) for v in raw_names]
            except Exception:
                try:
                    raw_names = transformer.get_feature_names_out()
                    candidate_names = [str(v) for v in raw_names]
                except Exception:
                    candidate_names = []

        if candidate_names:
            out.extend(
                [
                    value if value.startswith(f"{name}__") else f"{name}__{value}"
                    for value in candidate_names
                ]
            )
            continue

        out.extend([f"{name}__{col}" for col in cols_list])

    return out


def _to_json_rows(transformed: Any) -> list[list[Any]]:
    if transformed is None:
        return []

    matrix = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    rows: list[list[Any]] = []
    for row in arr:
        rows.append([_to_json_cell(value) for value in row])
    return rows


def _to_json_cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        cast = float(value)
        return cast if math.isfinite(cast) else None
    if isinstance(value, (float, np.floating)):
        cast = float(value)
        return cast if math.isfinite(cast) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)
