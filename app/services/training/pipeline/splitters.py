from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass(frozen=True)
class HoldoutSplit:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: Optional[pd.DataFrame]
    y_val: Optional[np.ndarray]
    X_test: pd.DataFrame
    y_test: np.ndarray
    warnings: tuple[str, ...] = ()
    attempts: int = 1
    random_state_used: int = 42


def _count_by_class(y: np.ndarray | None) -> dict[str, int]:
    if y is None:
        return {}
    vals, counts = np.unique(y, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def _is_binary_classification(task_type: str, y: np.ndarray) -> tuple[bool, np.ndarray]:
    if task_type != "classification":
        return False, np.asarray([])
    vals = np.unique(y)
    return bool(len(vals) == 2), vals


def _contains_all_classes(y: np.ndarray | None, classes: np.ndarray) -> bool:
    if y is None:
        return True
    observed = np.unique(y)
    return bool(len(observed) == len(classes) and np.all(np.isin(classes, observed)))


def _safe_stratify(y: np.ndarray, task_type: str):
    if task_type != "classification":
        return None
    try:
        vals, counts = np.unique(y, return_counts=True)
        # Stratify only if at least two classes with enough samples.
        if len(vals) < 2 or counts.min() < 2:
            return None
        return y
    except Exception:
        return None


def make_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task_type: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> HoldoutSplit:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        # Accept percentages as well.
        if abs((train_ratio + val_ratio + test_ratio) - 100.0) < 1e-6:
            train_ratio, val_ratio, test_ratio = train_ratio / 100.0, val_ratio / 100.0, test_ratio / 100.0
        else:
            raise RuntimeError("train/val/test ratios must sum to 1.0 (or 100)")

    n = len(X)
    if n < 10:
        raise RuntimeError(f"Not enough rows for holdout split (rows={n}).")

    is_binary_cls, global_classes = _is_binary_classification(task_type, np.asarray(y))
    max_attempts = 8
    split_warnings: list[str] = []

    def _split_once(seed: int, *, use_validation: bool) -> HoldoutSplit:
        stratify_main = y if is_binary_cls else _safe_stratify(y, task_type)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=max(0.01, (1.0 - train_ratio)),
            random_state=seed,
            stratify=stratify_main,
        )

        if not use_validation:
            return HoldoutSplit(
                X_train,
                y_train,
                None,
                None,
                X_temp,
                y_temp,
                warnings=(),
                attempts=1,
                random_state_used=seed,
            )

        denom = (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 1.0
        val_prop_in_temp = val_ratio / denom
        if is_binary_cls:
            vals_temp, counts_temp = np.unique(y_temp, return_counts=True)
            stratify_temp = y_temp if len(vals_temp) == 2 and counts_temp.min() >= 2 else None
        else:
            stratify_temp = _safe_stratify(y_temp, task_type)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=max(0.01, (1.0 - val_prop_in_temp)),
            random_state=seed,
            stratify=stratify_temp,
        )
        return HoldoutSplit(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            warnings=(),
            attempts=1,
            random_state_used=seed,
        )

    use_validation = bool(val_ratio > 0)
    for attempt in range(max_attempts):
        seed = int(random_state + attempt)
        try:
            split = _split_once(seed, use_validation=use_validation)
        except Exception as exc:
            split_warnings.append(
                f"Holdout split attempt {attempt + 1}/{max_attempts} failed (seed={seed}): {exc}"
            )
            continue

        if not is_binary_cls:
            return HoldoutSplit(
                split.X_train,
                split.y_train,
                split.X_val,
                split.y_val,
                split.X_test,
                split.y_test,
                warnings=tuple(split_warnings),
                attempts=attempt + 1,
                random_state_used=seed,
            )

        train_ok = _contains_all_classes(split.y_train, global_classes)
        val_ok = _contains_all_classes(split.y_val, global_classes)
        test_ok = _contains_all_classes(split.y_test, global_classes)
        if train_ok and val_ok and test_ok:
            return HoldoutSplit(
                split.X_train,
                split.y_train,
                split.X_val,
                split.y_val,
                split.X_test,
                split.y_test,
                warnings=tuple(split_warnings),
                attempts=attempt + 1,
                random_state_used=seed,
            )

        split_warnings.append(
            "Holdout split attempt "
            f"{attempt + 1}/{max_attempts} rejected (seed={seed}): "
            f"train={_count_by_class(split.y_train)}, "
            f"val={_count_by_class(split.y_val)}, "
            f"test={_count_by_class(split.y_test)}."
        )

    if use_validation and is_binary_cls:
        split_warnings.append(
            "Could not preserve both classes in train/val/test after retries; "
            "falling back to train/test split (validation disabled)."
        )
        for attempt in range(max_attempts):
            seed = int(random_state + attempt)
            try:
                split = _split_once(seed, use_validation=False)
            except Exception as exc:
                split_warnings.append(
                    f"Fallback split attempt {attempt + 1}/{max_attempts} failed (seed={seed}): {exc}"
                )
                continue

            train_ok = _contains_all_classes(split.y_train, global_classes)
            test_ok = _contains_all_classes(split.y_test, global_classes)
            if train_ok and test_ok:
                return HoldoutSplit(
                    split.X_train,
                    split.y_train,
                    None,
                    None,
                    split.X_test,
                    split.y_test,
                    warnings=tuple(split_warnings),
                    attempts=max_attempts + attempt + 1,
                    random_state_used=seed,
                )
            split_warnings.append(
                "Fallback split attempt "
                f"{attempt + 1}/{max_attempts} rejected (seed={seed}): "
                f"train={_count_by_class(split.y_train)}, "
                f"test={_count_by_class(split.y_test)}."
            )

    raise RuntimeError(
        "Unable to produce a robust holdout split with both classes present in required splits."
    )


def iter_kfold_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    split_method: str = "kfold",
    k_folds: int,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) pairs for k-fold cross-validation.

    split_method:
      - "kfold"            → plain KFold (works for classification and regression)
      - "stratified_kfold" → StratifiedKFold (classification only; keeps class ratios per fold)

    All data-leakage prevention (preprocessing fit, resampling) is the caller's
    responsibility: only fit on the train_idx slice, transform both.
    """
    if k_folds < 2:
        raise RuntimeError("kFolds must be >= 2.")

    n_samples = len(X)
    if k_folds > n_samples:
        raise RuntimeError(
            f"kFolds={k_folds} dépasse le nombre d'échantillons ({n_samples}). "
            f"Réduire kFolds à <= {n_samples}."
        )

    rng_kwarg: dict = {"random_state": random_state} if shuffle else {}

    if split_method == "stratified_kfold":
        vals, counts = np.unique(y, return_counts=True)
        if len(vals) < 2:
            raise RuntimeError(
                "stratified_kfold nécessite au moins 2 classes dans la cible."
            )
        min_class_count = int(counts.min())
        if min_class_count < k_folds:
            raise RuntimeError(
                f"stratified_kfold: la classe minoritaire n'a que {min_class_count} échantillon(s) "
                f"pour {k_folds} folds. Réduire kFolds à <= {min_class_count}."
            )
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=shuffle, **rng_kwarg)
    else:
        # Plain KFold — works for classification and regression alike.
        splitter = KFold(n_splits=k_folds, shuffle=shuffle, **rng_kwarg)

    yield from splitter.split(X, y)


def validate_kfold_config(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    split_method: str,
    k_folds: int,
    task_type: str,
) -> list[str]:
    """
    Return a list of human-readable error strings for the given CV config.
    Returns [] if the config is valid.
    """
    errors: list[str] = []
    n_samples = len(X)

    if k_folds < 2:
        errors.append("kFolds doit être >= 2.")
    elif k_folds > n_samples:
        errors.append(
            f"kFolds={k_folds} dépasse le nombre d'échantillons ({n_samples}). "
            f"Réduire kFolds à <= {n_samples}."
        )

    if split_method == "stratified_kfold":
        if task_type != "classification":
            errors.append(
                "stratified_kfold est réservé à la classification. "
                "Pour la régression, utiliser splitMethod='kfold'."
            )
        elif k_folds >= 2:
            vals, counts = np.unique(y, return_counts=True)
            if len(vals) < 2:
                errors.append(
                    "stratified_kfold nécessite au moins 2 classes dans la cible."
                )
            else:
                min_class_count = int(counts.min())
                min_class_label = str(vals[int(np.argmin(counts))])
                if min_class_count < k_folds:
                    errors.append(
                        f"stratified_kfold: la classe '{min_class_label}' n'a que "
                        f"{min_class_count} échantillon(s) pour {k_folds} folds. "
                        f"Réduire kFolds à <= {min_class_count} ou choisir kfold simple."
                    )

    return errors
