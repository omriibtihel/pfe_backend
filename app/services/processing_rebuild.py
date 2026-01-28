from __future__ import annotations

from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from app.crud import processing as crud_processing
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.df import read_df
from app.api.utils.processing_df import save_processed_df


def _ensure_cols_exist(df: pd.DataFrame, cols: list[str]) -> None:
    existing = set(map(str, df.columns))
    missing = [c for c in cols if c not in existing]
    if missing:
        raise ValueError(f"Colonnes introuvables: {missing}")


def _target_cols(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    # Si cols vide -> toutes les colonnes (Dataiku-like)
    if not cols:
        return [str(c) for c in df.columns]
    return [str(c) for c in cols]


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def _fill_missing(
    df: pd.DataFrame,
    cols: list[str],
    strategy: str,
    constant: object | None = None,
) -> pd.DataFrame:
    _ensure_cols_exist(df, cols)

    for c in cols:
        if strategy == "mean":
            if not _is_numeric(df, c):
                raise ValueError(f"'{c}' n'est pas numérique (mean impossible).")
            v = df[c].mean()
            df[c] = df[c].fillna(v)

        elif strategy == "median":
            if not _is_numeric(df, c):
                raise ValueError(f"'{c}' n'est pas numérique (median impossible).")
            v = df[c].median()
            df[c] = df[c].fillna(v)

        elif strategy == "mode":
            m = df[c].mode(dropna=True)
            v = m.iloc[0] if len(m) else None
            df[c] = df[c].fillna(v)

        elif strategy == "constant":
            df[c] = df[c].fillna(constant)

        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")

    return df


def _drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    _ensure_cols_exist(df, cols)
    return df.drop(columns=cols)


def _drop_duplicates(df: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
    if cols:
        _ensure_cols_exist(df, cols)
        return df.drop_duplicates(subset=cols)
    return df.drop_duplicates()


def _normalize(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    _ensure_cols_exist(df, cols)

    for c in cols:
        if not _is_numeric(df, c):
            raise ValueError(f"'{c}' n'est pas numérique (normalization impossible).")

        x = df[c].astype(float)

        if method == "minmax":
            mn, mx = np.nanmin(x), np.nanmax(x)
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            df[c] = (x - mn) / denom

        elif method == "zscore":
            mu, sd = np.nanmean(x), np.nanstd(x)
            denom = sd if sd != 0 else 1.0
            df[c] = (x - mu) / denom

        elif method == "robust":
            q1 = np.nanpercentile(x, 25)
            q3 = np.nanpercentile(x, 75)
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
            df[c] = (x - q1) / iqr

        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")

    return df


def _encode(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    _ensure_cols_exist(df, cols)

    if method == "label":
        for c in cols:
            s = df[c].astype("string")
            uniq = sorted(set([v for v in s.dropna().unique()]))
            mapping = {v: i for i, v in enumerate(uniq)}
            df[c] = s.map(mapping).astype("Int64")
        return df

    if method == "onehot":
        dummies = pd.get_dummies(df[cols].astype("string"), prefix=cols, dummy_na=True)
        df = df.drop(columns=cols)
        df = pd.concat([df, dummies], axis=1)
        return df

    raise ValueError(f"Méthode d'encodage inconnue: {method}")


def _legacy_infer_action(
    op_type: str,
    action: str | None,
    description: str | None,
    params: dict,
) -> tuple[str | None, dict]:
    """
    Compat rétro:
    - Certains anciens ops avaient params={} (donc pas d'action)
    - On essaye d'inférer l'action au lieu de lever une erreur.
    """
    desc = (description or "").lower()
    p = dict(params or {})

    if op_type == "cleaning" and not action:
        if any(k in desc for k in ["supprim", "drop"]):
            action = "drop_columns"
        elif "doubl" in desc or "duplicate" in desc:
            action = "drop_duplicates"
        else:
            action = "fill_missing"

        # compat ancienne clé fill_value => constant
        if "fill_value" in p and "strategy" not in p and "constant" not in p:
            p["strategy"] = "constant"
            p["constant"] = p.get("fill_value")

    if op_type == "other" and not action:
        action = "drop_duplicates"

    return action, p


def _apply_one(
    df: pd.DataFrame,
    op_type: str,
    cols: list[str],
    params: dict,
    *,
    description: str | None = None,
) -> pd.DataFrame:
    action = (params or {}).get("action")
    action, params = _legacy_infer_action(op_type, action, description, params)

    if op_type == "cleaning":
        if action == "drop_columns":
            return _drop_columns(df, cols)

        if action == "fill_missing":
            strategy = (params or {}).get("strategy") or (params or {}).get("method") or "mode"
            constant = (params or {}).get("constant", None)
            if constant is None and "fill_value" in (params or {}):
                constant = (params or {}).get("fill_value")
            return _fill_missing(df, cols, strategy=strategy, constant=constant)

        if action == "drop_duplicates":
            return _drop_duplicates(df, cols if cols else None)

        raise ValueError("Cleaning: action manquante ou inconnue.")

    if op_type == "imputation":
        method = (params or {}).get("method") or (params or {}).get("strategy") or "mode"
        constant = (params or {}).get("constant", None)
        if constant is None and "fill_value" in (params or {}):
            constant = (params or {}).get("fill_value")
        return _fill_missing(df, cols, strategy=method, constant=constant)

    if op_type == "normalization":
        method = (params or {}).get("method", "zscore")
        return _normalize(df, cols, method=method)

    if op_type == "encoding":
        method = (params or {}).get("method", "onehot")
        return _encode(df, cols, method=method)

    if op_type == "other":
        if action == "drop_duplicates":
            return _drop_duplicates(df, cols if cols else None)
        raise ValueError("Other: action manquante ou inconnue.")

    raise ValueError(f"Type d'opération inconnu: {op_type}")


def rebuild_processed(db: Session, project_id: int, dataset_id: int) -> None:
    """
    Recalcule le dataset traité à partir du fichier original,
    en rejouant toutes les opérations dans l’ordre.
    """
    ds = get_dataset_or_404(db, project_id, dataset_id)

    # ✅ Toujours repartir du brut (déterministe)
    df = read_df(ds.file_path)

    ops = crud_processing.list_operations(db, project_id, dataset_id)

    for op in ops:
        cols = _target_cols(df, op.columns)
        _ensure_cols_exist(df, cols)

        df = _apply_one(
            df,
            op.op_type,
            cols,
            op.params or {},
            description=getattr(op, "description", None),
        )

    save_processed_df(df, ds.file_path, dataset_id)
