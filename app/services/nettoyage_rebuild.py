# app/services/nettoyage_rebuild.py
from __future__ import annotations

from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from app.crud import nettoyage as crud_nettoyage
from app.api.utils.datasets import get_dataset_or_404
from app.api.utils.df import read_df
from app.api.utils.nettoyage_df import save_processed_df

# ---------------------------------------------------------------------
# LIMITED to "safe cleaning" operations.
# NO stat-based imputation / encoding / scaling here (avoid leakage).
# Those belong to the training pipeline (fit on TRAIN only).
# ---------------------------------------------------------------------

ALLOWED_CLEANING_ACTIONS = {
    "drop_columns",
    "drop_duplicates",
    "drop_empty_rows",
    "rename_columns",
    "strip_whitespace",
    "substitute_values",  # ✅ added
}

STRICT_REBUILD = False


def _split_existing_missing(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    existing = set(map(str, df.columns))
    cols = [str(c) for c in (cols or [])]
    present = [c for c in cols if c in existing]
    missing = [c for c in cols if c not in existing]
    return present, missing


def _drop_columns(df: pd.DataFrame, cols: list[str], *, strict: bool = STRICT_REBUILD) -> pd.DataFrame:
    if not cols:
        raise ValueError("drop_columns: sélectionne au moins une colonne.")
    present, missing = _split_existing_missing(df, cols)
    if missing and strict:
        raise ValueError(f"Colonnes introuvables: {missing}")
    if not present:
        return df
    return df.drop(columns=present)


def _drop_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
    *,
    strict: bool = STRICT_REBUILD,
) -> pd.DataFrame:
    if subset:
        present, missing = _split_existing_missing(df, subset)
        if missing and strict:
            raise ValueError(f"Colonnes introuvables: {missing}")
        return df.drop_duplicates(subset=present if present else None, keep=keep)
    return df.drop_duplicates(keep=keep)


def _drop_empty_rows(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    strict: bool = STRICT_REBUILD,
) -> pd.DataFrame:
    if cols:
        present, missing = _split_existing_missing(df, cols)
        if missing and strict:
            raise ValueError(f"Colonnes introuvables: {missing}")
        work = df[present].copy() if present else df.copy()
    else:
        work = df.copy()

    for c in work.columns:
        s = work[c]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            work[c] = s.astype("string").str.strip().replace("", pd.NA)

    mask_empty = work.isna().all(axis=1)
    return df.loc[~mask_empty].copy()


def _rename_columns(
    df: pd.DataFrame,
    mapping: dict,
    *,
    strict: bool = STRICT_REBUILD,
) -> tuple[pd.DataFrame, dict]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("rename_columns nécessite 'mapping' (dict) non vide.")

    old_keys = [str(k) for k in mapping.keys()]
    present, missing = _split_existing_missing(df, old_keys)
    if missing and strict:
        raise ValueError(f"Colonnes introuvables: {missing}")

    applied = {str(k): str(mapping[k]) for k in present}
    if not applied:
        return df, {}

    new_names = list(applied.values())
    if len(set(new_names)) != len(new_names):
        raise ValueError("rename_columns: deux colonnes ne peuvent pas avoir le même nouveau nom.")

    existing_cols = set(map(str, df.columns))
    old_set = set(applied.keys())
    for old, new in applied.items():
        if new in existing_cols and new not in old_set:
            raise ValueError(f"rename_columns: collision: '{new}' existe déjà (renommage '{old}' -> '{new}').")

    df2 = df.rename(columns=applied)
    if len(set(map(str, df2.columns))) != len(df2.columns):
        raise ValueError("rename_columns: collision de noms de colonnes après renommage.")

    return df2, applied


def _strip_whitespace(
    df: pd.DataFrame,
    cols: list[str] | None,
    *,
    strict: bool = STRICT_REBUILD,
) -> pd.DataFrame:
    df2 = df.copy()

    if not cols:
        cols = []
        for c in df2.columns:
            s = df2[c]
            if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                cols.append(str(c))

    if not cols:
        return df2

    present, missing = _split_existing_missing(df2, cols)
    if missing and strict:
        raise ValueError(f"Colonnes introuvables: {missing}")
    if not present:
        return df2

    for c in present:
        s = df2[c]
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            df2[c] = s.astype("string").str.strip()
    return df2


# -----------------------------
# ✅ SUBSTITUTE VALUES (exact)
# -----------------------------
def _coerce_scalar_for_series(s: pd.Series, value):
    """Try to coerce 'value' to the series dtype when possible."""
    if value is None:
        return None

    # numeric column -> try numeric
    if pd.api.types.is_numeric_dtype(s):
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            return float(value) if pd.api.types.is_float_dtype(s) else int(value)
        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return value
            num = pd.to_numeric([v], errors="coerce")[0]
            if pd.notna(num):
                # keep int if series is int-like
                if pd.api.types.is_integer_dtype(s):
                    try:
                        return int(num)
                    except Exception:
                        return num
                return float(num)
        return value

    # datetime column -> try parse
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            dt = pd.to_datetime([value], errors="coerce")[0]
            return dt if pd.notna(dt) else value
        except Exception:
            return value

    # bool column
    if pd.api.types.is_bool_dtype(s):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
        return value

    return value


def _substitute_values(
    df: pd.DataFrame,
    col: str,
    *,
    from_value=None,
    to_value=None,
    case_sensitive: bool = True,
    treat_from_as_null: bool = False,
    treat_to_as_null: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if col not in df.columns:
        raise ValueError(f"substitute_values: colonne introuvable: {col}")

    before_s = df[col]
    df2 = df.copy()

    # build mask
    if treat_from_as_null or from_value is None:
        mask = before_s.isna()
    else:
        fv = _coerce_scalar_for_series(before_s, from_value)

        # string/object compare
        if pd.api.types.is_object_dtype(before_s) or pd.api.types.is_string_dtype(before_s):
            s_str = before_s.astype("string")
            fv_str = str(from_value)
            if case_sensitive:
                mask = s_str == fv_str
            else:
                mask = s_str.str.lower() == fv_str.lower()
        else:
            mask = before_s == fv

    changed = int(mask.sum())

    # replacement value
    if treat_to_as_null or to_value is None:
        rep = pd.NA
    else:
        rep = _coerce_scalar_for_series(before_s, to_value)

        # if numeric column but replacement isn't numeric -> make column object
        if pd.api.types.is_numeric_dtype(before_s):
            if isinstance(rep, str):
                v = rep.strip()
                num = pd.to_numeric([v], errors="coerce")[0]
                if pd.isna(num):  # not numeric string
                    df2[col] = df2[col].astype("object")
            elif rep is not None and not isinstance(rep, (int, float, np.integer, np.floating)):
                df2[col] = df2[col].astype("object")

    missing_before = int(before_s.isna().sum())

    # apply
    df2.loc[mask, col] = rep

    missing_after = int(df2[col].isna().sum())

    effect = {
        "per_column": {
            col: {
                "changed_count": changed,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "filled": max(0, missing_before - missing_after),
            }
        }
    }
    return df2, effect


# -----------------------------
# Effects
# -----------------------------
def compute_operation_effect(before: pd.DataFrame, after: pd.DataFrame, params: dict) -> dict:
    before_cols = list(map(str, before.columns))
    after_cols = list(map(str, after.columns))
    added = [c for c in after_cols if c not in set(before_cols)]
    removed = [c for c in before_cols if c not in set(after_cols)]

    eff: dict = {
        "before_shape": {"rows": int(before.shape[0]), "cols": int(before.shape[1])},
        "after_shape": {"rows": int(after.shape[0]), "cols": int(after.shape[1])},
        "columns_added": added,
        "columns_removed": removed,
    }

    if after.shape[0] != before.shape[0]:
        delta = int(after.shape[0] - before.shape[0])
        eff["rows_delta"] = delta
        if delta < 0:
            eff["rows_removed"] = int(-delta)

    if (params or {}).get("action") == "rename_columns":
        eff["rename_mapping"] = (params or {}).get("mapping", {})

    return eff


def _legacy_infer_action(description: str | None) -> str | None:
    desc = (description or "").lower()
    if any(k in desc for k in ["rename", "renomm"]):
        return "rename_columns"
    if any(k in desc for k in ["doubl", "duplicate"]):
        return "drop_duplicates"
    if any(k in desc for k in ["vide", "empty", "blank"]):
        return "drop_empty_rows"
    if any(k in desc for k in ["strip", "trim", "espaces"]):
        return "strip_whitespace"
    if any(k in desc for k in ["substitut", "replace"]):
        return "substitute_values"
    if any(k in desc for k in ["supprim", "drop"]):
        return "drop_columns"
    return None


def _update_alias_map(alias: dict[str, str], applied_rename: dict[str, str]) -> None:
    if not applied_rename:
        return

    for old, new in applied_rename.items():
        old = str(old)
        new = str(new)

        for k, v in list(alias.items()):
            if v == old:
                alias[k] = new

        alias[old] = new


def _remap_cols(cols: list[str], alias: dict[str, str]) -> list[str]:
    out: list[str] = []
    for c in (cols or []):
        cc = str(c)
        out.append(alias.get(cc, cc))
    return out


def _apply_one(
    df: pd.DataFrame,
    op_type: str,
    cols: list[str],
    params: dict,
    alias_map: dict[str, str],
    *,
    description: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if op_type != "cleaning":
        raise ValueError(
            f"Opération '{op_type}' non supportée. "
            "Ce module est limité au nettoyage (no leakage)."
        )

    params = params or {}
    action = params.get("action") or _legacy_infer_action(description)
    if action not in ALLOWED_CLEANING_ACTIONS:
        raise ValueError(f"Cleaning: action inconnue: {action}. Allowed={sorted(ALLOWED_CLEANING_ACTIONS)}")

    cols = _remap_cols([str(c) for c in (cols or [])], alias_map)
    extra: dict = {}

    if action == "drop_columns":
        present, missing = _split_existing_missing(df, cols)
        if missing:
            extra["ignored_missing_cols"] = missing
        return _drop_columns(df, cols, strict=STRICT_REBUILD), extra

    if action == "drop_duplicates":
        keep = str(params.get("keep", "first"))
        subset = cols if cols else None
        present, missing = _split_existing_missing(df, subset or [])
        if subset and missing:
            extra["ignored_missing_cols"] = missing
        return _drop_duplicates(df, subset=subset, keep=keep, strict=STRICT_REBUILD), extra

    if action == "drop_empty_rows":
        subset = cols if cols else None
        if subset:
            present, missing = _split_existing_missing(df, subset)
            if missing:
                extra["ignored_missing_cols"] = missing
        return _drop_empty_rows(df, cols=subset, strict=STRICT_REBUILD), extra

    if action == "rename_columns":
        mapping = params.get("mapping") or {}
        if isinstance(mapping, dict) and mapping:
            remapped_mapping: dict[str, str] = {}
            for k, v in mapping.items():
                kk = alias_map.get(str(k), str(k))
                remapped_mapping[kk] = str(v)
            df2, applied = _rename_columns(df, mapping=remapped_mapping, strict=STRICT_REBUILD)
        else:
            df2, applied = _rename_columns(df, mapping=mapping, strict=STRICT_REBUILD)

        if applied:
            _update_alias_map(alias_map, applied)
            extra["applied_rename"] = applied
        else:
            extra["applied_rename"] = {}
        return df2, extra

    if action == "strip_whitespace":
        subset = cols if cols else None
        if subset:
            present, missing = _split_existing_missing(df, subset)
            if missing:
                extra["ignored_missing_cols"] = missing
        return _strip_whitespace(df, cols=subset, strict=STRICT_REBUILD), extra

    if action == "substitute_values":
        if len(cols) != 1:
            raise ValueError("substitute_values: l'opération doit cibler exactement 1 colonne (payload.columns).")
        col = cols[0]

        df2, sub_eff = _substitute_values(
            df,
            col,
            from_value=params.get("from_value", None),
            to_value=params.get("to_value", None),
            case_sensitive=bool(params.get("case_sensitive", True)),
            treat_from_as_null=bool(params.get("treat_from_as_null", False)),
            treat_to_as_null=bool(params.get("treat_to_as_null", False)),
        )
        extra.update(sub_eff)
        return df2, extra

    raise ValueError(f"Cleaning: action inconnue: {action}")


def rebuild_processed(db: Session, project_id: int, dataset_id: int) -> None:
    ds = get_dataset_or_404(db, project_id, dataset_id)
    df = read_df(ds.file_path)  # start from SOURCE file

    ops = crud_nettoyage.list_operations_by_type(db, project_id, dataset_id, op_type="cleaning")

    alias_map: dict[str, str] = {}

    for op in ops:
        raw_cols = [str(c) for c in (getattr(op, "columns", []) or [])]
        before = df.copy()

        df, extra = _apply_one(
            df=df,
            op_type=op.op_type,
            cols=raw_cols,
            params=op.params or {},
            alias_map=alias_map,
            description=getattr(op, "description", None),
        )

        try:
            effect = compute_operation_effect(before, df, op.params or {})
            if extra:
                effect.update(extra)
            crud_nettoyage.set_operation_result(db, op.id, effect)
        except Exception:
            pass

    save_processed_df(df, ds.file_path, dataset_id)
