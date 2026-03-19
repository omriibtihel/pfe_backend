from __future__ import annotations

import pandas as pd

from app.api.utils.nettoyage_df import save_processed_df
from app.services.processing_ops import (
    apply_cleaning,
    apply_imputation,
    apply_normalization,
    apply_encoding,
)


def _save_processed_df_compat(
    df: pd.DataFrame,
    dataset_id: int,
    dataset_file_path: str | None = None,
) -> None:

    try:
        # ancienne signature (ou ta signature actuelle)
        save_processed_df(df, dataset_id)
        return
    except TypeError:
        # nouvelle signature probable
        if dataset_file_path is None:
            raise TypeError(
                "save_processed_df attend (df, dataset_file_path, dataset_id) "
                "mais dataset_file_path n'a pas été fourni."
            )
        save_processed_df(df, dataset_file_path, dataset_id)


def apply_operation_to_df(
    df: pd.DataFrame,
    op_type: str,
    columns: list[str],
    params: dict | None,
) -> pd.DataFrame:

    op_type = (op_type or "").strip().lower()
    params = params or {}
    columns = columns or []

    if op_type == "cleaning":
        return apply_cleaning(df, columns, params)

    if op_type == "imputation":
        return apply_imputation(df, columns, params)

    if op_type == "normalization":
        return apply_normalization(df, columns, params)

    if op_type == "encoding":
        return apply_encoding(df, columns, params)

    if op_type == "other":
        return apply_cleaning(df, columns, params)

    raise ValueError(f"Unsupported operation type: {op_type}")


def apply_operation(
    df: pd.DataFrame,
    dataset_id: int,
    op_type: str,
    columns: list[str],
    params: dict | None,
    *,
    dataset_file_path: str | None = None,
) -> pd.DataFrame:
    """
    Applique l'opération et sauvegarde immédiatement le dataframe traité.
    """
    out = apply_operation_to_df(df, op_type, columns, params)
    _save_processed_df_compat(out, dataset_id, dataset_file_path=dataset_file_path)
    return out
