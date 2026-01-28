from __future__ import annotations

import pandas as pd

from app.api.utils.processing_df import save_processed_df
from app.services.processing_ops import apply_cleaning, apply_imputation, apply_normalization, apply_encoding


def apply_operation_to_df(df: pd.DataFrame, op_type: str, columns: list[str], params: dict) -> pd.DataFrame:
    """
    Applique UNE operation sur le dataframe.
    op_type: cleaning|imputation|normalization|encoding|other
    """
    op_type = (op_type or "").lower()

    if op_type == "cleaning":
        return apply_cleaning(df, columns, params)

    if op_type == "imputation":
        return apply_imputation(df, columns, params)

    if op_type == "normalization":
        return apply_normalization(df, columns, params)

    if op_type == "encoding":
        return apply_encoding(df, columns, params)

    if op_type == "other":
        # pour l'instant, on route vers cleaning si c'est une action simple
        return apply_cleaning(df, columns, params)

    raise ValueError(f"Unsupported operation type: {op_type}")


def apply_operation(
    df: pd.DataFrame,
    dataset_id: int,
    op_type: str,
    columns: list[str],
    params: dict,
) -> pd.DataFrame:
    """
    Applique l'opération et sauvegarde immédiatement le dataframe traité.
    """
    out = apply_operation_to_df(df, op_type, columns, params)
    save_processed_df(out, dataset_id)
    return out
