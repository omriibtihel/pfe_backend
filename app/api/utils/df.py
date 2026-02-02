# app/api/utils/df.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable
import pandas as pd


def _whitespace_only_to_na(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) == 0:
        return df

    df = df.copy()
    for c in obj_cols:
        # remplace uniquement les valeurs qui matchent "^\s*$" (espaces uniquement)
        df[c] = df[c].replace(to_replace=r"^\s*$", value=pd.NA, regex=True)
    return df


def read_df(
    path: Path,
    nrows: Optional[int] = None,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:

    path = Path(path)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Exception | None = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, nrows=nrows, encoding=enc, usecols=usecols)
            return _whitespace_only_to_na(df)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    # Fallback: keep going even if characters are weird
    try:
        df = pd.read_csv(
            path,
            nrows=nrows,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip",
            usecols=usecols,
        )
        return _whitespace_only_to_na(df)
    except Exception as e:
        last_err = e

    # Final fallback: replace invalid bytes (never crash API)
    df = pd.read_csv(
        path,
        nrows=nrows,
        encoding="utf-8",
        engine="python",
        encoding_errors="replace",
        on_bad_lines="skip",
        usecols=usecols,
    )
    return _whitespace_only_to_na(df)
