# app/api/utils/df.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable
import pandas as pd


def read_df(
    path: Path,
    nrows: Optional[int] = None,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Robust CSV reader:
    - tries common encodings (utf-8, utf-8-sig, cp1252, latin1)
    - uses engine="python" as fallback for weird separators/quotes
    """
    path = Path(path)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_err: Exception | None = None

    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, usecols=usecols)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            # could be separator issues, etc.
            last_err = e

    # Fallback: keep going even if characters are weird
    try:
        return pd.read_csv(
            path,
            nrows=nrows,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip",
            usecols=usecols,
        )
    except Exception as e:
        last_err = e

    # Final fallback: replace invalid bytes (never crash API)
    return pd.read_csv(
        path,
        nrows=nrows,
        encoding="utf-8",
        engine="python",
        encoding_errors="replace",
        on_bad_lines="skip",
        usecols=usecols,
    )
