from __future__ import annotations

from pathlib import Path
import pandas as pd

def write_df_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0
