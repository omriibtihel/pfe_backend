from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.api.utils_shared.df import read_df


def processed_path_for(dataset_file_path: str, dataset_id: int) -> Path:
    base = Path(dataset_file_path).parent
    # fichier de travail (ne touche pas l'original)
    return base / f"processed_dataset_{dataset_id}.csv"


def load_current_df(dataset_file_path: str, dataset_id: int) -> pd.DataFrame:
    p = processed_path_for(dataset_file_path, dataset_id)
    if p.exists():
        return read_df(p)
    return read_df(Path(dataset_file_path))


def _temporary_processed_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")


def save_processed_df(df: pd.DataFrame, dataset_file_path: str, dataset_id: int) -> Path:
    p = processed_path_for(dataset_file_path, dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_processed_path(p)

    # CSV-safe: convertir datetime en string propre si besoin
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        df.to_csv(tmp, index=False)

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise IOError(f"Cleaned file written but empty or missing: {p}")

        try:
            # Atomic rename — preferred path.
            tmp.replace(p)
        except OSError:
            # OneDrive / Windows Defender can lock the destination briefly.
            # Fall back to non-atomic copy then delete the tmp.
            shutil.copy2(str(tmp), str(p))
            tmp.unlink(missing_ok=True)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Failed to persist cleaned file to {p}: {e}") from e

    return p
