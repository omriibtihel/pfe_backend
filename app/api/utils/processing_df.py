from __future__ import annotations

from pathlib import Path
import pandas as pd

from app.api.utils.df import read_df

def processed_path_for(dataset_file_path: str, dataset_id: int) -> Path:
    base = Path(dataset_file_path).parent
    # fichier de travail (ne touche pas l'original)
    return base / f"processed_dataset_{dataset_id}.csv"

def load_current_df(dataset_file_path: str, dataset_id: int) -> pd.DataFrame:
    p = processed_path_for(dataset_file_path, dataset_id)
    if p.exists():
        # low_memory=False pour éviter types bizarres
        return pd.read_csv(p, low_memory=False)
    return read_df(Path(dataset_file_path))

def save_processed_df(df: pd.DataFrame, dataset_file_path: str, dataset_id: int) -> Path:
    p = processed_path_for(dataset_file_path, dataset_id)

    # CSV-safe: convertir datetime en string propre si besoin
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(p, index=False)
    return p
