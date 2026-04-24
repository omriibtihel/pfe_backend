from __future__ import annotations

import pandas as pd
import pytest

from app.services.nettoyage.df_utils import save_processed_df, processed_path_for


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _temp_files_for(final_path):
    return list(final_path.parent.glob(f".{final_path.name}.*.tmp"))


# ---------------------------------------------------------------------------
# test_save_processed_df_raises_on_empty_write
# ---------------------------------------------------------------------------

def test_save_processed_df_raises_on_empty_write(tmp_path, monkeypatch):
    """If to_csv produces an empty file, save_processed_df must raise IOError
    and must NOT leave a temp file on disk."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 99

    def _empty_write(self, path_or_buf=None, **kwargs):
        open(path_or_buf, "w").close()

    monkeypatch.setattr(pd.DataFrame, "to_csv", _empty_write)

    with pytest.raises(IOError, match="empty or missing"):
        save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    final = processed_path_for(str(raw_csv), dataset_id)
    assert not _temp_files_for(final), "Temp file was not cleaned up after empty write"
    assert not final.exists(), "Final processed file must not exist after failed write"


# ---------------------------------------------------------------------------
# test_save_processed_df_atomic_no_partial_file
# ---------------------------------------------------------------------------

def test_save_processed_df_atomic_no_partial_file(tmp_path, monkeypatch):
    """If to_csv raises mid-write, the original processed file must not be
    created or modified."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 42

    final = processed_path_for(str(raw_csv), dataset_id)

    def _crash_write(self, path_or_buf=None, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _crash_write)

    with pytest.raises(IOError, match="disk full"):
        save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    assert not _temp_files_for(final), "Temp file must be cleaned up on to_csv exception"
    assert not final.exists(), "Final processed file must not exist after failed to_csv"


# ---------------------------------------------------------------------------
# test_save_processed_df_happy_path
# ---------------------------------------------------------------------------

def test_save_processed_df_happy_path(tmp_path):
    """Normal write: final file exists, temp file does not, content is correct."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 7

    result_path = save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
    assert not _temp_files_for(result_path)

    loaded = pd.read_csv(result_path)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# test_save_processed_df_overwrites_existing_processed_file
# ---------------------------------------------------------------------------

def test_save_processed_df_overwrites_existing_processed_file(tmp_path):
    """Repeated cleaning saves must replace the prior processed CSV on Windows."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 7

    first = pd.DataFrame({"a": [1], "b": ["old"]})
    second = pd.DataFrame({"a": [2, 3], "b": ["new", "newer"]})

    result_path = save_processed_df(first, str(raw_csv), dataset_id)
    result_path = save_processed_df(second, str(raw_csv), dataset_id)

    assert result_path.exists()
    assert not _temp_files_for(result_path)

    loaded = pd.read_csv(result_path)
    assert loaded.to_dict(orient="list") == {"a": [2, 3], "b": ["new", "newer"]}
