from __future__ import annotations

import pandas as pd
import pytest

from app.services.nettoyage.df_utils import save_processed_df, processed_path_for


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# ---------------------------------------------------------------------------
# test_save_processed_df_raises_on_empty_write
# ---------------------------------------------------------------------------

def test_save_processed_df_raises_on_empty_write(tmp_path, monkeypatch):
    """If to_csv produces an empty file, save_processed_df must raise IOError
    and must NOT leave the .tmp file on disk."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 99

    # Monkeypatch to_csv so it writes an empty file (simulates a silent truncation)
    original_to_csv = pd.DataFrame.to_csv

    def _empty_write(self, path_or_buf=None, **kwargs):
        # Write empty content instead of real CSV
        open(path_or_buf, "w").close()

    monkeypatch.setattr(pd.DataFrame, "to_csv", _empty_write)

    with pytest.raises(IOError, match="empty or missing"):
        save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    # The .tmp file must not remain on disk
    tmp_file = processed_path_for(str(raw_csv), dataset_id).with_suffix(".tmp")
    assert not tmp_file.exists(), ".tmp file was not cleaned up after empty write"

    # The final .csv must not exist either (atomic: no partial file)
    final = processed_path_for(str(raw_csv), dataset_id)
    assert not final.exists(), "Final processed file must not exist after failed write"


# ---------------------------------------------------------------------------
# test_save_processed_df_atomic_no_partial_file
# ---------------------------------------------------------------------------

def test_save_processed_df_atomic_no_partial_file(tmp_path, monkeypatch):
    """If to_csv raises mid-write, the original processed file must not be
    created or modified (atomic guarantee via .tmp → rename)."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 42

    final = processed_path_for(str(raw_csv), dataset_id)
    tmp_file = final.with_suffix(".tmp")

    # Simulate a crash during to_csv
    def _crash_write(self, path_or_buf=None, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _crash_write)

    with pytest.raises(IOError, match="disk full"):
        save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    assert not tmp_file.exists(), ".tmp file must be cleaned up on to_csv exception"
    assert not final.exists(), "Final processed file must not exist after failed to_csv"


# ---------------------------------------------------------------------------
# test_save_processed_df_happy_path
# ---------------------------------------------------------------------------

def test_save_processed_df_happy_path(tmp_path):
    """Normal write: final file exists, .tmp does not, content is correct."""
    raw_csv = tmp_path / "data.csv"
    raw_csv.write_text("a,b\n1,x\n")
    dataset_id = 7

    result_path = save_processed_df(_simple_df(), str(raw_csv), dataset_id)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
    # No orphan .tmp file
    assert not result_path.with_suffix(".tmp").exists()
    # Content is readable and correct
    loaded = pd.read_csv(result_path)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 3
