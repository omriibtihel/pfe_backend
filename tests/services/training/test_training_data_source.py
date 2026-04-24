from __future__ import annotations

"""
Tests for training_service data source selection (Fix C).

run_training_session() creates its own DB session via SessionLocal(), so we patch
SessionLocal to inject a mock. All SQLAlchemy models must be imported before testing
to ensure mapper configuration succeeds.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Force mapper configuration by importing all related models before any test runs.
import app.models.dataset          # noqa: F401
import app.models.dataset_version  # noqa: F401
import app.models.training         # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "label": [0, 1] * 5})


def _make_training_session(session_id: int = 1, project_id: int = 1, version_id: int = 10):
    s = MagicMock()
    s.id = session_id
    s.project_id = project_id
    s.dataset_version_id = version_id
    s.config_json = {
        "targetColumn": "label",
        "models": ["logisticregression"],
        "metrics": ["accuracy"],
        "splitMethod": "holdout",
        "trainRatio": 70,
        "valRatio": 15,
        "testRatio": 15,
        "kFolds": 3,
        "taskType": "classification",
        "useSmote": False,
        "balancing": {"strategy": "none"},
        "preprocessing": {
            "defaults": {
                "numericImputation": "none",
                "numericScaling": "none",
                "categoricalImputation": "none",
                "categoricalEncoding": "onehot",
            },
            "columns": {},
        },
    }
    s.error_message = None
    return s


def _make_dataset_version(raw_csv_path: Path, source_ds_id: int = 5, version_id: int = 10):
    dv = MagicMock()
    dv.id = version_id
    dv.file_path = str(raw_csv_path)

    src_ds = MagicMock()
    src_ds.id = source_ds_id
    src_ds.file_path = str(raw_csv_path)
    dv.source_dataset = src_ds
    return dv


def _make_mock_db(training_session, dataset_version):
    """Return a mock DB whose .query().filter().first() returns the right object."""
    from app.models.training import TrainingSession as TS
    from app.models.dataset_version import DatasetVersion as DV

    db = MagicMock()
    db.close = MagicMock()

    def _query(model):
        q = MagicMock()
        if model is TS:
            q.filter.return_value.first.return_value = training_session
        elif model is DV:
            q.filter.return_value.first.return_value = dataset_version
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = _query
    return db


# ---------------------------------------------------------------------------
# test_uses_processed_file_when_exists
# ---------------------------------------------------------------------------

def test_uses_processed_file_when_exists(tmp_path):
    """When processed_dataset_<source_ds_id>.csv exists, load_dataframe must receive it."""
    raw_csv = tmp_path / "data.csv"
    _tiny_df().to_csv(raw_csv, index=False)

    source_ds_id = 5
    processed_csv = tmp_path / f"processed_dataset_{source_ds_id}.csv"
    _tiny_df().to_csv(processed_csv, index=False)   # cleaned file present

    dv = _make_dataset_version(raw_csv, source_ds_id=source_ds_id)
    s = _make_training_session()
    db = _make_mock_db(s, dv)

    loaded_paths: list[Path] = []

    def _fake_load(path):
        loaded_paths.append(Path(path))
        return _tiny_df()

    with patch("app.services.training.training_service.SessionLocal", return_value=db), \
         patch("app.services.training.training_service.resolve_dataset_path",
               return_value=(raw_csv, dv.id)), \
         patch("app.services.training.training_service.load_dataframe",
               side_effect=_fake_load), \
         patch("app.services.training.training_service.run_one_model",
               side_effect=RuntimeError("stop after load")), \
         patch("app.services.training.training_service.crud_vcs.get_map", return_value={}), \
         patch("app.services.training.training_service._update_session"), \
         patch("app.services.training.training_service.training_notifier"):

        from app.services.training.training_service import run_training_session
        run_training_session(session_id=1)

    assert len(loaded_paths) == 1, "load_dataframe must be called exactly once"
    assert loaded_paths[0] == processed_csv, (
        f"Expected processed path {processed_csv}, got {loaded_paths[0]}"
    )


# ---------------------------------------------------------------------------
# test_falls_back_to_raw_when_no_processed_file
# ---------------------------------------------------------------------------

def test_falls_back_to_raw_when_no_processed_file(tmp_path):
    """When no processed file exists, load_dataframe must receive the raw DatasetVersion path."""
    raw_csv = tmp_path / "data.csv"
    _tiny_df().to_csv(raw_csv, index=False)

    source_ds_id = 5
    # No processed file
    assert not (tmp_path / f"processed_dataset_{source_ds_id}.csv").exists()

    dv = _make_dataset_version(raw_csv, source_ds_id=source_ds_id)
    s = _make_training_session()
    db = _make_mock_db(s, dv)

    loaded_paths: list[Path] = []

    def _fake_load(path):
        loaded_paths.append(Path(path))
        return _tiny_df()

    with patch("app.services.training.training_service.SessionLocal", return_value=db), \
         patch("app.services.training.training_service.resolve_dataset_path",
               return_value=(raw_csv, dv.id)), \
         patch("app.services.training.training_service.load_dataframe",
               side_effect=_fake_load), \
         patch("app.services.training.training_service.run_one_model",
               side_effect=RuntimeError("stop after load")), \
         patch("app.services.training.training_service.crud_vcs.get_map", return_value={}), \
         patch("app.services.training.training_service._update_session"), \
         patch("app.services.training.training_service._append_session_message"), \
         patch("app.services.training.training_service.training_notifier"):

        from app.services.training.training_service import run_training_session
        run_training_session(session_id=1)

    assert len(loaded_paths) == 1
    assert loaded_paths[0] == raw_csv, (
        f"Expected raw path {raw_csv}, got {loaded_paths[0]}"
    )


# ---------------------------------------------------------------------------
# test_raw_warning_appended_to_session
# ---------------------------------------------------------------------------

def test_raw_warning_appended_to_session(tmp_path):
    """When raw path is used, a WARNING is appended to session.error_message."""
    raw_csv = tmp_path / "data.csv"
    _tiny_df().to_csv(raw_csv, index=False)

    source_ds_id = 5
    # No processed file
    assert not (tmp_path / f"processed_dataset_{source_ds_id}.csv").exists()

    dv = _make_dataset_version(raw_csv, source_ds_id=source_ds_id)
    s = _make_training_session()
    db = _make_mock_db(s, dv)

    appended_messages: list[str] = []

    def _fake_append(db_arg, session_arg, msg):
        appended_messages.append(msg)

    with patch("app.services.training.training_service.SessionLocal", return_value=db), \
         patch("app.services.training.training_service.resolve_dataset_path",
               return_value=(raw_csv, dv.id)), \
         patch("app.services.training.training_service.load_dataframe",
               return_value=_tiny_df()), \
         patch("app.services.training.training_service.run_one_model",
               side_effect=RuntimeError("stop after load")), \
         patch("app.services.training.training_service.crud_vcs.get_map", return_value={}), \
         patch("app.services.training.training_service._update_session"), \
         patch("app.services.training.training_service._append_session_message",
               side_effect=_fake_append), \
         patch("app.services.training.training_service.training_notifier"):

        from app.services.training.training_service import run_training_session
        run_training_session(session_id=1)

    warning_msgs = [m for m in appended_messages if "WARNING" in m and "unprocessed" in m]
    assert warning_msgs, (
        f"Expected a RAW_ORIGINAL warning in session messages, got: {appended_messages}"
    )
