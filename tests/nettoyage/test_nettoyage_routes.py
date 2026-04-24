from __future__ import annotations

"""
Tests for atomic apply / undo guarantees in the nettoyage routes.

These tests use mocking — they do NOT require a real DB or disk.
The invariants being verified:
  - apply:  disk write FIRST; if disk fails → no DB commit
  - apply:  if DB commit fails after disk write → processed file is removed
  - undo:   if disk rebuild fails → DB record is preserved (rollback)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_db():
    db = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.add = MagicMock()
    db.refresh = MagicMock()
    db.delete = MagicMock()
    return db


def _make_operation(op_type="cleaning", op_id=1):
    op = MagicMock()
    op.id = op_id
    op.op_type = op_type
    op.project_id = 1
    op.dataset_id = 10
    return op


# ---------------------------------------------------------------------------
# test_apply_disk_fail_does_not_commit
# ---------------------------------------------------------------------------

def test_apply_disk_fail_does_not_commit(tmp_path):
    """If rebuild_processed raises IOError, db.commit() must never be called."""
    from app.crud import nettoyage as crud_nettoyage

    db = _make_db()

    # create_operation with flush_only=True should flush, not commit
    op = _make_operation()
    created_op = MagicMock()
    created_op.id = 1

    with patch.object(crud_nettoyage, "create_operation", return_value=created_op) as mock_create, \
         patch("app.api.routes.nettoyage.rebuild_processed", side_effect=IOError("disk full")), \
         patch("app.api.routes.nettoyage.get_dataset_or_404") as mock_ds, \
         patch("app.api.routes.nettoyage.load_current_df", return_value=MagicMock()), \
         patch("app.api.routes.nettoyage._validate_cleaning_payload"):

        mock_ds.return_value = MagicMock(file_path=str(tmp_path / "data.csv"))

        from fastapi import HTTPException
        from app.api.routes.nettoyage import apply_operation
        from app.schemas.nettoyage import OperationIn

        payload = OperationIn(
            type="cleaning",
            description="drop rows",
            columns=[],
            params={"action": "drop_duplicates"},
        )

        current_user = MagicMock(id=1)

        with pytest.raises(HTTPException) as exc_info:
            apply_operation(
                project_id=1,
                dataset_id=10,
                payload=payload,
                db=db,
                current_user=current_user,
            )

        assert exc_info.value.status_code == 500
        assert "disk" in exc_info.value.detail.lower()

        # DB must have been rolled back, never committed
        db.rollback.assert_called()
        db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# test_apply_db_fail_removes_orphan_file
# ---------------------------------------------------------------------------

def test_apply_db_fail_removes_orphan_file(tmp_path):
    """If disk write succeeds but db.commit() raises, the processed file must be deleted."""
    from app.crud import nettoyage as crud_nettoyage
    from app.services.nettoyage.df_utils import processed_path_for

    db = _make_db()
    db.commit.side_effect = Exception("DB connection lost")

    dataset_file = str(tmp_path / "data.csv")

    # Simulate that rebuild_processed actually writes the file
    def _fake_rebuild(db, project_id, dataset_id):
        processed_path_for(dataset_file, dataset_id).write_text("a,b\n1,2\n")

    with patch("app.api.routes.nettoyage.rebuild_processed", side_effect=_fake_rebuild), \
         patch("app.api.routes.nettoyage.get_dataset_or_404") as mock_ds, \
         patch("app.api.routes.nettoyage.load_current_df", return_value=MagicMock()), \
         patch("app.api.routes.nettoyage._validate_cleaning_payload"), \
         patch.object(crud_nettoyage, "create_operation", return_value=MagicMock(id=1)):

        mock_ds.return_value = MagicMock(file_path=dataset_file)

        from fastapi import HTTPException
        from app.api.routes.nettoyage import apply_operation
        from app.schemas.nettoyage import OperationIn

        payload = OperationIn(
            type="cleaning",
            description="drop rows",
            columns=[],
            params={"action": "drop_duplicates"},
        )

        with pytest.raises(HTTPException) as exc_info:
            apply_operation(
                project_id=1,
                dataset_id=10,
                payload=payload,
                db=db,
                current_user=MagicMock(id=1),
            )

        assert exc_info.value.status_code == 500

        # The orphan processed file must have been cleaned up
        orphan = processed_path_for(dataset_file, 10)
        assert not orphan.exists(), "Orphan processed file must be removed on DB commit failure"


# ---------------------------------------------------------------------------
# test_undo_disk_fail_preserves_db_record
# ---------------------------------------------------------------------------

def test_undo_disk_fail_preserves_db_record(tmp_path):
    """If rebuild_processed fails during undo, db.rollback() must restore the deleted record."""
    db = _make_db()

    op = _make_operation(op_type="cleaning", op_id=99)

    # Simulate the DB query returning our operation
    query_mock = MagicMock()
    query_mock.filter_by.return_value.order_by.return_value.first.return_value = op
    db.query.return_value = query_mock

    with patch("app.api.routes.nettoyage.rebuild_processed", side_effect=IOError("no space")), \
         patch("app.api.routes.nettoyage.get_dataset_or_404") as mock_ds:

        mock_ds.return_value = MagicMock(file_path=str(tmp_path / "data.csv"))

        from fastapi import HTTPException
        from app.api.routes.nettoyage import undo_last

        with pytest.raises(HTTPException) as exc_info:
            undo_last(
                project_id=1,
                dataset_id=10,
                db=db,
                current_user=MagicMock(id=1),
            )

        assert exc_info.value.status_code == 500
        assert "preserved" in exc_info.value.detail

        # rollback must have been called to restore the staged deletion
        db.rollback.assert_called()
        # commit must never have been called
        db.commit.assert_not_called()
