from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.api.routes.training import _copy_artifacts_json, _session_to_front_session
from app.models.training import TrainedModel


def _make_model(
    *,
    model_id: int,
    session_id: int,
    model_type: str,
    saved: bool,
):
    return SimpleNamespace(
        id=model_id,
        session_id=session_id,
        project_id=9,
        model_type=model_type,
        task_type="classification",
        metrics_json={
            "test": {"accuracy": 0.91, "f1": 0.88},
            "primary_score": {"metric": "accuracy", "value": 0.91},
            "training_time_sec": 12.4,
        },
        artifacts_json={
            "saved": saved,
            "training_schema": {"feature_names": ["age", "bmi"]},
        },
        created_at=datetime(2026, 3, 6, 10, 30, tzinfo=timezone.utc),
    )


def test_session_payload_marks_saved_and_active_models() -> None:
    session = SimpleNamespace(
        id=17,
        project_id=9,
        dataset_version_id=4,
        status="succeeded",
        progress=100,
        current_model=None,
        error_message=None,
        config_json={"taskType": "classification"},
        created_at=datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc),
        started_at=datetime(2026, 3, 6, 10, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 3, 6, 10, 5, tzinfo=timezone.utc),
    )
    saved_active = _make_model(model_id=101, session_id=17, model_type="randomforest", saved=True)
    saved_inactive = _make_model(model_id=102, session_id=17, model_type="svm", saved=True)
    unsaved = _make_model(model_id=103, session_id=17, model_type="knn", saved=False)

    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.all.return_value = [saved_active, saved_inactive, unsaved]

    payload = _session_to_front_session(db, session, active_model_id=101)

    assert payload["activeModelId"] == "101"
    assert [item["id"] for item in payload["results"]] == ["101", "102", "103"]
    assert [item["isSaved"] for item in payload["results"]] == [True, True, False]
    assert [item["isActive"] for item in payload["results"]] == [True, False, False]
    assert payload["results"][0]["metrics"]["accuracy"] == 0.91
    db.query.assert_called_once_with(TrainedModel)


def test_copy_artifacts_json_returns_new_dict() -> None:
    original = {"saved": False, "training_schema": {"feature_names": ["age"]}}

    copied = _copy_artifacts_json(original)
    copied["saved"] = True

    assert copied is not original
    assert original["saved"] is False
    assert copied["saved"] is True
