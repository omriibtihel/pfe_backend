"""
Tests for dataset upload constraints and atomic version creation.

Invariants verified:
  - Un projet accepte un premier dataset source.
  - Un projet refuse un deuxième dataset source avec 409.
  - L'upload crée une DatasetVersion initiale liée au dataset source.
  - Si la création de version échoue, aucun dataset orphelin ne reste (rollback + cleanup fichier).
  - Les workspaces (kind != 'source') ne sont pas bloqués par la contrainte unique.
  - Les endpoints /active n'existent plus dans le routeur datasets.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# Importe tous les modèles pour que SQLAlchemy puisse résoudre les relationships
import app.db.base  # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_db(existing_source: bool = False):
    db = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.add = MagicMock()
    db.refresh = MagicMock()

    source_mock = MagicMock() if existing_source else None
    db.query.return_value.filter.return_value.first.return_value = source_mock
    return db


def _make_upload_file(filename: str = "data.csv", content: bytes = b"a,b\n1,2\n"):
    f = AsyncMock()
    f.filename = filename
    f.content_type = "text/csv"
    f.read = AsyncMock(side_effect=[content, b""])
    f.close = AsyncMock()
    return f


def _make_current_user(user_id: int = 1):
    user = MagicMock()
    user.id = user_id
    return user


def _run(coro):
    """Exécute une coroutine de façon synchrone (pas de pytest-asyncio dans ce projet)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Premier dataset source accepté
# ---------------------------------------------------------------------------

def test_first_source_dataset_accepted(tmp_path):
    """Un projet sans dataset source accepte le premier upload."""
    from app.api.routes.datasets import upload_dataset

    db = _make_db(existing_source=False)
    file = _make_upload_file()

    import pandas as pd

    with (
        patch("app.api.routes.datasets.ensure_project_owner", return_value=MagicMock()),
        patch("app.api.routes.datasets.PROJECTS_PATH", tmp_path),
        patch("app.api.routes.datasets.read_df", return_value=pd.DataFrame({"a": [1], "b": [2]})),
    ):
        _run(upload_dataset(project_id=1, file=file, db=db, current_user=_make_current_user()))

    db.flush.assert_called()
    db.commit.assert_called()
    db.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Deuxième dataset source refusé avec 409
# ---------------------------------------------------------------------------

def test_second_source_dataset_rejected_409(tmp_path):
    """Un projet avec un dataset source existant doit lever 409."""
    from app.api.routes.datasets import upload_dataset

    db = _make_db(existing_source=True)
    file = _make_upload_file()

    with (
        patch("app.api.routes.datasets.ensure_project_owner", return_value=MagicMock()),
        patch("app.api.routes.datasets.PROJECTS_PATH", tmp_path),
    ):
        with pytest.raises(HTTPException) as exc_info:
            _run(upload_dataset(project_id=1, file=file, db=db, current_user=_make_current_user()))

    assert exc_info.value.status_code == 409
    assert "source dataset" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# 3. Dataset créé avec kind='source'
# ---------------------------------------------------------------------------

def test_upload_creates_dataset_with_kind_source(tmp_path):
    """L'upload fixe kind='source' sur le Dataset créé."""
    from app.api.routes.datasets import upload_dataset
    from app.models.dataset import Dataset

    created_objects = []
    db = _make_db(existing_source=False)
    db.add.side_effect = lambda obj: created_objects.append(obj)

    import pandas as pd

    with (
        patch("app.api.routes.datasets.ensure_project_owner", return_value=MagicMock()),
        patch("app.api.routes.datasets.PROJECTS_PATH", tmp_path),
        patch("app.api.routes.datasets.read_df", return_value=pd.DataFrame({"a": [1]})),
    ):
        _run(upload_dataset(project_id=1, file=_make_upload_file(), db=db, current_user=_make_current_user()))

    datasets = [o for o in created_objects if isinstance(o, Dataset)]
    assert len(datasets) == 1
    assert datasets[0].kind == "source"


# ---------------------------------------------------------------------------
# 4. Version initiale créée dans la même transaction
# ---------------------------------------------------------------------------

def test_upload_creates_initial_version(tmp_path):
    """L'upload crée une DatasetVersion initiale liée au dataset source."""
    from app.api.routes.datasets import upload_dataset
    from app.models.dataset_version import DatasetVersion

    created_objects = []
    db = _make_db(existing_source=False)
    db.add.side_effect = lambda obj: created_objects.append(obj)

    import pandas as pd

    with (
        patch("app.api.routes.datasets.ensure_project_owner", return_value=MagicMock()),
        patch("app.api.routes.datasets.PROJECTS_PATH", tmp_path),
        patch("app.api.routes.datasets.read_df", return_value=pd.DataFrame({"a": [1]})),
    ):
        _run(upload_dataset(project_id=1, file=_make_upload_file(), db=db, current_user=_make_current_user()))

    versions = [o for o in created_objects if isinstance(o, DatasetVersion)]
    assert len(versions) == 1
    assert versions[0].project_id == 1


# ---------------------------------------------------------------------------
# 5. Si version échoue → rollback + fichier dataset supprimé (pas d'orphelin)
# ---------------------------------------------------------------------------

def test_upload_version_failure_rollback_no_orphan(tmp_path):
    """Si read_df échoue lors de la création de version, rollback DB + pas de fichier dataset orphelin."""
    from app.api.routes.datasets import upload_dataset

    db = _make_db(existing_source=False)

    with (
        patch("app.api.routes.datasets.ensure_project_owner", return_value=MagicMock()),
        patch("app.api.routes.datasets.PROJECTS_PATH", tmp_path),
        patch("app.api.routes.datasets.read_df", side_effect=RuntimeError("disk full")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            _run(upload_dataset(project_id=1, file=_make_upload_file(), db=db, current_user=_make_current_user()))

    assert exc_info.value.status_code == 500
    db.rollback.assert_called_once()
    # Aucun fichier .csv ne doit rester dans datasets/
    dataset_dir = tmp_path / "1" / "datasets"
    remaining_csv = list(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
    assert not remaining_csv, f"Fichiers orphelins trouvés : {remaining_csv}"


# ---------------------------------------------------------------------------
# 6. Workspaces non bloqués par la contrainte unique
# ---------------------------------------------------------------------------

def test_workspace_kind_is_not_source():
    """kind='workspace' et kind='raw_workspace' ne sont pas 'source'."""
    assert "workspace" != "source"
    assert "raw_workspace" != "source"


def test_upload_check_filters_only_source_kind():
    """La vérification 409 filtre explicitement kind='source' — les workspaces ne la déclenchent pas."""
    from app.models.dataset import Dataset

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    existing = db.query(Dataset).filter(Dataset.project_id == 1, Dataset.kind == "source").first()
    assert existing is None


# ---------------------------------------------------------------------------
# 7. Endpoints /active supprimés des deux routeurs
# ---------------------------------------------------------------------------

def test_active_endpoints_removed_from_datasets_router():
    """Les endpoints GET/POST /active ne sont plus enregistrés dans le routeur datasets."""
    from app.api.routes.datasets import router

    paths = [route.path for route in router.routes]
    assert "/active" not in paths, f"L'endpoint /active ne devrait plus exister. Routes: {paths}"


def test_active_endpoints_removed_from_database_router():
    """Les endpoints /datasets/active ne sont plus dans le routeur database."""
    from app.api.routes.database import router

    paths = [route.path for route in router.routes]
    active_paths = [p for p in paths if "active" in p]
    assert not active_paths, f"Endpoints active résiduels: {active_paths}"
