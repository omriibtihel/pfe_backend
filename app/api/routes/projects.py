from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import StaleDataError


from app.api.deps import get_db, get_current_user
from app.core.config import PROJECTS_PATH
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate

router = APIRouter()


def _get_owned_project(db: Session, project_id: int, user_id: int) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    if project.owner_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")
    return project


def _project_storage_dir(project_id: int) -> Path:
    return PROJECTS_PATH / str(project_id)


@router.get("", response_model=list[ProjectOut])
def list_my_projects(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(Project).filter(Project.owner_id == current_user.id)
    q = q.order_by(Project.created_at.desc()) if hasattr(Project, "created_at") else q.order_by(Project.id.desc())
    return q.all()


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = Project(
        name=payload.name,
        description=payload.description,
        owner_id=current_user.id,
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    try:
        project_dir = _project_storage_dir(project.id)
        (project_dir / "datasets").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        db.delete(project)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project storage directories",
        ) from e

    return project


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return _get_owned_project(db, project_id, current_user.id)


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: int,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = _get_owned_project(db, project_id, current_user.id)

    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None  # déjà supprimé — DELETE est idempotent
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")

    try:
        project.active_dataset_id = None
        db.flush()

        db.delete(project)
        db.commit()
        return None

    except StaleDataError:
        # Suppression concurrente — déjà supprimé par une autre requête
        db.rollback()
        return None

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project: {e}",
        ) from e
