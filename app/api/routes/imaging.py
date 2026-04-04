"""
Routes FastAPI pour le pipeline imagerie.
Monté dans main.py sous /api/projects/{project_id}/imaging
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, HTTPException,
    Query, UploadFile, status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import ensure_project_owner, get_current_user, get_current_user_sse, get_db
from app.core.config import PROJECTS_PATH
from app.crud import imaging as crud_imaging
from app.schemas.imaging import (
    ImagingConfigIn,
    ImagingModelResultOut,
    ImagingPredictionOut,
    ImagingSessionOut,
    ImageClassInfo,
    ImageListOut,
)
from app.services.imaging.data.dataset import ImageFolderDataset
from app.services.imaging.imaging_service import run_imaging_session
from app.services.training.notifier import training_notifier

router = APIRouter()

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _images_dir(project_id: int) -> Path:
    return PROJECTS_PATH / str(project_id) / "images"


# ──────────────────────────────────────────────────────────────────────────────
# Gestion des images
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_images(
    project_id: int,
    class_name: str = Query(..., min_length=1, max_length=128),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload une liste de fichiers image dans /images/{class_name}/."""
    ensure_project_owner(db, project_id, current_user.id)

    target_dir = _images_dir(project_id) / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded = 0
    skipped = 0
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            skipped += 1
            continue
        dest = target_dir / (f.filename or f"image_{uploaded}{suffix}")
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped, "class_name": class_name}


@router.get("/images", response_model=ImageListOut)
def list_images(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retourne la liste des classes et le nombre d'images par classe."""
    ensure_project_owner(db, project_id, current_user.id)

    images_dir = _images_dir(project_id)
    counts = ImageFolderDataset.count_per_class(str(images_dir))
    classes = [ImageClassInfo(name=name, count=cnt) for name, cnt in counts.items()]
    total = sum(c.count for c in classes)

    return ImageListOut(
        classes=classes,
        total_images=total,
        images_dir=str(images_dir),
    )


@router.delete("/images/{class_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image_class(
    project_id: int,
    class_name: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Supprime le dossier d'une classe entière."""
    ensure_project_owner(db, project_id, current_user.id)

    target = _images_dir(project_id) / class_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Classe '{class_name}' introuvable.")
    shutil.rmtree(target, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sessions d'entraînement imagerie
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=ImagingSessionOut, status_code=status.HTTP_201_CREATED)
def create_imaging_session(
    project_id: int,
    payload: ImagingConfigIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Lance une session d'entraînement imagerie en tâche de fond."""
    ensure_project_owner(db, project_id, current_user.id)

    images_dir = str(_images_dir(project_id))
    config_json = payload.model_dump()
    config_json["imagesDir"] = images_dir

    session = crud_imaging.create_session(
        db, project_id=project_id, config_json=config_json,
    )
    background_tasks.add_task(run_imaging_session, session.id)
    return ImagingSessionOut.from_orm_with_results(session)


@router.get("/sessions", response_model=List[ImagingSessionOut])
def list_imaging_sessions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    sessions = crud_imaging.list_sessions(db, project_id)
    return [ImagingSessionOut.from_orm_with_results(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=ImagingSessionOut)
def get_imaging_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    s = crud_imaging.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return ImagingSessionOut.from_orm_with_results(s)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_imaging_session(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    s = crud_imaging.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    crud_imaging.delete_session_and_files(db, s)


@router.get("/sessions/{session_id}/events")
async def stream_imaging_events(
    project_id: int,
    session_id: int,
    last_seq: int = Query(-1),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_sse),
):
    """Stream SSE des événements d'entraînement imagerie (réutilise training_notifier)."""
    ensure_project_owner(db, project_id, current_user.id)
    s = crud_imaging.get_session(db, session_id, project_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    return StreamingResponse(
        training_notifier.subscribe(session_id, last_seq=last_seq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Modèles entraînés
# ──────────────────────────────────────────────────────────────────────────────

@router.patch(
    "/sessions/{session_id}/models/{model_id}/save",
    response_model=ImagingModelResultOut,
)
def save_imaging_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    m = crud_imaging.get_model(db, model_id, session_id=session_id, project_id=project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Modèle introuvable.")
    crud_imaging.save_model(db, m)
    db.refresh(m)
    return ImagingModelResultOut.model_validate(m)


@router.patch(
    "/sessions/{session_id}/models/{model_id}/unsave",
    response_model=ImagingModelResultOut,
)
def unsave_imaging_model(
    project_id: int,
    session_id: int,
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    ensure_project_owner(db, project_id, current_user.id)
    m = crud_imaging.get_model(db, model_id, session_id=session_id, project_id=project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Modèle introuvable.")
    crud_imaging.unsave_model(db, m)
    db.refresh(m)
    return ImagingModelResultOut.model_validate(m)


# ──────────────────────────────────────────────────────────────────────────────
# Modèles sauvegardés (pour la page prédiction)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/models/saved", response_model=List[ImagingModelResultOut])
def list_saved_imaging_models(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retourne tous les modèles sauvegardés du projet (is_saved=True)."""
    ensure_project_owner(db, project_id, current_user.id)
    return crud_imaging.list_saved_models(db, project_id)


# ──────────────────────────────────────────────────────────────────────────────
# Prédiction imagerie
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=ImagingPredictionOut)
def predict_image(
    project_id: int,
    model_id: int = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Inférence sur une image avec un modèle sauvegardé."""
    ensure_project_owner(db, project_id, current_user.id)

    m = crud_imaging.get_model_by_id(db, model_id, project_id)
    if not m:
        raise HTTPException(status_code=404, detail="Modèle introuvable.")
    if not m.is_saved:
        raise HTTPException(status_code=400, detail="Ce modèle n'est pas sauvegardé.")

    arts = m.artifacts_json or {}
    model_pt = arts.get("model_pt")
    class_names = arts.get("class_names", [])
    image_size = arts.get("image_size", 224)
    model_name = arts.get("model_name", m.model_name)

    if not model_pt or not Path(model_pt).exists():
        raise HTTPException(status_code=404, detail="Fichier poids .pt introuvable.")
    if not class_names:
        raise HTTPException(status_code=400, detail="Noms de classes manquants dans le modèle.")

    try:
        from app.services.imaging.pipeline.predictor import predict_image as _predict
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PyTorch non installé : {exc}",
        ) from exc

    image_bytes = file.file.read()
    try:
        result = _predict(
            model_pt_path=model_pt,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            image_bytes=image_bytes,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'inférence : {exc}",
        ) from exc

    return ImagingPredictionOut(
        model_id=m.id,
        model_name=model_name,
        **result,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Capacités
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/capabilities")
def get_imaging_capabilities(
    project_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retourne les modèles et tâches supportés par le pipeline imagerie."""
    from app.services.imaging.pipeline.models import SUPPORTED_MODELS
    ensure_project_owner(db, project_id, current_user.id)
    return {
        "supportedModels": SUPPORTED_MODELS,
        "supportedTaskTypes": ["image_classification"],
        "maxImageSize": 512,
        "minImageSize": 64,
    }
