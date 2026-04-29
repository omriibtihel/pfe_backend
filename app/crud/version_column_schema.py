from sqlalchemy.orm import Session
from app.models.version_column_schema import VersionColumnSchema

ALLOWED = {"numeric", "categorical", "text", "binary", "datetime", "id", "other"}

def get_map(db: Session, dataset_version_id: int) -> dict[str, VersionColumnSchema]:
    rows = (
        db.query(VersionColumnSchema)
        .filter(VersionColumnSchema.dataset_version_id == dataset_version_id)
        .all()
    )
    return {r.column_name: r for r in rows}

def upsert_many_inferred(
    db: Session,
    project_id: int,
    dataset_version_id: int,
    inferred: dict[str, tuple[str, float]],
) -> dict[str, VersionColumnSchema]:
    """
    inferred: {col_name: (inferred_kind, confidence)}
    - upsert inferred_kind/confidence
    - DO NOT touch override_kind
    - commit once
    """
    rows = (
        db.query(VersionColumnSchema)
        .filter(VersionColumnSchema.dataset_version_id == dataset_version_id)
        .all()
    )
    mp = {r.column_name: r for r in rows}

    for col, (kind, conf) in inferred.items():
        col = str(col)
        kind = (kind or "other").lower()
        if kind not in ALLOWED:
            kind = "other"
        conf = float(conf or 0.0)

        r = mp.get(col)
        if r is None:
            r = VersionColumnSchema(
                project_id=project_id,
                dataset_version_id=dataset_version_id,
                column_name=col,
                inferred_kind=kind,
                confidence=conf,
            )
            db.add(r)
            mp[col] = r
        else:
            r.inferred_kind = kind
            r.confidence = conf

    db.commit()

    # refresh (optional)
    for r in mp.values():
        db.refresh(r)
    return mp

def set_overrides(
    db: Session,
    project_id: int,
    dataset_version_id: int,
    overrides: dict[str, str | None],
    commit: bool = True,
) -> dict[str, str | None]:
    """Persiste les overrides de kind. Retourne le diff effectivement appliqué
    {col: new_kind_or_None_for_clear} — utile pour générer le journal d'ops.

    Si `commit=False`, l'appelant gère la transaction (utile pour atomicité
    avec d'autres écritures, ex. ProcessingOperation).
    """
    rows = (
        db.query(VersionColumnSchema)
        .filter(VersionColumnSchema.dataset_version_id == dataset_version_id)
        .all()
    )
    mp = {r.column_name: r for r in rows}
    applied: dict[str, str | None] = {}

    for col, kind in overrides.items():
        col = str(col)
        if kind is not None:
            kind = str(kind).lower()
            if kind not in ALLOWED:
                continue

        r = mp.get(col)
        prev = r.override_kind if r is not None else None
        if prev == kind:
            continue  # no-op, ne pas générer d'op pour rien

        if r is None:
            r = VersionColumnSchema(
                project_id=project_id,
                dataset_version_id=dataset_version_id,
                column_name=col,
                inferred_kind="other",
                confidence=0.0,
                override_kind=kind,   # None => reset
            )
            db.add(r)
            mp[col] = r
        else:
            r.override_kind = kind

        applied[col] = kind

    if commit:
        db.commit()
    else:
        db.flush()
    return applied
