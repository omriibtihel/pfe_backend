# [REFACTOR] Extracted from app/api/routes/nettoyage.py to keep the route layer
# free of pandas/numpy/business logic. Routes now orchestrate; this module owns
# DataFrame manipulation, payload validation, and disk-side persistence.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import PROJECTS_PATH
from app.crud import nettoyage as crud_nettoyage
from app.models.dataset_version import DatasetVersion
from app.services.data.column_inference import (
    _ALL_NONE,
    detect_parasites,
    infer_kind_for_series,
    numeric_extra_stats,
)


# -----------------------------
# Domain exceptions
# -----------------------------
class CleaningValidationError(ValueError):
    """Raised when a cleaning payload is structurally invalid."""


class SchemaDecisionValidationError(ValueError):
    """Raised when a schema-decision payload is structurally invalid."""


# -----------------------------
# Pydantic in-payloads
# -----------------------------
class AlertConfigOut(BaseModel):
    missing_high: float = 0.20
    missing_low: float = 0.05
    high_cardinality_ratio: float = 0.90
    high_cardinality_min_uniq: int = 10
    outlier_high: float = 0.15
    outlier_moderate: float = 0.05
    skewness: float = 2.0


class SchemaDecisionIn(BaseModel):
    schema_action: str
    column: str | None = None
    kind: str | None = None
    alert_key: str | None = None
    verified: bool | None = None
    dismissed: bool | None = None


# -----------------------------
# Allowed-action contracts
# -----------------------------
ALLOWED_CLEANING_ACTIONS = {
    "drop_columns",
    "drop_duplicates",
    "drop_empty_rows",
    "drop_empty_cols",
    "rename_columns",
    "strip_whitespace",
    "substitute_values",
}

ALLOWED_KINDS = {"numeric", "categorical", "datetime", "binary", "text", "id", "other"}
ALLOWED_SCHEMA_ACTIONS = {"set_kind", "clear_kind", "verify_categorical", "dismiss_alert"}


# -----------------------------
# DataFrame -> JSON payloads
# -----------------------------
def build_preview_payload(df: pd.DataFrame, page: int, page_size: int) -> dict:
    page = max(1, page)
    page_size = min(max(1, page_size), 200)

    start = (page - 1) * page_size
    end = start + page_size

    chunk = df.iloc[start:end].copy()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.astype(object)
    chunk = chunk.where(pd.notnull(chunk), None)

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "rows": chunk.to_dict(orient="records"),
        "page": page,
        "page_size": page_size,
        "total_rows": int(df.shape[0]),
    }


def build_columns_meta_payload(df: pd.DataFrame) -> dict:
    cols = [str(c) for c in df.columns]
    total_rows = int(df.shape[0])

    out_cols: list[dict] = []
    counts = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "binary": 0,
        "text": 0,
        "id": 0,
        "other": 0,
    }

    for c in cols:
        s = df[c]
        s2 = s.replace([np.inf, -np.inf], np.nan)

        missing = int(pd.isna(s2).sum())
        non_null = s2.dropna()
        unique = int(non_null.nunique(dropna=True))

        inferred, _ = infer_kind_for_series(c, s2)
        kind = inferred if inferred in counts else "other"
        counts[kind] += 1

        try:
            sample_vals = non_null.astype(str).head(5).tolist()
        except Exception:
            sample_vals = []

        extra = (
            numeric_extra_stats(non_null)
            if pd.api.types.is_numeric_dtype(s2)
            else dict(_ALL_NONE)
        )

        out_cols.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "kind": kind,
                "inferred_kind": inferred,
                "override_kind": None,
                "missing": missing,
                "unique": unique,
                "total": total_rows,
                "sample": sample_vals,
                "parasites": detect_parasites(s2),
                **extra,
            }
        )

    return {"columns": out_cols, "counts": counts, "total_rows": total_rows}


def compute_column_distribution(df: pd.DataFrame, column: str, max_bins: int) -> dict:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    series = df[column].dropna()
    total = len(series)
    n_unique = series.nunique()

    if n_unique <= max_bins or not pd.api.types.is_numeric_dtype(series):
        counts = series.astype(str).value_counts().head(max_bins)
        bars = [{"label": str(lbl), "count": int(cnt)} for lbl, cnt in counts.items()]
        return {"type": "categorical", "column": column, "total": total, "bars": bars}

    hist_counts, edges = np.histogram(series.astype(float).dropna(), bins=max_bins)
    bars = [
        {
            "label": f"{edges[i]:.2g}–{edges[i+1]:.2g}",
            "count": int(hist_counts[i]),
            "rangeMin": float(edges[i]),
            "rangeMax": float(edges[i + 1]),
        }
        for i in range(len(hist_counts))
    ]
    return {"type": "histogram", "column": column, "total": total, "bars": bars}


def merge_kind_overrides(payload: dict, overrides: dict) -> dict:
    overrides = overrides or {}
    for col in payload.get("columns", []):
        name = col.get("name")
        if name in overrides:
            col["override_kind"] = overrides[name]
            col["kind"] = overrides[name]
    return payload


# -----------------------------
# Payload validators (raise ValueError subclasses)
# -----------------------------
def validate_cleaning_payload(payload: Any, df_current: pd.DataFrame) -> str:
    if payload.type != "cleaning":
        raise CleaningValidationError(
            "Ce module est limité au nettoyage (cleaning). "
            "Les étapes ML (imputation/encoding/normalization) se font dans l'entraînement (après split)."
        )

    params = payload.params or {}
    action = params.get("action")
    if action not in ALLOWED_CLEANING_ACTIONS:
        raise CleaningValidationError(
            f"Cleaning: params.action doit être dans {sorted(ALLOWED_CLEANING_ACTIONS)}"
        )

    existing = set(map(str, df_current.columns))
    cols = [str(c) for c in (payload.columns or [])]
    missing = [c for c in cols if c not in existing]
    if missing:
        raise CleaningValidationError(f"Colonnes introuvables: {missing}")

    if action == "rename_columns":
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise CleaningValidationError(
                "rename_columns: params.mapping doit être un dict non vide."
            )

        old_cols = [str(k) for k in mapping.keys()]
        old_missing = [c for c in old_cols if c not in existing]
        if old_missing:
            raise CleaningValidationError(
                f"rename_columns: colonnes introuvables: {old_missing}"
            )

        new_names = [str(v) for v in mapping.values()]
        if len(set(new_names)) != len(new_names):
            raise CleaningValidationError(
                "rename_columns: deux colonnes ne peuvent pas avoir le même nouveau nom."
            )

    if action == "drop_columns" and not cols:
        raise CleaningValidationError("drop_columns: payload.columns ne doit pas être vide.")

    if action == "substitute_values":
        if len(cols) != 1:
            raise CleaningValidationError(
                "substitute_values: payload.columns doit contenir exactement 1 colonne."
            )
        treat_from_as_null = bool(params.get("treat_from_as_null", False))
        if not treat_from_as_null and params.get("from_value", None) is None:
            raise CleaningValidationError(
                "substitute_values: 'from_value' est requis (ou coche treat_from_as_null=true)."
            )
        if "case_sensitive" in params and not isinstance(
            params.get("case_sensitive"), (bool, int)
        ):
            raise CleaningValidationError(
                "substitute_values: 'case_sensitive' doit être un bool."
            )

    return action


def validate_schema_decision(payload: SchemaDecisionIn, df_current: pd.DataFrame) -> dict:
    action = (payload.schema_action or "").strip()
    if action not in ALLOWED_SCHEMA_ACTIONS:
        raise SchemaDecisionValidationError(
            f"schema_action doit être dans {sorted(ALLOWED_SCHEMA_ACTIONS)}"
        )

    existing = set(map(str, df_current.columns))

    if action in {"set_kind", "clear_kind", "verify_categorical"}:
        if not payload.column or not payload.column.strip():
            raise SchemaDecisionValidationError("schema: 'column' est requis.")
        if payload.column not in existing:
            raise SchemaDecisionValidationError(
                f"schema: colonne introuvable: {payload.column}"
            )

    if action == "set_kind":
        k = (payload.kind or "").strip().lower()
        if k not in ALLOWED_KINDS:
            raise SchemaDecisionValidationError(
                f"schema: 'kind' doit être dans {sorted(ALLOWED_KINDS)}"
            )

    if action == "verify_categorical":
        if payload.verified is None:
            raise SchemaDecisionValidationError(
                "schema: 'verified' est requis pour verify_categorical."
            )

    if action == "dismiss_alert":
        if not payload.alert_key or not payload.alert_key.strip():
            raise SchemaDecisionValidationError(
                "schema: 'alert_key' est requis pour dismiss_alert."
            )
        if payload.dismissed is None:
            raise SchemaDecisionValidationError(
                "schema: 'dismissed' est requis pour dismiss_alert."
            )

    params: dict = {"schema_action": action}
    if payload.column:
        params["column"] = payload.column
    if payload.kind:
        params["kind"] = (payload.kind or "").strip().lower()
    if payload.alert_key:
        params["alert_key"] = payload.alert_key
    if payload.verified is not None:
        params["verified"] = bool(payload.verified)
    if payload.dismissed is not None:
        params["dismissed"] = bool(payload.dismissed)
    return params


def build_schema_decision_description(params: dict) -> str:
    action = params["schema_action"]
    if action == "set_kind":
        return f"Schema: {params.get('column')} → {params.get('kind')}"
    if action == "clear_kind":
        return f"Schema: clear override ({params.get('column')})"
    if action == "verify_categorical":
        v = params.get("verified")
        return (
            f"Schema: verified categorical ({params.get('column')})"
            if v is True
            else f"Schema: unverified categorical ({params.get('column')})"
        )
    d = params.get("dismissed")
    return (
        f"Schema: dismissed alert ({params.get('alert_key')})"
        if d is True
        else f"Schema: undismissed alert ({params.get('alert_key')})"
    )


# -----------------------------
# Disk-side persistence
# -----------------------------
def export_dataframe_to_csv(
    df: pd.DataFrame,
    project_id: int,
    dataset_id: int,
    original_name: str,
) -> tuple[Path, str]:
    """Write *df* to a fresh CSV in the project's exports directory.

    Returns (tmp_path, download_name).
    """
    export_dir = PROJECTS_PATH / str(project_id) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(original_name).stem or f"dataset_{dataset_id}"
    download_name = f"{safe_stem}_cleaned.csv"

    tmp_path = export_dir / f"{uuid4().hex}.csv"
    df.to_csv(tmp_path, index=False)
    return tmp_path, download_name


def _serialize_operation(op: Any) -> dict:
    created_at = getattr(op, "created_at", None)
    return {
        "id": getattr(op, "id", None),
        "op_type": getattr(op, "op_type", None),
        "description": getattr(op, "description", None),
        "columns": getattr(op, "columns", None),
        "params": getattr(op, "params", None),
        "created_at": created_at.isoformat() if created_at else None,
    }


def save_cleaned_as_version(
    db: Session,
    project_id: int,
    dataset_id: int,
    src: Any,
    df: pd.DataFrame,
    body_name: str,
) -> dict:
    """Persist *df* as a new DatasetVersion, capturing the operation chain."""
    ops = crud_nettoyage.list_operations(db, project_id, dataset_id)
    ops_payload = [_serialize_operation(o) for o in ops]

    versions_dir = PROJECTS_PATH / str(project_id) / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(src.original_name).stem or f"dataset_{dataset_id}"
    version_name = body_name.strip() if body_name and body_name.strip() else f"{safe_stem}_cleaned"

    stored_name = f"{uuid4().hex}.csv"
    dst_path = versions_dir / stored_name

    df.to_csv(dst_path, index=False)
    size_bytes = dst_path.stat().st_size

    target_value = getattr(src, "target_column", None)
    if target_value and target_value not in [str(c) for c in df.columns]:
        target_value = None
    can_predict = bool(target_value)

    operations_json = json.dumps(ops_payload, ensure_ascii=False, default=str)

    src_kind = getattr(src, "kind", "source")
    if src_kind == "raw_workspace" and getattr(src, "workspace_source_dataset_id", None):
        real_source_id = src.workspace_source_dataset_id
    else:
        real_source_id = src.id

    new_version = DatasetVersion(
        project_id=project_id,
        source_dataset_id=real_source_id,
        name=version_name,
        stored_name=stored_name,
        file_path=str(dst_path),
        content_type="text/csv",
        size_bytes=size_bytes,
        target_column=target_value,
        can_predict=can_predict,
        operations_json=operations_json,
    )

    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    return {
        "version_id": new_version.id,
        "project_id": new_version.project_id,
        "source_dataset_id": new_version.source_dataset_id,
        "name": new_version.name,
        "file_path": new_version.file_path,
        "can_predict": new_version.can_predict,
        "created_at": new_version.created_at.isoformat() if new_version.created_at else None,
    }
