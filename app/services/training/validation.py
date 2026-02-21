from __future__ import annotations

from typing import Any

import pandas as pd

from .config import SUPPORTED_SPLIT_METHODS, TrainingConfig, normalize_model_hyperparams
from .imbalance import class_counts, minority_ratio
from .models import MODEL_REGISTRY, list_available_models
from .preprocessing import resolve_effective_preprocessing_by_column

_MISSING_FRIENDLY_MODELS = {"xgboost", "lightgbm"}
_SKLEARN_MODELS = set(MODEL_REGISTRY._specs.keys()) - _MISSING_FRIENDLY_MODELS
_SCALING_SENSITIVE_MODELS = {"svm", "knn", "logisticregression"}


def _append_unique(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def _preview_columns(columns: list[str], limit: int = 6) -> str:
    if not columns:
        return ""
    sample = columns[:limit]
    tail = f" (+{len(columns) - limit})" if len(columns) > limit else ""
    return ", ".join(sample) + tail


def _infer_column_from_message(message: str, column_names: list[str]) -> str | None:
    haystack = str(message or "").lower()
    if not haystack:
        return None
    for col in sorted([str(c) for c in column_names], key=len, reverse=True):
        if col and col.lower() in haystack:
            return col
    return None


def _build_error_details(
    errors: list[str],
    warnings: list[str],
    column_names: list[str],
    *,
    skip_messages: set[str] | None = None,
    extra_details: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    skipped = skip_messages or set()
    details: list[dict[str, Any]] = []
    for message in errors:
        if str(message) in skipped:
            continue
        details.append(
            {
                "column": _infer_column_from_message(message, column_names),
                "severity": "error",
                "code": "validation_error",
                "message": str(message),
            }
        )
    for message in warnings:
        if str(message) in skipped:
            continue
        details.append(
            {
                "column": _infer_column_from_message(message, column_names),
                "severity": "warning",
                "code": "validation_warning",
                "message": str(message),
            }
        )
    if extra_details:
        details.extend(extra_details)
    return details


def _is_zero_one_binary(values: Any) -> bool:
    try:
        normalized = {float(v) for v in values}
    except Exception:
        return False
    return normalized == {0.0, 1.0}


def _normalize_selected_models(cfg: TrainingConfig) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    normalized: list[str] = []

    available = {str(m.get("name")): bool(m.get("installed", False)) for m in list_available_models()}
    known_models = set(MODEL_REGISTRY._specs.keys())

    for raw in cfg.models:
        name = MODEL_REGISTRY.normalize_name(str(raw))
        if name not in known_models:
            _append_unique(errors, f"Modele inconnu: '{raw}'.")
            continue
        if name in available and not available[name]:
            _append_unique(errors, f"Modele '{name}' non installe.")
            continue
        try:
            MODEL_REGISTRY.get_spec(name, cfg.task_type)
        except Exception as exc:
            _append_unique(errors, str(exc))
            continue
        if name not in normalized:
            normalized.append(name)

    return normalized, errors, [m for m in normalized if m in _MISSING_FRIENDLY_MODELS]


def _normalize_payload_model_hyperparams(
    cfg: TrainingConfig,
    *,
    selected_models: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    errors: list[str] = []
    details: list[dict[str, Any]] = []
    known_models = set(MODEL_REGISTRY._specs.keys())
    selected_set = set(str(m) for m in selected_models)
    requested = cfg.model_hyperparams if isinstance(cfg.model_hyperparams, dict) else {}

    detail_seen: set[tuple[str, str, str | None, str]] = set()

    def _push_detail(
        *,
        severity: str,
        code: str,
        message: str,
        model: str | None,
    ) -> None:
        key = (severity, code, model, message)
        if key in detail_seen:
            return
        detail_seen.add(key)
        details.append(
            {
                "model": model,
                "severity": severity,
                "code": code,
                "message": message,
            }
        )
        if severity == "error":
            _append_unique(errors, message)
        else:
            _append_unique(warnings, message)

    for model_name_raw in requested.keys():
        model_name = MODEL_REGISTRY.normalize_name(str(model_name_raw))
        if model_name not in known_models:
            _push_detail(
                severity="warning",
                code="HP_MODEL_UNKNOWN",
                model=str(model_name_raw),
                message=f"modelHyperparams.{model_name_raw} is unknown and was ignored.",
            )
            continue
        if model_name not in selected_set:
            _push_detail(
                severity="warning",
                code="HP_MODEL_NOT_SELECTED",
                model=model_name,
                message=f"modelHyperparams.{model_name} was provided but model '{model_name}' is not selected.",
            )

    normalized: dict[str, dict[str, Any]] = {}
    for model_name in selected_models:
        requested_params = requested.get(model_name, {})
        if not isinstance(requested_params, dict):
            _push_detail(
                severity="warning",
                code="HP_INVALID",
                model=model_name,
                message=f"modelHyperparams.{model_name} must be an object; using defaults.",
            )
            requested_params = {}

        result = normalize_model_hyperparams(
            model_name,
            requested_params,
            use_grid_search=bool(cfg.use_grid_search),
            task_type=cfg.task_type,
        )
        normalized[model_name] = dict(result.get("effective") or {})

        for detail in result.get("issues", []) or []:
            severity = str(detail.get("severity") or "warning").strip().lower()
            code = str(detail.get("code") or "HP").strip()
            message = str(detail.get("message") or "")
            model = str(detail.get("model") or model_name)
            if not message:
                continue
            _push_detail(
                severity="error" if severity == "error" else "warning",
                code=code,
                model=model,
                message=message,
            )

    return normalized, warnings, errors, details


def validate_training_config_payload(payload: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    cfg = TrainingConfig.from_front(payload)
    warnings: list[str] = []
    errors: list[str] = []
    hp_detail_messages: set[str] = set()
    hp_details: list[dict[str, Any]] = []
    effective_by_column: dict[str, dict[str, Any]] = {}
    all_feature_columns = [str(col) for col in df.columns if str(col) != str(cfg.target_column)]

    models, model_errors, missing_friendly_models = _normalize_selected_models(cfg)
    errors.extend(model_errors)
    normalized_model_hyperparams, hp_warnings, hp_errors, hp_details = _normalize_payload_model_hyperparams(
        cfg,
        selected_models=models,
    )
    warnings.extend(hp_warnings)
    errors.extend(hp_errors)
    hp_detail_messages = {str(d.get("message")) for d in hp_details if str(d.get("message", "")).strip()}

    if not cfg.target_column:
        _append_unique(errors, "targetColumn est requis.")
    if not models:
        _append_unique(errors, "Aucun modele valide selectionne.")
    if cfg.split_method not in SUPPORTED_SPLIT_METHODS:
        _append_unique(
            errors,
            f"splitMethod='{cfg.split_method}' non supporte. Methodes supportees: {', '.join(SUPPORTED_SPLIT_METHODS)}.",
        )

    if cfg.target_column and cfg.target_column not in df.columns:
        _append_unique(errors, f"La colonne cible '{cfg.target_column}' est introuvable dans le dataset.")

    if errors:
        normalized = cfg.as_dict()
        normalized["models"] = models
        normalized["modelHyperparams"] = normalized_model_hyperparams
        return {
            "normalized_config": normalized,
            "effective_preprocessing_by_column": effective_by_column,
            "warnings": warnings,
            "errors": errors,
            "error_details": _build_error_details(
                errors,
                warnings,
                all_feature_columns,
                skip_messages=hp_detail_messages,
                extra_details=hp_details,
            ),
        }

    df2 = df[df[cfg.target_column].notna()].copy()
    if df2.empty:
        _append_unique(errors, "Aucune ligne exploitable: la cible est vide (NaN) partout.")
        normalized = cfg.as_dict()
        normalized["models"] = models
        normalized["modelHyperparams"] = normalized_model_hyperparams
        return {
            "normalized_config": normalized,
            "effective_preprocessing_by_column": effective_by_column,
            "warnings": warnings,
            "errors": errors,
            "error_details": _build_error_details(
                errors,
                warnings,
                all_feature_columns,
                skip_messages=hp_detail_messages,
                extra_details=hp_details,
            ),
        }

    X = df2.drop(columns=[cfg.target_column])
    if cfg.task_type == "classification":
        y_values = df2[cfg.target_column].to_numpy()
        unique_values = pd.unique(y_values)
        if len(unique_values) == 2 and not _is_zero_one_binary(unique_values) and cfg.positive_label is None:
            _append_unique(
                warnings,
                "positiveLabel absent sur une classification binaire non {0,1}: "
                "le backend auto-resoudra la classe positive (instable). Fournis positiveLabel explicitement.",
            )

    effective = resolve_effective_preprocessing_by_column(X, cfg.preprocessing)
    effective_by_column = effective.effective_by_column

    active_columns = [col for col, col_cfg in effective_by_column.items() if bool(col_cfg.get("use", True))]
    if not active_columns:
        _append_unique(
            errors,
            "Aucune colonne active: toutes les colonnes sont marquees use=false. "
            "Active au moins une colonne pour lancer l'entrainement.",
        )

    sklearn_models = [m for m in models if m in _SKLEARN_MODELS]

    categorical_no_encoding_cols = [
        col
        for col, col_cfg in effective_by_column.items()
        if bool(col_cfg.get("use", True))
        and str(col_cfg.get("type")) in {"categorical", "ordinal"}
        and str(col_cfg.get("categoricalEncoding", "none")) == "none"
    ]
    if categorical_no_encoding_cols and sklearn_models:
        _append_unique(
            errors,
            "categoricalEncoding='none' sur colonnes categorielles/ordinales avec modeles sklearn "
            f"({', '.join(sklearn_models)}). Colonnes: {_preview_columns(categorical_no_encoding_cols)}.",
        )

    numeric_nan_cells = 0
    categorical_nan_cells = 0
    cols_with_nan_and_none_imputation: list[str] = []
    for col, col_cfg in effective_by_column.items():
        if not bool(col_cfg.get("use", True)):
            continue
        missing_cells = int(X[col].isna().sum())
        if missing_cells <= 0:
            continue

        col_type = str(col_cfg.get("type", "categorical"))
        if col_type == "numeric":
            numeric_nan_cells += missing_cells
            if str(col_cfg.get("numericImputation", "none")) == "none":
                cols_with_nan_and_none_imputation.append(col)
        else:
            categorical_nan_cells += missing_cells
            if str(col_cfg.get("categoricalImputation", "none")) == "none":
                cols_with_nan_and_none_imputation.append(col)

    if cols_with_nan_and_none_imputation:
        if sklearn_models:
            _append_unique(
                errors,
                "NaN detectes sur colonnes actives avec imputation='none' pour modeles sklearn "
                f"({', '.join(sklearn_models)}). Colonnes: {_preview_columns(cols_with_nan_and_none_imputation)}.",
            )
        if missing_friendly_models:
            _append_unique(
                warnings,
                "NaN detectes avec imputation='none' sur colonnes actives. "
                f"Modeles tolerants au missing selectionnes: {', '.join(missing_friendly_models)}. "
                f"Colonnes: {_preview_columns(cols_with_nan_and_none_imputation)}.",
            )

    numeric_scaling_none_cols = [
        col
        for col, col_cfg in effective_by_column.items()
        if bool(col_cfg.get("use", True))
        and str(col_cfg.get("type")) == "numeric"
        and str(col_cfg.get("numericScaling", "none")) == "none"
    ]
    sensitive_models = [m for m in models if m in _SCALING_SENSITIVE_MODELS]
    if numeric_scaling_none_cols and sensitive_models:
        _append_unique(
            warnings,
            "numericScaling='none' peut degrader les performances pour "
            f"{', '.join(sensitive_models)} sur colonnes numeriques actives: {_preview_columns(numeric_scaling_none_cols)}.",
        )

    numeric_imputation_enabled = [
        col
        for col, col_cfg in effective_by_column.items()
        if bool(col_cfg.get("use", True))
        and str(col_cfg.get("type")) == "numeric"
        and str(col_cfg.get("numericImputation", "none")) != "none"
    ]
    if numeric_imputation_enabled and numeric_nan_cells == 0:
        _append_unique(
            warnings,
            "Aucune valeur manquante numerique detectee sur colonnes actives: imputation numerique probablement inutile.",
        )

    categorical_imputation_enabled = [
        col
        for col, col_cfg in effective_by_column.items()
        if bool(col_cfg.get("use", True))
        and str(col_cfg.get("type")) in {"categorical", "ordinal"}
        and str(col_cfg.get("categoricalImputation", "none")) != "none"
    ]
    if categorical_imputation_enabled and categorical_nan_cells == 0:
        _append_unique(
            warnings,
            "Aucune valeur manquante categorielle/ordinale detectee sur colonnes actives: "
            "imputation categorielle probablement inutile.",
        )

    requested_rebalancing = str(getattr(cfg, "rebalance_method", "none") or "none").strip().lower()
    if requested_rebalancing != "none":
        if cfg.task_type != "classification":
            _append_unique(
                warnings,
                f"Reequilibrage '{requested_rebalancing}' non applicable hors classification.",
            )
        elif not cfg.use_smote:
            _append_unique(
                warnings,
                f"Reequilibrage '{requested_rebalancing}' demande mais useSmote=false.",
            )

    if cfg.use_smote:
        if cfg.task_type != "classification":
            _append_unique(warnings, "SMOTE non applicable hors classification.")
        else:
            y = df2[cfg.target_column].to_numpy()
            counts = class_counts(y)
            if len(counts) < 2:
                _append_unique(warnings, "SMOTE non applicable: une seule classe detectee.")
            elif min(counts.values()) < 6:
                _append_unique(warnings, "SMOTE potentiellement non applicable: classe minoritaire trop petite (<6).")
            ratio = minority_ratio(y)
            if ratio is not None and ratio >= 0.35:
                _append_unique(warnings, "SMOTE selectionne alors que le desequilibre semble modere.")

    normalized = cfg.as_dict()
    normalized["models"] = models
    normalized["modelHyperparams"] = normalized_model_hyperparams
    normalized["preprocessingSummary"] = {
        "numericColumns": int(len(effective.active_numeric_columns)),
        "categoricalColumns": int(len(effective.active_categorical_columns)),
        "numericMissingCells": int(numeric_nan_cells),
        "categoricalMissingCells": int(categorical_nan_cells),
    }

    return {
        "normalized_config": normalized,
        "effective_preprocessing_by_column": effective_by_column,
        "warnings": warnings,
        "errors": errors,
        "error_details": _build_error_details(
            errors,
            warnings,
            list(effective_by_column.keys()),
            skip_messages=hp_detail_messages,
            extra_details=hp_details,
        ),
    }
