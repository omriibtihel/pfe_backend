"""
Hyperparameter normalization and shared coercion helpers.

Exposes:
  - normalize_model_hyperparams()       — public, used by trainer / orchestrator
  - _coerce_scalar_with_schema()        — used internally + by normalize_model_hyperparams
  - Shared parsing helpers used by config/schema/ submodules:
      _as_dict, _norm_choice, _to_bool, _to_optional_int, _to_optional_float,
      _migrate_legacy_numeric_transform, _normalize_model_hyperparams_payload,
      _is_linear_kernel
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from scipy import stats as _scipy_stats

from app.services.training.pipeline.models import MODEL_REGISTRY
from app.services.training.config.schema.types import MODEL_HP_SCHEMA

# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _norm_choice(
    raw: Any,
    allowed: list[str],
    default: str,
    aliases: dict[str, str] | None = None,
) -> str:
    v = str(raw or "").strip().lower()
    if not v:
        return default
    if aliases:
        v = aliases.get(v, v)
    return v if v in allowed else default


def _migrate_legacy_numeric_transform(
    raw_scaling: Any,
    raw_power_transform: Any,
) -> tuple[Any, Any]:
    """
    Preserve backward compatibility for legacy payloads that used
    numericScaling for power transforms before numericPowerTransform existed.
    """
    scaling_value = str(raw_scaling or "").strip().lower()
    power_value = str(raw_power_transform or "").strip().lower()

    if not power_value and scaling_value in {"yeo_johnson", "box_cox"}:
        return None, scaling_value
    return raw_scaling, raw_power_transform


def _to_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    # Unknown object types should never implicitly enable options.
    return default


def _to_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _to_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _to_optional_str_list(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out if out else None


def _normalize_model_hyperparams_payload(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for model_name_raw, params_raw in raw.items():
        model_name = MODEL_REGISTRY.normalize_name(str(model_name_raw or "").strip().lower())
        if not model_name:
            continue
        if not isinstance(params_raw, Mapping):
            out[model_name] = {}
            continue
        out[model_name] = {str(k): v for k, v in params_raw.items()}
    return out


def _is_linear_kernel(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "linear"
    if isinstance(value, list) and value:
        normalized = [str(v).strip().lower() for v in value]
        return all(v == "linear" for v in normalized)
    return False


# ---------------------------------------------------------------------------
# Scalar coercion with schema
# ---------------------------------------------------------------------------


def _coerce_scalar_with_schema(
    field_name: str,
    raw_value: Any,
    schema: dict[str, Any],
) -> tuple[Any, str | None]:
    field_type = str(schema.get("type", "")).strip().lower()
    enum_values = [str(v).strip().lower() for v in (schema.get("enum") or [])]
    default_value = schema.get("default")

    def _as_float(v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("bool is not accepted")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                raise ValueError("empty string")
            return float(txt)
        raise ValueError("not a float")

    def _as_int(v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("bool is not accepted")
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if not float(v).is_integer():
                raise ValueError("float must be integer-like")
            return int(v)
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                raise ValueError("empty string")
            val_float = float(txt)
            if not val_float.is_integer():
                raise ValueError("string float must be integer-like")
            return int(val_float)
        raise ValueError("not an int")

    def _check_bounds(v: float | int) -> str | None:
        if not math.isfinite(float(v)):
            return f"{field_name} must be a finite number"
        min_value = schema.get("min")
        max_value = schema.get("max")
        gt_value = schema.get("gt")
        ge_value = schema.get("ge")
        lt_value = schema.get("lt")
        le_value = schema.get("le")

        if min_value is not None and v < min_value:
            return f"{field_name} must be >= {min_value}"
        if max_value is not None and v > max_value:
            return f"{field_name} must be <= {max_value}"
        if gt_value is not None and not (v > gt_value):
            return f"{field_name} must be > {gt_value}"
        if ge_value is not None and not (v >= ge_value):
            return f"{field_name} must be >= {ge_value}"
        if lt_value is not None and not (v < lt_value):
            return f"{field_name} must be < {lt_value}"
        if le_value is not None and not (v <= le_value):
            return f"{field_name} must be <= {le_value}"
        return None

    try:
        if field_type == "int":
            value = _as_int(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "int_or_none":
            if raw_value is None:
                return None, None
            if isinstance(raw_value, str) and raw_value.strip().lower() in {"none", "null"}:
                return None, None
            value = _as_int(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "float":
            value = _as_float(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "enum":
            value = str(raw_value or "").strip().lower()
            if value in enum_values:
                return value, None
            return default_value, f"{field_name} must be one of: {', '.join(enum_values)}"

        if field_type == "float_or_enum":
            if isinstance(raw_value, str):
                txt = raw_value.strip().lower()
                if txt in enum_values:
                    return txt, None
            value = _as_float(raw_value)
            bound_error = _check_bounds(value)
            if bound_error:
                return default_value, bound_error
            return value, None

        if field_type == "enum_or_null":
            if raw_value is None:
                return None, None
            if isinstance(raw_value, str) and raw_value.strip().lower() in {"none", "null", ""}:
                return None, None
            normalized = str(raw_value).strip().lower()
            if normalized in enum_values:
                return normalized, None
            allowed = ", ".join(f'"{v}"' for v in enum_values)
            return default_value, f"{field_name} must be one of: {allowed} or null"

        if field_type == "str":
            value = str(raw_value or "").strip()
            if value:
                return value, None
            return default_value, f"{field_name} must be a non-empty string"

        return raw_value, None
    except Exception:
        return default_value, f"{field_name} has an invalid type"


# ---------------------------------------------------------------------------
# Search-value coercion (RandomSearch / HalvingRandom)
# ---------------------------------------------------------------------------


def _coerce_search_value(
    field_name: str,
    value: Any,
    schema_field: dict[str, Any],
) -> Any:
    """
    Coerce a hyperparameter value for a distribution-based search.

    - ``__range__`` dict  → scipy distribution (uniform / loguniform / randint).
    - list               → returned as-is (already valid for param_distributions).
    - scalar             → delegates to ``_coerce_scalar_with_schema``.

    Returns ``None`` when a ``__range__`` dict is malformed.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, dict) and value.get("__range__") is True:
        try:
            min_val = float(value["min"])
            max_val = float(value["max"])
        except (KeyError, TypeError, ValueError):
            return None

        if not (math.isfinite(min_val) and math.isfinite(max_val)):
            return None
        if min_val >= max_val:
            return None

        field_type = str(schema_field.get("type", "")).strip().lower()
        distribution = str(value.get("distribution", "uniform")).strip().lower()

        # Integer fields → discrete uniform over [low, high] inclusive.
        if field_type in ("int", "int_or_none"):
            low = int(math.ceil(min_val))
            high = int(math.floor(max_val))
            if low > high:
                return None
            # int_or_none with includeNull: sklearn can sample from a plain list,
            # so mix None with a capped set of integer candidates.
            if field_type == "int_or_none" and bool(value.get("includeNull")):
                n_ints = min(high - low + 1, 20)
                step = max(1, (high - low + 1) // n_ints)
                int_values: list[Any] = list(range(low, high + 1, step))
                return [None] + int_values
            return _scipy_stats.randint(low, high + 1)

        # Log-uniform: scipy.stats.loguniform(a, b) samples in [a, b] log-uniformly.
        if distribution == "log_uniform":
            if min_val <= 0:
                return None  # loguniform requires strictly positive bounds
            return _scipy_stats.loguniform(min_val, max_val)

        # Default: uniform over [min_val, max_val].
        return _scipy_stats.uniform(min_val, max_val - min_val)

    # Scalar fallback.
    coerced, _ = _coerce_scalar_with_schema(field_name, value, schema_field)
    return coerced


# ---------------------------------------------------------------------------
# Public: normalize_model_hyperparams
# ---------------------------------------------------------------------------


def normalize_model_hyperparams(
    model_key: str,
    params: dict[str, Any] | None,
    *,
    use_grid_search: bool,
    task_type: str | None = None,
) -> dict[str, Any]:
    model_name = MODEL_REGISTRY.normalize_name(str(model_key or "").strip().lower())
    schema = MODEL_HP_SCHEMA.get(model_name, {})
    source_raw = params if isinstance(params, dict) else {}
    schema_key_lookup = {str(field).strip().lower(): str(field) for field in schema.keys()}
    source: dict[str, Any] = {}
    unknown_fields: list[str] = []
    for raw_field, raw_value in source_raw.items():
        candidate = str(raw_field).strip()
        canonical_field = schema_key_lookup.get(candidate.lower())
        if canonical_field is None:
            unknown_fields.append(candidate)
            continue
        source[canonical_field] = raw_value

    effective: dict[str, Any] = {name: spec.get("default") for name, spec in schema.items()}
    estimator_params: dict[str, Any] = {name: spec.get("default") for name, spec in schema.items()}
    param_grid: dict[str, list[Any]] = {}
    param_distributions: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    issue_keys: set[tuple[str, str, str]] = set()

    def _push_issue(severity: str, code: str, message: str) -> None:
        key = (severity, code, message)
        if key in issue_keys:
            return
        issue_keys.add(key)
        issues.append(
            {
                "model": model_name,
                "severity": severity,
                "code": code,
                "message": message,
            }
        )
        if severity == "error":
            errors.append(message)
        else:
            warnings.append(message)

    for unknown_field in unknown_fields:
        _push_issue(
            "warning",
            "HP_UNKNOWN",
            f"{model_name}.{unknown_field} is not supported and was ignored.",
        )

    for field, field_schema in schema.items():
        if field not in source:
            continue

        raw_value = source.get(field)

        # ── __range__ dict → scipy distribution for RandomSearch / HalvingRandom ──
        if isinstance(raw_value, dict) and raw_value.get("__range__") is True:
            if use_grid_search:
                dist = _coerce_search_value(field, raw_value, field_schema)
                if dist is not None:
                    param_distributions[field] = dist
                    effective[field] = raw_value  # keep raw dict for display / logging
                else:
                    _push_issue(
                        "error",
                        "HP_INVALID",
                        f"{model_name}.{field} has an invalid range specification: "
                        f"min={raw_value.get('min')}, max={raw_value.get('max')}.",
                    )
            else:
                _push_issue(
                    "warning",
                    "HP_RANGE_IGNORED",
                    f"{model_name}.{field} received a range specification while "
                    f"useGridSearch=false; ignored.",
                )
            continue

        if isinstance(raw_value, list):
            if len(raw_value) == 0:
                _push_issue(
                    "error",
                    "HP_INVALID",
                    f"{model_name}.{field} list cannot be empty.",
                )
                continue
            if use_grid_search:
                normalized_values: list[Any] = []
                for idx, item in enumerate(raw_value):
                    normalized_value, value_error = _coerce_scalar_with_schema(field, item, field_schema)
                    if value_error:
                        _push_issue(
                            "error",
                            "HP_INVALID",
                            f"{model_name}.{field}[{idx}] is invalid: {value_error}.",
                        )
                        continue
                    normalized_values.append(normalized_value)

                if normalized_values:
                    effective[field] = normalized_values
                    param_grid[field] = normalized_values
                continue

            _push_issue(
                "warning",
                "HP_LIST_IGNORED",
                f"{model_name}.{field} received a list while useGridSearch=false; using the first value.",
            )
            raw_value = raw_value[0]

        normalized_scalar, scalar_error = _coerce_scalar_with_schema(field, raw_value, field_schema)
        if scalar_error:
            _push_issue(
                "error",
                "HP_INVALID",
                f"{model_name}.{field} is invalid: {scalar_error}.",
            )
            continue

        effective[field] = normalized_scalar
        estimator_params[field] = normalized_scalar

    if model_name == "svm" and _is_linear_kernel(effective.get("kernel")) and "gamma" in schema:
        gamma_was_requested = "gamma" in source
        if gamma_was_requested or "gamma" in param_grid or "gamma" in param_distributions:
            _push_issue(
                "warning",
                "HP_SVM_GAMMA_IGNORED",
                "svm.gamma is ignored when svm.kernel='linear'; gamma was removed.",
            )
        effective.pop("gamma", None)
        estimator_params.pop("gamma", None)
        param_grid.pop("gamma", None)
        param_distributions.pop("gamma", None)

    if model_name == "decisiontree" and str(task_type or "").strip().lower() == "regression":
        criterion_was_requested = "criterion" in source
        if criterion_was_requested or "criterion" in param_grid or "criterion" in param_distributions:
            _push_issue(
                "warning",
                "HP_TASK_INCOMPATIBLE",
                "decisiontree.criterion is classification-only and was removed for regression.",
            )
        effective.pop("criterion", None)
        estimator_params.pop("criterion", None)
        param_grid.pop("criterion", None)
        param_distributions.pop("criterion", None)

    # class_weight is classification-only — strip it unconditionally in regression.
    if "class_weight" in schema and str(task_type or "").strip().lower() == "regression":
        if "class_weight" in source:
            _push_issue(
                "warning",
                "HP_TASK_INCOMPATIBLE",
                f"{model_name}.class_weight is classification-only and was removed for regression.",
            )
        effective.pop("class_weight", None)
        estimator_params.pop("class_weight", None)
        param_grid.pop("class_weight", None)
        param_distributions.pop("class_weight", None)

    for key in list(estimator_params.keys()):
        if key in param_grid and isinstance(param_grid.get(key), list):
            # Keep the estimator fixed for scalar params only; grid params are set by GridSearchCV.
            estimator_params.pop(key, None)

    return {
        "model": model_name,
        "effective": effective,
        "estimator_params": estimator_params,
        "param_grid": param_grid,
        "param_distributions": param_distributions,
        "issues": issues,
        "warnings": warnings,
        "errors": errors,
    }
