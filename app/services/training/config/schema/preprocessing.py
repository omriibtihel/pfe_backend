"""
PreprocessingDefaults and PreprocessingConfig frozen dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.training.config.normalization import (
    _as_dict,
    _migrate_legacy_numeric_transform,
    _norm_choice,
    _to_bool,
    _to_optional_str_list,
)
from app.services.training.config.schema.types import (
    CATEGORICAL_ENCODING_METHODS,
    CATEGORICAL_IMPUTATION_METHODS,
    DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_CATEGORICAL_IMPUTATION,
    DEFAULT_NUMERIC_IMPUTATION,
    DEFAULT_NUMERIC_POWER_TRANSFORM,
    DEFAULT_NUMERIC_SCALING,
    NUMERIC_IMPUTATION_METHODS,
    NUMERIC_POWER_TRANSFORM_METHODS,
    NUMERIC_SCALING_METHODS,
)


# ---------------------------------------------------------------------------
# Private helpers (local — only needed in this file)
# ---------------------------------------------------------------------------


def _norm_column_type(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip().lower()
    if value in {"numeric", "categorical", "ordinal"}:
        return value
    return fallback


def _copy_column_overrides(columns: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col, cfg in columns.items():
        cloned = dict(cfg)
        if isinstance(cloned.get("ordinalOrder"), list):
            cloned["ordinalOrder"] = [str(v) for v in cloned["ordinalOrder"]]
        out[str(col)] = cloned
    return out


# ---------------------------------------------------------------------------
# PreprocessingDefaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingDefaults:
    numeric_imputation: str = DEFAULT_NUMERIC_IMPUTATION
    categorical_imputation: str = DEFAULT_CATEGORICAL_IMPUTATION
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING
    numeric_scaling: str = DEFAULT_NUMERIC_SCALING
    numeric_power_transform: str = DEFAULT_NUMERIC_POWER_TRANSFORM

    def as_dict(self) -> dict[str, str]:
        return {
            "numericImputation": self.numeric_imputation,
            "numericPowerTransform": self.numeric_power_transform,
            "numericScaling": self.numeric_scaling,
            "categoricalImputation": self.categorical_imputation,
            "categoricalEncoding": self.categorical_encoding,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "imputation": {
                "numeric": self.numeric_imputation,
                "categorical": self.categorical_imputation,
            },
            "encoding": {
                "categorical": self.categorical_encoding,
            },
            "scaling": {
                "numeric": self.numeric_scaling,
            },
            "normalization": {
                "numeric": self.numeric_power_transform,
            },
        }


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingConfig:
    defaults: PreprocessingDefaults = field(default_factory=PreprocessingDefaults)
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Advanced params — exposed to frontend so nothing is hardcoded silently
    knn_n_neighbors: int = 5
    constant_fill_numeric: float = 0.0
    constant_fill_categorical: str = "__missing__"
    variance_threshold: float = 0.01

    @staticmethod
    def from_front(payload: dict[str, Any]) -> "PreprocessingConfig":
        pp = _as_dict(payload.get("preprocessing"))
        defaults = _as_dict(pp.get("defaults"))
        imp = _as_dict(pp.get("imputation"))
        enc = _as_dict(pp.get("encoding"))
        scl = _as_dict(pp.get("scaling"))
        nrm = _as_dict(pp.get("normalization"))
        raw_numeric_scaling = defaults.get(
            "numericScaling",
            pp.get("numericScaling", scl.get("numeric", payload.get("numericScaling"))),
        )
        raw_numeric_power_transform = defaults.get(
            "numericPowerTransform",
            pp.get("numericPowerTransform", nrm.get("numeric", payload.get("numericPowerTransform"))),
        )
        raw_numeric_scaling, raw_numeric_power_transform = _migrate_legacy_numeric_transform(
            raw_numeric_scaling,
            raw_numeric_power_transform,
        )

        numeric_imputation = _norm_choice(
            defaults.get("numericImputation", pp.get("numericImputation", imp.get("numeric", payload.get("numericImputation")))),
            NUMERIC_IMPUTATION_METHODS,
            DEFAULT_NUMERIC_IMPUTATION,
        )
        categorical_imputation = _norm_choice(
            defaults.get(
                "categoricalImputation",
                pp.get("categoricalImputation", imp.get("categorical", payload.get("categoricalImputation"))),
            ),
            CATEGORICAL_IMPUTATION_METHODS,
            DEFAULT_CATEGORICAL_IMPUTATION,
        )
        categorical_encoding = _norm_choice(
            defaults.get(
                "categoricalEncoding",
                pp.get("categoricalEncoding", enc.get("categorical", payload.get("categoricalEncoding"))),
            ),
            CATEGORICAL_ENCODING_METHODS,
            DEFAULT_CATEGORICAL_ENCODING,
            aliases={
                "labelencoding": "label",
                "label_encoder": "label",
                "ordinalencoding": "ordinal",
            },
        )
        numeric_scaling = _norm_choice(
            raw_numeric_scaling,
            NUMERIC_SCALING_METHODS,
            DEFAULT_NUMERIC_SCALING,
        )
        numeric_power_transform = _norm_choice(
            raw_numeric_power_transform,
            NUMERIC_POWER_TRANSFORM_METHODS,
            DEFAULT_NUMERIC_POWER_TRANSFORM,
        )

        resolved_defaults = PreprocessingDefaults(
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            categorical_encoding=categorical_encoding,
            numeric_scaling=numeric_scaling,
            numeric_power_transform=numeric_power_transform,
        )

        columns_raw = _as_dict(pp.get("columns"))
        normalized_columns: dict[str, dict[str, Any]] = {}
        for raw_col, raw_cfg in columns_raw.items():
            col = str(raw_col or "").strip()
            if not col:
                continue

            cfg_col = _as_dict(raw_cfg)
            out: dict[str, Any] = {}

            if "use" in cfg_col:
                out["use"] = _to_bool(cfg_col.get("use"), default=True)

            if "type" in cfg_col:
                normalized_type = str(cfg_col.get("type") or "").strip().lower()
                if normalized_type in {"numeric", "categorical", "ordinal"}:
                    out["type"] = normalized_type

            if "numericImputation" in cfg_col:
                out["numericImputation"] = _norm_choice(
                    cfg_col.get("numericImputation"),
                    NUMERIC_IMPUTATION_METHODS,
                    resolved_defaults.numeric_imputation,
                )
            raw_col_numeric_scaling, raw_col_numeric_power_transform = _migrate_legacy_numeric_transform(
                cfg_col.get("numericScaling"),
                cfg_col.get("numericPowerTransform"),
            )
            if "numericScaling" in cfg_col:
                out["numericScaling"] = _norm_choice(
                    raw_col_numeric_scaling,
                    NUMERIC_SCALING_METHODS,
                    resolved_defaults.numeric_scaling,
                )
            if "numericPowerTransform" in cfg_col or (
                "numericScaling" in cfg_col
                and str(cfg_col.get("numericScaling") or "").strip().lower() in {"yeo_johnson", "box_cox"}
            ):
                out["numericPowerTransform"] = _norm_choice(
                    raw_col_numeric_power_transform,
                    NUMERIC_POWER_TRANSFORM_METHODS,
                    resolved_defaults.numeric_power_transform,
                )
            if "categoricalImputation" in cfg_col:
                out["categoricalImputation"] = _norm_choice(
                    cfg_col.get("categoricalImputation"),
                    CATEGORICAL_IMPUTATION_METHODS,
                    resolved_defaults.categorical_imputation,
                )
            if "categoricalEncoding" in cfg_col:
                out["categoricalEncoding"] = _norm_choice(
                    cfg_col.get("categoricalEncoding"),
                    CATEGORICAL_ENCODING_METHODS,
                    resolved_defaults.categorical_encoding,
                    aliases={
                        "labelencoding": "label",
                        "label_encoder": "label",
                        "ordinalencoding": "ordinal",
                    },
                )

            if "ordinalOrder" in cfg_col:
                order = _to_optional_str_list(cfg_col.get("ordinalOrder"))
                if order is not None:
                    out["ordinalOrder"] = order

            # Per-column advanced params — override globals when explicitly set
            if "knnNeighbors" in cfg_col:
                try:
                    out["knnNeighbors"] = max(1, min(50, int(cfg_col["knnNeighbors"])))
                except (TypeError, ValueError):
                    pass
            if "constantFillNumeric" in cfg_col and cfg_col["constantFillNumeric"] is not None:
                try:
                    out["constantFillNumeric"] = float(cfg_col["constantFillNumeric"])
                except (TypeError, ValueError):
                    pass
            if "constantFillCategorical" in cfg_col:
                raw_cat = str(cfg_col["constantFillCategorical"] or "").strip()
                if raw_cat:
                    out["constantFillCategorical"] = raw_cat

            if out:
                normalized_columns[col] = out

        adv = _as_dict(pp.get("advancedParams"))
        knn_n_neighbors = max(1, min(50, int(adv.get("knnNeighbors") or 5)))
        constant_fill_numeric = float(adv.get("constantFillNumeric") if adv.get("constantFillNumeric") is not None else 0.0)
        raw_cat_fill = str(adv.get("constantFillCategorical") or "").strip()
        constant_fill_categorical = raw_cat_fill if raw_cat_fill else "__missing__"
        variance_threshold = max(0.0, min(1.0, float(adv.get("varianceThreshold") or 0.01)))

        return PreprocessingConfig(
            defaults=resolved_defaults,
            columns=normalized_columns,
            knn_n_neighbors=knn_n_neighbors,
            constant_fill_numeric=constant_fill_numeric,
            constant_fill_categorical=constant_fill_categorical,
            variance_threshold=variance_threshold,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults.as_dict(),
            "columns": _copy_column_overrides(self.columns),
            "advancedParams": {
                "knnNeighbors": self.knn_n_neighbors,
                "constantFillNumeric": self.constant_fill_numeric,
                "constantFillCategorical": self.constant_fill_categorical,
                "varianceThreshold": self.variance_threshold,
            },
        }

    @property
    def numeric_imputation(self) -> str:
        return self.defaults.numeric_imputation

    @property
    def numeric_scaling(self) -> str:
        return self.defaults.numeric_scaling

    @property
    def numeric_power_transform(self) -> str:
        return self.defaults.numeric_power_transform

    @property
    def categorical_imputation(self) -> str:
        return self.defaults.categorical_imputation

    @property
    def categorical_encoding(self) -> str:
        return self.defaults.categorical_encoding

    def resolve_column(self, column: str, inferred_type: str) -> dict[str, Any]:
        inferred = _norm_column_type(inferred_type, "categorical")
        raw = self.columns.get(column, {})
        cfg = raw if isinstance(raw, dict) else {}

        use = bool(cfg.get("use", True))
        final_type = _norm_column_type(cfg.get("type"), inferred)
        if final_type not in {"numeric", "categorical", "ordinal"}:
            final_type = inferred

        resolved = {
            "use": use,
            "inferredType": inferred,
            "type": final_type,
            "numericImputation": _norm_choice(
                cfg.get("numericImputation"),
                NUMERIC_IMPUTATION_METHODS,
                self.defaults.numeric_imputation,
            ),
            "numericPowerTransform": _norm_choice(
                cfg.get("numericPowerTransform"),
                NUMERIC_POWER_TRANSFORM_METHODS,
                self.defaults.numeric_power_transform,
            ),
            "numericScaling": _norm_choice(
                cfg.get("numericScaling"),
                NUMERIC_SCALING_METHODS,
                self.defaults.numeric_scaling,
            ),
            "categoricalImputation": _norm_choice(
                cfg.get("categoricalImputation"),
                CATEGORICAL_IMPUTATION_METHODS,
                self.defaults.categorical_imputation,
            ),
            "categoricalEncoding": _norm_choice(
                cfg.get("categoricalEncoding"),
                CATEGORICAL_ENCODING_METHODS,
                self.defaults.categorical_encoding,
                aliases={
                    "labelencoding": "label",
                    "label_encoder": "label",
                    "ordinalencoding": "ordinal",
                },
            ),
        }
        if final_type == "ordinal":
            order = _to_optional_str_list(cfg.get("ordinalOrder"))
            if order is not None:
                resolved["ordinalOrder"] = order

        # Per-column advanced params (fall back to global defaults when absent)
        resolved["knnNeighbors"] = int(cfg["knnNeighbors"]) if "knnNeighbors" in cfg else self.knn_n_neighbors
        resolved["constantFillNumeric"] = float(cfg["constantFillNumeric"]) if "constantFillNumeric" in cfg else self.constant_fill_numeric
        resolved["constantFillCategorical"] = str(cfg["constantFillCategorical"]) if "constantFillCategorical" in cfg else self.constant_fill_categorical

        return resolved

    def defaults_dict(self) -> dict[str, str]:
        return {
            "numericImputation": self.defaults.numeric_imputation,
            "numericPowerTransform": self.defaults.numeric_power_transform,
            "numericScaling": self.defaults.numeric_scaling,
            "categoricalImputation": self.defaults.categorical_imputation,
            "categoricalEncoding": self.defaults.categorical_encoding,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        return self.defaults.as_legacy_dict()
