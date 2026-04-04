from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from app.services.training.config.schema import PreprocessingConfig


@dataclass(frozen=True)
class PreprocessSpec:
    preprocessor: Any
    numeric_cols: List[str]
    categorical_cols: List[str]
    selected_methods: dict[str, Any]
    effective_by_column: dict[str, dict[str, Any]]
    dropped_columns: List[str]
    column_types: dict[str, str]
    feature_selector: Any = None  # VarianceThreshold — fitted in orchestrator, not here


@dataclass(frozen=True)
class EffectivePreprocessing:
    effective_by_column: dict[str, dict[str, Any]]
    dropped_columns: List[str]
    column_types: dict[str, str]
    active_numeric_columns: List[str]
    active_categorical_columns: List[str]


def infer_columns(X: pd.DataFrame, numeric_threshold: float = 0.85) -> Tuple[List[str], List[str]]:
    """
    Règles pro (simples mais robustes) :
    1) Si dtype est déjà numérique -> numeric
    2) Sinon, on tente conversion en numérique et on calcule la fraction convertible
       en ignorant les NaN d'origine.
    """
    num_cols: List[str] = []
    cat_cols: List[str] = []

    for col in X.columns:
        s = X[col]

        # 1) dtype numérique => numeric (même avec NaN)
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num_cols.append(col)
            continue

        # 2) tentative conversion
        s_non_null = s.dropna()
        if len(s_non_null) == 0:
            # colonne vide -> catégoriel par défaut
            cat_cols.append(col)
            continue

        converted = pd.to_numeric(s_non_null, errors="coerce")
        frac_numeric = converted.notna().mean()

        if frac_numeric >= numeric_threshold:
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols


def _build_numeric_imputer(method: str, knn_n_neighbors: int = 5, constant_fill: float = 0.0):
    if method == "none":
        return "passthrough"
    if method == "knn":
        return KNNImputer(n_neighbors=knn_n_neighbors)
    if method == "constant":
        return SimpleImputer(strategy="constant", fill_value=constant_fill)
    return SimpleImputer(strategy=method)


def _build_numeric_scaler(method: str):
    if method == "none":
        return "passthrough"
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler(with_centering=True, with_scaling=True)
    if method == "maxabs":
        return MaxAbsScaler()
    # standard (default): z-score with centering for numeric columns.
    # Numeric branches run on dense dataframe subsets, so centering is expected.
    return StandardScaler(with_mean=True, with_std=True)


def _build_categorical_imputer(method: str, constant_fill: str = "__missing__"):
    if method == "none":
        return "passthrough"
    if method == "constant":
        return SimpleImputer(strategy="constant", fill_value=constant_fill)
    return SimpleImputer(strategy=method)


def _build_categorical_encoder(method: str, ordinal_order: list[str] | None = None, n_features: int = 1):
    if method == "none":
        return "passthrough"
    if method == "onehot":
        # Keep a dedicated column for each observed category.
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    # "label" and "ordinal" both map to OrdinalEncoder for feature columns.
    if ordinal_order:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            categories=[list(ordinal_order) for _ in range(max(1, int(n_features)))],
        )
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )


def resolve_effective_preprocessing_by_column(X: pd.DataFrame, prep_cfg: PreprocessingConfig) -> EffectivePreprocessing:
    numeric_cols, categorical_cols = infer_columns(X)
    inferred_types: dict[str, str] = {col: "numeric" for col in numeric_cols}
    inferred_types.update({col: "categorical" for col in categorical_cols})

    effective_by_column: dict[str, dict[str, Any]] = {}
    dropped_columns: List[str] = []
    column_types: dict[str, str] = {}
    active_numeric_columns: List[str] = []
    active_categorical_columns: List[str] = []

    for col in X.columns:
        inferred = inferred_types.get(col, "categorical")
        resolved = prep_cfg.resolve_column(str(col), inferred)
        final_type = str(resolved.get("type", inferred))
        use = bool(resolved.get("use", True))

        if final_type not in {"numeric", "categorical", "ordinal"}:
            final_type = inferred if inferred in {"numeric", "categorical"} else "categorical"
            resolved["type"] = final_type

        passthrough = False
        if use:
            if final_type == "numeric":
                active_numeric_columns.append(str(col))
                passthrough = (
                    str(resolved.get("numericImputation", "none")) == "none"
                    and str(resolved.get("numericScaling", "none")) == "none"
                )
            else:
                active_categorical_columns.append(str(col))
                passthrough = (
                    str(resolved.get("categoricalImputation", "none")) == "none"
                    and str(resolved.get("categoricalEncoding", "none")) == "none"
                )
        else:
            dropped_columns.append(str(col))

        resolved["passthrough"] = bool(passthrough and use)
        effective_by_column[str(col)] = resolved
        column_types[str(col)] = final_type

    return EffectivePreprocessing(
        effective_by_column=effective_by_column,
        dropped_columns=dropped_columns,
        column_types=column_types,
        active_numeric_columns=active_numeric_columns,
        active_categorical_columns=active_categorical_columns,
    )


def build_preprocessor(X: pd.DataFrame, prep_cfg: PreprocessingConfig) -> PreprocessSpec:
    if X is None or len(X) == 0:
        raise RuntimeError("build_preprocessor expects a non-empty TRAIN split.")

    effective = resolve_effective_preprocessing_by_column(X, prep_cfg)
    effective_by_column = effective.effective_by_column

    active_columns = [col for col in X.columns if bool(effective_by_column.get(str(col), {}).get("use", True))]
    if not active_columns:
        raise RuntimeError(
            "No feature columns are enabled for preprocessing/training. "
            "Set at least one column to use=true."
        )

    # Keys include per-column advanced params so columns with different
    # k / fill_value get separate Pipeline instances.
    numeric_groups: dict[tuple[str, str, int, float], list[str]] = {}
    categorical_groups: dict[tuple[str, str, tuple[str, ...], str], list[str]] = {}
    passthrough_cols: list[str] = []

    for col in X.columns:
        col_name = str(col)
        cfg = effective_by_column[col_name]
        if not bool(cfg.get("use", True)):
            continue

        col_type = str(cfg.get("type", "categorical"))
        is_passthrough = bool(cfg.get("passthrough", False))

        if col_type == "numeric":
            if is_passthrough:
                passthrough_cols.append(col_name)
                continue
            key_num = (
                str(cfg.get("numericImputation", prep_cfg.numeric_imputation)),
                str(cfg.get("numericScaling", prep_cfg.numeric_scaling)),
                int(cfg.get("knnNeighbors", prep_cfg.knn_n_neighbors)),
                float(cfg.get("constantFillNumeric", prep_cfg.constant_fill_numeric)),
            )
            numeric_groups.setdefault(key_num, []).append(col_name)
            continue

        if is_passthrough:
            passthrough_cols.append(col_name)
            continue

        raw_order = cfg.get("ordinalOrder") if col_type == "ordinal" else []
        order_tuple = tuple([str(v) for v in raw_order]) if isinstance(raw_order, list) else tuple()
        key_cat = (
            str(cfg.get("categoricalImputation", prep_cfg.categorical_imputation)),
            str(cfg.get("categoricalEncoding", prep_cfg.categorical_encoding)),
            order_tuple,
            str(cfg.get("constantFillCategorical", prep_cfg.constant_fill_categorical)),
        )
        categorical_groups.setdefault(key_cat, []).append(col_name)

    transformers: list[tuple[str, Any, list[str]]] = []

    for idx, ((num_imp, num_scale, knn_k, const_fill_num), cols) in enumerate(numeric_groups.items()):
        steps: list[tuple[str, Any]] = []
        if num_imp != "none":
            steps.append(("imputer", _build_numeric_imputer(
                num_imp,
                knn_n_neighbors=knn_k,
                constant_fill=const_fill_num,
            )))
        if num_scale != "none":
            steps.append(("scaler", _build_numeric_scaler(num_scale)))
        if steps:
            name = "num" if idx == 0 else f"num_{idx}"
            transformers.append((name, Pipeline(steps=steps), cols))

    for idx, ((cat_imp, cat_enc, order_tuple, const_fill_cat), cols) in enumerate(categorical_groups.items()):
        steps: list[tuple[str, Any]] = []
        if cat_imp != "none":
            steps.append(("imputer", _build_categorical_imputer(
                cat_imp,
                constant_fill=const_fill_cat,
            )))
        if cat_enc != "none":
            steps.append(
                (
                    "encoder",
                    _build_categorical_encoder(
                        cat_enc,
                        ordinal_order=list(order_tuple) if order_tuple else None,
                        n_features=len(cols),
                    ),
                )
            )
        if steps:
            name = "cat" if idx == 0 else f"cat_{idx}"
            transformers.append((name, Pipeline(steps=steps), cols))

    if passthrough_cols:
        transformers.append(("passthrough_cols", "passthrough", passthrough_cols))

    preprocessor: Any = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return PreprocessSpec(
        preprocessor=preprocessor,
        numeric_cols=effective.active_numeric_columns,
        categorical_cols=effective.active_categorical_columns,
        selected_methods={
            **prep_cfg.as_dict(),
            "legacy": prep_cfg.as_legacy_dict(),
        },
        effective_by_column=effective.effective_by_column,
        dropped_columns=effective.dropped_columns,
        column_types=effective.column_types,
        feature_selector=VarianceThreshold(threshold=prep_cfg.variance_threshold),
    )
