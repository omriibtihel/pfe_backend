"""
TrainingConfig frozen dataclass — top-level config for a training run.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services.training.config.normalization import (
    _normalize_model_hyperparams_payload,
    _to_bool,
    _to_optional_int,
)
from app.services.training.config.schema.types import SplitMethod, TaskType
from app.services.training.config.schema.feature_def import FeatureEngineeringConfig
from app.services.training.config.schema.balancing import BalancingConfig
from app.services.training.config.schema.preprocessing import PreprocessingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers (local — only needed in this file)
# ---------------------------------------------------------------------------

_VALID_SEARCH_TYPES = {"none", "grid", "random", "halving_random"}
_CV_METHODS: frozenset[str] = frozenset({
    "kfold", "stratified_kfold",
    "repeated_stratified_kfold",
    "group_kfold", "stratified_group_kfold",
    "loo",
})


def _parse_search_type(payload: dict[str, Any]) -> str:
    """Parse search_type from payload with backward-compat fallback from useGridSearch."""
    raw = str(payload.get("searchType", "")).strip().lower()
    if raw in _VALID_SEARCH_TYPES:
        return raw
    # Legacy fallback: useGridSearch=true → "grid"
    return "grid" if _to_bool(payload.get("useGridSearch"), default=False) else "none"


def _int_or_default(raw: Any, default: int) -> int:
    """Return int(raw) when raw is not None, else default. Never use `or default`."""
    if raw is None:
        return default
    return int(raw)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str
    task_type: TaskType
    models: Sequence[str]
    metrics: Sequence[str]

    split_method: SplitMethod
    train_ratio: float
    val_ratio: float
    test_ratio: float
    k_folds: int

    use_grid_search: bool
    use_smote: bool
    shuffle: bool
    balancing: BalancingConfig
    preprocessing: PreprocessingConfig
    model_hyperparams: dict[str, dict[str, Any]] = field(default_factory=dict)
    positive_label: Any = None
    debug: bool = False
    random_state: int = 42
    dataset_version_id: int | None = None
    custom_code: str = ""
    search_type: str = "none"          # "none" | "grid" | "random"
    n_iter_random_search: int = 40     # only used when search_type == "random"
    inner_cv_folds: int = 3            # folds for inner GridSearch in nested CV
    n_repeats: int = 3                 # only used when split_method == "repeated_stratified_kfold"
    group_column: str | None = None    # only used when split_method in ("group_kfold", "stratified_group_kfold")
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    resolved_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError for out-of-range or logically invalid parameter combinations."""
        if self.split_method in _CV_METHODS and self.k_folds < 2:
            raise ValueError(
                f"kFolds must be >= 2 for CV split method '{self.split_method}', got {self.k_folds}"
            )

        test_ratio_pct = self.test_ratio * 100.0
        if not (0.0 <= test_ratio_pct <= 95.0):
            raise ValueError(
                f"testRatio must be between 0 and 95, got {test_ratio_pct:.1f}"
            )

        if self.split_method == "holdout":
            total_pct = (self.train_ratio + self.val_ratio + self.test_ratio) * 100.0
            if abs(total_pct - 100.0) > 1.0:
                raise ValueError(
                    f"trainRatio + valRatio + testRatio must sum to 100, got {total_pct:.1f}"
                )

        if self.split_method == "repeated_stratified_kfold" and self.n_repeats < 1:
            raise ValueError(
                f"nRepeats must be >= 1 for repeated CV, got {self.n_repeats}"
            )

    @staticmethod
    def from_front(payload: dict[str, Any]) -> "TrainingConfig":
        def to_ratio(x: Any, default: float) -> float:
            try:
                v = float(x)
            except Exception:
                return default
            if v > 1:
                v = v / 100.0
            return max(0.0, min(v, 0.99))

        split_raw = str(payload.get("splitMethod", "holdout")).strip().lower()
        if split_raw in _CV_METHODS:
            split_method: SplitMethod = split_raw  # type: ignore[assignment]
        else:
            split_method = "holdout"

        legacy_use_smote = _to_bool(payload.get("useSmote"), default=False)
        balancing_cfg = BalancingConfig.from_front(
            payload.get("balancing"),
            legacy_use_smote=legacy_use_smote,
        )
        use_smote = bool(str(balancing_cfg.strategy) in {"smote", "smote_tomek"})

        train_ratio = to_ratio(payload.get("trainRatio", 0.70), 0.70)
        val_ratio = to_ratio(payload.get("valRatio", 0.15), 0.15)
        test_ratio = to_ratio(
            payload.get("testRatio", 0.0 if split_method != "holdout" else 0.15),
            0.0 if split_method != "holdout" else 0.15,
        )

        # None-check pattern — never use `or DEFAULT` for numeric params.
        # `or DEFAULT` silently replaces any falsy value (0, False) with the default,
        # which corrupts semantically valid inputs like random_state=0.
        k_folds = _int_or_default(payload.get("kFolds"), 5)
        n_iter_random_search = _int_or_default(payload.get("nIterRandomSearch"), 40)
        _gc = payload.get("gridCvFolds")
        _ic = payload.get("innerCvFolds")
        inner_cv_folds = int(_gc) if _gc is not None else _int_or_default(_ic, 3)
        random_state = _int_or_default(payload.get("randomState"), 42)
        n_repeats = _int_or_default(payload.get("nRepeats"), 3)

        shuffle = _to_bool(payload.get("shuffle"), default=True)
        debug = _to_bool(payload.get("debug", payload.get("trainingDebug")), default=False)

        logger.debug(
            "param_resolved | kFolds=%s | nIterRandomSearch=%s | innerCvFolds=%s"
            " | randomState=%s | nRepeats=%s | shuffle=%s | debug=%s"
            " | trainRatio=%.4f | valRatio=%.4f | testRatio=%.4f",
            k_folds, n_iter_random_search, inner_cv_folds,
            random_state, n_repeats, shuffle, debug,
            train_ratio, val_ratio, test_ratio,
        )

        resolved_params: dict[str, Any] = {
            "kFolds": k_folds,
            "nIterRandomSearch": n_iter_random_search,
            "innerCvFolds": inner_cv_folds,
            "randomState": random_state,
            "nRepeats": n_repeats,
            "shuffle": shuffle,
            "trainRatio": round(train_ratio * 100, 1),
            "valRatio": round(val_ratio * 100, 1),
            "testRatio": round(test_ratio * 100, 1),
            "splitMethod": split_method,
        }

        config = TrainingConfig(
            target_column=str(payload.get("targetColumn", "")).strip(),
            task_type=str(payload.get("taskType", "classification")).strip().lower(),  # type: ignore
            models=[str(m).strip().lower() for m in (payload.get("models") or [])],
            metrics=[str(m).strip().lower() for m in (payload.get("metrics") or [])],
            split_method=split_method,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            k_folds=k_folds,
            search_type=_parse_search_type(payload),
            use_grid_search=_parse_search_type(payload) != "none",
            n_iter_random_search=n_iter_random_search,
            inner_cv_folds=inner_cv_folds,
            use_smote=use_smote,
            shuffle=shuffle,
            balancing=balancing_cfg,
            preprocessing=PreprocessingConfig.from_front(payload),
            model_hyperparams=_normalize_model_hyperparams_payload(payload.get("modelHyperparams")),
            positive_label=payload.get("positiveLabel", payload.get("positive_label")),
            debug=debug,
            random_state=random_state,
            dataset_version_id=_to_optional_int(payload.get("datasetVersionId")),
            custom_code=str(payload.get("customCode", "") or ""),
            n_repeats=n_repeats,
            group_column=str(payload.get("groupColumn", "") or "").strip() or None,
            feature_engineering=FeatureEngineeringConfig.from_front(
                payload.get("featureEngineering")
            ),
            resolved_params=resolved_params,
        )
        config.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "datasetVersionId": self.dataset_version_id,
            "targetColumn": self.target_column,
            "taskType": self.task_type,
            "models": [str(m) for m in self.models],
            "metrics": [str(m) for m in self.metrics],
            "splitMethod": self.split_method,
            "trainRatio": self.train_ratio,
            "valRatio": self.val_ratio,
            "testRatio": self.test_ratio,
            "kFolds": self.k_folds,
            "shuffle": bool(self.shuffle),
            "searchType": self.search_type,
            "nIterRandomSearch": self.n_iter_random_search,
            "innerCvFolds": self.inner_cv_folds,
            "useGridSearch": bool(self.use_grid_search),  # backward compat
            "useSmote": bool(str(self.balancing.strategy) in {"smote", "smote_tomek"}),
            "balancing": self.balancing.as_dict(),
            "preprocessing": self.preprocessing.as_dict(),
            "modelHyperparams": copy.deepcopy(self.model_hyperparams),
            "positiveLabel": self.positive_label,
            "debug": bool(self.debug),
            "randomState": self.random_state,
            "customCode": self.custom_code,
            "nRepeats": self.n_repeats,
            "groupColumn": self.group_column,
            "featureEngineering": self.feature_engineering.as_dict(),
            "resolved_params": dict(self.resolved_params),
        }
