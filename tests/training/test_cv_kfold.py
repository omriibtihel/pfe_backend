"""
Tests for K-Fold and Stratified K-Fold cross-validation.

Covers:
1. kfold classification — N folds, aggregated metrics returned
2. stratified_kfold — class proportions approximately preserved per fold
3. Error: stratified_kfold on regression → RuntimeError
4. Error: kFolds > n_samples → RuntimeError in splitter
5. Error: stratified_kfold with minority class too small → RuntimeError
6. Resampling (SMOTE) applied only on train_fold, not val_fold
7. Holdout regression not broken
8. Sparse + resampling path (OneHot → dense conversion before SMOTE)
9. validate_kfold_config returns correct errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.training.config import TrainingConfig
from app.services.training.splitters import iter_kfold_splits, validate_kfold_config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_classification_df(n: int = 200, n_minority: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_majority = n - n_minority
    y = np.array([0] * n_majority + [1] * n_minority)
    rng.shuffle(y)
    age = rng.normal(52, 10, n)
    bmi = rng.normal(25, 5, n)
    sex = np.where(rng.random(n) < 0.5, "F", "M")
    return pd.DataFrame({"age": age, "bmi": bmi, "sex": sex, "label": y})


def _make_multiclass_df(n: int = 300, n_classes: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n // n_classes)
    rng.shuffle(y)
    x1 = rng.normal(0, 1, len(y))
    x2 = rng.normal(0, 1, len(y))
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _make_regression_df(n: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _cfg_kfold(
    *,
    df: pd.DataFrame,
    target: str,
    task_type: str = "classification",
    split_method: str = "kfold",
    k: int = 5,
    strategy: str = "none",
    models: list[str] | None = None,
) -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": target,
            "taskType": task_type,
            "models": models or ["randomforest"],
            "metrics": ["accuracy", "f1"] if task_type == "classification" else ["r2", "mae"],
            "splitMethod": split_method,
            "kFolds": k,
            "shuffle": True,
            "useGridSearch": False,
            "balancing": {"strategy": strategy},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. kfold classification — N folds, aggregated metrics returned
# ──────────────────────────────────────────────────────────────────────────────

def test_kfold_classification_runs_and_returns_cv_metrics():
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=200, n_minority=50)
    cfg = _cfg_kfold(df=df, target="label", split_method="kfold", k=5)

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    # Check CV flag
    assert result.metrics_json.get("cv") is True
    assert result.metrics_json.get("k_folds") == 5

    # fold_results: one entry per fold
    fold_results = result.metrics_json.get("fold_results", [])
    assert len(fold_results) == 5

    # At least 4 out of 5 should succeed
    n_ok = sum(1 for fr in fold_results if fr.get("status") == "ok")
    assert n_ok >= 4, f"Too many fold failures: {fold_results}"

    # cv_summary has mean/std
    cv_summary = result.metrics_json.get("cv_summary", {})
    assert "mean" in cv_summary
    assert "std" in cv_summary
    assert cv_summary.get("n_folds_ok") == n_ok

    # "test" key holds the mean CV val score (compat alias)
    assert "test" in result.metrics_json

    # fitted_pipeline is a valid sklearn Pipeline
    from sklearn.pipeline import Pipeline
    assert isinstance(result.fitted_pipeline, Pipeline)

    # artifacts contain split_info with CV fields
    split_info = result.artifacts_json.get("split_info", {})
    assert split_info.get("method") == "kfold"
    assert split_info.get("k_folds") == 5

    # refit_info: final model was refit on full data
    refit_info = result.artifacts_json.get("refit_info", {})
    assert refit_info.get("refit_on_full_data") is True


# ──────────────────────────────────────────────────────────────────────────────
# 2. stratified_kfold — class proportions approximately preserved
# ──────────────────────────────────────────────────────────────────────────────

def test_stratified_kfold_preserves_class_proportions():
    """Each fold's val set should have both classes (no total class absence)."""
    df = _make_classification_df(n=300, n_minority=60)
    X = df.drop(columns=["label"])
    y = df["label"].to_numpy()

    folds = list(
        iter_kfold_splits(
            X, y,
            split_method="stratified_kfold",
            k_folds=5,
            shuffle=True,
            random_state=42,
        )
    )
    assert len(folds) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        y_val = y[val_idx]
        classes_in_val = set(y_val.tolist())
        assert 0 in classes_in_val, f"Fold {fold_idx}: class 0 missing from val"
        assert 1 in classes_in_val, f"Fold {fold_idx}: class 1 missing from val"

    # Global ratio check: minority ratio should be similar in val folds
    global_ratio = float((y == 1).sum()) / len(y)
    fold_ratios = []
    for train_idx, val_idx in folds:
        y_val = y[val_idx]
        fold_ratios.append(float((y_val == 1).sum()) / len(y_val))
    mean_fold_ratio = float(np.mean(fold_ratios))
    # Allow ±10% deviation from global ratio
    assert abs(mean_fold_ratio - global_ratio) < 0.10, (
        f"Mean fold minority ratio {mean_fold_ratio:.3f} deviates too much from "
        f"global {global_ratio:.3f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. stratified_kfold on regression → RuntimeError from validate_kfold_config
# ──────────────────────────────────────────────────────────────────────────────

def test_stratified_kfold_regression_raises_in_validation():
    df = _make_regression_df(n=100)
    X = df.drop(columns=["y"])
    y = df["y"].to_numpy()

    errors = validate_kfold_config(
        X, y,
        split_method="stratified_kfold",
        k_folds=5,
        task_type="regression",
    )
    assert len(errors) > 0
    assert any("régression" in e or "classification" in e for e in errors), errors


def test_stratified_kfold_regression_raises_in_orchestrator():
    import app.services.training.orchestrator as orch

    df = _make_regression_df(n=100)
    cfg = _cfg_kfold(
        df=df,
        target="y",
        task_type="regression",
        split_method="stratified_kfold",
        k=5,
    )
    with pytest.raises(RuntimeError, match="stratified_kfold"):
        orch.run_one_model(df, cfg, model_type="randomforest")


# ──────────────────────────────────────────────────────────────────────────────
# 4. kFolds > n_samples → RuntimeError
# ──────────────────────────────────────────────────────────────────────────────

def test_kfolds_greater_than_n_samples_raises():
    df = _make_regression_df(n=30)
    X = df.drop(columns=["y"])
    y = df["y"].to_numpy()

    errors = validate_kfold_config(
        X, y,
        split_method="kfold",
        k_folds=50,
        task_type="regression",
    )
    assert any("50" in e or "30" in e for e in errors), errors


def test_kfolds_greater_than_n_samples_raises_in_splitter():
    df = _make_regression_df(n=20)
    X = df.drop(columns=["y"])
    y = df["y"].to_numpy()

    with pytest.raises(RuntimeError, match="kFolds"):
        list(iter_kfold_splits(X, y, split_method="kfold", k_folds=50))


# ──────────────────────────────────────────────────────────────────────────────
# 5. stratified_kfold minority class too small → RuntimeError
# ──────────────────────────────────────────────────────────────────────────────

def test_stratified_kfold_minority_too_small():
    """Minority class has 3 samples but k=5 → should raise."""
    rng = np.random.default_rng(0)
    n = 100
    y = np.array([0] * 97 + [1] * 3)
    rng.shuffle(y)
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})

    errors = validate_kfold_config(
        X, y,
        split_method="stratified_kfold",
        k_folds=5,
        task_type="classification",
    )
    assert len(errors) > 0
    assert any("3" in e or "minoritaire" in e or "minority" in e.lower() for e in errors), errors


def test_stratified_kfold_minority_too_small_in_splitter():
    rng = np.random.default_rng(0)
    n = 50
    y = np.array([0] * 47 + [1] * 3)
    rng.shuffle(y)
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})

    with pytest.raises(RuntimeError, match="3"):
        list(iter_kfold_splits(X, y, split_method="stratified_kfold", k_folds=5))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Resampling applied only on train_fold — never on val_fold
# ──────────────────────────────────────────────────────────────────────────────

def test_smote_applied_only_on_train_fold_not_val():
    """
    Run orchestrator with SMOTE. Verify that val_fold sizes match the original
    index sizes (not inflated by SMOTE).
    """
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=200, n_minority=30)
    cfg = _cfg_kfold(
        df=df,
        target="label",
        split_method="kfold",
        k=3,
        strategy="smote",
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    fold_results = result.metrics_json.get("fold_results", [])
    ok_folds = [fr for fr in fold_results if fr.get("status") == "ok"]
    assert len(ok_folds) >= 1

    total_rows = len(df.dropna(subset=["label"]))
    expected_val_size = total_rows // 3  # approximately

    for fr in ok_folds:
        val_size = fr.get("val_size", 0)
        # Val size must NOT be inflated: it should be close to 1/k of total data
        assert val_size <= expected_val_size * 1.1, (
            f"Fold {fr['fold']}: val_size={val_size} looks inflated (expected ≈{expected_val_size}). "
            "SMOTE may have leaked into val fold."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Holdout not broken (regression)
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_regression_still_works():
    import app.services.training.orchestrator as orch

    df = _make_regression_df(n=120)
    cfg = TrainingConfig.from_front(
        {
            "targetColumn": "y",
            "taskType": "regression",
            "models": ["randomforest"],
            "metrics": ["r2", "mae"],
            "splitMethod": "holdout",
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": 5,
            "useGridSearch": False,
            "balancing": {"strategy": "none"},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalEncoding": "none",
            },
        }
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    # Must NOT be a CV result
    assert result.metrics_json.get("cv") is not True
    assert "train" in result.metrics_json
    assert "test" in result.metrics_json

    split_info = result.artifacts_json.get("split_info", {})
    assert split_info.get("method") == "holdout"


# ──────────────────────────────────────────────────────────────────────────────
# 7b. Holdout classification still works
# ──────────────────────────────────────────────────────────────────────────────

def test_holdout_classification_still_works():
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=200, n_minority=50)
    cfg = TrainingConfig.from_front(
        {
            "targetColumn": "label",
            "taskType": "classification",
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1"],
            "splitMethod": "holdout",
            "trainRatio": 70,
            "valRatio": 15,
            "testRatio": 15,
            "kFolds": 5,
            "useGridSearch": False,
            "balancing": {"strategy": "none"},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")
    assert result.metrics_json.get("cv") is not True
    assert "test" in result.metrics_json
    assert result.artifacts_json.get("split_info", {}).get("method") == "holdout"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Sparse + resampling: OneHot produces sparse → SMOTE requires dense
# ──────────────────────────────────────────────────────────────────────────────

def test_sparse_to_dense_conversion_before_smote_no_crash():
    """
    Dataset with a categorical column → OneHot produces sparse matrix.
    SMOTE strategy requires dense. The orchestrator must convert silently.
    """
    import app.services.training.orchestrator as orch

    rng = np.random.default_rng(99)
    n = 150
    y = np.array([0] * 120 + [1] * 30)
    rng.shuffle(y)
    cat_col = np.where(rng.random(n) < 0.5, "A", "B")
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(0, 1, n),
            "cat": cat_col,
            "label": y,
        }
    )
    cfg = _cfg_kfold(
        df=df,
        target="label",
        split_method="kfold",
        k=3,
        strategy="smote",
    )

    # Should not crash
    result = orch.run_one_model(df, cfg, model_type="randomforest")
    assert result.metrics_json.get("cv") is True
    ok_folds = [fr for fr in result.metrics_json.get("fold_results", []) if fr.get("status") == "ok"]
    assert len(ok_folds) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# 9. validate_kfold_config comprehensive
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_kfold_config_valid_returns_empty():
    df = _make_classification_df(n=200, n_minority=60)
    X = df.drop(columns=["label"])
    y = df["label"].to_numpy()

    errors = validate_kfold_config(
        X, y,
        split_method="stratified_kfold",
        k_folds=5,
        task_type="classification",
    )
    assert errors == [], f"Unexpected errors: {errors}"


def test_validate_kfold_config_k_less_than_2():
    df = _make_classification_df(n=50)
    X = df.drop(columns=["label"])
    y = df["label"].to_numpy()

    errors = validate_kfold_config(
        X, y,
        split_method="kfold",
        k_folds=1,
        task_type="classification",
    )
    assert len(errors) > 0
    assert any("2" in e for e in errors), errors


def test_stratified_kfold_multiclass_works():
    """stratified_kfold should work with 3+ classes."""
    import app.services.training.orchestrator as orch

    df = _make_multiclass_df(n=300, n_classes=3)
    cfg = _cfg_kfold(
        df=df,
        target="target",
        split_method="stratified_kfold",
        k=5,
        strategy="none",
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")
    assert result.metrics_json.get("cv") is True
    ok_folds = [fr for fr in result.metrics_json.get("fold_results", []) if fr.get("status") == "ok"]
    assert len(ok_folds) >= 4


# ──────────────────────────────────────────────────────────────────────────────
# Holdout test set in CV mode (testRatio > 0)
# ──────────────────────────────────────────────────────────────────────────────

def _cfg_kfold_with_test(
    *,
    df: pd.DataFrame,
    target: str,
    task_type: str = "classification",
    split_method: str = "kfold",
    k: int = 5,
    test_ratio: int = 20,
    strategy: str = "none",
) -> TrainingConfig:
    return TrainingConfig.from_front(
        {
            "targetColumn": target,
            "taskType": task_type,
            "models": ["randomforest"],
            "metrics": ["accuracy", "f1"] if task_type == "classification" else ["r2", "mae"],
            "splitMethod": split_method,
            "kFolds": k,
            "shuffle": True,
            "testRatio": test_ratio,
            "useGridSearch": False,
            "balancing": {"strategy": strategy},
            "preprocessing": {
                "numericImputation": "median",
                "numericScaling": "standard",
                "categoricalImputation": "most_frequent",
                "categoricalEncoding": "onehot",
            },
        }
    )


def test_cv_with_holdout_test_produces_test_metrics():
    """kfold + testRatio=20% should produce holdout_test_metrics and has_holdout_test=True."""
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=300, n_minority=80)
    cfg = _cfg_kfold_with_test(df=df, target="label", split_method="kfold", k=5, test_ratio=20)

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    assert result.metrics_json.get("cv") is True
    assert result.metrics_json.get("has_holdout_test") is True

    # holdout_test_metrics must be present and non-empty
    holdout_metrics = result.metrics_json.get("holdout_test_metrics")
    assert holdout_metrics is not None, "holdout_test_metrics missing from metrics_json"
    assert isinstance(holdout_metrics, dict)

    # "test" key should point to the holdout metrics (not CV mean)
    test_block = result.metrics_json.get("test")
    assert test_block is holdout_metrics or test_block == holdout_metrics, (
        '"test" key should alias holdout_test_metrics when has_holdout_test=True'
    )

    # cv_mean is still available separately
    cv_mean = result.metrics_json.get("cv_mean")
    assert isinstance(cv_mean, dict) and len(cv_mean) > 0

    # refit_info: refit was NOT on full data (only on non-test portion)
    refit_info = result.artifacts_json.get("refit_info", {})
    assert refit_info.get("refit_on_full_data") is False

    # split_info: has_holdout_test and test_rows populated
    split_info = result.artifacts_json.get("split_info", {})
    assert split_info.get("has_holdout_test") is True
    assert isinstance(split_info.get("test_rows"), int) and split_info["test_rows"] > 0
    assert split_info.get("n_samples_cv", 0) < split_info.get("n_samples", 0)


def test_cv_testRatio_zero_is_unchanged():
    """testRatio=0 (default) should behave exactly as before: refit on all data."""
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=200, n_minority=50)
    cfg = _cfg_kfold_with_test(df=df, target="label", split_method="kfold", k=5, test_ratio=0)

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    assert result.metrics_json.get("has_holdout_test") is False
    assert result.metrics_json.get("holdout_test_metrics") is None

    refit_info = result.artifacts_json.get("refit_info", {})
    assert refit_info.get("refit_on_full_data") is True

    split_info = result.artifacts_json.get("split_info", {})
    assert split_info.get("has_holdout_test") is False
    assert split_info.get("test_rows") is None


def test_cv_holdout_test_not_contaminated_in_folds():
    """
    When testRatio=20%, val_fold sizes should reflect 80% of the dataset (CV portion),
    not the full 100%. The holdout test set must NOT appear inside any fold.
    """
    import app.services.training.orchestrator as orch

    total = 300
    test_pct = 20
    df = _make_classification_df(n=total, n_minority=75)
    cfg = _cfg_kfold_with_test(
        df=df, target="label", split_method="kfold", k=5, test_ratio=test_pct
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")

    fold_results = result.metrics_json.get("fold_results", [])
    ok_folds = [fr for fr in fold_results if fr.get("status") == "ok"]
    assert len(ok_folds) >= 4

    n_cv = total * (100 - test_pct) // 100  # ≈ 240
    # Each val_fold should be ≈ n_cv / k = 240 / 5 = 48
    expected_val = n_cv / 5
    for fr in ok_folds:
        val_size = fr.get("val_size", 0)
        # Must be clearly smaller than total/k (which would be 60 if test weren't removed)
        assert val_size < total // 5 + 5, (
            f"Fold {fr['fold']}: val_size={val_size} too large — holdout test may have leaked."
        )
        # Must be close to expected_val (within 2 rows of rounding)
        assert abs(val_size - expected_val) <= 3, (
            f"Fold {fr['fold']}: val_size={val_size} far from expected ≈{expected_val:.0f}"
        )


def test_cv_holdout_test_regression():
    """testRatio > 0 should work for regression tasks as well."""
    import app.services.training.orchestrator as orch

    df = _make_regression_df(n=200)
    cfg = _cfg_kfold_with_test(
        df=df, target="y", task_type="regression",
        split_method="kfold", k=5, test_ratio=20,
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")
    assert result.metrics_json.get("has_holdout_test") is True
    assert result.metrics_json.get("holdout_test_metrics") is not None
    assert result.artifacts_json.get("refit_info", {}).get("refit_on_full_data") is False


def test_cv_holdout_test_stratified_kfold():
    """testRatio > 0 + stratified_kfold: both class proportions and test set are stratified."""
    import app.services.training.orchestrator as orch

    df = _make_classification_df(n=300, n_minority=90)
    cfg = _cfg_kfold_with_test(
        df=df, target="label", split_method="stratified_kfold", k=5, test_ratio=20
    )

    result = orch.run_one_model(df, cfg, model_type="randomforest")
    assert result.metrics_json.get("cv") is True
    assert result.metrics_json.get("has_holdout_test") is True
    ok_folds = [fr for fr in result.metrics_json.get("fold_results", []) if fr.get("status") == "ok"]
    assert len(ok_folds) >= 4
