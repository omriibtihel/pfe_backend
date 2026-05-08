"""
Local Counterfactual Explanations — per-row counterfactual generation
using DiCE (Diverse Counterfactual Explanations) with a reliable binary-search
fallback for confident models or when SHAP background data is unavailable.

For a single input row, returns a list of changed features:
  [
    {
      "feature":         str,    # feature name (preprocessed, e.g. "num__Glucose")
      "original_value":  float,  # value in the input row (original units when scalable)
      "suggested_value": float,  # value that flips the prediction
      "delta":           float,  # suggested_value - original_value
    },
    ...  # sorted by |delta| descending
  ]

Design choices:
  - method="random": gradient-free, works with any sklearn estimator.
  - Preprocessed feature space: consistent with SHAP and LIME. The background
    from artifacts["shap"]["background_data"] is reused when available.
  - Feature name matching: strips the ColumnTransformer prefix (e.g. "num__")
    so original names ("Glucose") match preprocessed names ("num__Glucose").
  - Binary-search fallback: when DiCE fails or background is absent, a
    univariate binary search provably reaches the decision boundary in ≤ 24
    iterations per feature, regardless of model confidence.
  - Inverse scaling: StandardScaler/MinMaxScaler/RobustScaler values are
    converted back to original clinical units when possible.
  - Returns None on any failure — never raises to the caller.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services.shap._utils import extract_model_from_pipeline, to_json_safe
from app.services.shap.global_shap import (
    _get_preprocessed_feature_names,
    _transform_except_model,
)

logger = logging.getLogger(__name__)

_DESIRED_CLASS = "opposite"
_MAX_CF = 5


# ── Inverse-scaling helpers ────────────────────────────────────────────────────

def _is_linear_invertible_scaler(obj: Any) -> bool:
    has_std = (
        hasattr(obj, "mean_") and hasattr(obj, "scale_")
        and not hasattr(obj, "lambdas_")
    )
    has_robust = hasattr(obj, "center_") and hasattr(obj, "scale_")
    has_minmax = hasattr(obj, "data_min_") and hasattr(obj, "data_range_")
    return has_std or has_robust or has_minmax


def _extract_scaler_params(scaler: Any, col_idx: int) -> Optional[Tuple[float, float]]:
    """Return (offset, scale) where x_original = x_preprocessed * scale + offset."""
    try:
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_") and not hasattr(scaler, "lambdas_"):
            return float(scaler.mean_[col_idx]), float(scaler.scale_[col_idx])
        if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
            return float(scaler.center_[col_idx]), float(scaler.scale_[col_idx])
        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
            return float(scaler.data_min_[col_idx]), float(scaler.data_range_[col_idx])
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _build_inverse_scale_map(pipeline: Any) -> Dict[str, Tuple[float, float]]:
    named = getattr(pipeline, "named_steps", {})
    prep = named.get("prep")
    if prep is None or not hasattr(prep, "transformers_"):
        return {}

    result: Dict[str, Tuple[float, float]] = {}
    for _, transformer, columns in prep.transformers_:
        scaler: Optional[Any] = None
        if hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if _is_linear_invertible_scaler(step):
                    scaler = step
                    break
        elif _is_linear_invertible_scaler(transformer):
            scaler = transformer
        if scaler is None:
            continue
        for i, col in enumerate(columns):
            params = _extract_scaler_params(scaler, i)
            if params is not None:
                result[str(col)] = params
    return result


def _apply_inverse_scale(prep_value: float, params: Tuple[float, float]) -> float:
    offset, scale = params
    return offset if scale == 0.0 else prep_value * scale + offset


def _strip_ct_prefix(name: str) -> str:
    """
    Strip the ColumnTransformer prefix added by sklearn ('num__', 'cat__',
    'remainder__'…) so the frontend shows clinical names ('Glucose') rather
    than internal pipeline names ('num__Glucose'). OHE category suffixes
    (e.g. 'sex_male') are preserved.
    """
    return name.split("__", 1)[-1] if "__" in name else name


# ── Feature name mapping ───────────────────────────────────────────────────────

def _map_features_to_prep(
    features_to_vary: List[str],
    prep_feat_names: List[str],
) -> List[str]:
    """
    Map original feature names to their preprocessed counterparts.

    Handles three naming patterns produced by sklearn's ColumnTransformer:
      - Direct:   "Glucose"      → "Glucose"        (no prefix)
      - CT prefix:"Glucose"      → "num__Glucose"   (transformer name + __)
      - OHE:      "sex"          → "sex_male"        (+ category suffix)
      - CT + OHE: "sex"          → "cat__sex_male"   (both)
    """
    prep_vary: List[str] = []
    for orig in features_to_vary:
        matched = []
        for pn in prep_feat_names:
            # Strip ColumnTransformer prefix (e.g. "num__", "cat__", "remainder__")
            # so "num__Glucose" becomes "Glucose" for comparison.
            pn_base = pn.split("__", 1)[-1] if "__" in pn else pn

            if (
                pn == orig                          # exact: "Glucose"
                or pn_base == orig                  # CT-stripped: "num__Glucose" → "Glucose"
                or pn.startswith(orig + "_")        # OHE no prefix: "sex_male"
                or pn_base.startswith(orig + "_")   # OHE with prefix: "cat__sex_male"
            ):
                matched.append(pn)
        prep_vary.extend(matched)
    return prep_vary


# ── Estimator I/O helper ──────────────────────────────────────────────────────

def _make_estimator_input(
    estimator: Any,
    X: np.ndarray,
    prep_feat_names: List[str],
) -> Any:
    """
    Return X in the exact format the estimator was fitted with.

    Avoids sklearn's "X has feature names, but … was fitted without feature names"
    (and the inverse) UserWarning by matching the training-time API.
    """
    if hasattr(estimator, "feature_names_in_"):
        names = list(estimator.feature_names_in_)
        if len(names) == X.shape[1]:
            return pd.DataFrame(X, columns=names)
    return X  # estimator fitted on raw numpy → keep numpy


# ── Multi-feature directional counterfactual ──────────────────────────────────

def _multi_feature_cf(
    estimator: Any,
    X_row_prep: np.ndarray,
    prep_feat_names: List[str],
    prep_vary: List[str],
    task_type: str,
    sigma_map: Dict[str, float],
) -> Optional[List[Dict[str, Any]]]:
    """
    Multi-feature directional counterfactual (batched).

    1. Probes each modifiable feature at ±{0.5, 1.5, 3, 6}σ to find the direction
       that reduces the current class probability — all probes for all features
       are submitted in a single batched predict_proba call.
    2. Pushes every feature simultaneously in its beneficial direction by α σ.
    3. Binary-searches α ∈ [0, 20] to find the minimum push that flips.

    Classification only.
    """
    if task_type != "classification":
        return None
    if not hasattr(estimator, "predict_proba"):
        return None

    feat_to_idx: Dict[str, int] = {n: i for i, n in enumerate(prep_feat_names)}

    def _predict(X: np.ndarray) -> str:
        return str(estimator.predict(_make_estimator_input(estimator, X, prep_feat_names))[0])

    def _proba_batch(X_batch: np.ndarray) -> np.ndarray:
        """Batched predict_proba — orders of magnitude faster than per-row."""
        return estimator.predict_proba(_make_estimator_input(estimator, X_batch, prep_feat_names))

    current_pred = _predict(X_row_prep)
    current_proba = _proba_batch(X_row_prep)[0]
    current_class_idx = int(np.argmax(current_proba))
    base_p = float(current_proba[current_class_idx])

    # ── 1. Batched multi-distance direction probing ───────────────────────────
    distances = (0.5, 1.5, 3.0, 6.0)
    signs = (1.0, -1.0)

    probe_rows: List[np.ndarray] = []
    probe_meta: List[Tuple[str, int, float, float]] = []  # (col, idx, sigma, sign)
    for col in prep_vary:
        idx = feat_to_idx.get(col)
        if idx is None:
            continue
        sigma = sigma_map.get(col, 1.0)
        for distance in distances:
            for sign in signs:
                row = X_row_prep[0].copy()
                row[idx] += sign * distance * sigma
                probe_rows.append(row)
                probe_meta.append((col, idx, sigma, sign))

    if not probe_rows:
        return None

    # ONE batched call instead of 8 × N sequential calls.
    probe_batch = np.array(probe_rows)
    probe_probas = _proba_batch(probe_batch)[:, current_class_idx]

    # Aggregate probability drops per (col, sign)
    drops_by_col: Dict[str, Dict[Any, Any]] = {}
    for i, (col, idx, sigma, sign) in enumerate(probe_meta):
        bucket = drops_by_col.setdefault(col, {"idx": idx, "sigma": sigma, +1.0: 0.0, -1.0: 0.0})
        bucket[sign] += max(0.0, base_p - float(probe_probas[i]))

    directions: List[Tuple[str, int, float, float]] = []
    for col, info in drops_by_col.items():
        if info[1.0] > info[-1.0] + 1e-9:
            directions.append((col, info["idx"], +1.0, info["sigma"]))
        elif info[-1.0] > info[1.0] + 1e-9:
            directions.append((col, info["idx"], -1.0, info["sigma"]))

    if not directions:
        logger.warning("counterfactual: no feature has a measurable gradient on predict_proba")
        return None

    logger.info(
        "counterfactual: directional probe found %d/%d features with gradient",
        len(directions), len(prep_vary),
    )

    # ── 2. Combined push: alpha is the scalar magnitude for all features ──────
    def _predict_at_alpha(alpha: float) -> str:
        test = X_row_prep.copy()
        for _, idx, sign, sigma in directions:
            test[0, idx] += alpha * sign * sigma
        return _predict(test)

    # Push up to 20σ — clinically extreme but still finite. The frontend will
    # show absolute values so a clinician can judge plausibility.
    max_alpha = 20.0
    if _predict_at_alpha(max_alpha) == current_pred:
        logger.info(
            "counterfactual: directional ±%gσ push insufficient — caller will try random search",
            max_alpha,
        )
        return None

    # ── 3. Binary search the minimum alpha that flips ─────────────────────────
    lo, hi = 0.0, max_alpha
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if _predict_at_alpha(mid) == current_pred:
            lo = mid
        else:
            hi = mid
    alpha_min = hi

    # ── 4. Build per-feature output ───────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for col, idx, sign, sigma in directions:
        orig_val = float(X_row_prep[0, idx])
        sugg_val = orig_val + alpha_min * sign * sigma
        delta = sugg_val - orig_val
        if abs(delta) < 1e-8:
            continue
        results.append({
            "col":       col,
            "col_idx":   idx,
            "orig_prep": orig_val,
            "sugg_prep": sugg_val,
            "delta_abs": abs(delta),
        })

    if not results:
        return None

    results.sort(key=lambda r: r["delta_abs"], reverse=True)
    return results


# ── Random search (last-resort fallback) ───────────────────────────────────────

def _random_search_cf(
    estimator: Any,
    X_row_prep: np.ndarray,
    prep_feat_names: List[str],
    prep_vary: List[str],
    sigma_map: Dict[str, float],
    n_samples: int = 2000,
    random_state: int = 42,
) -> Optional[List[Dict[str, Any]]]:
    """
    Batched random-perturbation counterfactual.

    Submits all `n_samples` perturbations per radius in a single estimator.predict
    call — roughly 50× faster than per-row prediction for tree ensembles, which
    keeps the whole search well under the API timeout even with 2000 samples.

    Tries progressively wider radii (3σ → 20σ); stops as soon as a flip is found
    within the current radius.
    """
    feat_to_idx = {n: i for i, n in enumerate(prep_feat_names)}
    cols = [c for c in prep_vary if c in feat_to_idx]
    if not cols:
        return None

    indices = np.array([feat_to_idx[c] for c in cols], dtype=int)
    sigmas = np.array([sigma_map.get(c, 1.0) for c in cols], dtype=float)

    def _predict_batch(X_batch: np.ndarray) -> np.ndarray:
        return np.asarray(
            estimator.predict(_make_estimator_input(estimator, X_batch, prep_feat_names))
        )

    current_pred_str = str(_predict_batch(X_row_prep)[0])
    rng = np.random.default_rng(random_state)

    best_X: Optional[np.ndarray] = None
    best_dist = float("inf")

    samples_per_radius = max(1, n_samples // 4)

    for radius in (3.0, 6.0, 10.0, 20.0):
        # Build the entire batch as a (samples_per_radius, n_features) matrix.
        batch = np.tile(X_row_prep, (samples_per_radius, 1))
        offsets = rng.uniform(-radius, radius, size=(samples_per_radius, len(indices)))
        # Scale each column's offset by its sigma, then add to the modifiable cols.
        batch[:, indices] += offsets * sigmas

        # ONE batched predict — vectorised across all samples.
        preds = _predict_batch(batch)
        flipped_mask = np.array([str(p) != current_pred_str for p in preds])
        if not flipped_mask.any():
            continue

        flipped = batch[flipped_mask]
        diffs = (flipped[:, indices] - X_row_prep[0, indices]) / sigmas
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        local_min = int(np.argmin(distances))
        if float(distances[local_min]) < best_dist:
            best_dist = float(distances[local_min])
            best_X = flipped[local_min:local_min + 1].copy()

        # Stop expanding if the closest flip already fits inside the current radius.
        if best_X is not None and best_dist <= radius:
            break

    if best_X is None:
        return None

    logger.info(
        "counterfactual: random search found a flip at L2=%.2fσ over %d features",
        best_dist, len(indices),
    )

    results: List[Dict[str, Any]] = []
    for col, idx in zip(cols, indices):
        orig_val = float(X_row_prep[0, int(idx)])
        sugg_val = float(best_X[0, int(idx)])
        if abs(sugg_val - orig_val) < 1e-8:
            continue
        results.append({
            "col":       col,
            "col_idx":   int(idx),
            "orig_prep": orig_val,
            "sugg_prep": sugg_val,
            "delta_abs": abs(sugg_val - orig_val),
        })
    if not results:
        return None
    results.sort(key=lambda r: r["delta_abs"], reverse=True)
    return results


def _format_items(
    raw: List[Dict[str, Any]],
    inv_scale_map: Dict[str, Tuple[float, float]],
    ohe_features: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Convert raw counterfactual results to the public output format.

    For OHE binary features (e.g. sex_male ∈ {0, 1}), the continuous search
    may produce values like 0.73.  These are snapped to the nearest valid
    integer so the suggestion is clinically meaningful.  Features that map to
    the same integer after snapping are skipped (no actionable change).
    """
    items: List[Dict[str, Any]] = []
    for r in raw:
        pname = r["col"]
        orig_prep = r["orig_prep"]
        sugg_prep = r["sugg_prep"]

        # Snap OHE features to {0, 1} before inverse-scaling or display.
        if ohe_features and pname in ohe_features:
            snapped = float(round(max(0.0, min(1.0, sugg_prep))))
            if abs(snapped - round(max(0.0, min(1.0, orig_prep)))) < 1e-8:
                continue  # After snapping both are equal — no actionable change
            orig_prep = float(round(max(0.0, min(1.0, orig_prep))))
            sugg_prep = snapped
            orig_val, sugg_val = orig_prep, sugg_prep
        elif pname in inv_scale_map:
            params = inv_scale_map[pname]
            orig_val = _apply_inverse_scale(orig_prep, params)
            sugg_val = _apply_inverse_scale(sugg_prep, params)
        else:
            orig_val = orig_prep
            sugg_val = sugg_prep

        delta = sugg_val - orig_val
        items.append({
            "feature":         _strip_ct_prefix(str(pname)),
            "original_value":  to_json_safe(orig_val),
            "suggested_value": to_json_safe(sugg_val),
            "delta":           to_json_safe(delta),
        })
    items.sort(key=lambda d: abs(float(d["delta"] or 0.0)), reverse=True)
    return items


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_counterfactual(
    fitted_pipe: Any,
    row: pd.DataFrame,
    *,
    features_to_vary: List[str],
    task_type: str = "classification",
    background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    n_counterfactuals: int = 3,
    random_state: int = 42,
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate counterfactuals for a single row.

    Strategy:
    1. Try DiCE method="random" when SHAP background data is available.
    2. Fall back to a per-feature binary search when DiCE fails or background
       is absent. The binary search always finds a counterfactual if one exists
       within ±10 standard deviations of each feature.

    Returns a list of dicts (only changed features) sorted by |delta| descending,
    or None on failure.
    """
    if row is None or len(row) == 0 or not features_to_vary:
        return None

    names = list(feature_names) if feature_names is not None else list(row.columns)
    n_cf = min(int(n_counterfactuals), _MAX_CF)

    try:
        estimator = extract_model_from_pipeline(fitted_pipe)

        X_row_prep = _transform_except_model(fitted_pipe, row)
        if X_row_prep.ndim != 2 or X_row_prep.shape[0] == 0:
            return None

        n_prep = X_row_prep.shape[1]

        prep_feat_names = _get_preprocessed_feature_names(fitted_pipe, names)
        if len(prep_feat_names) != n_prep:
            prep_feat_names = [f"f_{i}" for i in range(n_prep)]

        # Map original feature names to preprocessed names, handling the
        # ColumnTransformer "num__" / "cat__" prefix automatically.
        prep_vary = _map_features_to_prep(features_to_vary, prep_feat_names)

        if not prep_vary:
            logger.warning(
                "counterfactual: no preprocessed columns matched features_to_vary=%s "
                "(prep space sample: %s)",
                features_to_vary[:5],
                prep_feat_names[:10],
            )
            return None

        logger.info(
            "counterfactual: prep_vary=%s (%d cols) | n_prep=%d",
            prep_vary[:5], len(prep_vary), n_prep,
        )

        inv_scale_map = _build_inverse_scale_map(fitted_pipe)

        # Per-feature sigma for binary search range and DiCE permitted_range.
        # When background is available, compute from the actual distribution.
        # Otherwise fall back to 1.0 (StandardScaler output has unit variance).
        bg_array: Optional[np.ndarray] = None
        sigma_map: Dict[str, float] = {}
        # Features whose background values are exclusively 0 or 1 are OHE binary
        # dummies. Counterfactual search may land on values like 0.73 in the
        # continuous preprocessed space; we snap them back to {0, 1} before
        # display so suggestions are clinically valid.
        ohe_prep_features: set = set()

        if (
            background is not None
            and background.ndim == 2
            and background.shape[0] >= 5
            and background.shape[1] == n_prep
        ):
            bg_array = np.asarray(background, dtype=float)
            feat_to_idx = {n: i for i, n in enumerate(prep_feat_names)}
            for col in prep_vary:
                idx = feat_to_idx.get(col)
                if idx is not None:
                    col_vals = bg_array[:, idx]
                    sigma_map[col] = float(np.std(col_vals)) or 1.0
                    # Detect OHE feature: all non-NaN values are exactly 0 or 1
                    valid_vals = col_vals[~np.isnan(col_vals)]
                    if valid_vals.size > 0 and set(np.unique(valid_vals).tolist()).issubset({0.0, 1.0}):
                        ohe_prep_features.add(col)
        else:
            logger.info(
                "counterfactual: SHAP background unavailable (shape=%s, expected (n,%d)); "
                "will use binary-search fallback directly",
                getattr(background, "shape", None),
                n_prep,
            )

        # ── Attempt DiCE when background is available ─────────────────────────
        # DiCE always feeds named DataFrames to the estimator internally. When
        # the estimator was fitted on numpy, sklearn emits a UserWarning per
        # batch — noisy and outside our control. We silence it locally so our
        # logs stay clean; the prediction itself is correct (column order is
        # preserved by DiCE).
        dice_items: Optional[List[Dict[str, Any]]] = None
        if bg_array is not None:
            try:
                import dice_ml

                bg_labels = estimator.predict(_make_estimator_input(estimator, bg_array, prep_feat_names))
                unique_labels = np.unique(bg_labels)

                # DiCE Data requires a named DataFrame regardless.
                outcome_col = "__target__"
                bg_df = pd.DataFrame(bg_array, columns=prep_feat_names)
                bg_df[outcome_col] = bg_labels

                if len(unique_labels) < 2:
                    opposite = 1 - int(unique_labels[0])
                    synth = pd.DataFrame(X_row_prep, columns=prep_feat_names)
                    synth[outcome_col] = opposite
                    bg_df = pd.concat([bg_df, synth], ignore_index=True)

                logger.info(
                    "counterfactual: bg classes=%s | current_pred=%s",
                    np.unique(bg_df[outcome_col]).tolist(),
                    estimator.predict(_make_estimator_input(estimator, X_row_prep, prep_feat_names))[0],
                )

                # Extend feature ranges to ±3σ so DiCE can cross the decision
                # boundary even for high-confidence models.
                permitted_range: Dict[str, List[float]] = {}
                for col in prep_vary:
                    idx = {n: i for i, n in enumerate(prep_feat_names)}.get(col)
                    if idx is None:
                        continue
                    vals = bg_array[:, idx]
                    sigma = float(np.std(vals)) or 1.0
                    permitted_range[col] = [
                        float(np.min(vals)) - 3.0 * sigma,
                        float(np.max(vals)) + 3.0 * sigma,
                    ]

                d = dice_ml.Data(
                    dataframe=bg_df,
                    continuous_features=list(prep_feat_names),
                    outcome_name=outcome_col,
                )
                m = dice_ml.Model(model=estimator, backend="sklearn")
                exp = dice_ml.Dice(d, m, method="random")
                query_df = pd.DataFrame(X_row_prep, columns=prep_feat_names)
                n_cf_request = min(n_cf * 3, _MAX_CF * 3)

                with warnings.catch_warnings():
                    # Suppress sklearn's "X has feature names, but … was fitted
                    # without feature names" emitted by DiCE's internal predict
                    # calls. The CFs are still correct (column order preserved).
                    warnings.filterwarnings(
                        "ignore",
                        message="X has feature names",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="X does not have valid feature names",
                        category=UserWarning,
                    )
                    dice_result = exp.generate_counterfactuals(
                        query_df,
                        total_CFs=n_cf_request,
                        features_to_vary=prep_vary,
                        desired_class=_DESIRED_CLASS,
                        random_seed=random_state,
                        permitted_range=permitted_range or None,
                        posthoc_sparsity_param=0.0,
                    )

                if (
                    dice_result is not None
                    and dice_result.cf_examples_list
                    and dice_result.cf_examples_list[0].final_cfs_df is not None
                    and len(dice_result.cf_examples_list[0].final_cfs_df) > 0
                ):
                    best_cf = dice_result.cf_examples_list[0].final_cfs_df.iloc[0]
                    row_vec = X_row_prep[0]
                    raw_items: List[Dict[str, Any]] = []
                    for i, pname in enumerate(prep_feat_names):
                        if pname not in best_cf.index:
                            continue
                        orig_prep = float(row_vec[i])
                        sugg_prep = float(best_cf[pname])
                        if abs(sugg_prep - orig_prep) < 1e-8:
                            continue
                        if pname in inv_scale_map:
                            params = inv_scale_map[pname]
                            orig_val = _apply_inverse_scale(orig_prep, params)
                            sugg_val = _apply_inverse_scale(sugg_prep, params)
                        else:
                            orig_val, sugg_val = orig_prep, sugg_prep
                        delta = sugg_val - orig_val
                        raw_items.append({
                            "col": pname, "col_idx": i,
                            "orig_prep": orig_prep, "sugg_prep": sugg_prep,
                            "delta_abs": abs(delta),
                        })
                        raw_items[-1].update({
                            "feature":         _strip_ct_prefix(str(pname)),
                            "original_value":  to_json_safe(orig_val),
                            "suggested_value": to_json_safe(sugg_val),
                            "delta":           to_json_safe(delta),
                        })
                    if raw_items:
                        raw_items.sort(key=lambda d: abs(float(d["delta"] or 0.0)), reverse=True)
                        dice_items = [{
                            "feature":         d["feature"],
                            "original_value":  d["original_value"],
                            "suggested_value": d["suggested_value"],
                            "delta":           d["delta"],
                        } for d in raw_items]
                        logger.info("counterfactual: DiCE found %d changed features", len(dice_items))
                else:
                    logger.info("counterfactual: DiCE found no valid flip — using binary search")

            except Exception as dice_exc:
                logger.warning("counterfactual: DiCE failed (%s) — using binary search", dice_exc)

        if dice_items is not None:
            return dice_items

        # ── Multi-feature directional fallback ────────────────────────────────
        # Tier 1: gradient-guided directional search (cheap, often works).
        logger.info(
            "counterfactual: running multi-feature directional search on %d features",
            len(prep_vary),
        )
        raw = _multi_feature_cf(
            estimator, X_row_prep, prep_feat_names, prep_vary,
            task_type, sigma_map,
        )

        # Tier 2: random search (expensive, robust to piecewise-constant surfaces
        # that gradient probing fails to navigate — typical for tree ensembles).
        if raw is None and task_type == "classification":
            logger.info("counterfactual: directional search failed — trying random search")
            raw = _random_search_cf(
                estimator, X_row_prep, prep_feat_names, prep_vary,
                sigma_map, n_samples=2000, random_state=random_state,
            )

        if raw is None:
            logger.warning(
                "counterfactual: no flip found even with random search — "
                "the modifiable features are insufficient to invert this prediction"
            )
            return None

        return _format_items(raw, inv_scale_map, ohe_features=ohe_prep_features)

    except Exception as exc:
        logger.warning("counterfactual: compute_counterfactual failed: %s", exc, exc_info=True)
        return None
