import math

import pandas as pd
import numpy as np

KINDS = {"numeric","categorical","text","binary","datetime","id","other"}


_ALL_NONE = {
    "skewness": None, "outlier_count": None, "outlier_ratio": None, "has_negative": None,
    "min_val": None, "max_val": None, "mean_val": None, "median_val": None,
    "q1_val": None, "q3_val": None,
}


def _safe_float(value: float) -> float | None:
    # NaN/Inf are not JSON-compliant under starlette's allow_nan=False.
    return value if math.isfinite(value) else None


def numeric_extra_stats(non_null: pd.Series) -> dict:
    """Full numeric summary for a series (already stripped of NaN/inf by caller).

    Basic stats (min/max/mean/median/quartiles/has_negative) require ≥ 1 value.
    Skewness and IQR-based outlier detection require ≥ 4 values.
    Returns all-None for an empty series.
    """
    from scipy.stats import skew as _skew  # lazy import

    data = non_null.to_numpy(dtype=float)
    if len(data) == 0:
        return dict(_ALL_NONE)

    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    basic = {
        "min_val":    _safe_float(float(data.min())),
        "max_val":    _safe_float(float(data.max())),
        "mean_val":   _safe_float(float(data.mean())),
        "median_val": _safe_float(float(np.median(data))),
        "q1_val":     _safe_float(q1),
        "q3_val":     _safe_float(q3),
        "has_negative": bool(float(data.min()) <= 0),
    }

    if len(data) < 4:
        return {**basic, "skewness": None, "outlier_count": None, "outlier_ratio": None}

    skewness = _safe_float(float(_skew(data)))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((data < lower) | (data > upper)).sum())
    return {
        **basic,
        "skewness":     skewness,
        "outlier_count": outlier_count,
        "outlier_ratio": _safe_float(float(outlier_count / len(data))),
    }



def detect_parasites(s: pd.Series) -> dict | None:
    """
    Detect non-numeric parasitic values in a column that should be numeric.

    A column is "suspicious" when its dtype is object but >= 50% of non-null
    values convert to a number. The non-convertible values are the parasites.

    Returns:
        {"count": int, "distinct": list[str] (max 5), "convertible_ratio": float}
        or None if the column is already numeric / not suspicious / no parasites.
    """
    if (
        pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_datetime64_any_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    ):
        return None

    non_null = s.dropna()
    if len(non_null) == 0:
        return None

    converted = pd.to_numeric(non_null, errors="coerce")
    convertible_ratio = float(converted.notna().mean())

    if convertible_ratio < 0.50:
        return None

    parasite_mask = converted.isna()
    count = int(parasite_mask.sum())
    if count == 0:
        return None

    distinct = (
        non_null[parasite_mask]
        .astype(str)
        .value_counts()
        .head(5)
        .index.tolist()
    )
    return {
        "count": count,
        "distinct": distinct,
        "convertible_ratio": round(convertible_ratio, 4),
    }

_BIN_STR = {"0","1","true","false","yes","no","y","n","oui","non","vrai","faux"}

def _norm_vals(values):
    out = []
    for x in values:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(s)
    return out[:80]

def _looks_like_id(col_name: str, s: pd.Series, unique: int, nn: int) -> bool:
    name = (col_name or "").lower()
    if name == "id" or name.endswith("_id") or name.startswith("id_"):
        return True
    # heuristique : quasi unique + pas trop de NaN
    if nn > 0 and unique >= int(0.98 * nn) and nn >= 50:
        return True
    return False

def _binary_from_values(sample):
    if not sample:
        return False
    low = [v.lower() for v in sample]
    return all(v in _BIN_STR for v in low)

def _numeric_ratio(sample):
    if not sample:
        return 0.0
    ok = 0
    for v in sample:
        vv = v.replace(",", ".")
        try:
            float(vv)
            ok += 1
        except:
            pass
    return ok / len(sample)

def infer_kind_for_series(name: str, s: pd.Series) -> tuple[str, float]:
    col = name or ""
    non_null = s.dropna()
    nn = int(len(non_null))
    unique = int(non_null.nunique(dropna=True)) if nn else 0

    # datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", 0.90

    # bool dtype => binary
    if pd.api.types.is_bool_dtype(s):
        return "binary", 0.90

    # id heuristic
    if _looks_like_id(col, s, unique, nn):
        return "id", 0.85

    # numeric dtype
    if pd.api.types.is_numeric_dtype(s):
        if unique == 2:
            return "binary", 0.80
        # codes (0/1/2...) => souvent catégoriel si faible cardinalité
        ratio = (unique / nn) if nn else 1.0
        if 0 < unique <= 12 and ratio <= 0.05:
            return "categorical", 0.70
        return "numeric", 0.85

    # object/string
    sample = _norm_vals(non_null.astype(str).head(80).tolist())

    # binaire string
    if unique == 2 and _binary_from_values(sample):
        return "binary", 0.85

    # numeric stored as strings
    nr = _numeric_ratio(sample)
    if nr >= 0.90:
        if unique == 2:
            return "binary", 0.70
        ratio = (unique / nn) if nn else 1.0
        if 0 < unique <= 12 and ratio <= 0.05:
            return "categorical", 0.65
        return "numeric", 0.70

    # text vs categorical
    ratio = (unique / nn) if nn else 1.0
    avg_len = (sum(len(x) for x in sample) / len(sample)) if sample else 0.0
    has_spaces = any(" " in x for x in sample)
    long_vals = sum(1 for x in sample if len(x) >= 25)

    # categorical if low cardinality
    if 0 < unique <= 25 and ratio <= 0.05:
        return "categorical", 0.80

    # text if looks like free text
    if avg_len >= 18 or has_spaces or long_vals >= 1:
        return "text", 0.78

    # fallback
    return "categorical", 0.55
