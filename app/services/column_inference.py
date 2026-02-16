import pandas as pd

KINDS = {"numeric","categorical","text","binary","datetime","id","other"}

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
