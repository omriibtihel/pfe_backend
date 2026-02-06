# app/utils/json_sanitize.py
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(obj: Any):
    """Convertit récursivement un objet en structure JSON-serializable."""
    # None / primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas types
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (pd.Series,)):
        return to_json_safe(obj.to_list())
    if isinstance(obj, (pd.DataFrame,)):
        return to_json_safe(obj.to_dict(orient="records"))

    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]

    # fallback (string)
    return str(obj)
