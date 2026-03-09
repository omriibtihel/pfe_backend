"""Shared utilities for the training pipeline."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def log_event(event: str, **payload: Any) -> None:
    """Emit a structured JSON log entry."""
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str, ensure_ascii=False))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_python_scalar(v: Any) -> Any:
    """Convert a numpy scalar to a Python builtin; replace nan/inf with None."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None
    return v


def safe_json_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = 20,
) -> Any:
    """Recursively convert *value* to a JSON-serializable form (depth-limited).

    Handles numpy scalars, dicts, lists, arrays, and callables gracefully.
    """
    value = to_python_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        if depth >= max_depth:
            return {"type": "dict", "size": int(len(value))}
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                out["__truncated__"] = int(len(value) - max_items)
                break
            out[str(k)] = safe_json_value(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if depth >= max_depth:
            return {"type": type(value).__name__, "length": int(len(seq))}
        out_list = [
            safe_json_value(v, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out_list.append(f"...(+{len(seq) - max_items} more)")
        return out_list

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return {"type": value.__class__.__name__, "shape": [int(s) for s in shape]}
        except Exception:
            return {"type": value.__class__.__name__}

    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__)
        return f"<callable:{name}>"

    return f"<{value.__class__.__name__}>"
