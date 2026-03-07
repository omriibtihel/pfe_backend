"""Shared utilities for the training pipeline."""
from __future__ import annotations

import json
import logging
from typing import Any

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
