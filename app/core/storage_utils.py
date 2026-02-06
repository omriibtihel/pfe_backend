from __future__ import annotations
from uuid import uuid4

def make_unique_stored_name(prefix: str = "") -> str:
    return f"{prefix}{uuid4().hex}"
