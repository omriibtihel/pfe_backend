"""LLM client protocol — every backend conforms to ``LLMClient.generate``.

The protocol returns a structured ``LLMReport`` (a plain dict shaped per the
JSON schema in the integration plan). Whether the real client streams under
the hood or not, the public API is "give me the full validated report".
The streaming concern lives one layer up in ``ReportService``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

LLMReport = dict[str, Any]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every report backend must satisfy.

    The ``reinforced`` kwarg is set to True on the second attempt when the
    validator rejected the first (schema / blocklist failure). Backends with
    a prompt builder pass it through; deterministic backends (TemplateClient)
    ignore it.
    """

    name: str

    def generate(self, context: "Any", *, reinforced: bool = False) -> LLMReport: ...  # noqa: F821
