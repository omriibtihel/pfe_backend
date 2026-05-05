"""OllamaClient — calls a local Ollama server with JSON-mode forced.

The client returns a fully parsed dict, so ``ReportService.generate`` can use
the same contract for local and hosted LLM backends.

Failure modes raise ``OllamaError`` so the service / router can decide whether
to retry, escalate, or fall back. The client never tries to "fix" a malformed
response — that responsibility belongs to ``ReportValidator``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests

from app.services.reporting.context_builder import ReportContext
from app.services.reporting.llm_client.base import LLMReport
from app.services.reporting.prompts import PromptBuilder

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised on any failure to obtain a usable string from Ollama."""


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    timeout_s: int = 30
    # Lower temperature → more deterministic, fewer schema deviations.
    temperature: float = 0.2
    # Bound the generation to keep latency predictable on CPU.
    num_predict: int = 1024


class OllamaClient:
    """LLMClient backend for a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        config: OllamaConfig,
        *,
        prompt_builder: Optional[PromptBuilder] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._cfg = config
        self._prompts = prompt_builder or PromptBuilder()
        self._http = session or requests.Session()

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, context: ReportContext, *, reinforced: bool = False) -> LLMReport:
        prompts = self._prompts.build(context, reinforced=reinforced)
        raw_text = self._call_generate(
            system=prompts["system"],
            user=prompts["user"],
        )
        try:
            parsed = json.loads(raw_text)
        except (ValueError, TypeError) as exc:
            # Don't try to repair here — the validator handles partial JSON.
            # We just surface the raw string, wrapped in a sentinel dict so
            # the validator sees the bytes and can run its extraction.
            raise OllamaError(f"non_json_response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("response_not_object")
        return parsed

    def warmup(self) -> bool:
        """Best-effort warm call to load the model into memory.

        Ollama lazily loads weights on first use (5–15 s on CPU), so this is
        called at server startup to amortize that cost. Returns False if the
        daemon is unreachable; caller decides whether that is fatal.
        """
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/api/generate",
                json={
                    "model": self._cfg.model,
                    "prompt": "ok",
                    "stream": False,
                    "options": {"num_predict": 1, "temperature": 0.0},
                },
                timeout=self._cfg.timeout_s,
            )
            return resp.ok
        except requests.RequestException as exc:
            logger.info("ollama.warmup_unreachable: %s", exc)
            return False

    # ── Internals ────────────────────────────────────────────────────────────

    def _call_generate(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "format": "json",  # constrains output to valid JSON when supported
            "options": {
                "temperature": self._cfg.temperature,
                "num_predict": self._cfg.num_predict,
            },
        }
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/api/generate",
                json=payload,
                timeout=self._cfg.timeout_s,
            )
        except requests.Timeout as exc:
            raise OllamaError("timeout") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"network: {exc}") from exc

        if not resp.ok:
            raise OllamaError(f"http_{resp.status_code}: {resp.text[:200]}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise OllamaError("malformed_envelope") from exc

        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise OllamaError("empty_response")
        return text


__all__ = ["OllamaClient", "OllamaConfig", "OllamaError"]
