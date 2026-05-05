"""GroqClient — primary LLM backend via Groq's managed API.

Groq serves Llama 3.3 70B with sub-second latency, no local GPU required,
and no warm-up penalty. It is the default active backend in production.

Activation: set ``REPORTING_GROQ_API_KEY`` in the environment. When the key
is empty the client raises ``GroqError("disabled")`` at construction time so
the router skips it silently and falls back to TemplateClient.

Ollama (on-prem) can be added in front of Groq in the chain by setting
``REPORTING_OLLAMA_URL`` and ``REPORTING_OLLAMA_MODEL`` — see service.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from app.services.reporting.context_builder import ReportContext
from app.services.reporting.llm_client.base import LLMReport
from app.services.reporting.prompts import PromptBuilder

logger = logging.getLogger(__name__)


class GroqError(RuntimeError):
    """Raised on any failure to obtain a usable JSON object from Groq."""


@dataclass
class GroqConfig:
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_s: int = 30
    temperature: float = 0.2
    max_tokens: int = 1024


class GroqClient:
    """LLMClient backend for Groq's OpenAI-compatible chat-completions API.

    The API speaks OpenAI-style: messages array + ``response_format`` for
    JSON-mode constraint. We use ``json_object`` mode whenever the model
    supports it (all Llama 3 instruct variants do as of the integration
    plan's writing — January 2026).
    """

    name = "groq"

    def __init__(
        self,
        config: GroqConfig,
        *,
        prompt_builder: Optional[PromptBuilder] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not config.api_key:
            raise GroqError("disabled")
        self._cfg = config
        self._prompts = prompt_builder or PromptBuilder()
        self._http = session or requests.Session()

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, context: ReportContext, *, reinforced: bool = False) -> LLMReport:
        prompts = self._prompts.build(context, reinforced=reinforced)
        text = self._chat_completion(system=prompts["system"], user=prompts["user"])
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise GroqError(f"non_json_response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise GroqError("response_not_object")
        return parsed

    # ── Internals ────────────────────────────────────────────────────────────

    def _chat_completion(self, *, system: str, user: str) -> str:
        body = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._cfg.api_key}"},
                timeout=self._cfg.timeout_s,
            )
        except requests.Timeout as exc:
            raise GroqError("timeout") from exc
        except requests.RequestException as exc:
            raise GroqError(f"network: {exc}") from exc

        if not resp.ok:
            raise GroqError(f"http_{resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise GroqError("malformed_envelope") from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqError("missing_content") from exc

        if not isinstance(text, str) or not text.strip():
            raise GroqError("empty_response")
        return text


__all__ = ["GroqClient", "GroqConfig", "GroqError"]
