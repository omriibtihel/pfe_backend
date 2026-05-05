"""LLM client implementations and routing."""
from __future__ import annotations

from app.services.reporting.llm_client.base import LLMClient, LLMReport
from app.services.reporting.llm_client.groq import GroqClient, GroqConfig, GroqError
from app.services.reporting.llm_client.ollama import (
    OllamaClient,
    OllamaConfig,
    OllamaError,
)
from app.services.reporting.llm_client.router import LLMRouter, RouterResult
from app.services.reporting.llm_client.template import TemplateClient

__all__ = [
    "GroqClient",
    "GroqConfig",
    "GroqError",
    "LLMClient",
    "LLMReport",
    "LLMRouter",
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "RouterResult",
    "TemplateClient",
]
