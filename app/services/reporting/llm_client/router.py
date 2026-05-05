"""LLMRouter — walks the fallback chain Groq → Template (Ollama optional).

The router owns the state machine in plan §8:

- For each client (in declared order):
  - Call ``generate(ctx, reinforced=False)``.
  - On exception (timeout / network / non-JSON): escalate to next client.
  - Validate the response.
    - PASS → return.
    - REJECTED with ``invalid_json``: escalate to next client (no retry).
    - REJECTED with schema / blocklist: retry the same client once with
      ``reinforced=True`` and re-validate.
      - PASS → return.
      - Still REJECTED → escalate to next client.
- TemplateClient is appended at the end of the chain if not already present;
  it is deterministic and never validated, so the router *always* succeeds.

The router is injected with a ``ReportValidator`` so tests can swap in a
stricter or laxer one without touching the chain logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from app.services.reporting.context_builder import ReportContext
from app.services.reporting.llm_client.base import LLMClient, LLMReport
from app.services.reporting.llm_client.template import TemplateClient
from app.services.reporting.validator import ReportValidator

logger = logging.getLogger(__name__)


@dataclass
class RouterAttempt:
    client: str
    reinforced: bool
    outcome: str           # "passed" | "exception:..." | reason from validator


@dataclass
class RouterResult:
    provider: str
    report: LLMReport
    retries: int                       # retries on the *winning* client
    fallback_reason: str | None        # last failure encountered, None if first try passed
    attempts: list[RouterAttempt] = field(default_factory=list)


class LLMRouter:
    """Stateless walker over a list of LLM clients."""

    def __init__(
        self,
        clients: Iterable[LLMClient],
        *,
        validator: ReportValidator | None = None,
    ) -> None:
        chain = list(clients)
        # Idempotent insert of the deterministic safety net at the bottom.
        if all(getattr(c, "name", None) != "template" for c in chain):
            chain.append(TemplateClient())
        self._chain: list[LLMClient] = chain
        self._validator = validator or ReportValidator()

    @property
    def chain(self) -> list[LLMClient]:
        return list(self._chain)

    def generate(self, context: ReportContext) -> RouterResult:
        attempts: list[RouterAttempt] = []
        last_failure: str | None = None

        for client in self._chain:
            # The template client is the canonical floor — its output is
            # already canonical, so we skip validation and short-circuit here.
            if client.name == "template":
                report = client.generate(context)
                return RouterResult(
                    provider="template",
                    report=report,
                    retries=0,
                    fallback_reason=last_failure,
                    attempts=attempts,
                )

            verdict = self._try_client(client, context, attempts)
            if verdict is not None:
                provider, report, retries = verdict
                return RouterResult(
                    provider=provider,
                    report=report,
                    retries=retries,
                    fallback_reason=last_failure,
                    attempts=attempts,
                )
            # Capture the latest reason for upstream observability
            if attempts:
                last_failure = f"{attempts[-1].client}:{attempts[-1].outcome}"

        # Defensive — TemplateClient is always appended, so we shouldn't get
        # here. But if the chain was tampered with, fall back to a fresh one.
        logger.warning("reporting.router_exhausted attempts=%d", len(attempts))
        report = TemplateClient().generate(context)
        return RouterResult(
            provider="template",
            report=report,
            retries=0,
            fallback_reason=last_failure or "router_exhausted",
            attempts=attempts,
        )

    # ── Per-client state machine ─────────────────────────────────────────────

    def _try_client(
        self,
        client: LLMClient,
        context: ReportContext,
        attempts: list[RouterAttempt],
    ) -> tuple[str, LLMReport, int] | None:
        """Returns (provider, report, retries) if the client succeeds, else None."""
        for attempt in range(2):  # 1 try + 1 reinforced retry max
            reinforced = attempt > 0
            try:
                raw = client.generate(context, reinforced=reinforced)
            except Exception as exc:
                outcome = f"exception:{type(exc).__name__}"
                attempts.append(RouterAttempt(client.name, reinforced, outcome))
                logger.info(
                    "reporting.client_exception client=%s reinforced=%s err=%s",
                    client.name, reinforced, exc,
                )
                return None  # escalate to next client — no retry on exception

            result = self._validator.validate(raw, context.lang)
            if result.status == "passed" and result.report is not None:
                attempts.append(RouterAttempt(client.name, reinforced, "passed"))
                return client.name, result.report, attempt

            outcome = result.reason or "rejected"
            attempts.append(RouterAttempt(client.name, reinforced, outcome))

            # Plan §8: invalid_json doesn't get a reinforced retry on the
            # same client — escalate immediately. Schema/blocklist failures
            # do retry once.
            if result.reason == "invalid_json":
                return None

        # Both attempts (initial + reinforced) failed validation → escalate.
        return None


__all__ = ["LLMRouter", "RouterResult", "RouterAttempt"]
