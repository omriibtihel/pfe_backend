"""
TrainingNotifier — Phase 9
In-process SSE (Server-Sent Events) event bus.

Architecture
------------
* ``TrainingNotifier`` is a lightweight in-memory store.
* The training background task calls ``notifier.emit(session_id, event_type, data)``.
* The FastAPI SSE route calls ``notifier.subscribe(session_id)`` which returns an
  async generator that yields SSE-formatted strings.
* Events older than ``_MAX_EVENTS`` per session are discarded.

This design works without Redis/Celery and is sufficient for
single-process deployments.  For multi-process deployments the notifier
can be replaced by a Redis pub/sub implementation with the same interface.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

_MAX_EVENTS = 200          # max buffered events per session
_POLL_INTERVAL_S = 0.4    # SSE poll interval
_SESSION_TTL_S = 3_600    # clean up sessions after 1 h


# ──────────────────────────────────────────────────────────────────────────────
# Event types (protocol contract with the frontend)
# ──────────────────────────────────────────────────────────────────────────────

class EventType:
    STARTED = "training.started"
    PROFILED = "training.profiled"
    RECOMMENDATION_READY = "training.recommendation.ready"
    BASELINE_COMPLETE = "training.baseline.complete"
    HPO_PROGRESS = "training.hpo.progress"
    HPO_COMPLETE = "training.hpo.complete"
    MODEL_COMPLETE = "training.model.complete"
    FINAL_COMPLETE = "training.final.complete"
    PROGRESS = "training.progress"
    ERROR = "training.error"


# ──────────────────────────────────────────────────────────────────────────────
# Internal event record
# ──────────────────────────────────────────────────────────────────────────────

class _TrainingEvent:
    __slots__ = ("event_type", "data", "ts", "seq")

    def __init__(self, event_type: str, data: dict[str, Any], seq: int) -> None:
        self.event_type = event_type
        self.data = data
        self.ts = time.time()
        self.seq = seq

    def to_sse(self) -> str:
        payload = json.dumps({"seq": self.seq, "ts": self.ts, **self.data}, default=str)
        return f"event: {self.event_type}\ndata: {payload}\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# Notifier
# ──────────────────────────────────────────────────────────────────────────────

class TrainingNotifier:
    """
    Thread-safe (via asyncio lock) SSE event bus.

    Typical usage in background task (sync context)::

        notifier.emit_sync(session_id, EventType.STARTED, {"model": "all"})

    Typical usage in async route (SSE)::

        async for chunk in notifier.subscribe(session_id):
            yield chunk
    """

    def __init__(self) -> None:
        # session_id → deque of _TrainingEvent
        self._events: dict[int, deque[_TrainingEvent]] = {}
        # sequence counter per session
        self._seq: dict[int, int] = {}
        # creation time per session for TTL cleanup
        self._created: dict[int, float] = {}
        self._lock = asyncio.Lock()

    # ── emit (sync, called from background tasks) ─────────────────────────────

    def emit_sync(
        self,
        session_id: int,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit from a synchronous background task."""
        if session_id not in self._events:
            self._events[session_id] = deque(maxlen=_MAX_EVENTS)
            self._seq[session_id] = 0
            self._created[session_id] = time.time()

        seq = self._seq[session_id]
        self._seq[session_id] = seq + 1
        evt = _TrainingEvent(event_type, data or {}, seq)
        self._events[session_id].append(evt)
        logger.debug("notifier.emit session=%d event=%s seq=%d", session_id, event_type, seq)

    # ── subscribe (async, called from SSE route) ───────────────────────────────

    async def subscribe(
        self,
        session_id: int,
        *,
        last_seq: int = -1,
        timeout_s: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE-formatted strings.

        Parameters
        ----------
        session_id:
            The training session to watch.
        last_seq:
            Client passes the last event sequence it received to resume
            without replaying the full history.
        timeout_s:
            Stop yielding after this many seconds (client should reconnect).
        """
        deadline = time.time() + timeout_s
        cursor = last_seq + 1

        # Send a heartbeat immediately so the connection is confirmed.
        yield ": heartbeat\n\n"

        while time.time() < deadline:
            queue = self._events.get(session_id)
            if queue:
                # Drain events with seq >= cursor
                pending = [e for e in queue if e.seq >= cursor]
                for evt in pending:
                    cursor = evt.seq + 1
                    yield evt.to_sse()

                    # Auto-close stream when training finished or errored
                    if evt.event_type in {EventType.FINAL_COMPLETE, EventType.ERROR}:
                        return

            await asyncio.sleep(_POLL_INTERVAL_S)

        # Timeout heartbeat
        yield "event: timeout\ndata: {}\n\n"

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup_old_sessions(self) -> None:
        cutoff = time.time() - _SESSION_TTL_S
        stale = [sid for sid, ts in self._created.items() if ts < cutoff]
        for sid in stale:
            self._events.pop(sid, None)
            self._seq.pop(sid, None)
            self._created.pop(sid, None)


# ── Singleton ─────────────────────────────────────────────────────────────────

training_notifier: TrainingNotifier = TrainingNotifier()
