"""Runtime signals the agent would otherwise swallow.

A response cut short by the output-token ceiling is reported only as a provider stop
reason on the raw message. Nothing raises, so a clipped draft or a half-written critique
is indistinguishable from a finished one.

That matters most for **subagents**. deepagents turns a subagent run into a ``task`` tool
result by walking back to its last message with text, so:

* a critique truncated *after* some text is handed back and acted on as if complete; and
* one truncated *before* any text — entirely possible, because extended thinking bills
  against the same ceiling — comes back as an empty string with ``status="success"``.

Neither case logs anything on its own. :class:`TruncationWarner` watches every model call
in the graph, orchestrator and subagents alike, so the CLI can say so out loud.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Every provider spells "I ran out of output tokens" differently, and `SPEECHWRITER_MODEL`
# is free-form — `init_chat_model` infers the provider from the id, so a non-Anthropic
# model is one env var away. Matching only Anthropic's `stop_reason` would silently return
# truncation detection to zero for those, which is the exact failure this class exists to
# catch. Compared lowercased, since Gemini reports `MAX_TOKENS`.
_TRUNCATION_REASONS = frozenset({"max_tokens", "max_output_tokens", "length"})
_STOP_REASON_KEYS = ("stop_reason", "finish_reason")


class TruncationWarner(BaseCallbackHandler):
    """Counts model responses that hit the output-token ceiling.

    Owned by :class:`~speechwriter.agent.SpeechwriterAgent`; attach it to an invocation
    via ``bundle.turn_config(...)``. It sees nested subagent calls too, since callbacks
    propagate down the graph — which is also why the counter is lock-guarded: tool calls
    within one turn can execute concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.truncated = 0

    def reset(self) -> None:
        """Zero the counter so each turn reports only its own truncations."""
        with self._lock:
            self.truncated = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record any generation whose provider stop reason was the token ceiling."""
        hits = sum(
            1
            for generations in response.generations
            for generation in generations
            if _was_truncated(getattr(generation, "message", None))
        )
        if not hits:
            return

        with self._lock:
            self.truncated += hits
        logger.warning(
            "%d model response(s) truncated at the output-token ceiling; "
            "raise SPEECHWRITER_MAX_TOKENS.",
            hits,
        )


def _was_truncated(message: object) -> bool:
    """True when a message's provider metadata reports an output-token cutoff."""
    if message is None:
        return False
    metadata = getattr(message, "response_metadata", None) or {}
    return any(
        isinstance(value, str) and value.lower() in _TRUNCATION_REASONS
        for value in (metadata.get(key) for key in _STOP_REASON_KEYS)
    )
