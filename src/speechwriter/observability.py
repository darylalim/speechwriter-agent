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

It counts **refusals** for the same reason. When a Claude safety classifier declines a call
that the server-side fallback could not rescue, the response is an ordinary HTTP 200 with
``stop_reason="refusal"`` and little or no text — which, inside a subagent, reaches the
orchestrator as the same empty ``status="success"`` tool result a truncation does.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Every provider spells "I ran out of output tokens" differently. The agent builds only Claude
# clients now, but this class is public API (`speechwriter.TruncationWarner`) and a consumer
# may attach it to any LangChain model, so matching only Anthropic's `stop_reason` would
# silently return truncation detection to zero for theirs. Compared lowercased, since Gemini
# reports `MAX_TOKENS`.
_TRUNCATION_REASONS = frozenset({"max_tokens", "max_output_tokens", "length"})
_STOP_REASON_KEYS = ("stop_reason", "finish_reason")
# Anthropic's spelling only: no other provider reports a classifier decline as a stop reason.
_REFUSAL = "refusal"


class TruncationWarner(BaseCallbackHandler):
    """Counts model responses that hit the output-token ceiling, or were refused.

    Owned by :class:`~speechwriter.agent.SpeechwriterAgent`; attach it to an invocation
    via ``bundle.turn_config(...)``. It sees nested subagent calls too, since callbacks
    propagate down the graph — which is also why the counter is lock-guarded: tool calls
    within one turn can execute concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.truncated = 0
        self.refused = 0
        # The classifier categories behind this turn's refusals ("cyber", "general_harms", …),
        # when the API named one — what a reader needs to tell a benign false positive from a
        # brief that genuinely strayed.
        self.refusal_categories: list[str] = []

    def reset(self) -> None:
        """Zero the counters so each turn reports only its own truncations and refusals."""
        with self._lock:
            self.truncated = 0
            self.refused = 0
            self.refusal_categories = []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record any generation that hit the token ceiling or was refused."""
        messages = [
            getattr(generation, "message", None)
            for generations in response.generations
            for generation in generations
        ]
        hits = sum(1 for message in messages if _was_truncated(message))
        refusals = [_refusal_category(message) for message in messages if _was_refused(message)]

        if hits:
            with self._lock:
                self.truncated += hits
            logger.warning(
                "%d model response(s) truncated at the output-token ceiling; "
                "raise SPEECHWRITER_MAX_TOKENS.",
                hits,
            )
        if refusals:
            named = [category for category in refusals if category]
            with self._lock:
                self.refused += len(refusals)
                self.refusal_categories.extend(named)
            logger.warning(
                "%d model response(s) refused by a safety classifier%s.",
                len(refusals),
                f" ({', '.join(named)})" if named else "",
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


def _was_refused(message: object) -> bool:
    """True when a message's provider metadata reports a safety-classifier decline."""
    if message is None:
        return False
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("stop_reason") == _REFUSAL


def _refusal_category(message: object) -> str | None:
    """The classifier category a refusal named, if any — ``stop_details`` is optional."""
    metadata = getattr(message, "response_metadata", None) or {}
    details = metadata.get("stop_details")
    category = details.get("category") if isinstance(details, dict) else None
    return category if isinstance(category, str) else None
