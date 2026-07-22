"""Runtime signals the agent would otherwise swallow.

A response cut short by the output-token ceiling is reported only as
``stop_reason == "max_tokens"`` on the raw message. Nothing raises, so a clipped draft
or a half-written critique is indistinguishable from a finished one.

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
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TruncationWarner(BaseCallbackHandler):
    """Counts model responses that hit the output-token ceiling.

    Attach it per invocation via ``config={"callbacks": [warner]}``; it sees nested
    subagent calls too, since callbacks propagate down the graph.
    """

    def __init__(self) -> None:
        self.truncated = 0

    def reset(self) -> None:
        """Zero the counter so each turn reports only its own truncations."""
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
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                metadata = getattr(message, "response_metadata", None) or {}
                if metadata.get("stop_reason") == "max_tokens":
                    self.truncated += 1
                    logger.warning(
                        "Model response truncated at the output-token ceiling "
                        "(stop_reason=max_tokens); raise SPEECHWRITER_MAX_TOKENS."
                    )
