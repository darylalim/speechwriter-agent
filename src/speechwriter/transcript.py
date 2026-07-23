"""Turning a streamed LangChain message into renderable events — shared by both front ends.

The CLI (Rich) and the web UI (Streamlit) render to different backends but must agree on the
*interpretation* of a streamed message: which parts are tool calls, how a tool result's
success is read, what counts as prose. That decode lives here, once, as UI-neutral
:class:`Event` objects; each front end owns only its own formatting — truncation width,
colour, escaping. Kept free of any Streamlit or Rich import so the CLI never drags the web
stack in and vice versa.

``Event.text`` is the *raw* string (full tool arguments, full result, full prose). Clipping
is deliberately not baked in: the CLI clips to a narrower width without escaping (Rich markup
tolerates backticks), while the web UI clips wider and neutralises the backtick that would
break an inline-code span. :func:`clip` is the one piece of that they do share.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class Event:
    """One renderable thing the agent did: a tool call, its result, or prose.

    ``text`` is raw and unclipped; a renderer applies :func:`clip` (and any escaping) itself.
    """

    kind: Literal["call", "result", "prose"]
    text: str
    name: str = ""
    ok: bool = True


def iter_events(message: Any) -> Iterator[Event]:
    """Decode one streamed message into zero or more renderable events.

    Only ``AIMessage`` (tool calls, then prose) and ``ToolMessage`` (a result) produce
    output. Anything else in the streamed value — the user's echoed prompt, a system
    message — yields nothing, which is why there is no explicit skip for those kinds.
    """
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            arguments = json.dumps(call.get("args", {}), ensure_ascii=False, default=str)
            yield Event(kind="call", name=call.get("name", "tool"), text=arguments)
        text = message.text.strip()
        if text:
            yield Event(kind="prose", text=text)
    elif isinstance(message, ToolMessage):
        yield Event(
            kind="result",
            name=message.name or "tool",
            text=message.text,
            ok=message.status != "error",
        )


def clip(text: str, length: int) -> str:
    """Collapse runs of whitespace and clip to ``length`` characters with an ellipsis.

    The single source for the transcript-preview truncation both front ends need; each wraps
    it with its own width and (for the web UI) escaping.
    """
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"
