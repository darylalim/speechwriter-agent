"""Streamlit glue for the speechwriter agent — the web counterpart to :mod:`speechwriter.cli`.

The terminal UI already lives inside this package; the browser UI belongs beside it for the
same reason. Both are thin by design: they own rendering and session bookkeeping, and defer
durability (``bundle.persist()``) and observability (``bundle.turn_config()``) to the bundle
that owns them.

Streamlit's execution model changes two things versus the CLI's ``while True`` loop:

* **The whole script reruns on every interaction.** So the agent must be built once and
  shared — hence ``@st.cache_resource``. Rebuilding per rerun would not just be slow; it
  would construct a fresh ``InMemoryStore`` each time and silently discard every voice
  profile learned so far this session, which is the one thing memory exists to prevent.
* **Nothing survives a rerun unless it is in ``st.session_state``.** So a turn is recorded
  as *data* (:class:`Turn`) and replayed, rather than existing only as the side effect of
  having streamed it. Live rendering and replay go through the same functions, so the two
  cannot drift.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from streamlit.delta_generator import DeltaGenerator

from speechwriter.agent import SpeechwriterAgent, build_agent

# Session-state keys. Named constants because two page scripts read them.
_TRANSCRIPT = "transcript"
_SEEN = "seen_message_ids"
_THREAD = "thread_id"

_PREVIEW_LEN = 110

# Material Symbols, coloured via Streamlit's Markdown directives rather than CSS.
_CALL_ICON = ":blue[:material/bolt:]"
_OK_ICON = ":green[:material/check_circle:]"
_ERROR_ICON = ":red[:material/error:]"


@dataclass
class Event:
    """One renderable thing the agent did: a tool call, its result, or prose."""

    kind: Literal["call", "result", "prose"]
    text: str
    name: str = ""
    ok: bool = True


@dataclass
class Turn:
    """One user commission and everything the agent emitted in response."""

    prompt: str
    events: list[Event] = field(default_factory=list)
    truncated: int = 0
    error: str = ""


@st.cache_resource(show_spinner="Starting the speechwriter…")
def get_bundle() -> SpeechwriterAgent:
    """Build the agent once and share it across reruns, pages, and browser sessions.

    ``build_agent()`` touches neither the model nor the network, so this is cheap and safe
    — the cost being cached is importing the deepagents/LangChain stack and rehydrating the
    memory snapshot.

    Sharing across *sessions* is deliberate: the Store, and therefore the learned voice
    profiles, is global to the app, exactly as it is for the CLI writing one snapshot file.
    The caveat is that ``bundle.warner`` is shared too, so two browser tabs running turns at
    the same instant would pool their truncation counts. That is acceptable for a local
    single-user app and is the honest trade for routing observability through
    ``turn_config()`` instead of hand-building a per-session config.
    """
    return build_agent()


def init_session() -> None:
    """Ensure this browser session has a transcript, a seen-set, and its own thread."""
    st.session_state.setdefault(_TRANSCRIPT, [])
    st.session_state.setdefault(_SEEN, set())
    # Guarded rather than `setdefault(...)` so the id is not re-minted on every rerun just
    # to be thrown away — and so it is obvious that the thread does *not* rotate per run.
    if _THREAD not in st.session_state:
        st.session_state[_THREAD] = _new_thread_id()


def transcript() -> list[Turn]:
    """The conversation so far, as replayable data."""
    return st.session_state[_TRANSCRIPT]


def thread_id() -> str:
    """The LangGraph thread this session resumes on each turn."""
    return st.session_state[_THREAD]


def reset_conversation() -> None:
    """Start a fresh thread, deliberately dropping prior context.

    Mirrors the CLI's post-interrupt rotation: a new ``thread_id`` means the checkpointer is
    never asked to resume a graph that may be half-executed. The transcript and seen-set go
    with it, since both are scoped to the thread that produced them.
    """
    st.session_state[_TRANSCRIPT] = []
    st.session_state[_SEEN] = set()
    st.session_state[_THREAD] = _new_thread_id()


def run_turn(bundle: SpeechwriterAgent, prompt: str) -> Turn:
    """Stream one commission, rendering as it arrives, and return the recorded turn.

    Call inside an ``st.chat_message("assistant")`` block: it opens a status container for
    the agent's tool activity and a plain container beneath it for prose.

    Exceptions are caught rather than propagated. A provider timeout mid-draft should mark
    the turn failed and leave the rest of the app usable, not replace the page with a
    traceback and lose the transcript.
    """
    turn = Turn(prompt=prompt)
    bundle.warner.reset()
    seen: set[str] = st.session_state[_SEEN]

    status = st.status("Working…", expanded=True)
    prose = st.container()

    payload = {"messages": [{"role": "user", "content": prompt}]}
    try:
        stream = bundle.agent.stream(
            payload,
            config=bundle.turn_config(thread_id()),
            stream_mode="values",
        )
        for chunk in stream:
            for message in chunk.get("messages", []):
                for event in _new_events(message, seen):
                    turn.events.append(event)
                    _render_event(event, status, prose)
    except Exception as exc:
        turn.error = f"{type(exc).__name__}: {exc}"
        status.update(label="Run failed", state="error", expanded=False)
    else:
        status.update(label=_activity_label(turn.events), state="complete", expanded=False)

    # Read after the run either way: a turn that raised still clipped whatever it clipped.
    turn.truncated = bundle.warner.truncated
    _render_footnotes(turn, bundle, prose)
    return turn


def render_turn(turn: Turn) -> None:
    """Replay a recorded turn. Shares every rendering path with the live stream."""
    with st.chat_message("user"):
        st.markdown(turn.prompt)

    with st.chat_message("assistant"):
        status = st.status(
            _activity_label(turn.events) if not turn.error else "Run failed",
            state="error" if turn.error else "complete",
            expanded=False,
        )
        prose = st.container()
        for event in turn.events:
            _render_event(event, status, prose)
        _render_footnotes(turn, get_bundle(), prose)


def _new_thread_id() -> str:
    return f"web-{uuid.uuid4().hex[:8]}"


def _new_events(message: Any, seen: set[str]) -> Iterator[Event]:
    """Yield events for a message not yet rendered, recording it as seen.

    ``stream_mode="values"`` replays the *entire* message list on every step, and the
    checkpointer carries earlier turns forward, so without this filter each rerun of the
    graph would re-emit the whole conversation.
    """
    identifier = getattr(message, "id", None) or str(id(message))
    if identifier in seen:
        return
    seen.add(identifier)

    # Only these two kinds produce output. Anything else in the list — the user's own
    # prompt, a system message — falls through and yields nothing, which is why there is
    # no explicit skip for them.
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            arguments = json.dumps(call.get("args", {}), ensure_ascii=False, default=str)
            yield Event(kind="call", name=call.get("name", "tool"), text=_preview(arguments))
        text = message.text.strip()
        if text:
            yield Event(kind="prose", text=text)
    elif isinstance(message, ToolMessage):
        yield Event(
            kind="result",
            name=message.name or "tool",
            text=_preview(message.text),
            ok=message.status != "error",
        )


def _render_event(event: Event, status: DeltaGenerator, prose: DeltaGenerator) -> None:
    """Write one event into the activity log, or into the prose beneath it."""
    if event.kind == "prose":
        prose.markdown(event.text)
        return

    if event.kind == "call":
        icon = _CALL_ICON
    else:
        icon = _OK_ICON if event.ok else _ERROR_ICON
    status.markdown(f"{icon} **{event.name}** `{event.text}`")


def _render_footnotes(turn: Turn, bundle: SpeechwriterAgent, prose: DeltaGenerator) -> None:
    """Surface the two failures a finished-looking answer would otherwise hide."""
    if turn.truncated:
        prose.warning(
            f"{turn.truncated} model response(s) hit the output-token ceiling and were cut "
            f"off, so the text above may be incomplete. Raise `SPEECHWRITER_MAX_TOKENS` "
            f"(currently {bundle.ceiling_label}).",
            icon=":material/content_cut:",
        )
    if turn.error:
        prose.error(turn.error, icon=":material/error:")


def _activity_label(events: list[Event]) -> str:
    steps = sum(1 for event in events if event.kind == "call")
    return f"Done — {steps} step{'' if steps == 1 else 's'}" if steps else "Done"


def _preview(text: str, length: int = _PREVIEW_LEN) -> str:
    """Collapse whitespace and clip, neutralising the one char that breaks inline code.

    Tool arguments and results are arbitrary text — JSON, file contents, a critique. Left as
    Markdown, a stray backtick or asterisk would reflow the activity log, so previews render
    as inline code and the character that could escape that wrapper is replaced.
    """
    collapsed = " ".join(text.split()).replace("`", "'")
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"
