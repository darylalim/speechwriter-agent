"""Streamlit glue for the speechwriter agent — the web counterpart to :mod:`speechwriter.cli`.

The terminal UI already lives inside this package; the browser UI belongs beside it for the
same reason. Both are thin by design: they own rendering and session bookkeeping, and defer
durability (``bundle.persist()``) and observability (``bundle.turn_config()``) to the bundle
that owns them. The message→event decode and the preview-clip they both need live in
:mod:`speechwriter.transcript`, so the two front ends cannot drift on *what* a message means.

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

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from speechwriter import workspace
from speechwriter.agent import SpeechwriterAgent, build_agent
from speechwriter.transcript import Event, clip, iter_events

# Session-state keys. Named constants because two page scripts read them.
_TRANSCRIPT = "transcript"
_SEEN = "seen_message_ids"
_THREAD = "thread_id"
# Set while a turn is streaming, cleared once it finishes. If it is still set at the start of
# the *next* turn, the previous one was cancelled mid-flight (see `_rotate_if_interrupted`).
_PENDING = "turn_in_flight"

_PREVIEW_LEN = 110

# The Write page's suggestion pills: the widget's own key, and the plain key a click hands to
# the next rerun. Both live here rather than in the page because `reset_conversation` has to be
# able to disarm the queue — a brief left armed by a run the reader walked away from would
# otherwise commission a speech, and spend tokens, on the next visit to the page.
SUGGESTION_KEY = "suggestion"
_QUEUED = "queued_prompt"

# How many measured drafts `spoken_length` keeps. Exported because `browse.py` bounds its
# "already measured" flags to the same number: a flag that outlives its cache entry sends the
# next page render straight back into a ~9s synthesis, which is what the button exists to stop.
MEASURE_CACHE_ENTRIES = 8

# Material Symbols, coloured via Streamlit's Markdown directives rather than CSS.
_CALL_ICON = ":blue[:material/bolt:]"
_OK_ICON = ":green[:material/check_circle:]"
_ERROR_ICON = ":red[:material/error:]"


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


# `max_entries` bounds the cache: the key includes each file's mtime, and the agent revises a
# draft in place, so every revision mints a *new* entry holding the full parsed text of the
# folder — with no bound, those dead snapshots accumulate for the life of the process. The
# live working set is tiny (one entry per directory, currently two), so this is a generous
# safety ceiling, not a tuning knob; eviction only ever discards a signature the file has
# already moved past, never the current one we re-read on a miss.
@st.cache_data(show_spinner=False, max_entries=32)
def _parse_documents(
    directory: str, signature: tuple[tuple[str, float], ...]
) -> list[workspace.Document]:
    """Read and front-matter-parse every draft in ``directory`` (cached).

    ``signature`` — each file's name and mtime — is a cache-key-only argument: it makes the
    expensive read+parse re-run only when the directory's contents actually change, never on
    a plain rerun. It is intentionally unused in the body; the cheap stat that produced it
    already ran in :func:`documents`.
    """
    return workspace.load_documents(Path(directory))


def documents(directory: Path) -> list[workspace.Document]:
    """Cached listing of a workspace subdirectory, newest first.

    The glob+stat that builds the cache signature is cheap and runs every rerun; the full
    file reads and front-matter parsing behind it run only when a draft is added, removed, or
    rewritten. Without this, every rerun on the Workspace page — picking a different draft,
    switching views — re-read and re-parsed every file in the folder just to show one.
    """
    stamps: list[tuple[str, float]] = []
    if directory.is_dir():
        for path in directory.glob("*.md"):
            try:
                stamps.append((path.name, path.stat().st_mtime))
            except OSError:
                # The agent writes into this folder while the page renders, so a name can
                # survive the glob and fail the stat. Any key built from here would then
                # describe a folder we could not fully see — and `load_documents` re-globs, so
                # the parse behind that key can disagree with it. Worse, a later render hitting
                # the same race would rebuild the same key and be served the stale parse. So
                # read straight through instead: an unreadable file costs the cache, never
                # correctness, and the race clears itself on the next render. Raising here
                # would take the whole Workspace page down over it.
                return workspace.load_documents(directory)
    return _parse_documents(str(directory), tuple(sorted(stamps)))


# `cache_resource`, not `cache_data`: each entry carries the synthesised WAV — roughly 2.9 MB
# per spoken minute at 24 kHz mono — and `cache_data` returns a *copy* on every hit, so each
# rerun of the page would unpickle ~8.6 MB of audio just to render one decimal. `cache_resource`
# hands back the object itself, which is safe here because `SpokenLength` is frozen and its
# payload is immutable bytes. Bounded so the store cannot grow with every draft measured.
@st.cache_resource(show_spinner=False, max_entries=MEASURE_CACHE_ENTRIES)
def spoken_length(text: str) -> workspace.SpokenLength:
    """Measured delivery time for a draft, synthesised once per distinct text.

    Keyed on the draft's own text rather than its path or slug, so a revised speech — which
    the agent rewrites *in place* — measures again while a mere rerun does not. That matters
    more here than for :func:`documents`: this costs a TTS pass (~9s for a three-minute
    speech), where that costs a file read.

    The caching lives here rather than in :mod:`speechwriter.workspace` for the usual reason:
    that module stays free of any Streamlit import.
    """
    return workspace.measure_spoken_length(text)


def init_session() -> None:
    """Ensure this browser session has a transcript, a seen-set, and its own thread."""
    st.session_state.setdefault(_TRANSCRIPT, [])
    st.session_state.setdefault(_SEEN, set())
    st.session_state.setdefault(_PENDING, False)
    # Guarded rather than `setdefault(...)` so the id is not re-minted on every rerun just
    # to be thrown away — and so it is obvious that the thread does *not* rotate per run.
    if _THREAD not in st.session_state:
        st.session_state[_THREAD] = _new_thread_id()


def transcript() -> list[Turn]:
    """The conversation so far, as replayable data."""
    return st.session_state[_TRANSCRIPT]


def queue_suggestion(brief: str) -> None:
    """Hand a picked suggestion to the next rerun."""
    st.session_state[_QUEUED] = brief


def take_queued_suggestion() -> str | None:
    """Pop the queued brief, if any.

    Draining is the whole point, so this is never a plain read: a brief that stayed in the
    queue would fire again on a later rerun as a commission nobody asked for.
    """
    queued = st.session_state.pop(_QUEUED, None)
    return queued if isinstance(queued, str) else None


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
    st.session_state[_PENDING] = False
    st.session_state[_THREAD] = _new_thread_id()
    # A queued suggestion is part of the conversation being dropped. Left armed, "New
    # conversation" would be followed immediately by the commission it was meant to abandon.
    st.session_state.pop(_QUEUED, None)


def run_turn(bundle: SpeechwriterAgent, prompt: str) -> Turn:
    """Stream one commission, rendering as it arrives, and return the recorded turn.

    Call inside an ``st.chat_message("assistant")`` block: it opens a status container for
    the agent's tool activity and a plain container beneath it for prose.

    Exceptions are caught rather than propagated. A provider timeout mid-draft should mark
    the turn failed and leave the rest of the app usable, not replace the page with a
    traceback and lose the transcript.
    """
    _rotate_if_interrupted()
    turn = Turn(prompt=prompt)
    bundle.warner.reset()
    # Mark the turn in-flight *before* streaming. A user clicking the stop button raises a
    # BaseException that sails past the `except Exception` below, so this flag is the only
    # trace a cancelled turn leaves — `_rotate_if_interrupted` reads it on the next turn.
    st.session_state[_PENDING] = True
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

    # Through `_status_shape` rather than inline, for the same reason `_render_event` is
    # shared: `render_turn` draws this header again on replay, and the two computing it
    # separately is exactly how a live turn and its replay drift.
    label, state = _status_shape(turn)
    status.update(label=label, state=state, expanded=False)

    # The turn finished (cleanly or with a caught error); it is no longer in-flight. Only a
    # stop leaves _PENDING set, and only that triggers a thread rotation next turn — matching
    # the CLI, which rotates on interrupt but not on ordinary errors.
    st.session_state[_PENDING] = False
    # Read after the run either way: a turn that raised still clipped whatever it clipped.
    turn.truncated = bundle.warner.truncated
    _render_footnotes(turn, bundle, prose)
    return turn


def render_turn(turn: Turn) -> None:
    """Replay a recorded turn. Shares every rendering path with the live stream."""
    with st.chat_message("user"):
        st.markdown(turn.prompt)

    with st.chat_message("assistant"):
        label, state = _status_shape(turn)
        status = st.status(label, state=state, expanded=False)
        prose = st.container()
        for event in turn.events:
            _render_event(event, status, prose)
        _render_footnotes(turn, get_bundle(), prose)


def _new_thread_id() -> str:
    return f"web-{uuid.uuid4().hex[:8]}"


def _rotate_if_interrupted() -> None:
    """Recover the thread if the previous turn was cancelled with the stop button.

    ``submit_mode="stop"`` raises a ``BaseException`` that propagates past ``run_turn``'s
    ``except Exception``, so a cancelled turn never records, never persists, and never
    rotates — leaving the LangGraph thread mid-execution. Run before the next turn: if the
    previous one is still flagged in-flight, rotate to a fresh thread so we never resume a
    half-executed graph, the web analog of the CLI's post-interrupt rotation. The visible
    transcript is display data and stays; only the dropped graph context differs, and the
    seen-set is scoped to the abandoned thread so it goes too.
    """
    if st.session_state.get(_PENDING):
        st.session_state[_THREAD] = _new_thread_id()
        st.session_state[_SEEN] = set()
        st.session_state[_PENDING] = False


def _new_events(message: object, seen: set[str]) -> list[Event]:
    """Decode a not-yet-seen message into events, recording it as seen.

    ``stream_mode="values"`` replays the *entire* message list on every step, and the
    checkpointer carries earlier turns forward, so without this filter each step would
    re-emit the whole conversation.
    """
    identifier = getattr(message, "id", None)
    if identifier is None:
        # Graph state always assigns message ids, so this is defensive. Key off type +
        # content rather than str(id(message)): a raw object id can be recycled by CPython
        # after GC and cause a *different* later message to be skipped, whereas a content
        # key at worst coalesces two byte-identical messages — harmless in the activity log.
        identifier = f"{type(message).__name__}:{getattr(message, 'content', '')!r}"
    if identifier in seen:
        return []
    seen.add(identifier)
    return list(iter_events(message))


def _render_event(event: Event, status: DeltaGenerator, prose: DeltaGenerator) -> None:
    """Write one event into the activity log, or into the prose beneath it."""
    if event.kind == "prose":
        prose.markdown(event.text)
        return

    if event.kind == "call":
        icon = _CALL_ICON
    else:
        icon = _OK_ICON if event.ok else _ERROR_ICON
    # A tool result can legitimately be empty — deepagents forwards an empty `status="success"`
    # when a subagent spends its whole ceiling thinking and emits no text, which is precisely
    # the failure `TruncationWarner` exists to surface. Wrapped unconditionally, that renders
    # as two bare backticks: an empty code span that reads as a glitch rather than as the
    # signal it is.
    body = _preview(event.text)
    status.markdown(f"{icon} **{event.name}** " + (f"`{body}`" if body else "_no output_"))


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


def _status_shape(turn: Turn) -> tuple[str, Literal["complete", "error"]]:
    """Label and state for a finished turn's activity log — one source for both paths."""
    if turn.error:
        return "Run failed", "error"
    return _activity_label(turn.events), "complete"


def _activity_label(events: list[Event]) -> str:
    steps = sum(1 for event in events if event.kind == "call")
    return f"Done — {steps} step{'' if steps == 1 else 's'}" if steps else "Done"


def _preview(text: str, length: int = _PREVIEW_LEN) -> str:
    """Clip a raw event string for the activity log, neutralising inline-code breakout.

    Tool arguments and results are arbitrary text — JSON, file contents, a critique. Rendered
    inside a `` ` `` span, a stray backtick would end the span and let the rest reflow as
    Markdown, so the one character that can escape the wrapper is replaced after clipping.
    """
    return clip(text, length).replace("`", "'")
