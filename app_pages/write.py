"""Commission a speech — the conversational front end to the orchestrator."""

import streamlit as st

from speechwriter.webui import get_bundle, render_turn, run_turn, transcript

# Concrete openers, not feature labels: the orchestrator's first job is intake, and a brief
# that already names speaker, audience, occasion, and length skips a round of questions.
SUGGESTIONS = {
    ":blue[:material/celebration:] Wedding toast": (
        "Write a 3-minute wedding toast from the bride's older brother. Warm and funny, "
        "built around one story, landing on something sincere."
    ),
    ":green[:material/school:] Commencement": (
        "Draft a 6-minute commencement address for a state university, delivered by an "
        "alum who started their career late. Hopeful without being saccharine."
    ),
    ":violet[:material/campaign:] Product keynote": (
        "Write a 5-minute keynote opening for a founder launching a developer tool. "
        "Confident and concrete, with no hype words."
    ),
}

bundle = get_bundle()
history = transcript()
# Not `anthropic_api_key` directly — a model served over SPEECHWRITER_BASE_URL needs no key
# of ours, and gating the chat input on one would disable a working configuration.
has_key = bundle.settings.model_credentials_present

# Session keys. The pill's own selection cannot carry the commission — see `_queue_suggestion`.
_PICKED = "suggestion"
_QUEUED = "queued_prompt"


def _queue_suggestion() -> None:
    """Hand the picked suggestion to the next rerun, exactly once.

    Runs as the pills' ``on_change``, so it fires on a real click and never on a rerun. That
    is the whole point. Relying on the widget disappearing after the first turn was not
    enough: a turn cancelled with the stop button raises a ``BaseException`` past `run_turn`,
    so nothing is appended to the transcript, the welcome block renders again with the pill
    still lit — and the same commission fired a second time, unasked. A queue that is
    *popped* by the run consuming it cannot do that, whatever the pill still looks like.
    """
    picked = st.session_state.get(_PICKED)
    if picked:
        st.session_state[_QUEUED] = SUGGESTIONS[picked]


if not history:
    # A centered welcome while the transcript is empty; it collapses back to the ordinary
    # left-aligned chat column the moment the first turn is recorded and this branch stops
    # rendering.
    with st.container(horizontal_alignment="center"):
        # The container's horizontal_alignment centers content-width children (the pills), but
        # the full-width title and caption keep their own text_alignment to center their text
        # — dropping it left-aligns them, so the two are not redundant.
        st.title("Write a speech", text_alignment="center")
        st.caption(
            "Name the speaker, the audience, the occasion, the goal, and the length — "
            "I'll ask if something essential is missing, then plan, draft, critique, and revise.",
            text_alignment="center",
        )
        # Rendered only on an empty transcript, so the widget stops existing after the first
        # turn and Streamlit drops its state — which is what leaves the pills unselected again
        # after "New conversation". The commission itself travels through `_QUEUED`, not
        # through this selection.
        st.pills(
            "Try one of these",
            list(SUGGESTIONS),
            key=_PICKED,
            on_change=_queue_suggestion,
            label_visibility="collapsed",
            # Matched to the chat input below. A pill that takes the click and then silently
            # drops it — which is all the `and has_key` guard below can do — is worse than one
            # that shows it is not available.
            disabled=not has_key,
        )

if not has_key:
    st.error(
        "No `ANTHROPIC_API_KEY` found. Add it to a local `.env` file "
        "(`ANTHROPIC_API_KEY=sk-ant-...`) and restart the app — or run a local model "
        "instead by setting `SPEECHWRITER_BASE_URL` to an OpenAI-compatible server.",
        icon=":material/key_off:",
    )

for turn in history:
    render_turn(turn)

# `submit_mode="stop"` turns the send button into a stop button while a turn is running, so
# a commission that goes long can be cancelled instead of being waited out.
typed = st.chat_input("Describe your speech…", submit_mode="stop", disabled=not has_key)
# Drained on its own line, before the choice. Folding the pop into `typed or ...` short-
# circuits it: whenever a typed brief wins, the pop never runs and the queued suggestion stays
# armed to fire as a second, unasked commission on a later rerun. Streamlit coalesces a pending
# rerun with a new one and ships every widget state on each message, and the chat box stays
# typeable while a turn runs — so a pill click and a typed brief really do arrive together.
# Popping unconditionally keeps the queue single-use; a typed message still supersedes a pill.
queued = st.session_state.pop(_QUEUED, None)
prompt: str | None = typed or queued

if prompt and has_key:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        completed = run_turn(bundle, prompt)
    history.append(completed)

    # Snapshot now rather than on shutdown. The CLI persists in a `finally`, but a browser
    # tab closing or a server being killed runs no teardown, so "save at the end" would
    # mean "usually never" — and a voice profile the agent just learned would be lost.
    bundle.persist()

    # Re-render from the recorded turn instead of keeping what was just streamed, so the
    # replay path is exercised on every turn and cannot quietly drift from the live one.
    st.rerun()
