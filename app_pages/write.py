"""Commission a speech — the conversational front end to the orchestrator."""

import streamlit as st

from speechwriter.webui import (
    SUGGESTION_KEY,
    get_bundle,
    queue_suggestion,
    render_turn,
    run_turn,
    take_queued_suggestion,
    transcript,
)

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
# Not "is a credential present" — a locally served model needs none of ours, so that question
# became true for every configuration when the hosted client left. What can still be wrong
# before a socket is opened is the endpoint's *shape*, which is what this gates on.
endpoint_usable = bundle.settings.model_endpoint_usable
# Drained here, before anything renders or can raise. The queue has to be emptied by the run
# that sees it, and the further down the script that happens the more ways there are to leave
# it armed — a render that raises, a page switch, a stop landing during a cold start. Losing a
# brief to an abandoned run is the safe direction; firing one is not.
queued = take_queued_suggestion()


def _queue_suggestion() -> None:
    """Hand the picked suggestion to the next rerun, exactly once.

    Runs as the pills' ``on_change``, so it fires on a real click and never on a rerun. That
    is the whole point. Relying on the widget disappearing after the first turn was not
    enough: a turn cancelled with the stop button raises a ``BaseException`` past `run_turn`,
    so nothing is appended to the transcript, the welcome block renders again with the pill
    still lit — and the same commission fired a second time, unasked. A queue that is
    *popped* by the run consuming it cannot do that, whatever the pill still looks like.
    """
    picked = st.session_state.get(SUGGESTION_KEY)
    # `.get`, not a subscript: this runs inside a callback, where a KeyError would replace the
    # page with a traceback instead of being ignored. A label that is no longer one of ours —
    # a stale widget value surviving a hot reload after the suggestions were renamed — should
    # queue nothing, which is exactly what the pre-queue code did.
    brief = SUGGESTIONS.get(picked) if isinstance(picked, str) else None
    if brief:
        queue_suggestion(brief)


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
            key=SUGGESTION_KEY,
            on_change=_queue_suggestion,
            label_visibility="collapsed",
            # Matched to the chat input below. A pill that takes the click and then silently
            # drops it — which is all the `and endpoint_usable` guard below can do — is worse
            # than one that shows it is not available.
            disabled=not endpoint_usable,
        )

if not endpoint_usable:
    # Narrow, and honest about being narrow. This is not "the server is down" — nothing here
    # probes, and the sidebar already names the endpoint the agent is pointed at. It is the one
    # failure visible without a socket: a `SPEECHWRITER_BASE_URL` that `urllib` cannot even
    # build a request from, which `config._configured_endpoint` deliberately passes through
    # unrewritten rather than silently replacing with the default.
    #
    # Reachable two ways: a malformed value in the environment, or one typed into the endpoint
    # field this session. The fix is the same control either way, so unlike its predecessor
    # this needs no branch on how the reader got here.
    st.error(
        f"`{bundle.settings.base_url}` is not an endpoint the agent can call, so no turn will "
        "reach it. Open **Local endpoint** in the sidebar and point it at an OpenAI-compatible "
        "server (it needs an `http`/`https` scheme and a host), or set a valid "
        "`SPEECHWRITER_BASE_URL` in a local dotenv file and restart the app.",
        icon=":material/link_off:",
    )

for turn in history:
    render_turn(turn)

# `submit_mode="stop"` turns the send button into a stop button while a turn is running, so
# a commission that goes long can be cancelled instead of being waited out.
typed = st.chat_input("Describe your speech…", submit_mode="stop", disabled=not endpoint_usable)
# The queue was already drained at the top of the script, so this only chooses. Reading it
# here as `typed or st.session_state.pop(...)` would short-circuit the pop whenever a typed
# brief wins and leave the suggestion armed — Streamlit coalesces a pending rerun with a new
# one and ships every widget state on each message, and the chat box stays typeable while a
# turn runs, so a pill click and a typed brief really do arrive together.
prompt: str | None = typed or queued

if prompt and endpoint_usable:
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
