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
has_key = bool(bundle.settings.anthropic_api_key)

suggested = None
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
        # turn. Streamlit drops an unrendered widget's state, which is what stops a selected
        # pill from re-firing the same commission on every later rerun.
        suggested = st.pills("Try one of these", list(SUGGESTIONS), label_visibility="collapsed")

if not has_key:
    st.error(
        "No `ANTHROPIC_API_KEY` found. Add it to a local `.env` file "
        "(`ANTHROPIC_API_KEY=sk-ant-...`) and restart the app.",
        icon=":material/key_off:",
    )

for turn in history:
    render_turn(turn)

# `submit_mode="stop"` turns the send button into a stop button while a turn is running, so
# a commission that goes long can be cancelled instead of being waited out.
typed = st.chat_input("Describe your speech…", submit_mode="stop", disabled=not has_key)
prompt = typed or (SUGGESTIONS.get(suggested) if isinstance(suggested, str) else None)

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
