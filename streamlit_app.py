"""Entry point for the speechwriter web app.

    uv run streamlit run streamlit_app.py

This file is the router, not a page. Streamlit executes it on every rerun *before* the
selected page, so it holds only what every page shares: page config, the status sidebar,
and the navigation bar. Page content lives in ``app_pages/``.

Configuration is read the same way the CLI reads it — ``build_agent()`` calls
``load_settings()``, which loads the project's dotenv. Streamlit's own ``st.secrets`` is
deliberately unused: a second config source would be one more place for the model id or an
API key to disagree with itself.

The model picker is the one exception, and it is a narrow one. It reads configuration from
nowhere new: it *overrides* the model for this session, in memory, and the environment remains
the durable setting the next start reads. There is still exactly one place a credential can be
written down.
"""

import streamlit as st

from speechwriter.webui import (
    MODEL_KEY,
    available_choices,
    build_error,
    get_bundle,
    init_session,
    reset_conversation,
    switch_model,
)

st.set_page_config(
    page_title="Speechwriter",
    page_icon=":material/edit_note:",
    layout="centered",
)

bundle = get_bundle()
init_session()
settings = bundle.settings
# Popped before the sidebar renders, so a pick that could not be built is reported once and the
# picker is drawn normally beneath it. `get_bundle` recovers by falling back to the environment
# rather than raising, because a raise here is above the sidebar and takes the page with it.
failed_pick = build_error()

with st.sidebar:
    with st.container(horizontal=True):
        # `model_credentials_present`, the property the chat input also gates on and the one
        # `config.py` names as the contract for both front ends.
        if settings.model_credentials_present:
            st.badge("Ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("No API key", icon=":material/key_off:", color="red")

        if settings.research_enabled:
            st.badge("Research", icon=":material/travel_explore:", color="blue")
        else:
            st.badge("No research", icon=":material/travel_explore:", color="gray")

    # The run-config lines grouped as one bordered card, so they read as a single unit rather
    # than loose captions floating above the primary action beneath them. The model line *is*
    # the picker: it was already the place the model was named, and a separate control would
    # have left a caption that could disagree with it.
    with st.container(border=True):
        choices = available_choices(settings)
        # The index is resolved from the *built* bundle's settings rather than from the widget's
        # memory, so the control always opens on the model actually in force — including on the
        # very first render, before anything has been picked. Streamlit prefers an existing
        # `session_state` value over `index`, which is what makes a pick stick across reruns.
        current = next((c for c in choices if c.is_current(settings)), choices[0])
        st.selectbox(
            "Model",
            choices,
            index=choices.index(current),
            format_func=lambda choice: choice.label,
            key=MODEL_KEY,
            on_change=switch_model,
            help=(
                "Rebuilds the agent on the chosen model. Learned voice profiles are saved "
                "first; the current conversation restarts, because the rebuilt agent cannot "
                "resume it."
            ),
        )
        if failed_pick:
            # Reported next to the picker that caused it, not at the top of the page: the reader
            # needs to see which entry failed and choose again in one glance.
            st.caption(f":red[Could not switch — {failed_pick}]")
        st.caption(f"Output ceiling — {bundle.ceiling_label}")
        # Only when on: the CLI banner lists every setting, but this sidebar is for what the
        # reader can act on, and "not tracing" is the default rather than something to fix.
        if bundle.tracing is not None:
            st.caption(f"Traces — `{bundle.tracing.label}`")
        if bundle.ceiling_exceeds_model:
            # Its own line, not a suffix on the caption above: the ceiling is a number, this is
            # "that number will be refused". SPEECHWRITER_MAX_TOKENS is global and outlives the
            # model it was sized for, so an override sized for a roomier model is rejected here.
            st.caption(
                f":red[Above this model's {bundle.profiled_max_tokens:,}-token maximum — lower "
                f"or unset `SPEECHWRITER_MAX_TOKENS`.]"
            )

    # The two on-disk locations are diagnostic, not glanceable, and long absolute paths
    # wrap awkwardly in a narrow sidebar — so they live one fold down rather than crowding
    # the model line and the primary action beneath it.
    with st.expander("Storage", icon=":material/database:", type="compact"):
        st.caption(f"Workspace — `{settings.workspace_dir}`")
        st.caption(f"Memory — `{settings.store_path}`")

    st.button(
        "New conversation",
        icon=":material/restart_alt:",
        on_click=reset_conversation,
        help="Start a fresh thread. Drops the current conversation's context.",
        width="stretch",
    )

page = st.navigation(
    [
        st.Page("app_pages/write.py", title="Write", icon=":material/edit_note:"),
        st.Page("app_pages/browse.py", title="Workspace", icon=":material/folder_open:"),
    ],
    position="top",
)
page.run()
