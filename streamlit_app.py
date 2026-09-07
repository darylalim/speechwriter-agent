"""Entry point for the speechwriter web app.

    uv run streamlit run streamlit_app.py

This file is the router, not a page. Streamlit executes it on every rerun *before* the
selected page, so it holds only what every page shares: page config, the status sidebar,
and the navigation bar. Page content lives in ``app_pages/``.

Configuration is read the same way the CLI reads it — ``build_agent()`` calls
``load_settings()``, which loads the project's dotenv. Streamlit's own ``st.secrets`` is
deliberately unused: a second config source would be one more place for the model id or an
API key to disagree with itself.

The model picker in the sidebar is the one exception, and it is a narrow one. It does not read
configuration from anywhere new — it *overrides* one field for this session, in memory, and the
environment remains the durable setting the next start reads. There is still exactly one place
a credential or an endpoint can be written down.
"""

import streamlit as st

from speechwriter.webui import (
    MODEL_KEY,
    available_choices,
    base_settings,
    build_error,
    detect_models,
    detected,
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
        # `model_credentials_present`, not `anthropic_api_key` — the property the chat input
        # already gates on, and the one `config.py` names as the contract for both front ends.
        # A model served over SPEECHWRITER_BASE_URL needs no key of ours, so reading the
        # narrower field flagged a perfectly working local setup as broken.
        if settings.model_credentials_present:
            st.badge("Ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("No credentials", icon=":material/key_off:", color="red")

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
        if bundle.ceiling_exceeds_model:
            # Its own line, not a suffix on the caption above: the ceiling is a number, this is
            # "that number will be refused at the first turn". Only reachable since the model
            # became switchable, because SPEECHWRITER_MAX_TOKENS is global and outlives the
            # model it was sized for.
            st.caption(
                f":red[Above this model's {bundle.profiled_max_tokens:,} — "
                f"unset `SPEECHWRITER_MAX_TOKENS` or pick a larger model.]"
            )

        # Gated on the *configured* endpoint rather than the selected model's, so selecting a
        # Claude entry does not remove the controls that lead back to the local one. Shown only
        # when an endpoint is configured at all, so the default Anthropic sidebar is unchanged —
        # and worth its own line for the reason the CLI banner gives it one: a local server that
        # is simply not running looks like a hung turn unless the UI said where it pointed.
        endpoint = base_settings().base_url
        if endpoint:
            serving = " (serving the selected model)" if settings.uses_local_endpoint else ""
            st.caption(f"Endpoint — `{endpoint}`{serving}")
            st.button(
                "Detect models",
                icon=":material/sync:",
                on_click=detect_models,
                help="Ask this endpoint which models it serves, and add them to the list above.",
                width="stretch",
            )
            found = detected()
            if found:
                st.caption(f"Found {len(found)} model(s) at this endpoint.")
            elif found is not None:
                # Distinct from "never asked": an endpoint that answered with nothing is a real
                # result, and reads as a broken button if it renders the same as silence.
                st.caption("That endpoint listed no models.")

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
