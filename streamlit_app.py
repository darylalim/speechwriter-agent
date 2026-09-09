"""Entry point for the speechwriter web app.

    uv run streamlit run streamlit_app.py

This file is the router, not a page. Streamlit executes it on every rerun *before* the
selected page, so it holds only what every page shares: page config, the status sidebar,
and the navigation bar. Page content lives in ``app_pages/``.

Configuration is read the same way the CLI reads it — ``build_agent()`` calls
``load_settings()``, which loads the project's dotenv. Streamlit's own ``st.secrets`` is
deliberately unused: a second config source would be one more place for the model id or an
API key to disagree with itself.

The two sidebar controls — the model picker and the endpoint field — are the exception, and it
is a narrow one. Neither reads configuration from anywhere new: they *override* a field for this
session, in memory, and the environment remains the durable setting the next start reads. There
is still exactly one place a credential can be written down.

The endpoint field is deliberately not a second config source. It decides which server the
Detect button *asks*, never which model is running — that changes only when the picker says so
— and it is seeded from ``SPEECHWRITER_BASE_URL`` so the box always shows what will be asked.
It is drawn unconditionally because the reader it exists for has configured nothing at all.
"""

import streamlit as st

from speechwriter.config import DEFAULT_LOCAL_ENDPOINT
from speechwriter.webui import (
    ENDPOINT_KEY,
    MODEL_KEY,
    apply_endpoint,
    available_choices,
    base_settings,
    build_error,
    detect_models,
    detections,
    get_bundle,
    init_session,
    reset_conversation,
    session_endpoint,
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
        if settings.uses_local_endpoint:
            # Beside the ceiling, not inside the fold below: this names where the model that is
            # *running* is served, which the endpoint field does not — that field holds whatever
            # server the reader is currently pointing Detect at, and the two differ the moment
            # they go looking at a second one. A local server that is simply not running looks
            # like a hung turn unless the UI said where it pointed, which is the same reason the
            # CLI banner gives `endpoint` a line of its own.
            st.caption(f"Endpoint — `{settings.base_url}`")
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

        # Always drawn, and that is the point of it: an endpoint that could only be reached
        # when `SPEECHWRITER_BASE_URL` was already set left the one reader this is for — the
        # one running a local server and no Anthropic key — editing a dotenv and restarting.
        # Folded away rather than inline because it is a setup step, not a per-turn control,
        # and the sidebar's primary job is naming the model that is running.
        with st.expander("Local endpoint", icon=":material/dns:", type="compact"):
            st.text_input(
                "OpenAI-compatible server",
                key=ENDPOINT_KEY,
                on_change=apply_endpoint,
                placeholder=DEFAULT_LOCAL_ENDPOINT,
                help=(
                    "A local `mlx_lm.server`, vLLM, LM Studio or Ollama. Detect adds the "
                    "models it serves to the list above, for this session only — the "
                    "environment stays the durable setting."
                ),
            )
            configured = base_settings()
            target = session_endpoint()
            typed = (st.session_state.get(ENDPOINT_KEY) or "").strip()
            found = detections()
            st.button(
                "Detect models",
                icon=":material/sync:",
                on_click=detect_models,
                help="Ask this endpoint which models it serves, and add them to the list above.",
                width="stretch",
                # Same rule the suggestion pills follow in `write.py`: a control that takes the
                # click and silently drops it is worse than one that shows it is unavailable.
                # Nothing askable in the box means there is no server to ask.
                disabled=target is None,
            )
            if typed and target is None:
                # Said before a probe is attempted, because "that is not an endpoint" and "that
                # endpoint answered nothing" are different facts and only one of them is worth
                # retyping the URL over.
                #
                # It names a working example rather than only the verdict, because the reader
                # facing this did not necessarily type it: the browser restores form fields, and
                # a malformed `SPEECHWRITER_BASE_URL` is seeded verbatim on purpose. Arriving at
                # a red caption about text you have no memory of writing is only actionable if
                # the caption says what the shape should be.
                st.caption(f":red[Not an HTTP endpoint — try `{DEFAULT_LOCAL_ENDPOINT}`.]")
            elif found:
                st.caption(f"Found {len(found)} model(s) — pick one above.")
            elif found is not None:
                # Distinct from "never asked": an endpoint that answered with nothing is a real
                # result, and reads as a broken button if it renders the same as silence.
                st.caption("That endpoint listed no models — is the server running?")
            withheld = (
                target and configured.openai_api_key and not configured.endpoint_api_key_for(target)
            )
            if withheld:
                # Otherwise a hosted endpoint 401s and renders as "listed no models", which
                # sends the reader to check a server that is answering perfectly well.
                st.caption(
                    ":orange[No credential sent — `OPENAI_API_KEY` reaches only the endpoint "
                    "set in the environment.]"
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
