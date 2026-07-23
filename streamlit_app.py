"""Entry point for the speechwriter web app.

    uv run streamlit run streamlit_app.py

This file is the router, not a page. Streamlit executes it on every rerun *before* the
selected page, so it holds only what every page shares: page config, the status sidebar,
and the navigation bar. Page content lives in ``app_pages/``.

Configuration is read the same way the CLI reads it — ``build_agent()`` calls
``load_settings()``, which loads the project's ``.env``. Streamlit's own ``st.secrets`` is
deliberately unused: a second config source would be one more place for the model id or an
API key to disagree with itself.
"""

import streamlit as st

from speechwriter.webui import get_bundle, init_session, reset_conversation

st.set_page_config(
    page_title="Speechwriter",
    page_icon=":material/edit_note:",
    layout="centered",
)

bundle = get_bundle()
init_session()
settings = bundle.settings

with st.sidebar:
    with st.container(horizontal=True):
        if settings.anthropic_api_key:
            st.badge("Ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("No API key", icon=":material/key_off:", color="red")

        if settings.research_enabled:
            st.badge("Research", icon=":material/travel_explore:", color="blue")
        else:
            st.badge("No research", icon=":material/travel_explore:", color="gray")

    # The two run-config lines grouped as one bordered card, so they read as a single unit
    # rather than two loose captions floating above the primary action beneath them.
    with st.container(border=True):
        st.caption(f"Model — `{settings.model}`")
        st.caption(f"Output ceiling — {bundle.ceiling_label}")

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
