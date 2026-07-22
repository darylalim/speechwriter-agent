"""Browse what the agent produced: speech drafts, research notes, and learned voice profiles.

Read-only, and free — nothing here calls the model. The three views are rendered in
mutually exclusive branches rather than as tabs so only the selected one does any file or
Store reading; `st.tabs` would compute all three on every rerun.

Named `browse.py`, not `workspace.py`, so it is never confused with the
`speechwriter.workspace` module it imports or the `workspace/` output directory it reads.
"""

import streamlit as st

from speechwriter import workspace
from speechwriter.webui import get_bundle

bundle = get_bundle()
settings = bundle.settings


def document_browser(documents: list[workspace.Document], *, spoken: bool, empty: str) -> None:
    """Pick-and-read over a list of Markdown documents, newest first."""
    if not documents:
        st.caption(empty)
        return

    labels = {f"{doc.slug}  ·  {doc.modified:%b %d, %H:%M}": doc for doc in documents}
    picked = st.selectbox("Document", list(labels), label_visibility="collapsed")
    document = labels.get(picked) if isinstance(picked, str) else None
    if document is None:
        return

    with st.container(horizontal=True, vertical_alignment="center"):
        st.metric("Words", f"{document.words:,}")
        # Only meaningful for something meant to be said out loud; a research brief is not.
        if spoken:
            st.metric("Spoken length", f"~{document.minutes:.1f} min")
        st.download_button(
            "Download",
            document.text,
            file_name=document.path.name,
            icon=":material/download:",
        )

    # Rendered as metadata rather than passed through st.markdown: a `---` fence directly
    # after the header's last line would otherwise turn the whole block into one setext H2.
    if document.front_matter:
        st.caption(
            "  \n".join(
                f"**{label}** — {value}" if label else value
                for label, value in document.front_matter
            )
        )

    with st.container(border=True):
        st.markdown(document.body)


st.title("Workspace")

view = st.segmented_control(
    "View",
    ["Speeches", "Research", "Memory"],
    default="Speeches",
    label_visibility="collapsed",
)

if view == "Research":
    document_browser(
        workspace.research_notes(settings),
        spoken=False,
        empty="No research notes yet. They appear here when the researcher subagent runs.",
    )

elif view == "Memory":
    # Read from the live Store, not the JSON snapshot, so a profile learned this session
    # shows up before anything has called persist().
    entries = workspace.memories(bundle.store)
    if not entries:
        st.caption(
            "No voice profiles yet. The agent writes one after it delivers a speech, and "
            "reads it back on the next commission for the same speaker."
        )
    else:
        st.caption(f"Persisted to `{settings.store_path}`")
        for entry in entries:
            with st.expander(entry.key, icon=":material/record_voice_over:"):
                st.markdown(entry.text)

else:
    document_browser(
        workspace.speeches(settings),
        spoken=True,
        empty="No speeches yet. Commission one on the Write page.",
    )
