"""Browse what the agent produced: speech drafts, research notes, and learned voice profiles.

Read-only, and free — nothing here calls the model. The three views are rendered in
mutually exclusive branches rather than as tabs so only the selected one does any file or
Store reading; `st.tabs` would compute all three on every rerun.

Named `browse.py`, not `workspace.py`, so it is never confused with the
`speechwriter.workspace` module it imports or the `workspace/` output directory it reads.
"""

import streamlit as st

from speechwriter import workspace
from speechwriter.config import WORDS_PER_MINUTE
from speechwriter.webui import documents, get_bundle, spoken_length

bundle = get_bundle()
settings = bundle.settings


def _humanize(key: str) -> str:
    """Render a memory slug like `david-best-man.md` as a readable `David Best Man`.

    Display-only: the entry's ``key`` stays the source of truth — the Store is keyed on it
    and ``persist()`` writes it verbatim — this just softens the filename for the header.
    Falls back to the raw key if stripping leaves nothing to show.
    """
    label = key.removesuffix(".md").replace("-", " ").replace("_", " ").strip()
    return label.title() or key


def _measured_length(document: workspace.Document) -> workspace.SpokenLength | None:
    """The draft's real delivery time, once the reader asks for it.

    Gated behind a button rather than computed with the page. Synthesis runs at roughly RTF
    0.06 — about nine seconds for a three-minute speech — which is fine to wait for
    deliberately and far too slow to pay on every rerun of a page whose whole job is
    browsing. `webui.spoken_length` caches the result on the draft's text, so re-picking the
    same document is instant while a revised one is measured again.
    """
    flag = f"measured:{document.slug}"
    if not st.session_state.get(flag):
        # Rerun on click so the button is *replaced* by the result rather than sitting beside
        # it; without it the row would show a stale control next to the number it produced.
        if st.button(
            "Measure",
            key=f"measure-{document.slug}",
            icon=":material/graphic_eq:",
            help="Synthesise the draft and time it, instead of estimating from word count.",
        ):
            st.session_state[flag] = True
            st.rerun()
        return None

    try:
        with st.spinner("Synthesising…"):
            return spoken_length(document.text)
    except workspace.AudioUnavailable as exc:
        # A normal state to explain, not an error to raise at someone: the audio extra is
        # genuinely optional and the rest of the page works perfectly without it.
        st.info(str(exc), icon=":material/volume_off:")
        return None


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

    measured = None
    with st.container(horizontal=True, vertical_alignment="bottom"):
        # The tooltip names what the count leaves out, matching workspace.py: the `---` header
        # block is stripped and bracketed delivery cues (`[pause]`) are dropped as unspoken.
        st.metric(
            "Words",
            f"{document.words:,}",
            border=True,
            width="content",
            help="Body word count. Excludes the header block and bracketed delivery cues.",
        )
        # Only meaningful for something meant to be said out loud; a research brief is not.
        if spoken:
            # Sourced from config, not hardcoded, so the tooltip can never quote a pace that
            # disagrees with the estimate it explains — workspace.py computes `minutes` from
            # the same constant.
            st.metric(
                "Spoken length",
                f"~{document.minutes:.1f} min",
                border=True,
                width="content",
                help=f"Estimated at about {WORDS_PER_MINUTE} words per minute.",
            )
            measured = _measured_length(document)
            if measured is not None:
                # Showing both is the point: one constant cannot know that this draft is
                # dense with long words and that one is short and punchy. `delta_color="off"`
                # because drift in either direction is information, not good or bad news.
                drift = measured.seconds - document.minutes * 60
                st.metric(
                    "Measured",
                    f"{measured.minutes:.1f} min",
                    delta=f"{drift:+.0f}s vs estimate",
                    delta_color="off",
                    border=True,
                    width="content",
                    help="Synthesised with Kokoro. Times the words only — a bracketed cue "
                    "adds no silence, so this is time-to-say, not time-on-stage.",
                )
        # Pushed to the far edge so it reads as an action, not a third stat card.
        with st.container(horizontal_alignment="right"):
            st.download_button(
                "Download",
                document.text,
                file_name=document.path.name,
                icon=":material/download:",
            )

    # The synthesis already happened, so playing it back costs nothing extra — and hearing a
    # draft is the fastest way to catch what the critic's "speakability" pass can only infer.
    if measured is not None and measured.wav:
        st.audio(measured.wav, format="audio/wav")

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
# One orienting line under the title, mirroring the Write page's welcome caption so the two
# top-level pages read as a pair.
st.caption("Speeches, research notes, and the speaker voices the agent has learned.")

view = st.segmented_control(
    "View",
    ["Speeches", "Research", "Memory"],
    default="Speeches",
    label_visibility="collapsed",
)

if view == "Research":
    document_browser(
        documents(workspace.research_dir(settings)),
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
            # Humanized for the header; the raw slug still keys the Store behind it.
            with st.expander(_humanize(entry.key), icon=":material/record_voice_over:"):
                st.markdown(entry.text)

else:
    document_browser(
        documents(workspace.speeches_dir(settings)),
        spoken=True,
        empty="No speeches yet. Commission one on the Write page.",
    )
