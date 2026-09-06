"""Browse what the agent produced: speech drafts, research notes, and learned voice profiles.

Read-only, and free — nothing here calls the model. The three views are rendered in
mutually exclusive branches rather than as tabs so only the selected one does any file or
Store reading; `st.tabs` would compute all three on every rerun.

Named `browse.py`, not `workspace.py`, so it is never confused with the
`speechwriter.workspace` module it imports or the `workspace/` output directory it reads.
"""

import hashlib

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


def _measure_flag(document: workspace.Document) -> str:
    """Session key for "the reader asked to measure *this* draft".

    Keyed on the text's digest, not just the slug. The agent revises a speech **in place**, so
    a slug-only key stays True across a rewrite — and since `spoken_length` is cached on the
    text, the next visit would miss the cache and silently run a full ~9s synthesis on page
    render, which is exactly what gating it behind a button was meant to prevent.
    """
    digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()[:12]
    return f"measured:{document.slug}:{digest}"


def _measure_if_requested(
    document: workspace.Document,
) -> tuple[workspace.SpokenLength | None, str]:
    """Synthesise the draft if asked, returning ``(measurement, notice)``.

    Returns the notice rather than rendering it: this runs *outside* the metric row, because
    an `st.info` emitted inside a `horizontal=True` container becomes a third flex column and
    wraps its install command across a narrow strip — and that string's whole job is to be
    readable.

    Any failure clears the flag. Without that the page is unrecoverable: `st.cache_data` does
    not cache exceptions, so a sticky flag re-attempts and re-raises on every rerun, and a
    traceback out of here stops the rest of the page rendering. Clearing it puts the button
    back, which is also what you want after installing the extra the first branch complains
    about.
    """
    flag = _measure_flag(document)
    if not st.session_state.get(flag):
        return None, ""
    try:
        with st.spinner("Synthesising the draft…"):
            return spoken_length(document.text), ""
    except workspace.AudioUnavailable as exc:
        st.session_state.pop(flag, None)
        return None, str(exc)
    except Exception as exc:
        # Broad on purpose. Everything past the import guard is someone else's failure mode —
        # a first-run model download with no network, an HF rate limit, a full disk, mlx
        # raising on a pathological draft — and none of them should take the page down.
        st.session_state.pop(flag, None)
        return None, f"Could not measure this draft: {exc}"


def _measure_button(document: workspace.Document) -> None:
    """The control that opts into a measurement, rendered inside the metric row."""
    flag = _measure_flag(document)
    # Rerun on click so the button is *replaced* by the result rather than sitting beside it.
    if st.button(
        "Measure",
        key=f"button-{flag}",
        icon=":material/graphic_eq:",
        help="Synthesise the draft and time it, instead of estimating from word count.",
    ):
        st.session_state[flag] = True
        st.rerun()


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

    # Measured before the row opens: the spinner and any failure notice need full width.
    measured, notice = _measure_if_requested(document) if spoken else (None, "")
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
            if measured is None:
                _measure_button(document)
            else:
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

    if notice:
        st.info(notice, icon=":material/volume_off:")

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
