"""Browse what the agent produced: speech drafts, research notes, and learned voice profiles.

Read-only, and free — nothing here calls the model. The three views are rendered in
mutually exclusive branches rather than as tabs so only the selected one does any file or
Store reading; `st.tabs` would compute all three on every rerun.

Named `browse.py`, not `workspace.py`, so it is never confused with the
`speechwriter.workspace` module it imports or the `workspace/` output directory it reads.

The layout keeps one thing constant, the width of the text being read, and lets everything
around it move. Streamlit cannot tell Python the viewport size (`st.context` has none), so
nothing here branches on the screen. A rail and a `READING_WIDTH` column share one wrapping
row inside a centred `PAGE_WIDTH` frame. They sit side by side when the row has room and
stack, rail first, when it has not. That room is what the browser actually lays out, so it
counts the sidebar too: closing the sidebar can move the page from stacked to side by side
at a window width that does not change. A rule keyed to screen width could not do that.
"""

import hashlib

import streamlit as st

from speechwriter import workspace
from speechwriter.config import WORDS_PER_MINUTE
from speechwriter.webui import (
    MEASURE_CACHE_ENTRIES,
    READING_WIDTH,
    documents,
    get_bundle,
    spoken_length,
)

# Wide here, while the router's `st.set_page_config` keeps Write centred: a page may override
# the layout for itself, and moving between the two switches it each way. Chat is one column,
# so only this page needs the room.
st.set_page_config(layout="wide")

# The rail stretches to fill the row when the page is stacked, and this frame caps it when
# the page is wide; without the cap it grew to 1096px on a wide screen. The rail's figure is a
# ceiling, not a fixed width: a fixed rail stacks at its own width and wastes the row beside
# it, which pushed the speech down to 647px on a tablet.
RAIL_WIDTH = 272
# What `gap="large"` renders at the 16px root size, and it scales with that size. The gap is
# what sets the wrap point: a stretchy container shrinks to about 128px, and with
# `gap="medium"` a 768px tablet went side by side with a 144px rail.
LARGE_GAP = 64
PAGE_WIDTH = RAIL_WIDTH + LARGE_GAP + READING_WIDTH

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


# Ordered, oldest first, and bounded to `MEASURE_CACHE_ENTRIES`. One list rather than a
# boolean per digest, because the agent revises in place: every rewrite mints a fresh digest,
# so per-digest keys would accumulate for the life of the session and — worse — outlive the
# LRU cache they stand for. A flag whose WAV has been evicted sends the next page render
# straight back into a billed synthesis with no button pressed, which is the one thing gating
# the measurement behind a button was meant to prevent. Bounded to the cache's own size and
# refreshed on every hit, so the list ages the way the cache does — `st.cache_resource` is LRU
# and reorders on read, where a list that only ever appended would evict by *first* request.
# The two still cannot be made identical: that cache is process-global while these flags are
# per-session, so another tab can evict a WAV this session is flagging. When that happens the
# next render of that draft pays one synthesis and both are consistent again — the bound buys
# a list that cannot grow without limit and a divergence that costs one extra synthesis, not a
# guarantee.
_MEASURED = "measured"


def _measured() -> list[str]:
    """The drafts this reader has asked to measure, oldest first."""
    return st.session_state.setdefault(_MEASURED, [])


def _remember(flag: str) -> None:
    """Mark a draft measured — most recently used last — evicting past the cache's bound."""
    flags = _measured()
    if flag in flags:
        flags.remove(flag)
    flags.append(flag)
    del flags[:-MEASURE_CACHE_ENTRIES]


def _forget(flag: str) -> None:
    """Drop a flag after a failed measurement, putting the button back."""
    flags = _measured()
    if flag in flags:
        flags.remove(flag)


def _measure_flag(document: workspace.Document) -> str:
    """Session key for "the reader asked to measure *this* draft".

    Keyed on the text's digest, not just the slug. The agent revises a speech **in place**, so
    a slug-only key stays True across a rewrite — and since `spoken_length` is cached on the
    text, the next visit would miss the cache and silently run a full billed synthesis on page
    render, which is exactly what gating it behind a button was meant to prevent.
    """
    digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()[:12]
    return f"measured:{document.slug}:{digest}"


def _measure_if_requested(
    document: workspace.Document,
) -> tuple[workspace.SpokenLength | None, str]:
    """Synthesise the draft if asked, returning ``(measurement, notice)``.

    Returns the notice rather than rendering it: this runs before the stats row opens, and the
    notice is drawn after it closes, because an `st.info` emitted inside a `horizontal=True`
    container becomes one more flex item and wraps its text down a narrow strip, and that
    string's whole job is to be readable. It still lands in the rail, under the row.

    Any failure forgets the draft. Without that the page is unrecoverable: `st.cache_resource`
    does not cache exceptions, so a sticky flag re-attempts and re-raises on every rerun, and a
    traceback out of here stops the rest of the page rendering. Clearing it puts the button
    back, which is also what you want after setting the key the first branch complains
    about.
    """
    flag = _measure_flag(document)
    if flag not in _measured():
        return None, ""
    try:
        # `show_time` because this is the one wait in the app long enough to look hung — a
        # three-minute speech is a few seconds of Deepgram requests, and the cache declares
        # `show_spinner=False`, so this is the only progress the reader gets.
        with st.spinner("Synthesising the draft…", show_time=True):
            measured = spoken_length(document.text, get_bundle().settings.deepgram_api_key)
        # That read just refreshed the cache entry, so refresh the flag with it — otherwise
        # the flag list evicts by first request while the cache evicts by last use, and a
        # still-flagged draft whose WAV has gone re-synthesises on plain page render.
        _remember(flag)
        return measured, ""
    except workspace.AudioUnavailable as exc:
        _forget(flag)
        return None, str(exc)
    except Exception as exc:
        # Broad on purpose. Everything past the key check is someone else's failure mode — a
        # rejected key, a rate limit, no network — and none of them should take the page down.
        # `SynthesisFailed` carries Deepgram's own error text, which names the fix.
        _forget(flag)
        return None, f"Could not measure this draft: {exc}"


def _measure_button(document: workspace.Document) -> None:
    """The control that opts into a measurement, rendered inside the metric row."""
    flag = _measure_flag(document)
    # Rerun on click so the button is *replaced* by the result rather than sitting beside it.
    # Disabled, not hidden, without a key: the reader learns the feature exists and what turns
    # it on. The same rule the suggestion pills follow — a control that takes a click and then
    # only complains is worse than one that shows it is unavailable. Knowable up front now that
    # the requirement is a key rather than an installed extra.
    enabled = get_bundle().settings.measurement_enabled
    if st.button(
        "Measure",
        key=f"button-{flag}",
        icon=":material/graphic_eq:",
        help=(
            "Synthesise the draft with Deepgram and time it, instead of estimating from word "
            "count. Billed per character."
            if enabled
            else "Set DEEPGRAM_API_KEY to time drafts by synthesising them."
        ),
        disabled=not enabled,
    ):
        _remember(flag)
        st.rerun()


def document_browser(
    documents: list[workspace.Document], *, kind: str, spoken: bool, empty: str
) -> None:
    """Pick-and-read over a list of Markdown documents, newest first.

    ``kind`` keys the picker. Two things follow from it. The key is distinct per view, so a
    draft chosen under Speeches cannot surface as the selection under Research — a shared key
    would carry one across, since the key *is* the widget's identity. And with
    ``persist_state="session"`` the choice survives the widget unmounting: switching views, or
    stepping over to Write and back, returns to the draft that was open rather than to the
    top of the list. A selection whose label has since changed — the agent revises in place,
    and the label carries the mtime — falls back to the newest entry rather than raising.
    """
    if not documents:
        st.caption(empty)
        return

    labels = {f"{doc.slug}  ·  {doc.modified:%b %d, %H:%M}": doc for doc in documents}
    # `gap="large"` is load-bearing, not decoration: it is what makes a tablet stack rather
    # than squeeze the rail beside the speech. See `LARGE_GAP`.
    with st.container(horizontal=True, gap="large"):
        # The rail. Stretchy, so it fills the row when the page is stacked.
        with st.container():
            picked = st.selectbox(
                "Document",
                list(labels),
                key=f"document-{kind}",
                persist_state="session",
                label_visibility="collapsed",
            )
            document = labels.get(picked) if isinstance(picked, str) else None
            if document is None:
                return

            # Measured before the stats row opens, so the spinner spans the rail rather than
            # becoming one more item in the row.
            measured, notice = _measure_if_requested(document) if spoken else (None, "")
            # One wrapping row: stats and buttons share a line wherever they fit. Unbordered,
            # because on a phone the borders cost height the speech needs.
            with st.container(horizontal=True, vertical_alignment="bottom"):
                # The tooltip names what the count leaves out, matching workspace.py: the `---`
                # header block is stripped and bracketed delivery cues (`[pause]`) are dropped
                # as unspoken.
                st.metric(
                    "Words",
                    f"{document.words:,}",
                    width="content",
                    help="Body word count. Excludes the header block and bracketed delivery cues.",
                )
                # Only meaningful for something meant to be said out loud; a research brief is
                # not.
                if spoken:
                    # Sourced from config, not hardcoded, so the tooltip can never quote a pace
                    # that disagrees with the estimate it explains — workspace.py computes
                    # `minutes` from the same constant.
                    st.metric(
                        "Spoken length",
                        f"~{document.minutes:.1f} min",
                        width="content",
                        help=f"Estimated at about {WORDS_PER_MINUTE} words per minute.",
                    )
                    if measured is None:
                        _measure_button(document)
                    else:
                        # Showing both is the point: one constant cannot know that this draft
                        # is dense with long words and that one is short and punchy.
                        # `delta_color="off"` because drift in either direction is information,
                        # not good or bad news.
                        drift = measured.seconds - document.minutes * 60
                        st.metric(
                            "Measured",
                            f"{measured.minutes:.1f} min",
                            delta=f"{drift:+.0f}s",
                            # The qualifier goes in the parameter built for it, so it renders
                            # muted and caption-sized beside the number instead of at the
                            # delta's weight.
                            delta_description="vs estimate",
                            delta_color="off",
                            width="content",
                            help="Synthesised with Deepgram Aura-2. Times the words only — a "
                            "bracketed cue adds no silence, so this is time-to-say, not "
                            "time-on-stage.",
                        )
                st.download_button(
                    "Download",
                    document.text,
                    file_name=document.path.name,
                    icon=":material/download:",
                    # The bytes are already in hand and nothing here changes state, so the
                    # default rerun buys nothing and costs a re-stat of the folder plus — for a
                    # draft already measured — a "Synthesising…" spinner flashed over a cache
                    # hit.
                    on_click="ignore",
                )

            if notice:
                st.info(notice, icon=":material/volume_off:")

            # The synthesis already happened, so playing it back costs nothing extra — and
            # hearing a draft is the fastest way to catch what the critic's "speakability" pass
            # can only infer.
            if measured is not None and measured.wav:
                st.audio(measured.wav, format="audio/wav")

            # Rebuilt from the parsed pairs rather than passed through raw: a `---` fence
            # directly after the header's last line would otherwise turn the whole block into
            # one setext H2. Rebuilt, it holds no fence, so st.markdown is safe. And it must be
            # st.markdown, not st.caption: Streamlit draws captions as a 60% mix of text and
            # background, which is under AA in both modes, and speaker, audience and occasion
            # are what a reader needs before reading on.
            if document.front_matter:
                st.markdown(
                    "  \n".join(
                        f"**{label}** — {value}" if label else value
                        for label, value in document.front_matter
                    )
                )

        # `wrap=False` so a wide table in a research note scrolls inside this column rather
        # than taking the page sideways with it; prose still wraps at the column width. It
        # only suits a container with one child, because `wrap=False` lays children out in a
        # row. AppTest does not render, so a browser check is the real guard on this.
        with st.container(width=READING_WIDTH, horizontal=True, wrap=False):
            st.markdown(document.body)


# The page sits in a centred frame no wider than the rail, the gap and the reading column.
with st.container(horizontal_alignment="center"):
    with st.container(width=PAGE_WIDTH):
        st.title("Workspace")

        # `required` so the lit segment cannot be clicked off: without it a second click
        # returns None, which matches no branch below and lands in the `else` — rendering
        # Speeches with nothing selected, the control and the page disagreeing. It also makes
        # the return `str` rather than `str | None`, so the branches below are total by
        # construction. It names the three views, which is why the page has no caption under
        # its title: on a phone that line cost height the speech needs.
        view = st.segmented_control(
            "View",
            ["Speeches", "Research", "Memory"],
            default="Speeches",
            required=True,
            key="workspace-view",
            persist_state="session",
            label_visibility="collapsed",
        )

        if view == "Research":
            document_browser(
                documents(workspace.research_dir(settings)),
                kind="research",
                spoken=False,
                empty="No research notes yet. They appear here when the researcher subagent runs.",
            )

        elif view == "Memory":
            # Read from the live Store, not the JSON snapshot, so a profile learned this
            # session shows up before anything has called persist().
            entries = workspace.memories(bundle.store)
            with st.container(width=READING_WIDTH):
                if not entries:
                    st.caption(
                        "No voice profiles yet. The agent writes one after it delivers a "
                        "speech, and reads it back on the next commission for the same speaker."
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
                kind="speeches",
                spoken=True,
                empty="No speeches yet. Commission one on the Write page.",
            )
