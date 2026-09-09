"""Offline tests for the web app and the workspace reader.

Same bargain as ``test_build.py``: building the agent calls neither the model nor the
network, so Streamlit can render the whole app headlessly for free. That is what makes it
worth testing at all — there is no CI here, so a page that raises on import would otherwise
be discovered by a human opening a browser.
"""

from __future__ import annotations

import importlib
import os
import re
import tomllib
from dataclasses import replace

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from streamlit.testing.v1 import AppTest

from speechwriter import cli, config, webui, workspace
from speechwriter.config import load_settings
from speechwriter.prompts import orchestrator_prompt, researcher_prompt

_REPO_ROOT = config._PKG_DIR.parents[1]


def _write(path, text: str, *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_documents_are_listed_newest_first(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    speeches = settings.workspace_dir / config.SPEECHES_SUBDIR

    _write(speeches / "older.md", "first draft", mtime=1_000_000)
    _write(speeches / "newer.md", "second draft", mtime=2_000_000)
    # Not Markdown: must not appear, or the browser lists the agent's stray scratch files.
    _write(speeches / "notes.txt", "ignore me", mtime=3_000_000)

    found = workspace.speeches(settings)
    assert [doc.slug for doc in found] == ["newer", "older"]


def test_missing_workspace_dir_reads_as_empty(monkeypatch, tmp_path):
    # A fresh checkout has no speeches/ until the agent's first write. "Nothing yet" is the
    # normal state, not an error the page should raise on.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    assert workspace.speeches(settings) == []
    assert workspace.research_notes(settings) == []


def test_spoken_length_uses_the_pace_the_prompt_prescribes(monkeypatch, tmp_path):
    # Both halves of the pace seam, not the reader's own constant twice over: the prompt
    # *tells* the agent a pace and `Document.minutes` *estimates* at one. Parsing the
    # rendered prompt is what makes re-typing "~150 words per minute" into prompts.py fail
    # here, rather than silently disagreeing with every duration the browser prints.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    prescribed = re.search(r"~(\d+) words per minute", orchestrator_prompt(settings))
    assert prescribed, "the orchestrator prompt no longer states a pace in words per minute"
    pace = int(prescribed.group(1))
    assert pace == config.WORDS_PER_MINUTE, (
        f"the prompt prescribes {pace} wpm but the browser estimates at "
        f"{config.WORDS_PER_MINUTE} — interpolate WORDS_PER_MINUTE, do not re-type it"
    )

    words = pace * 3
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "toast.md",
        " ".join(["word"] * words),
        mtime=1_000_000,
    )

    document = workspace.speeches(settings)[0]
    assert document.words == words
    assert document.minutes == 3.0


def test_front_matter_is_split_off_rather_than_rendered_or_counted(monkeypatch, tmp_path):
    # The agent fences its header block with `---`. In CommonMark a `---` line directly
    # after a paragraph makes that paragraph a setext H2, so passing the raw file to
    # st.markdown renders the whole header as one run-on heading — and counting its words
    # inflates the very spoken-length estimate the header itself quotes.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    body_words = config.WORDS_PER_MINUTE * 2
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "toast.md",
        "---\n"
        "Speaker: Daryl (best man)\n"
        "Word count: ~260 words | Est. time: ~2:00 at 130 wpm\n"
        "---\n\n" + " ".join(["word"] * body_words),
        mtime=1_000_000,
    )

    document = workspace.speeches(settings)[0]
    assert document.front_matter[0] == ("Speaker", "Daryl (best man)")
    # Partitioned on the *first* colon, so a value carrying its own colons survives whole.
    assert document.front_matter[1] == (
        "Word count",
        "~260 words | Est. time: ~2:00 at 130 wpm",
    )
    assert not document.body.startswith("---")
    assert document.words == body_words  # header excluded from the count
    assert document.minutes == 2.0
    assert document.text.startswith("---")  # a download still gets the file as written


def test_stage_directions_do_not_count_as_spoken_words(monkeypatch, tmp_path):
    # The delivery-and-cadence skill asks for a marked-up script, so `[pause]` cues are
    # expected in a finished draft. They are acted on, not said.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "cued.md",
        "Good evening.\n\n[pause]\n\nThank you all.\n\n[beat, slow down here]\n\nGoodnight.",
        mtime=1_000_000,
    )

    # "Good evening. Thank you all. Goodnight." — the two cues contribute nothing.
    assert workspace.speeches(settings)[0].words == 6


def test_document_without_front_matter_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "plain.md",
        "Just prose, no header.",
        mtime=1_000_000,
    )

    document = workspace.speeches(settings)[0]
    assert document.front_matter == ()
    assert document.body == "Just prose, no header."


def test_prompts_point_the_agent_at_the_folders_the_browser_reads(monkeypatch, tmp_path):
    # The cross-subsystem check that matters now that a reader exists: the prompts *tell*
    # the agent where to save, and the browse page reads real directories back. Each case
    # derives the virtual path from the reader function the page actually calls, so
    # re-hardcoding either side fails here rather than silently producing a browser that
    # lists nothing. Research needs its own case: `browse.py` lists `research_dir()` just
    # as it lists `speeches_dir()`, and the researcher subagent — which writes there — is
    # steered by a prompt of its own that the orchestrator's wording cannot vouch for.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    cases = (
        ("speeches", workspace.speeches_dir(settings), "orchestrator", orchestrator_prompt),
        ("research", workspace.research_dir(settings), "orchestrator", orchestrator_prompt),
        ("research", workspace.research_dir(settings), "researcher", researcher_prompt),
    )

    for label, read_dir, prompt_name, render in cases:
        as_the_agent_sees_it = "/" + read_dir.relative_to(settings.project_root).as_posix()
        # The trailing slash is load-bearing: without it this passes on a prompt that says
        # `/workspace/speeches-drafts/`, which is exactly the drift being guarded against.
        assert f"{as_the_agent_sees_it}/" in render(settings), (
            f"the {prompt_name} prompt does not name {as_the_agent_sees_it}/ — "
            f"the {label} browser reads {read_dir}"
        )


def test_memory_entries_render_known_and_unknown_payloads():
    # deepagents' StoreBackend owns the payload shape and `file_format` is left at its
    # default, so an unrecognised value must still render as *something* — an empty panel
    # would read as "no memory saved" when a profile is in fact stored.
    store = InMemoryStore()
    store.put(("speechwriter", "memories"), "a-doc.md", {"content": "warm, plainspoken"})
    store.put(("speechwriter", "memories"), "b-lines.md", {"content": ["line one", "line two"]})
    store.put(("speechwriter", "memories"), "c-odd.md", {"unexpected": 42})

    entries = {entry.key: entry.text for entry in workspace.memories(store)}
    assert entries["a-doc.md"] == "warm, plainspoken"
    assert entries["b-lines.md"] == "line one\nline two"
    assert "unexpected" in entries["c-odd.md"]  # fell back to JSON rather than rendering blank


def test_memory_listing_pages_past_the_store_search_limit():
    # Regression: `Store.search` truncates at 10 by default. The web UI must go through
    # `memory.all_items`, not a hand-rolled walk, or it shows a partial memory as whole.
    store = InMemoryStore()
    for i in range(25):
        store.put(("speechwriter", "memories"), f"speaker-{i:02d}.md", {"content": f"v{i}"})

    assert len(workspace.memories(store)) == 25


def test_each_streamed_message_is_rendered_exactly_once():
    # `stream_mode="values"` replays the *entire* message list on every step, and the
    # checkpointer carries earlier turns forward, so without the seen-set the transcript
    # would re-render the whole conversation on every step of every turn.
    seen: set[str] = set()
    reply = AIMessage(
        content="Here is the draft.",
        id="ai-1",
        tool_calls=[{"name": "write_todos", "args": {"todos": ["intake"]}, "id": "call-1"}],
    )

    events = list(webui._new_events(reply, seen))
    assert [(event.kind, event.name) for event in events] == [
        ("call", "write_todos"),
        ("prose", ""),
    ]
    assert events[1].text == "Here is the draft."
    assert list(webui._new_events(reply, seen)) == []  # same message next step: no repeat


def test_prompt_echoes_are_not_rendered_as_agent_output():
    # The user's own message comes back in every streamed value, and the page has already
    # shown it in its own bubble. Nothing skips it explicitly — it simply is neither an
    # AIMessage nor a ToolMessage — so this pins the behaviour against a future `else`
    # branch that tried to render unrecognised message types.
    seen: set[str] = set()
    assert list(webui._new_events(HumanMessage(content="A toast", id="h-1"), seen)) == []


def test_failed_tool_results_are_marked_as_failures():
    seen: set[str] = set()
    failed = ToolMessage(content="boom", id="t-9", name="task", tool_call_id="c-1", status="error")
    (event,) = webui._new_events(failed, seen)
    assert (event.kind, event.name, event.ok) == ("result", "task", False)


def test_previews_cannot_break_out_of_their_code_span():
    # Tool arguments are arbitrary text. A backtick left intact would end the inline-code
    # span and let the rest of a tool result reflow the activity log as Markdown.
    assert "`" not in webui._preview('{"cmd": "echo `whoami`"}')


def test_a_recorded_turn_replays_with_its_truncation_warning(monkeypatch, tmp_path):
    # Replay is the path the user sees after every turn, and it runs without the model —
    # so it is worth proving that a turn which hit the ceiling still says so on redraw.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60)
    app.session_state["transcript"] = [
        webui.Turn(
            prompt="A toast for Ana",
            events=[
                webui.Event(kind="call", name="write_todos", text="{}"),
                webui.Event(kind="prose", text="Here is the toast."),
            ],
            truncated=1,
        )
    ]
    app.run()

    assert not app.exception
    assert any("Here is the toast." in block.value for block in app.markdown)
    assert any("output-token ceiling" in warning.value for warning in app.warning)


def test_web_app_renders_without_network_or_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Same rule, and here it is load-bearing rather than defensive: an exported base URL makes
    # `model_credentials_present` true, so the setup error this asserts on never renders.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    # The bundle is cached across the process, so a leftover from another test would pin
    # this run to the wrong SPEECHWRITER_HOME.
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    # Missing key must be reported in the page, not crash it — the workspace stays browsable.
    assert any("ANTHROPIC_API_KEY" in error.value for error in app.error)


def test_both_pages_render(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    settings = load_settings()
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "ana-toast.md",
        "# Toast\n\n" + " ".join(["word"] * 260),
        mtime=1_000_000,
    )

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.error  # a key is present, so no setup error

    app.switch_page("app_pages/browse.py").run()
    assert not app.exception
    assert any("ana-toast" in str(option) for option in app.main.selectbox[0].options)


def test_markdown_link_label_counts_as_spoken_words(monkeypatch, tmp_path):
    # The stage-direction strip must not eat a Markdown link's label: `[our report](url)` is
    # spoken, `[pause]` is not. Regression for the `(?!\\()` lookahead on _STAGE_DIRECTION;
    # without it the whole `[our report]` is deleted and only the bare URL is counted.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    _write(
        settings.workspace_dir / config.SPEECHES_SUBDIR / "linked.md",
        "[our report](http://x.com)",
        mtime=1_000_000,
    )
    # Two real words ("our report") survive; strip the label and it collapses to one.
    assert workspace.speeches(settings)[0].words == 2


def test_a_cancelled_turn_rotates_the_thread(monkeypatch):
    # A stop mid-turn leaves _PENDING set (the BaseException sails past run_turn's except),
    # so the next turn must rotate to a fresh thread and never resume a half-executed graph.
    state = {"turn_in_flight": True, "thread_id": "web-dirty", "seen_message_ids": {"m1"}}
    monkeypatch.setattr(webui.st, "session_state", state)

    webui._rotate_if_interrupted()

    assert state["thread_id"] != "web-dirty"  # rotated to a fresh thread
    assert state["seen_message_ids"] == set()  # scoped to the abandoned thread, so dropped
    assert state["turn_in_flight"] is False


def test_a_completed_turn_keeps_its_thread(monkeypatch):
    # The mirror case: a turn that finished cleanly must NOT rotate, or every turn would
    # start a new thread and the conversation could never build across turns.
    state = {"turn_in_flight": False, "thread_id": "web-keep", "seen_message_ids": {"m1"}}
    monkeypatch.setattr(webui.st, "session_state", state)

    webui._rotate_if_interrupted()

    assert state["thread_id"] == "web-keep"
    assert state["seen_message_ids"] == {"m1"}


def test_document_reader_reflects_a_newly_written_draft(monkeypatch, tmp_path):
    # The cached listing must invalidate when the folder changes, or a draft the agent just
    # saved would never appear. The name+mtime signature is what forces the re-read.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_data.clear()
    settings = load_settings()
    speeches = workspace.speeches_dir(settings)

    _write(speeches / "first.md", "one", mtime=1_000_000)
    assert [doc.slug for doc in webui.documents(speeches)] == ["first"]

    _write(speeches / "second.md", "two", mtime=2_000_000)
    assert [doc.slug for doc in webui.documents(speeches)] == ["second", "first"]


def test_memory_view_renders_a_seeded_profile(monkeypatch, tmp_path):
    # The browse page's Memory branch (an expander per profile) was never driven by a test,
    # so a crash there would only surface when a human opened the page.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    bundle = webui.get_bundle()
    bundle.store.put(("speechwriter", "memories"), "mayor.md", {"content": "warm, plainspoken"})

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.switch_page("app_pages/browse.py").run()
    app.segmented_control[0].set_value("Memory").run()

    assert not app.exception
    assert any("warm, plainspoken" in block.value for block in app.markdown)


def test_streamlit_config_parses_and_offers_both_theme_modes():
    # AppTest does not parse the project theme config, so a TOML typo or a dropped
    # [theme.dark] table would otherwise reach a human running `streamlit run` — exactly the
    # manual discovery the rest of this suite exists to pre-empt. This and
    # `test_theme_links_clear_wcag_aa_and_stay_visible_without_color` are the only checks that
    # read .streamlit/config.toml at all: it is not Python, so ruff-ty-gate.sh never sees it.
    # pytest-gate.sh watches .streamlit/ for that reason, which is what puts this assertion in
    # the inner loop rather than only in CI.
    data = tomllib.loads((_REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    # Both mode tables must exist or Streamlit locks to a single mode and the light/dark
    # toggle silently disappears.
    assert "light" in data["theme"]
    assert "dark" in data["theme"]
    # The security invariant the file's own header comment documents: bind loopback only, so
    # the budget-spending agent is never put on the network by an "External URL".
    assert data["server"]["address"] == "localhost"


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance for an ``#rrggbb`` string."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(one: str, other: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colors, 1.0 to 21.0."""
    first, second = _relative_luminance(one), _relative_luminance(other)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def test_theme_links_clear_wcag_aa_and_stay_visible_without_color():
    # The palette in .streamlit/config.toml already replaced Solarized Light *for failing AA*
    # — its body text was 4.13:1 — and the link color then slipped past the same audit at
    # 3.88:1, because nothing measured it. Both halves are asserted here rather than the hex
    # literals, so a future repalette is free to pick any colors that pass.
    data = tomllib.loads((_REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    theme = data["theme"]
    # Read once and named: underlining satisfies F73 for every mode at once, so this is the
    # switch that decides whether the per-mode contrast rule below has any work to do.
    underlined = theme.get("linkUnderline", True)

    for mode in ("light", "dark"):
        palette = theme[mode]
        link, body = palette["linkColor"], palette["textColor"]
        # Every surface a link can be read on, not just the page ground: chat bubbles,
        # expanders and the whole sidebar sit on secondaryBackgroundColor, and in light mode
        # that is the tightest of them — 4.67:1, against 5.38:1 on the background that used to
        # be the only one checked.
        surfaces = {v for k, v in palette.items() if k.endswith("ackgroundColor")}
        surfaces |= {
            v for k, v in palette.get("sidebar", {}).items() if k.endswith("ackgroundColor")
        }
        for surface in sorted(surfaces):
            # 1.4.3 Contrast (Minimum), for text.
            assert _contrast(link, surface) >= 4.5, f"{mode} linkColor is under AA on {surface}"
        # F73: a link may not be marked by color alone. Either it is underlined, or it stands
        # 3:1 clear of the prose around it — #61afef sits 1.11:1 against dark-mode body text,
        # which is why the underline is what carries this today.
        assert underlined or _contrast(link, body) >= 3.0, (
            f"{mode} links are distinguished by color alone"
        )


def test_the_status_badge_gates_on_credentials_not_an_anthropic_key(monkeypatch, tmp_path):
    # `config.py` documents `model_credentials_present` as the contract for both front ends,
    # and the chat input honours it — but the sidebar badge read `anthropic_api_key` directly,
    # so a model served over SPEECHWRITER_BASE_URL ran perfectly under a red "no key" badge.
    # Two independent literals with nothing structural tying them, which is what this closes.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    # Badges render as Markdown directives, so this reads the rendered text rather than a
    # `st.badge` accessor, which AppTest does not expose.
    assert any("Ready]" in block.value for block in app.markdown)
    # And the page must not tell a working local setup to go and find an API key.
    assert not app.error
    assert not app.chat_input[0].disabled

    # The mirror. Without it, hardcoding the badge to "Ready" passes the whole suite — the
    # same lie this test exists to catch, told in the other direction.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL")
    monkeypatch.delenv("SPEECHWRITER_MODEL")
    st.cache_resource.clear()
    bare = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not bare.exception
    assert any("No credentials]" in block.value for block in bare.markdown)
    assert bare.chat_input[0].disabled


def _picked(app) -> config.ModelChoice:
    """The whole ``ModelChoice`` the sidebar picker is showing.

    The isinstance check is the point as well as the narrowing: the widget carries the model and
    the endpoint as one record, and a refactor that reduced it to a bare id string would take
    the two apart — which is exactly the state ``ModelChoice`` exists to make unrepresentable.
    """
    value = app.sidebar.selectbox[0].value
    assert isinstance(value, config.ModelChoice), value
    return value


def test_the_sidebar_picker_rebuilds_the_agent_on_the_chosen_model(monkeypatch, tmp_path):
    # The whole feature, end to end. Picking a model has to *rebuild* the bundle, not merely
    # record a preference: the resolved output ceiling is read off the constructed client, so if
    # the rebuild does not happen the caption keeps quoting the previous model's. Haiku 4.5 is
    # the discriminating pick — Sonnet 5 and Opus 5 are both profiled at 128k, so a switch
    # between those two would pass with no rebuild at all.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    picker = app.sidebar.selectbox[0]
    assert list(picker.options) == [choice.label for choice in config.MODEL_CHOICES]
    assert _picked(app).model == config.DEFAULT_MODEL
    assert "128,000" in app.sidebar.caption[0].value

    app.sidebar.selectbox[0].select("Haiku 4.5").run()

    assert not app.exception
    assert _picked(app).model == "claude-haiku-4-5"
    assert "64,000" in app.sidebar.caption[0].value


def test_switching_models_in_the_browser_saves_before_it_invalidates(monkeypatch, tmp_path):
    # The web half of the persist-before-rebuild order. CLAUDE.md claims this is asserted from
    # both front ends; before this test only the CLI's was, and deleting `bundle.persist()` from
    # `switch_model` passed the entire suite — the exact silent data loss the ordering exists to
    # prevent, uncovered.
    #
    # Driven through `webui` directly rather than through AppTest, because the callback runs
    # between reruns and the widget only reports where it ended up, not what it did on the way.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    st.cache_resource.clear()

    # Warm the cache *before* the pick is recorded, which is the real sequence: the callback
    # fires while the bundle for the previous model is still cached. Setting the choice first
    # would make `get_bundle()` build the new model inside `switch_model`, and the guard against
    # switching to what is already running would (correctly) return early.
    webui.get_bundle()

    order: list[str] = []
    monkeypatch.setattr(webui.SpeechwriterAgent, "persist", lambda self: order.append("persist"))
    monkeypatch.setattr(
        type(webui.get_bundle), "clear", lambda self, *a, **k: order.append("invalidate")
    )
    monkeypatch.setattr(webui, "reset_conversation", lambda: order.append("reset"))

    st.session_state[webui.MODEL_KEY] = config.MODEL_CHOICES[2]
    try:
        webui.switch_model()
    finally:
        st.session_state.pop(webui.MODEL_KEY, None)

    assert order == ["persist", "invalidate", "reset"], (
        "the switch must save the Store before the rebuild reloads it from disk, and rotate the "
        "thread after — `build_agent` mints a fresh Store and a fresh checkpointer"
    )


def test_switching_models_resets_even_when_the_bundle_is_not_cached(monkeypatch, tmp_path):
    # `switch_model` used to ask `get_bundle()` whether the pick was already running. That
    # reads the *new* choice out of session state, so on a cold cache it built the new model,
    # matched itself, and returned before `reset_conversation()` — leaving the page showing a
    # transcript whose thread names a checkpoint the freshly-minted MemorySaver never saw.
    #
    # A cold cache at callback time is ordinary, not exotic: `cache_resource` is app-global, so
    # one tab switching clears it for every other tab, and Streamlit also drops it after an edit.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    st.cache_resource.clear()

    order: list[str] = []
    monkeypatch.setattr(webui.SpeechwriterAgent, "persist", lambda self: order.append("persist"))
    monkeypatch.setattr(
        type(webui.get_bundle), "clear", lambda self, *a, **k: order.append("invalidate")
    )
    monkeypatch.setattr(webui, "reset_conversation", lambda: order.append("reset"))

    # No `get_bundle()` first: the cache is cold, exactly as it is for a second tab.
    st.session_state[webui.MODEL_KEY] = config.MODEL_CHOICES[2]
    try:
        webui.switch_model()
    finally:
        st.session_state.pop(webui.MODEL_KEY, None)

    assert order == ["persist", "invalidate", "reset"], (
        "a cold cache must not skip the thread rotation — the transcript would outlive the "
        "checkpoint it belongs to"
    )


def test_a_model_that_cannot_be_built_leaves_the_page_usable(monkeypatch, tmp_path):
    # `get_bundle()` runs at module scope in streamlit_app.py, *above* the sidebar, so an
    # unbuildable pick would take the page down before the picker that would let the reader undo
    # it is ever drawn — and the pick survives in session state, so every rerun raises again.
    # The CLI already guards the identical call; this is the browser's half.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    st.cache_resource.clear()

    # An id whose provider cannot be inferred: `init_chat_model` raises at *construction*.
    st.session_state[webui.MODEL_KEY] = config.ModelChoice("Broken", "no-such-provider-model")
    try:
        bundle = webui.get_bundle()
        reported = webui.build_error()
    finally:
        st.session_state.pop(webui.MODEL_KEY, None)

    # Fell back to the environment's model rather than raising...
    assert bundle.settings.model == config.DEFAULT_MODEL
    # ...told the page which pick failed, exactly once...
    assert reported is not None and "Broken" in reported
    assert webui.build_error() is None
    # ...and dropped the bad selection, so the next rerun does not raise again.
    assert webui.selected_choice() is None


def test_detected_models_join_the_roster_exactly_once(monkeypatch, tmp_path):
    # `available_choices` is the dedup that stops Streamlit silently resetting the selection:
    # `ModelChoice` is a NamedTuple, so a detected entry that differed from the synthesised one
    # by so much as its label would appear twice *and* leave the selected value unfindable
    # among the options. Nothing exercised it before.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://localhost:1234/v1")
    st.cache_resource.clear()

    settings = load_settings()
    endpoint = "http://localhost:1234/v1"
    # As if "Detect models" had answered: the configured model, plus one the reader has not
    # seen. Whole `ModelChoice` records, because that is the shape the session now stores —
    # seeding bare ids here is what the old pairing code accepted and this one must not.
    # Three, sorted as `detect_models` stores them, and only one of them configured. Two is not
    # enough to see the ordering bug below: with a single unconfigured detection it lands last
    # whether detections are merged before or after the offered pairs, so the assertion passes
    # under both. The third entry is what makes the two orders differ.
    st.session_state[webui.DETECTED_KEY] = [
        config.local_choice("local/granite", endpoint),
        config.local_choice("local/qwen", endpoint),
        config.local_choice("local/zephyr", endpoint),
    ]
    try:
        offered = webui.available_choices(settings)
    finally:
        st.session_state.pop(webui.DETECTED_KEY, None)

    labels = [choice.label for choice in offered]
    assert labels.count("local/qwen (local)") == 1, labels
    assert "local/granite (local)" in labels

    # Order is fixed, and stays fixed once a detected model is the one selected. Detections are
    # merged *before* the offered configurations for exactly this reason: `offered` varies with
    # the selection and `detected` does not, so a detection placed after it would be promoted
    # past the others the moment it was picked — and `index=choices.index(current)` names a row
    # number that has to mean the same thing on the next render.
    on_zephyr = replace(settings, model="local/zephyr")
    st.session_state[webui.DETECTED_KEY] = [
        config.local_choice("local/granite", endpoint),
        config.local_choice("local/qwen", endpoint),
        config.local_choice("local/zephyr", endpoint),
    ]
    try:
        assert [c.label for c in webui.available_choices(on_zephyr)] == labels
    finally:
        st.session_state.pop(webui.DETECTED_KEY, None)
    # Detected entries carry the endpoint they were found at, or picking one would build an
    # Anthropic client for a model only that server has.
    granite = next(c for c in offered if c.model == "local/granite")
    assert granite.base_url == endpoint


def test_detections_stay_bound_to_the_server_that_answered(monkeypatch, tmp_path):
    # The failure the whole `list[str]` -> `list[ModelChoice]` change exists to prevent. While
    # the endpoint could not change, pairing ids with `base_settings().base_url` at *read* time
    # was sound. It can change now, so ids detected at one server would be relabelled as served
    # by another the moment the field was retyped — a pair naming a server the model is not on,
    # which `_build_model` turns into a client pointed straight at it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    st.cache_resource.clear()

    # Detected at 8080, while the field has since been retyped to point at 1234.
    st.session_state[webui.DETECTED_KEY] = [config.local_choice("qwen", "http://127.0.0.1:8080/v1")]
    st.session_state[webui.ENDPOINT_KEY] = "http://127.0.0.1:1234/v1"
    try:
        offered = webui.available_choices(load_settings())
    finally:
        st.session_state.pop(webui.DETECTED_KEY, None)
        st.session_state.pop(webui.ENDPOINT_KEY, None)

    qwen = next(c for c in offered if c.model == "qwen")
    assert qwen.base_url == "http://127.0.0.1:8080/v1", (
        "a detection was relabelled with the endpoint the field happens to hold now"
    )


def test_editing_the_endpoint_is_not_a_model_switch(monkeypatch, tmp_path):
    # `switch_model`'s three steps are load-bearing and sit one function away, so a contributor
    # wiring up the endpoint field will reasonably wonder whether they belong here too. They do
    # not: pointing at a server changes what may be *offered*, never what is running. Doing them
    # anyway would drop the conversation and rotate the thread every time the box lost focus.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    st.cache_resource.clear()

    calls: list[str] = []
    monkeypatch.setattr(type(webui.get_bundle), "clear", lambda self: calls.append("invalidate"))
    monkeypatch.setattr(webui, "reset_conversation", lambda: calls.append("reset"))

    st.session_state[webui.ENDPOINT_KEY] = "localhost:8080"
    st.session_state[webui.DETECTED_KEY] = [config.local_choice("qwen", "http://elsewhere/v1")]
    try:
        webui.apply_endpoint()
        # Normalised in place, so the reader sees the URL that will actually be asked rather
        # than the app quietly asking one the box never showed.
        assert st.session_state[webui.ENDPOINT_KEY] == "http://localhost:8080/v1"
        # And the previous server's models are gone: kept, they would put two rows reading
        # "qwen (local)" in the picker, of which `resolve_choice` matches the first by label.
        assert webui.detections() is None
    finally:
        st.session_state.pop(webui.ENDPOINT_KEY, None)
        st.session_state.pop(webui.DETECTED_KEY, None)

    assert calls == [], f"editing the endpoint ran a model switch: {calls}"


def test_junk_in_the_endpoint_field_is_left_alone_rather_than_rewritten(monkeypatch, tmp_path):
    # Two halves, and the second is the one that bites. `normalize_endpoint` must not raise —
    # `session_endpoint()` runs on every rerun, so a `ValueError` here is not a bad caption but
    # a page that throws on every rerun with the offending text still in session state, which is
    # the unrecoverable shape `get_bundle`'s own `except` exists to prevent. And text that does
    # not normalise stays exactly as typed, so a typo remains legible instead of being rewritten
    # into something confidently wrong.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    st.session_state[webui.ENDPOINT_KEY] = "http://[::1"
    try:
        assert webui.session_endpoint() is None
        webui.apply_endpoint()
        assert st.session_state[webui.ENDPOINT_KEY] == "http://[::1"
    finally:
        st.session_state.pop(webui.ENDPOINT_KEY, None)


def test_a_configured_local_model_survives_a_rerun(monkeypatch, tmp_path):
    # Streamlit replaces a `session_state` value that is not among a widget's options with
    # option zero, raising nothing — so a roster that did not carry the configured pair would
    # take a reader on a keyless local endpoint and silently retarget them onto claude-sonnet-5,
    # flipping a working app into "No credentials".
    #
    # Both runs matter, and not equally: the first proves the configured pair is offered at all,
    # the second that a *stored* selection is not then silently overwritten — a state the widget
    # can only reach once it has a value, and one the existing badge test (which runs once)
    # cannot see.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert _picked(app).model == "local/qwen"

    app.run()

    assert not app.exception
    assert _picked(app).model == "local/qwen"
    assert _picked(app).base_url == "http://localhost:1234/v1"
    assert any("Ready]" in block.value for block in app.markdown)
    # Detect is offered only against a configured endpoint, and it must not have *run* on
    # render: the suite is offline by construction and CI renders this page dozens of times.
    # Asserted through the caption the sidebar only draws once an answer exists, not through
    # `webui.detected()` — that reads this process's session state rather than the rendered
    # app's, so it returns None either way and the check passed with detection wired to render.
    assert [button.label for button in app.sidebar.button] == ["Detect models", "New conversation"]
    captions = [caption.value for caption in app.sidebar.caption]
    assert not any("listed no models" in caption for caption in captions), captions
    assert not any("Found" in caption for caption in captions), captions


def test_a_lit_suggestion_pill_cannot_recommission_on_a_rerun(monkeypatch, tmp_path):
    # The pill used to be consumed by *not being rendered* once the transcript filled. A turn
    # cancelled with the stop button raises a BaseException past `run_turn`, so nothing is
    # appended, the welcome block draws again with the pill still lit — and the same
    # commission fired a second time, unasked. This is that exact state: selection present,
    # transcript empty. Only an `on_change` click may queue a brief now, never a bare rerun.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    # Taken from the rendered widget rather than restated here, so the test cannot drift from
    # the page's own suggestion labels.
    app.session_state["suggestion"] = app.pills[0].options[0]
    app.session_state["transcript"] = []
    app.run()

    assert not app.exception
    assert app.session_state["transcript"] == []
    assert "queued_prompt" not in app.session_state


def test_an_unreadable_file_bypasses_the_cache_instead_of_keying_against_it(monkeypatch, tmp_path):
    # The agent writes into this folder while the page renders, so a name can survive the glob
    # and fail the stat. Any key built from what remains describes a folder we could not fully
    # see — and `load_documents` re-globs, so the parse behind it can disagree. Worse, a later
    # render hitting the same race rebuilds the same key and is served that stale parse. A
    # dangling symlink is the same race, deterministically.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    speeches = settings.workspace_dir / config.SPEECHES_SUBDIR
    _write(speeches / "kept.md", "a draft", mtime=1_000_000)

    keyed: list[tuple[tuple[str, float], ...]] = []
    real = webui._parse_documents
    monkeypatch.setattr(
        webui, "_parse_documents", lambda directory, sig: keyed.append(sig) or real(directory, sig)
    )

    assert [doc.slug for doc in webui.documents(speeches)] == ["kept"]
    assert len(keyed) == 1  # a folder we can see whole is cached as usual

    (speeches / "racing.md").symlink_to(speeches / "gone.md")
    found = webui.documents(speeches)

    assert len(keyed) == 1, "an unstattable file must not mint a cache key"
    # And the page still gets the drafts it can read — the race costs the cache, not the view.
    assert [doc.slug for doc in found] == ["kept"]


def test_a_typed_brief_does_not_leave_a_suggestion_queued(monkeypatch, tmp_path):
    # The first version of the queue read `typed or st.session_state.pop(...)`, and `or` never
    # evaluates the pop when a typed brief wins — so the suggestion stayed armed and fired as a
    # second, unasked commission on a later rerun. The two arriving together is not contrived:
    # Streamlit coalesces a pending rerun with a new one and ships every widget state on each
    # message, and the chat box stays typeable while a pill's turn is still running.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    commissioned: list[str] = []

    def _record(_bundle, prompt: str) -> webui.Turn:
        commissioned.append(prompt)
        return webui.Turn(prompt=prompt)

    # The page re-imports `run_turn` on every run, so patching the module reaches it — and
    # keeps this test as free and offline as the rest of the suite.
    monkeypatch.setattr(webui, "run_turn", _record)

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.pills[0].set_value(app.pills[0].options[0])
    app.chat_input[0].set_value("A toast for Ana")
    app.run()

    assert not app.exception
    assert commissioned == ["A toast for Ana"]
    # The queue must be drained by the run that saw it, whether or not it won.
    assert "queued_prompt" not in app.session_state

    app.run()  # a plain rerun must not commission anything further
    assert commissioned == ["A toast for Ana"]


def _click_measure(app) -> None:
    """Press the Measure button, which shares `app.button` with the sidebar's reset."""
    next(button for button in app.button if button.label == "Measure").click().run()


def test_the_measured_set_is_bounded_and_a_failure_puts_the_button_back(monkeypatch, tmp_path):
    # `_measured`/`_remember`/`_forget` are the most intricate new logic on the page and the
    # audio tests never reach them — they call `workspace.measure_spoken_length` directly.
    # Stubbing the synthesis puts the flag path under test without the audio extra, which CI
    # never installs. The bound is lowered rather than measuring nine drafts.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    settings = load_settings()
    for index in range(3):
        _write(
            settings.workspace_dir / config.SPEECHES_SUBDIR / f"draft-{index}.md",
            f"draft {index} " + " ".join(["word"] * 120),
            mtime=1_000_000 + index,
        )

    monkeypatch.setattr(webui, "MEASURE_CACHE_ENTRIES", 2)
    monkeypatch.setattr(
        webui,
        "spoken_length",
        lambda text: workspace.SpokenLength(seconds=90.0, wav=b"", sample_rate=24_000),
    )

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60)
    app.run().switch_page("app_pages/browse.py").run()

    for option in list(app.main.selectbox[0].options):
        app.main.selectbox[0].select(option).run()
        _click_measure(app)

    assert not app.exception
    # A measured draft reports the synthesised figure beside the estimate...
    assert any(metric.label == "Measured" for metric in app.metric)
    # ...and the flag list is held to the cache's own size, so a flag cannot outlive its WAV
    # by more than the bound. Without `del flags[:-MEASURE_CACHE_ENTRIES]` this would be 3.
    assert len(app.session_state["measured"]) == 2

    # Re-viewing an already-measured draft refreshes its flag, so the list ages the way the
    # cache does: `st.cache_resource` is LRU and reorders on read, where a list that only ever
    # appended would evict by *first* request and drop a draft whose WAV is still warm.
    oldest = app.session_state["measured"][0]
    slug = oldest.split(":")[1]
    revisited = next(option for option in app.main.selectbox[0].options if option.startswith(slug))
    app.main.selectbox[0].select(revisited).run()

    assert app.session_state["measured"][-1] == oldest, "a read must refresh the flag"

    # A failed synthesis must forget the draft, or the flag re-raises on every rerun with no
    # way back to the button — the page becomes unrecoverable rather than merely unmeasured.
    def _unavailable(_text: str) -> workspace.SpokenLength:
        raise workspace.AudioUnavailable("install the audio extra")

    monkeypatch.setattr(webui, "spoken_length", _unavailable)
    app.main.selectbox[0].select(app.main.selectbox[0].options[0]).run()
    _click_measure(app)

    assert not app.exception
    assert any("install the audio extra" in info.value for info in app.info)
    # Forgotten — the count drops rather than the failed draft staying flagged forever.
    assert len(app.session_state["measured"]) == 1
    # And the button is back, which is also what you want after installing the extra.
    assert any(button.label == "Measure" for button in app.button)


def test_new_conversation_disarms_a_queued_suggestion(monkeypatch, tmp_path):
    # A brief queued by a pill click is drained by the Write page — but only if that page runs.
    # Leave the queue armed (a render that raised, a stop during a cold start, a nav away) and
    # walk to Workspace, and "New conversation" is the one control that says "drop all this".
    # If it does not clear the queue, the next visit to Write commissions the speech the reader
    # believed they had abandoned, and spends tokens doing it. Asserted from Workspace on
    # purpose: on the Write page the drain would hide the bug.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60)
    app.run().switch_page("app_pages/browse.py").run()
    app.session_state["queued_prompt"] = "Write a 3-minute wedding toast."
    next(button for button in app.button if button.label == "New conversation").click().run()

    assert not app.exception
    assert "queued_prompt" not in app.session_state


def test_the_workspace_view_control_cannot_be_deselected(monkeypatch, tmp_path):
    # Without `required`, a second click on the lit segment returns None — which matches
    # neither the "Research" nor the "Memory" branch and falls through to the `else`, drawing
    # Speeches with no segment highlighted. Asserted on the widget rather than by driving a
    # deselect, because AppTest's `unselect` bypasses the frontend rule it is testing.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    # CLAUDE.md: SPEECHWRITER_BASE_URL changes the client type, and AppTest builds a model.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60)
    app.run().switch_page("app_pages/browse.py").run()

    assert not app.exception
    assert app.segmented_control[0].proto.required


def _without_the_audio_extra(monkeypatch):
    """Make every `mlx_audio` import fail, as it does in CI and any default install."""
    real = importlib.import_module

    def blocked(name, *args, **kwargs):
        if name.startswith("mlx_audio"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    # The per-process model cache would otherwise satisfy the call before the import runs.
    monkeypatch.setattr(workspace, "_TTS_MODELS", {})


def test_measuring_without_the_audio_extra_names_the_install_command(monkeypatch):
    # The extra is genuinely optional, so this path is a normal state the UI has to explain.
    # A bare ImportError escaping here would surface in the browser as a red traceback on a
    # page whose other five features work fine.
    _without_the_audio_extra(monkeypatch)

    with pytest.raises(workspace.AudioUnavailable) as excinfo:
        workspace.measure_spoken_length("Good evening, and thank you all for coming.")

    assert "--extra audio" in str(excinfo.value)


def test_measuring_an_unspoken_draft_needs_no_model_at_all(monkeypatch):
    # A header-only file has nothing to say, and loading a TTS model to discover that would
    # cost seconds for a guaranteed zero. Asserted with imports blocked, so a regression that
    # moved the short-circuit below the model load fails here rather than merely getting slow.
    _without_the_audio_extra(monkeypatch)

    measured = workspace.measure_spoken_length("---\nspeaker: Ana\n---\n\n[pause]\n")

    assert measured.seconds == 0.0
    assert measured.wav == b""


def test_measured_and_estimated_lengths_describe_the_same_words():
    # The two figures the browser prints side by side must be derived from one corpus, or
    # they differ for a reason the reader cannot see. This pins the shared-corpus property
    # without synthesising anything: both go through `_spoken_text`.
    draft = "---\nspeaker: Ana\n---\n\nGood evening. [pause] Thank you all for coming.\n"
    _, body = workspace._split_front_matter(draft)

    assert "[pause]" not in workspace._spoken_text(body)
    assert workspace.spoken_words(draft) == len(workspace._spoken_text(body).split())


@pytest.mark.skipif(
    not os.environ.get("SPEECHWRITER_TEST_AUDIO"),
    reason="needs `uv sync --extra audio` and downloads a TTS model; set SPEECHWRITER_TEST_AUDIO=1",
)
def test_measured_length_is_in_the_right_ballpark():
    # Opt-in, because it is the one test here that is neither free nor offline: the first run
    # downloads Kokoro. Asserts a *range* rather than a figure -- the point is that the
    # measurement is real and roughly agrees with the words-per-minute estimate, not that a
    # particular voice hits a particular duration.
    words = "Good evening, and thank you all for coming out tonight. " * 10
    measured = workspace.measure_spoken_length(words)

    assert measured.sample_rate > 0
    assert measured.wav.startswith(b"RIFF")
    estimate = workspace.spoken_words(words) / config.WORDS_PER_MINUTE * 60
    assert 0.5 * estimate < measured.seconds < 2.0 * estimate


def test_the_endpoint_field_is_offered_when_nothing_is_configured(monkeypatch, tmp_path):
    # The whole reason this feature exists. Every local-model path was gated on
    # `SPEECHWRITER_BASE_URL` already being set, so the reader it was for — a local server, no
    # Anthropic key, nothing configured — had to edit a dotenv and restart to reach a control
    # whose entire job is sparing them that. Drawn unconditionally, or it is unreachable.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    assert [field.label for field in app.sidebar.text_input] == ["OpenAI-compatible server"]
    assert "Detect models" in [button.label for button in app.sidebar.button]
    # Empty, not prefilled: the placeholder is a hint about the shape of the answer, and putting
    # a guessed URL in the *value* would make Detect probe a port nobody named.
    assert app.sidebar.text_input[0].value == ""
    # And the picker is untouched while nothing has been detected — a bare environment still
    # offers exactly the curated roster, which `model_choices` promises and the picker's index
    # arithmetic depends on.
    assert list(app.sidebar.selectbox[0].options) == [c.label for c in config.MODEL_CHOICES]
    # Nothing was probed on render. The suite is offline by construction and CI renders this
    # page dozens of times per run; asserted through the captions the sidebar only draws once
    # an answer exists, since `webui.detections()` reads this process's session state rather
    # than the rendered app's and would return None either way.
    captions = [caption.value for caption in app.sidebar.caption]
    assert not any("listed no models" in caption for caption in captions), captions
    assert not any("Found" in caption for caption in captions), captions

    # And a value that is not an endpoint says what one looks like. The reader seeing this did
    # not necessarily type it — browsers restore form fields, and a malformed
    # SPEECHWRITER_BASE_URL is seeded verbatim by design — so a bare verdict about text they do
    # not remember writing leaves them with nothing to do. Reported by a reader who hit exactly
    # that: a restored `http://[::1` and "Not an HTTP endpoint."
    app.sidebar.text_input[0].set_value("http://[::1").run()
    complaint = next(c.value for c in app.sidebar.caption if "Not an HTTP endpoint" in c.value)
    assert config.DEFAULT_LOCAL_ENDPOINT in complaint, complaint


def test_a_typed_endpoint_survives_a_rerun_and_reaches_the_picker(monkeypatch, tmp_path):
    # Two reruns, because one cannot see the failure that matters. Streamlit *silently* rewrites
    # a selection absent from a widget's options to option zero, so a detected entry that were
    # not re-offered on the next render would take the reader off the model they just picked
    # with no error anywhere.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()
    monkeypatch.setattr(webui.endpoints, "list_models", lambda url, **kw: ["local/qwen"])

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.sidebar.text_input[0].set_value("127.0.0.1:8080").run()
    # Normalised in place, so the box shows the URL that will actually be asked.
    assert app.sidebar.text_input[0].value == "http://127.0.0.1:8080/v1"

    app.sidebar.button[0].click().run()
    assert "local/qwen (local)" in list(app.sidebar.selectbox[0].options)

    app.sidebar.selectbox[0].select("local/qwen (local)").run()
    assert _picked(app).base_url == "http://127.0.0.1:8080/v1"
    # A keyless machine can now run: the badge gates on `model_credentials_present`, which a
    # local endpoint satisfies without any key of ours.
    assert any("Ready]" in block.value for block in app.markdown)

    app.run()

    assert not app.exception
    assert _picked(app).model == "local/qwen"
    assert _picked(app).base_url == "http://127.0.0.1:8080/v1"


def test_both_front_ends_offer_the_same_roster(monkeypatch, tmp_path):
    # The seam this feature widens. `cli._roster` and `webui.available_choices` compose the same
    # arguments into one `model_choices` call, and nothing structural keeps them doing so — they
    # are two functions in two modules that must agree on *order*, because the REPL prints row
    # numbers a reader types back and the picker resolves `index=choices.index(current)`. They
    # already drifted once: the browser learned to widen the roster with detections and the
    # terminal did not.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    st.cache_resource.clear()

    configured = load_settings()
    detected = [
        config.local_choice("local/granite", "http://127.0.0.1:8080/v1"),
        config.local_choice("local/zephyr", "http://127.0.0.1:1234/v1"),
    ]
    bundle = webui.get_bundle()
    st.session_state[webui.DETECTED_KEY] = detected
    try:
        browser = webui.available_choices(bundle.settings)
    finally:
        st.session_state.pop(webui.DETECTED_KEY, None)
    terminal = cli._roster(configured, bundle, detected)

    assert terminal == browser, "the two front ends disagree about the roster"


def test_detect_says_nothing_about_a_server_it_never_contacted(monkeypatch, tmp_path):
    # `detect_models` wrote `[]` when the field held nothing askable, which looked like the tidy
    # answer and lied: `[]` is the "asked, and the server offered nothing" state, which the
    # sidebar reports as "That endpoint listed no models — is the server running?" about a server
    # that was never contacted. The button is disabled in that state too, so this is the belt to
    # that braces — and it is the half a contributor could remove without the page looking wrong.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    st.cache_resource.clear()
    probes: list[str] = []
    monkeypatch.setattr(webui.endpoints, "list_models", lambda url, **kw: probes.append(url) or [])

    st.session_state[webui.ENDPOINT_KEY] = ""
    try:
        webui.detect_models()
        assert webui.detections() is None, "an empty field recorded a probe that never happened"
    finally:
        st.session_state.pop(webui.ENDPOINT_KEY, None)
        st.session_state.pop(webui.DETECTED_KEY, None)
    assert probes == [], probes

    # And the button says so before the click, rather than taking it and dropping it — the rule
    # `write.py`'s suggestion pills already follow.
    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    detect = next(b for b in app.sidebar.button if b.label == "Detect models")
    assert detect.disabled, "Detect is clickable with no endpoint to ask"


def test_a_no_op_edit_keeps_the_models_already_found(monkeypatch, tmp_path):
    # `apply_endpoint` dropped detections on any text change at all. An edit that lands on the
    # same server — pasting the URL back, a trailing slash, a stray space — then emptied the
    # roster and made the reader sit through another probe (up to DEFAULT_TIMEOUT) of a server
    # that never stopped answering. Only a change of *target* may clear them.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    st.cache_resource.clear()
    found = [config.local_choice("qwen", "http://localhost:8080/v1")]

    st.session_state[webui.ENDPOINT_KEY] = "http://localhost:8080/v1"
    st.session_state[webui.DETECTED_KEY] = found
    try:
        # Same server, spelled differently.
        st.session_state[webui.ENDPOINT_KEY] = "  http://localhost:8080/v1/  "
        webui.apply_endpoint()
        assert st.session_state[webui.ENDPOINT_KEY] == "http://localhost:8080/v1"
        assert webui.detections() == found, "a no-op edit re-probed a server that was answering"

        # A different server does clear them, which is the half that must keep working.
        st.session_state[webui.ENDPOINT_KEY] = "http://localhost:1234/v1"
        webui.apply_endpoint()
        assert webui.detections() is None
    finally:
        st.session_state.pop(webui.ENDPOINT_KEY, None)
        st.session_state.pop(webui.DETECTED_KEY, None)


def test_the_sidebar_names_where_the_running_model_is_served(monkeypatch, tmp_path):
    # The endpoint field holds whatever server Detect is pointed at, which is not necessarily
    # where the *running* model is served — they differ the moment the reader goes looking at a
    # second one. With the old "Serving the selected model." caption gone in that state and the
    # label carrying no host, nothing on the page named the endpoint being called. That is the
    # affordance the CLI banner gives its own line: a local server that is simply not running
    # looks like a hung turn unless the UI said where it pointed.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    captions = [caption.value for caption in app.sidebar.caption]
    assert any("http://127.0.0.1:8080/v1" in caption for caption in captions), captions

    # Still named after the reader points the field somewhere else entirely.
    app.sidebar.text_input[0].set_value("http://127.0.0.1:1234/v1").run()
    captions = [caption.value for caption in app.sidebar.caption]
    assert any("http://127.0.0.1:8080/v1" in caption for caption in captions), captions
