"""Offline tests for the web app and the workspace reader.

Same bargain as ``test_build.py``: building the agent calls neither the model nor the
network, so Streamlit can render the whole app headlessly for free. That is what makes it
worth testing at all — there is no CI here, so a page that raises on import would otherwise
be discovered by a human opening a browser.
"""

from __future__ import annotations

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from streamlit.testing.v1 import AppTest

from speechwriter import config, webui, workspace
from speechwriter.config import load_settings
from speechwriter.prompts import orchestrator_prompt

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
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    words = config.WORDS_PER_MINUTE * 3
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


def test_prompt_points_the_agent_at_the_folder_the_browser_reads(monkeypatch, tmp_path):
    # The cross-subsystem check that matters now that a reader exists: the prompt *tells*
    # the agent where to save drafts, and `workspace.speeches()` reads a real directory.
    # This derives the virtual path from that real directory, so re-hardcoding either side
    # fails here rather than silently producing a browser that lists nothing.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    read_dir = settings.workspace_dir / config.SPEECHES_SUBDIR
    as_the_agent_sees_it = "/" + read_dir.relative_to(settings.project_root).as_posix()

    # The trailing slash is load-bearing: without it this passes on a prompt that says
    # `/workspace/speeches-drafts/`, which is exactly the drift being guarded against.
    assert f"{as_the_agent_sees_it}/" in orchestrator_prompt(settings)


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
    # The bundle is cached across the process, so a leftover from another test would pin
    # this run to the wrong SPEECHWRITER_HOME.
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    # Missing key must be reported in the page, not crash it — the workspace stays browsable.
    assert any("ANTHROPIC_API_KEY" in error.value for error in app.error)


def test_both_pages_render(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
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
    assert any("ana-toast" in str(option) for option in app.selectbox[0].options)


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
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    st.cache_resource.clear()

    bundle = webui.get_bundle()
    bundle.store.put(("speechwriter", "memories"), "mayor.md", {"content": "warm, plainspoken"})

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.switch_page("app_pages/browse.py").run()
    app.segmented_control[0].set_value("Memory").run()

    assert not app.exception
    assert any("warm, plainspoken" in block.value for block in app.markdown)
