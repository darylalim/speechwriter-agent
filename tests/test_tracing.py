"""Tracing to LangSmith — offline, like the rest of the suite.

No LangSmith is contacted. The endpoint is a throwaway loopback HTTP server that records what
arrives, which is stricter than the hosted service would be: it asserts the wire — the path, the
API key, and the project and thread inside the multipart body — rather than whatever a UI chose
to show.

LangSmith keeps process-wide state in three places, and every test here resets all three through
the ``home`` fixture: the ``LANGSMITH_*`` variables (monkeypatched), langsmith's ``lru_cache``d
env reads (cleared by ``load_settings`` and again at teardown), and the cached client in
``langsmith.run_trees`` (reset, so no test inherits an earlier test's loopback endpoint).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import langsmith.run_trees
import pytest
import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import utils as langsmith_utils
from rich.console import Console
from streamlit.testing.v1 import AppTest

from speechwriter import cli, config, tracing
from speechwriter.agent import build_agent

_REPO_ROOT = config._PKG_DIR.parents[1]

_LANGSMITH_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


def _clear_langsmith_caches() -> None:
    for cached in (langsmith_utils.get_env_var, langsmith_utils.get_tracer_project):
        cache_clear = getattr(cached, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An isolated home with nothing traced, and no tracing state left behind afterwards."""
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    for name in _LANGSMITH_VARS:
        # A developer's shell may export any of these, and real env wins over the dotenv.
        monkeypatch.delenv(name, raising=False)
    # Private, and deliberately so: the tracer's client is a module global, and one built
    # against an earlier test's loopback port would otherwise receive this test's runs.
    monkeypatch.setattr(langsmith.run_trees, "_CLIENT", None)
    yield tmp_path
    # monkeypatch restores the variables after this; the caches would still hold them.
    monkeypatch.undo()
    _clear_langsmith_caches()


class _Stub(BaseChatModel):
    """Answers once with no tool calls, so a turn is one model call and then ends."""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


@contextlib.contextmanager
def _langsmith():
    """A loopback LangSmith that answers everything with ``{}`` and keeps every request."""
    received: list[tuple[str, str, dict[str, str], bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def _record(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            received.append((self.command, self.path, dict(self.headers), body))
            payload = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PATCH = _record  # noqa: N815 - BaseHTTPRequestHandler's spelling

        def log_message(self, format, *args):  # noqa: A002 - the base class's own parameter name
            """Silence the default stderr access log."""

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tracing_is_off_until_langsmith_tracing_is_set(home):
    # Unset means off: no label, nothing for the front ends to print, and no client built.
    bundle = build_agent()

    assert bundle.tracing is None
    assert langsmith.run_trees._CLIENT is None


def test_a_turn_reaches_langsmith_with_its_project_thread_and_key(home, monkeypatch):
    # The whole feature on the wire, because none of it is our code and so none of it can be
    # asserted any other way: LangChain starts the tracer from the environment, LangGraph copies
    # `thread_id` into run metadata, and the client uploads both with the key. A CLI thread
    # rotated after an interrupt, or a browser "New conversation", must start a new LangSmith
    # thread exactly when the agent does — which is only true while `thread_id` reaches the run.
    monkeypatch.setattr("speechwriter.agent._build_model", lambda settings: _Stub())
    with _langsmith() as (endpoint, received):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", endpoint)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-dummy")
        monkeypatch.setenv("LANGSMITH_PROJECT", "wire-test")

        bundle = build_agent()
        assert bundle.tracing is not None
        bundle.agent.invoke(
            {"messages": [{"role": "user", "content": "A toast, please."}]},
            config=bundle.turn_config("thread-wire"),
        )
        tracing.flush_traces()
        langsmith.run_trees.get_cached_client().flush()

    uploads = [
        (path, headers, body) for method, path, headers, body in received if method == "POST"
    ]
    assert uploads, f"no run reached LangSmith: {[(m, p) for m, p, _, _ in received]}"
    assert all(path.startswith("/runs") for path, _, _ in uploads), uploads
    assert all(
        {k.lower(): v for k, v in headers.items()}.get("x-api-key") == "lsv2-dummy"
        for _, headers, _ in uploads
    )
    body = b"".join(body for _, _, body in uploads)
    assert b"wire-test" in body, "the runs did not name the configured project"
    assert b"thread-wire" in body, "thread_id did not reach the run, so threads will not group"


def test_the_project_defaults_to_this_agents_own(home):
    # LangSmith's own default is a project literally called "default", shared by everything a
    # reader traces without naming one. `load_settings` puts ours in its place, and clears
    # langsmith's cache so a read that happened before the dotenv loaded cannot pin "default".
    #
    # Poisoned first, or this passes vacuously: `get_tracer_project` is `lru_cache`d, and a
    # read of the *unset* variable — a module-level client, an import-time check — is exactly
    # what would otherwise stick for the life of the process.
    assert langsmith_utils.get_tracer_project() == "default"

    config.load_settings()

    assert langsmith_utils.get_tracer_project() == config.DEFAULT_LANGSMITH_PROJECT


def test_a_named_project_is_never_overridden(home, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-dummy")
    monkeypatch.setenv("LANGSMITH_PROJECT", "theirs")

    bundle = build_agent()

    assert bundle.tracing is not None
    assert bundle.tracing.project == "theirs"


def test_building_with_tracing_on_opens_no_socket(home, monkeypatch):
    # The offline-build invariant, with tracing on: reporting it reads the environment only.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-dummy")
    opened: list[object] = []
    real_connect = socket.socket.connect

    def spy(self, address):
        opened.append(address)
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    bundle = build_agent()

    assert bundle.tracing is not None
    assert opened == [], f"building the agent opened {opened}"


def test_tracing_without_a_key_is_reported_not_hidden(home, monkeypatch, caplog):
    # LangChain still starts a tracer, and every upload is rejected — a warning per batch, far
    # from the cause. The label is where the reader looks before a turn, so it says so there.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    with caplog.at_level(logging.WARNING, logger="speechwriter.tracing"):
        bundle = build_agent()

    assert bundle.tracing is not None
    assert bundle.tracing.has_api_key is False
    assert "no LANGSMITH_API_KEY" in bundle.tracing.label
    assert "LANGSMITH_API_KEY is not set" in caplog.text


def test_the_banner_says_where_traces_go(home):
    # Both front ends print where the agent is pointed before a turn is spent; "am I tracing,
    # and to which project" is the same kind of fact as "which model".
    bundle = build_agent()
    console = Console(record=True, width=200)

    cli._banner(console, bundle)
    # Matched as a row, not as a word: `research   off` is on the banner too. (`export_text`
    # also clears the recording, so it is read exactly once per render.)
    assert re.search(r"traces\s+off", console.export_text())

    on = dataclasses.replace(
        bundle,
        tracing=tracing.Tracing(
            "speechwriter-agent", "https://api.smith.langchain.com", has_api_key=True
        ),
    )
    console = Console(record=True, width=200)
    cli._banner(console, on)
    assert "LangSmith · speechwriter-agent" in console.export_text()


def test_a_self_hosted_endpoint_is_named_on_the_label(home):
    hosted = tracing.Tracing("p", "https://api.smith.langchain.com", has_api_key=True)
    local = tracing.Tracing("p", "http://langsmith.internal:1984", has_api_key=True)

    assert hosted.label == "LangSmith · p"
    assert local.label == "LangSmith · p · http://langsmith.internal:1984"


def test_the_sidebar_says_where_traces_go(home, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-dummy")
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    traces = [c.value for c in app.sidebar.caption if c.value.startswith("Traces")]
    assert traces == [f"Traces — `LangSmith · {config.DEFAULT_LANGSMITH_PROJECT}`"], traces


def test_flushing_never_raises_when_nothing_is_traced(home):
    # Called from the CLI's exit path beside the memory save; it must be a no-op, not an error,
    # on the ordinary untraced run.
    tracing.flush_traces()
