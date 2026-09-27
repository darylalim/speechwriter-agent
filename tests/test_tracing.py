"""Tracing to a self-hosted Phoenix — offline, like the rest of the suite.

No Phoenix runs here. The collector is a throwaway loopback HTTP server that records what
arrives, which is stricter than a Phoenix would be: it asserts the wire — the path, the bearer,
and the project and session inside the protobuf body — rather than whatever a UI chose to show.

Tracing is process-wide (LangChain is instrumented globally), so every test here goes through
the ``home`` fixture, whose teardown calls ``disable_tracing()``. A test that left tracing on
would have every later test in the process exporting spans at a dead port.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith.utils import get_env_var
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from rich.console import Console
from streamlit.testing.v1 import AppTest

from speechwriter import cli, config, tracing
from speechwriter.agent import build_agent

_REPO_ROOT = config._PKG_DIR.parents[1]

_PHOENIX_VARS = (
    "PHOENIX_COLLECTOR_ENDPOINT",
    "PHOENIX_PROJECT",
    "PHOENIX_PROJECT_NAME",
    "PHOENIX_API_KEY",
)
_LANGSMITH_SWITCHES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")

# Port 9 is `discard`, which nothing on a development machine or a CI runner listens on — a
# collector that is guaranteed to refuse. Used only where no span is ever exported.
_DEAD_COLLECTOR = "http://127.0.0.1:9"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An isolated home with nothing traced, and no tracing left behind afterwards."""
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # Pins the documented default pair, as every test that builds a model does.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    for name in (*_PHOENIX_VARS, *_LANGSMITH_SWITCHES):
        # A developer's shell may export any of these, and real env wins over the dotenv.
        monkeypatch.delenv(name, raising=False)
    yield tmp_path
    tracing.disable_tracing()


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
def _collector():
    """A loopback OTLP/HTTP collector that keeps every request it is sent."""
    received: list[tuple[str, str | None, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's own spelling
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.path, self.headers.get("Authorization"), body))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

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


def _attributes(attributes) -> dict[str, str]:
    return {attr.key: attr.value.string_value for attr in attributes}


def test_tracing_is_off_until_a_collector_is_named(home):
    # No default endpoint, on purpose: a default would ship every draft to whatever holds a
    # well-known port without anyone having asked for traces. Unset has to mean untouched —
    # including LangChain itself, which is patched process-wide once tracing is on.
    bundle = build_agent()

    assert bundle.tracing is None
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


def test_a_turn_reaches_the_collector_with_its_project_session_and_key(home, monkeypatch):
    # The end-to-end property, asserted on the wire. Three things must arrive together for a
    # trace to be useful in Phoenix: the project (resource attribute) so it lands where the
    # reader looks, the session (span attribute) so one conversation reads as one, and the
    # model call itself as an LLM span — the thing a reader opens a trace to see.
    monkeypatch.setattr("speechwriter.agent._build_model", lambda _settings: _Stub())
    with _collector() as (endpoint, received):
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)
        monkeypatch.setenv("PHOENIX_PROJECT", "speech-tests")
        monkeypatch.setenv("PHOENIX_API_KEY", "px-test")

        bundle = build_agent()
        assert bundle.tracing is not None
        bundle.agent.invoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            config=bundle.turn_config("thread-under-test"),
        )
        assert bundle.tracing.provider.force_flush(10_000), "spans never left the processor"

    assert received, "the collector never heard from the exporter"
    # Phoenix's convention: the variable names the server, `/v1/traces` is where OTLP goes.
    assert {path for path, _, _ in received} == {"/v1/traces"}
    assert {auth for _, auth, _ in received} == {"Bearer px-test"}

    requests = [ExportTraceServiceRequest.FromString(body) for _, _, body in received]
    resources = [
        _attributes(rs.resource.attributes) for req in requests for rs in req.resource_spans
    ]
    spans = [
        _attributes(span.attributes)
        for req in requests
        for rs in req.resource_spans
        for scope in rs.scope_spans
        for span in scope.spans
    ]
    assert {r.get("openinference.project.name") for r in resources} == {"speech-tests"}
    # Every span, not just some: LangGraph copies `thread_id` into run metadata and OpenInference
    # reads the session from there, so a span without it would be an orphan in the Sessions view.
    assert {s.get("session.id") for s in spans} == {"thread-under-test"}
    assert "LLM" in {s.get("openinference.span.kind") for s in spans}, spans


def test_building_with_tracing_on_opens_no_socket(home, monkeypatch):
    # The offline invariant, with tracing on. The exporter connects when its first batch
    # leaves; if setup ever started probing the collector, every build — and so the whole
    # suite, and CI — would depend on a Phoenix being up.
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)
    connected: list[object] = []
    real_connect = socket.socket.connect

    def spy(sock, address):
        connected.append(address)
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", spy)

    bundle = build_agent()

    assert bundle.tracing is not None
    assert connected == [], f"building with tracing on connected to {connected}"


def test_the_exporter_is_batched_and_bounded(home, monkeypatch):
    # Both are measured choices, and both fail silently. A synchronous processor — the default
    # of the `register()` this module declined to use — puts a network round trip inside every
    # span of a turn. And the SDK's own 10s retry window held `exit` for 7.5s against a Phoenix
    # that was not running; a local collector answers in milliseconds.
    exporters: list[dict[str, object]] = []
    processors: list[object] = []

    class Exporter(tracing.OTLPSpanExporter):
        def __init__(self, **kwargs):
            exporters.append(kwargs)
            super().__init__(**kwargs)

    class Processor(tracing.BatchSpanProcessor):
        def __init__(self, exporter, **kwargs):
            processors.append(exporter)
            super().__init__(exporter, **kwargs)

    monkeypatch.setattr(tracing, "OTLPSpanExporter", Exporter)
    monkeypatch.setattr(tracing, "BatchSpanProcessor", Processor)
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)

    assert build_agent().tracing is not None

    assert len(processors) == 1, "spans are no longer batched"
    [exporter] = exporters
    assert exporter["timeout"] == tracing.EXPORT_TIMEOUT_SECONDS
    assert tracing.EXPORT_TIMEOUT_SECONDS < 10, "no tighter than the SDK default it replaces"
    # No key configured, no header sent — not an empty bearer.
    assert exporter["headers"] is None


def test_tracing_is_enabled_once_and_reported_by_every_later_bundle(home, monkeypatch):
    # Tracing is process-wide, so a bundle reports what is in force rather than what its own
    # settings asked for. A model switch rebuilds the bundle; it must neither instrument
    # LangChain a second time nor start claiming the process is untraced.
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)
    first = build_agent()
    # A model switch: same environment, fresh bundle. The case a first version of this test
    # skipped — it only rebuilt with the variable gone, which passed with the once-only guard
    # deleted, because the rebuild that would have tripped over our own instrumentation and
    # reported "off" after every switch never happened.
    switched = build_agent()
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT")
    unset = build_agent()

    assert first.tracing is not None
    assert switched.tracing is first.tracing
    assert unset.tracing is first.tracing


def test_a_langchain_someone_else_instrumented_is_not_claimed(home, monkeypatch, caplog):
    # A library consumer may run its own OpenInference setup. Instrumenting again is a silent
    # no-op upstream, so our provider would receive nothing while the banner said "tracing to
    # Phoenix" — a signal that has stopped running but still reads as passing.
    theirs = tracing.TracerProvider()
    LangChainInstrumentor().instrument(tracer_provider=theirs)
    try:
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)
        with caplog.at_level(logging.WARNING, logger="speechwriter.tracing"):
            bundle = build_agent()
    finally:
        LangChainInstrumentor().uninstrument()

    assert bundle.tracing is None
    assert "already instrumented" in caplog.text


def test_a_malformed_collector_is_a_warning_not_a_crash(home, monkeypatch, caplog):
    # A typo in an observability setting must not stop a speech from being written. The two
    # spellings a reader actually types without a scheme read differently to `urlsplit`, which
    # is why the check is `usable_endpoint`'s rather than a scheme test of its own.
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "localhost:6006")

    with caplog.at_level(logging.WARNING, logger="speechwriter.tracing"):
        bundle = build_agent()

    assert bundle.tracing is None
    assert "Not tracing" in caplog.text and "localhost:6006" in caplog.text
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry


def test_the_collector_url_follows_phoenix_convention_and_never_raises():
    expected = {
        "http://localhost:6006": "http://localhost:6006/v1/traces",
        "http://localhost:6006/": "http://localhost:6006/v1/traces",
        # A Phoenix behind a reverse proxy keeps its prefix.
        "https://tools.example/phoenix": "https://tools.example/phoenix/v1/traces",
        # Already the OTLP path: used as written, not doubled.
        "http://localhost:6006/v1/traces": "http://localhost:6006/v1/traces",
        "http://localhost:6006/v1/traces/": "http://localhost:6006/v1/traces",
        "  http://localhost:6006  ": "http://localhost:6006/v1/traces",
        "localhost:6006": None,
        "127.0.0.1:6006": None,
        "file:///tmp/spans": None,
        "http://[::1": None,
        "": None,
    }
    for raw, want in expected.items():
        assert tracing.collector_url(raw) == want, raw


def test_the_server_url_is_the_collector_url_run_backwards():
    # The eval harness finds Phoenix's REST API through the same variable tracing uses, so a
    # reader who wrote the OTLP path into it must still reach the API at the server's root --
    # a client handed the collector URL asks for /v1/traces/v1/datasets and gets a 404.
    expected = {
        "http://localhost:6006": "http://localhost:6006",
        "http://localhost:6006/": "http://localhost:6006",
        "http://localhost:6006/v1/traces": "http://localhost:6006",
        "http://localhost:6006/v1/traces/": "http://localhost:6006",
        # A reverse-proxy prefix is kept, exactly as collector_url keeps it.
        "https://tools.example/phoenix/v1/traces": "https://tools.example/phoenix",
        "localhost:6006": None,
        "file:///tmp/spans": None,
        "http://[::1": None,
        "": None,
    }
    for raw, want in expected.items():
        assert tracing.server_url(raw) == want, raw
        if want is not None:
            # And the two stay inverse: from either spelling, the collector is the same place.
            assert tracing.collector_url(want) == tracing.collector_url(raw), raw


def test_a_dead_collector_is_reported_once_per_outage(caplog):
    # A batch leaves every five seconds during a turn, and a failed one makes the OTLP exporter
    # log each retry — measured at ~4 lines a batch, printed into the REPL's transcript. The
    # first failing batch keeps its own diagnostics (they name the reason); after that the
    # exporter is quiet until a batch lands, and the next outage is reported afresh.
    chatter = logging.getLogger(tracing._EXPORTER_LOGGER)

    class Scripted(SpanExporter):
        def __init__(self, results):
            self.results = list(results)

        def export(self, spans):
            result = self.results.pop(0)
            if result is SpanExportResult.FAILURE:
                chatter.warning("Transient error: connection refused")
            return result

    fail, ok = SpanExportResult.FAILURE, SpanExportResult.SUCCESS
    exporter = tracing._ReportingExporter(Scripted([fail, fail, fail, ok, fail]), _DEAD_COLLECTOR)
    quiet = tracing._QuietWhileReported(exporter)
    chatter.addFilter(quiet)
    try:
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                exporter.export([])
    finally:
        chatter.removeFilter(quiet)

    ours = [r.getMessage() for r in caplog.records if r.name == "speechwriter.tracing"]
    theirs = [r for r in caplog.records if r.name == tracing._EXPORTER_LOGGER]
    assert sum("not accepting traces" in m for m in ours) == 2, ours  # two outages
    assert sum("accepting traces again" in m for m in ours) == 1, ours
    # The first batch of each outage speaks; the two repeats of the first outage do not.
    assert len(theirs) == 2, [r.getMessage() for r in theirs]


def test_langsmith_left_on_beside_phoenix_is_called_out(home, monkeypatch, caplog):
    # Phoenix replaced LangSmith, but LangSmith's tracer switches itself on from the
    # environment alone — so a dotenv written for the old setup keeps sending every draft to a
    # hosted service with nothing in this code asking it to. Say so where the reader will see it.
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    try:
        with caplog.at_level(logging.WARNING, logger="speechwriter.tracing"):
            assert build_agent().tracing is not None
    finally:
        # langsmith memoises env reads; never leave "tracing on" cached for a later test's turn.
        cache_clear = getattr(get_env_var, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    assert "LANGSMITH_TRACING" in caplog.text and "as well as Phoenix" in caplog.text


def test_the_banner_says_where_traces_go(home):
    # Both front ends print where the agent is pointed before a turn is spent; "am I tracing,
    # and to which project" is the same kind of fact as "which endpoint".
    bundle = build_agent()
    console = Console(record=True, width=200)

    cli._banner(console, bundle)
    # Matched as a row, not as a word: `research   off` is on the banner too. (`export_text`
    # also clears the recording, so it is read exactly once per render.)
    assert re.search(r"traces\s+off", console.export_text())

    on = dataclasses.replace(
        bundle,
        tracing=tracing.Tracing(
            "http://localhost:6006", "speechwriter-agent", tracing.TracerProvider()
        ),
    )
    console = Console(record=True, width=200)
    cli._banner(console, on)
    assert "Phoenix · speechwriter-agent · http://localhost:6006" in console.export_text()


def test_the_sidebar_says_where_traces_go(home, monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", _DEAD_COLLECTOR)
    st.cache_resource.clear()

    app = AppTest.from_file(str(_REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()

    assert not app.exception
    traces = [c.value for c in app.sidebar.caption if c.value.startswith("Traces")]
    assert traces == [f"Traces — `Phoenix · speechwriter-agent · {_DEAD_COLLECTOR}`"], traces
