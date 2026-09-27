"""Trace every turn to a self-hosted Phoenix, over plain OpenTelemetry.

Off unless ``PHOENIX_COLLECTOR_ENDPOINT`` names a collector. When it does, :func:`enable_tracing`
instruments LangChain once per process, so every model call, tool call and subagent run in the
graph becomes a span — orchestrator and subagents alike, since LangChain callbacks propagate
down it. Spans are grouped into Phoenix *sessions* by ``thread_id`` for free: LangGraph copies
it into run metadata, and OpenInference reads it from there. So a CLI thread rotated after an
interrupt, or a browser conversation reset, starts a new session exactly when the agent does.

This replaced LangSmith tracing, which was driven entirely by ``LANGSMITH_*`` environment
variables and had no code here at all. The eval harness followed: its datasets are mirrored to,
and its experiments recorded in, the same Phoenix (``evals/sync_datasets.py``, ``--phoenix`` on
``evals/run_experiment.py``), found through the same variable via :func:`server_url`.

Three decisions are load-bearing, and each was measured rather than assumed.

**No ``phoenix.otel.register()``.** It is the documented one-liner, and against the
OpenTelemetry this project resolves (1.45) it raises ``AttributeError`` on every call: it reads
the exporter's private ``_headers``, which 1.45 removed. It also walks *up* from the working
directory for a ``.env.phoenix`` file to take credentials from — the upward walk
``load_settings()`` refuses for the project's own dotenv — prints a banner to stdout, and pulls
in ``grpcio`` for a transport this never uses. What it does is four lines of the public SDK, so
those four lines are here instead, configured only from :class:`~speechwriter.config.Settings`.

**Batched, with a short export timeout.** ``register()`` defaults to a *synchronous* processor,
which would put a network round trip inside every span of a turn. Batched, a turn never waits
on the collector — but the final flush at exit does, and against a Phoenix that is not running
the SDK's default 10s retry window held ``exit`` for 7.5s (measured). A local collector answers
in milliseconds, so :data:`EXPORT_TIMEOUT_SECONDS` bounds that wait instead.

**One warning per outage, not four lines per batch.** A failed export makes the OTLP exporter
log each retry, and a batch leaves every five seconds during a turn — measured at ~4 lines per
batch, printed straight into the REPL's streamed transcript. :class:`_ReportingExporter` lets
the first failing batch's own diagnostics through (they carry the reason: refused, 401, …),
adds one line saying what to do, and then keeps the exporter quiet until a batch lands again.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from speechwriter import endpoints
from speechwriter.config import Settings

logger = logging.getLogger(__name__)

# How long one batch may spend reaching the collector, retries included — and so the most a
# down collector can hold up the flush at exit. See the module docstring for the measurement.
EXPORT_TIMEOUT_SECONDS = 3.0

# OTLP over HTTP. Phoenix also takes gRPC on 4317, which would need `grpcio` for nothing: the
# HTTP collector is on the same port as the UI, so one URL is both what to open and where to send.
_TRACES_PATH = "/v1/traces"

# Where the OTLP HTTP exporter logs every retry and failure. Named rather than derived from
# `OTLPSpanExporter.__module__` because the retry loop logs through a logger the exporter hands
# it, and that handing-over is what this spells.
_EXPORTER_LOGGER = "opentelemetry.exporter.otlp.proto.http.trace_exporter"

_SERVICE = "speechwriter-agent"


@dataclass(frozen=True)
class Tracing:
    """Where this process is sending traces. Exists only while tracing is actually on."""

    endpoint: str  # as configured — the Phoenix UI answers here too
    project: str
    provider: TracerProvider

    @property
    def label(self) -> str:
        """One line for a banner: which Phoenix project, and where to open it."""
        return f"Phoenix · {self.project} · {self.endpoint}"


def collector_url(endpoint: str) -> str | None:
    """Where spans are POSTed for ``PHOENIX_COLLECTOR_ENDPOINT``, or ``None`` if it is no URL.

    Phoenix's own convention, followed rather than invented: the variable names the server, and
    ``/v1/traces`` is appended while any path prefix is kept, so a Phoenix behind a reverse
    proxy at ``http://host/phoenix`` receives at ``http://host/phoenix/v1/traces``. A value
    already ending in ``/v1/traces`` is used as written.

    The shape check is :func:`~speechwriter.endpoints.usable_endpoint`'s, for the reason it
    exists: a missing scheme or host fails at *export*, on a background thread, far from the
    typo — and ``file://`` has no business being a place drafts are sent.
    """
    usable = endpoints.usable_endpoint(endpoint)
    if usable is None:
        return None
    parts = urlsplit(usable)
    path = parts.path.rstrip("/")
    if not path.endswith(_TRACES_PATH):
        path += _TRACES_PATH
    return urlunsplit(parts._replace(path=path))


def server_url(endpoint: str) -> str | None:
    """The Phoenix *server* ``PHOENIX_COLLECTOR_ENDPOINT`` names, for its REST API.

    :func:`collector_url` run backwards: the eval harness talks to the same Phoenix the traces
    go to, so one variable answers both questions. A value written as the OTLP path has
    ``/v1/traces`` removed — the API lives at the server's root, and a client handed the
    collector URL would ask for ``/v1/traces/v1/datasets``. A proxy prefix is kept, as there.
    """
    usable = endpoints.usable_endpoint(endpoint)
    if usable is None:
        return None
    parts = urlsplit(usable)
    path = parts.path.rstrip("/")
    if path.endswith(_TRACES_PATH):
        path = path[: -len(_TRACES_PATH)]
    return urlunsplit(parts._replace(path=path))


_lock = threading.Lock()
_active: Tracing | None = None
_quieter: _QuietWhileReported | None = None


def enable_tracing(settings: Settings) -> Tracing | None:
    """Start sending traces to Phoenix if ``settings`` names a collector; what is now in force.

    Called by :func:`~speechwriter.agent.build_agent`, so the CLI, the web UI, the eval harness
    and library consumers are all traced alike — the property LangSmith's environment variables
    had, kept. Safe to call on every build: tracing is process-wide (LangChain is instrumented
    globally), so the first call that enables it wins and later calls report it.

    **Opens no socket.** The exporter connects when its first batch leaves, never here, so the
    offline-build invariant holds with tracing on.

    **Never raises.** A malformed endpoint, or anything the SDK objects to, is a log line and a
    ``None``: a typo in an observability setting must not stop a speech from being written.
    """
    global _active, _quieter
    with _lock:
        if _active is not None or settings.phoenix_endpoint is None:
            return _active

        url = collector_url(settings.phoenix_endpoint)
        if url is None:
            logger.warning(
                "Not tracing: PHOENIX_COLLECTOR_ENDPOINT=%r needs an http(s) scheme and a host, "
                "e.g. http://localhost:6006.",
                settings.phoenix_endpoint,
            )
            return None

        instrumentor = LangChainInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            # Instrumenting again would be a silent no-op, leaving our provider with no spans
            # while a banner said otherwise. Whoever got there first owns LangChain's traces.
            logger.warning(
                "Not tracing to Phoenix: LangChain is already instrumented by another "
                "OpenTelemetry setup in this process."
            )
            return None

        # Sent only to the collector the same operator named; there is no default endpoint for
        # a key to leak to, which is also why tracing is off rather than aimed at localhost.
        headers = (
            {"authorization": f"Bearer {settings.phoenix_api_key}"}
            if settings.phoenix_api_key
            else None
        )
        try:
            exporter = _ReportingExporter(
                OTLPSpanExporter(endpoint=url, headers=headers, timeout=EXPORT_TIMEOUT_SECONDS),
                settings.phoenix_endpoint,
            )
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        SERVICE_NAME: _SERVICE,
                        ResourceAttributes.PROJECT_NAME: settings.phoenix_project,
                    }
                )
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            # Our own provider, never installed as the global one: a library consumer's
            # OpenTelemetry setup keeps whatever provider it already had.
            instrumentor.instrument(tracer_provider=provider)
        except Exception:
            logger.warning("Not tracing: could not set up the Phoenix exporter.", exc_info=True)
            return None

        _quieter = _QuietWhileReported(exporter)
        logging.getLogger(_EXPORTER_LOGGER).addFilter(_quieter)
        _warn_if_langsmith_still_traces()

        _active = Tracing(settings.phoenix_endpoint, settings.phoenix_project, provider)
        logger.info("Tracing to %s (project %s).", url, settings.phoenix_project)
        return _active


def disable_tracing() -> None:
    """Undo :func:`enable_tracing`: stop instrumenting LangChain, then flush and close.

    For tests, which must not leave a process-wide patch behind for the next one, and for a
    long-lived host that wants its traces delivered before it moves on. A process that simply
    exits needs neither — the provider flushes itself at exit.
    """
    global _active, _quieter
    with _lock:
        if _active is None:
            return
        LangChainInstrumentor().uninstrument()
        _active.provider.shutdown()
        if _quieter is not None:
            logging.getLogger(_EXPORTER_LOGGER).removeFilter(_quieter)
        _active = None
        _quieter = None


class _ReportingExporter(SpanExporter):
    """Delegates to the OTLP exporter, and says once — not once per batch — that it failed."""

    def __init__(self, inner: SpanExporter, endpoint: str) -> None:
        self._inner = inner
        self._endpoint = endpoint
        # Read by `_QuietWhileReported` from the logging call sites inside `export`; written only
        # by the batch processor's single worker thread (and by shutdown, after it has joined).
        self.reported = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = self._inner.export(spans)
        if result is SpanExportResult.FAILURE:
            if not self.reported:
                self.reported = True
                logger.warning(
                    "Phoenix at %s is not accepting traces, so they are being dropped until it "
                    "does. Start it (see README: Tracing with Phoenix), or unset "
                    "PHOENIX_COLLECTOR_ENDPOINT to stop tracing.",
                    self._endpoint,
                )
        elif self.reported:
            self.reported = False
            logger.warning("Phoenix at %s is accepting traces again.", self._endpoint)
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


class _QuietWhileReported(logging.Filter):
    """Drop the exporter's own retry chatter once :class:`_ReportingExporter` has spoken.

    Keyed on state rather than on message text, so the first failing batch's diagnostics —
    which name the actual reason — still get through, and a wording change upstream cannot
    turn this into a filter that silently swallows something new.
    """

    def __init__(self, exporter: _ReportingExporter) -> None:
        super().__init__()
        self._exporter = exporter

    def filter(self, record: logging.LogRecord) -> bool:
        return not self._exporter.reported


def _warn_if_langsmith_still_traces() -> None:
    """Say so when LangSmith tracing is left on beside Phoenix — every turn would go to both.

    Phoenix *replaced* LangSmith here, but LangSmith's tracer is switched on by environment
    alone (``langsmith`` arrives with ``langchain-core``), so a dotenv written for the old setup
    keeps shipping every draft to a hosted service with nothing in this code asking it to.
    """
    names = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
    on = [name for name in names if os.environ.get(name, "").strip().lower() == "true"]
    if on:
        logger.warning(
            "%s is still set, so traces go to LangSmith as well as Phoenix. Remove it to trace "
            "to Phoenix only.",
            " / ".join(on),
        )
