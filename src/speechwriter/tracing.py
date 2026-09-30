"""Report — and flush — the LangSmith tracing LangChain does on its own.

LangSmith tracing needs no code to *happen*: LangChain's callback manager starts a
``LangChainTracer`` for every run whenever ``LANGSMITH_TRACING=true`` is in the environment, and
the tracer uploads to ``LANGSMITH_ENDPOINT`` (the hosted API by default) with
``LANGSMITH_API_KEY``. Every model call, tool call and subagent run in the graph becomes a run in
the trace, orchestrator and subagents alike, since callbacks propagate down it. Runs are grouped
into LangSmith *threads* by ``thread_id`` for free: LangGraph copies it into run metadata, and
LangSmith groups on that key — so a CLI thread rotated after an interrupt, or a browser
conversation reset, starts a new thread exactly when the agent does.

What this module adds is the part LangChain does not: **saying so**. :func:`current_tracing`
reports what is in force so both front ends can print it before a turn is spent — the banner's
``traces`` row and the sidebar's **Traces** caption — and :func:`flush_traces` waits for queued
runs before a short-lived process exits, because the tracer uploads on a background thread.

It replaced a Phoenix exporter over plain OpenTelemetry, which is why the front ends already had
a place to print this. The eval harness followed back too: datasets are mirrored to, and
experiments recorded in, the same LangSmith workspace (``evals/sync_datasets.py``,
``--langsmith`` on ``evals/run_experiment.py``).

Two things are load-bearing:

**The environment is read through langsmith's own caches, after they are cleared.** Both
``get_env_var`` and ``get_tracer_project`` are ``lru_cache``d, so the first read of a variable
sticks for the life of the process — including a read of "unset" that happened before the dotenv
was loaded. :func:`speechwriter.config.load_settings` clears both right after loading it, so by
the time anything here runs, what langsmith reports is what the dotenv says.

**A switched-on tracer with no key is reported, not hidden.** LangChain will still start a tracer
and every upload will be rejected, a warning per batch, far from the cause. The label says so
instead, on the one line the reader looks at before a turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from speechwriter.config import Settings

logger = logging.getLogger(__name__)

# LangSmith's hosted API, which the client uses when `LANGSMITH_ENDPOINT` is unset. Named so the
# label can say "hosted" rather than printing the default URL at every reader.
_HOSTED_ENDPOINT = "https://api.smith.langchain.com"


@dataclass(frozen=True)
class Tracing:
    """Where this process's LangChain runs are being traced. Exists only while tracing is on."""

    project: str
    endpoint: str
    has_api_key: bool

    @property
    def label(self) -> str:
        """One line for a banner: which LangSmith project, where, and whether it can upload."""
        where = "" if self.endpoint.rstrip("/") == _HOSTED_ENDPOINT else f" · {self.endpoint}"
        missing = "" if self.has_api_key else " · no LANGSMITH_API_KEY, uploads will be rejected"
        return f"LangSmith · {self.project}{where}{missing}"


def current_tracing(settings: Settings) -> Tracing | None:
    """What LangSmith tracing is in force for this process, or ``None`` when it is off.

    Called by :func:`~speechwriter.agent.build_agent`, so the CLI, the web UI, the eval harness
    and library consumers all report it alike. It opens no socket and constructs no client, so
    the offline-build invariant holds with tracing on.

    ``settings`` is taken for the shape of the call rather than read: tracing is LangSmith's
    process-wide environment contract, and a second copy of it on ``Settings`` would be one more
    place for the two to disagree. What it guarantees is ordering — a ``Settings`` exists only
    after :func:`~speechwriter.config.load_settings` has loaded the dotenv and cleared
    langsmith's caches, which is what makes the reads below mean what the dotenv says.
    """
    del settings  # see the docstring: the argument orders the call, it carries no data
    from langsmith import utils

    if utils.tracing_is_enabled() is not True:
        return None
    tracing = Tracing(
        project=utils.get_tracer_project() or "default",
        endpoint=utils.get_env_var("ENDPOINT", default=_HOSTED_ENDPOINT) or _HOSTED_ENDPOINT,
        has_api_key=bool(utils.get_env_var("API_KEY")),
    )
    if not tracing.has_api_key:
        logger.warning(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set: every trace upload to "
            "%s will be rejected. Set the key, or unset LANGSMITH_TRACING.",
            tracing.endpoint,
        )
    return tracing


def flush_traces() -> None:
    """Wait for queued runs to reach LangSmith. Call before a short-lived process exits.

    The tracer uploads on a background thread, so a CLI session that ends right after its last
    turn could otherwise exit with that turn's runs still queued. Never raises: a trace that
    could not be delivered is not a reason to lose the exit-time memory save beside it.
    """
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as exc:  # pragma: no cover - defensive; the save must still run
        logger.warning("Could not flush LangSmith traces: %s: %s", type(exc).__name__, exc)
