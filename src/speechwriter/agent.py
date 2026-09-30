"""The speechwriter agent factory — where every layer composes into one graph.

This is the single place that assembles the Deep Agent:

* **model**        — a Claude model (default ``claude-sonnet-5-5``) with an explicit
                     output-token ceiling, effort level, and thinking configuration.
* **system_prompt**— the speechwriting method (see :mod:`speechwriter.prompts`).
* **subagents**    — ``researcher`` (Tavily) + ``style-critic`` (see :mod:`speechwriter.subagents`).
* **skills**       — the on-demand rhetoric library under ``/skills``.
* **backend**      — a ``CompositeBackend`` routing ``/memories/`` to a persistent
                     ``StoreBackend`` and everything else to real disk via ``FilesystemBackend``.
* **store**        — a JSON-snapshotted ``InMemoryStore`` for durable voice profiles.
* **checkpointer** — ``MemorySaver``, required so multi-turn conversation state and any
                     human-in-the-loop interrupts have somewhere to persist per thread.
* **tracing**      — every turn traced to LangSmith when ``LANGSMITH_TRACING=true``; LangChain
                     does the tracing, the bundle reports it (see :mod:`speechwriter.tracing`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from speechwriter.config import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    Settings,
    load_settings,
)
from speechwriter.memory import load_store, save_store
from speechwriter.observability import TruncationWarner
from speechwriter.prompts import orchestrator_prompt
from speechwriter.subagents import build_subagents
from speechwriter.tracing import Tracing, current_tracing

logger = logging.getLogger(__name__)

# How the model thinks on every call. Three things, each load-bearing:
#
# * `adaptive` is the only on-mode the 5.5 models accept (`budget_tokens` and `disabled` are
#   400s), and it is what `effort` steers.
# * `display: "summarized"` because the default on these models is `"omitted"` — thinking blocks
#   arrive with empty text — and on Sonnet 5.5 the notes the model writes *between tool calls*
#   come back as thinking blocks too. Omitted, a long plan/draft/critique turn reads as silence.
# * `block_binding: drop_block` because thinking blocks are bound to the exact history that
#   produced them, and accounts created on or after 2026-08-31 get a **400** when an edited
#   history replays one. deepagents' `SummarizationMiddleware` edits history by design — it
#   replaces older turns with a summary — so without this the first compaction in a long session
#   would fail the next call outright. `drop_block` drops the stale blocks instead: the request
#   succeeds and the model runs without that earlier reasoning, which is what compaction meant
#   anyway. It needs the beta header below.
_THINKING: dict[str, Any] = {
    "type": "adaptive",
    "display": "summarized",
    "block_binding": {"prefix_mismatch_behavior": "drop_block"},
}

# Betas every call carries — one per feature above that needs one. Named so a request log's
# `anthropic-beta` header greps back to the reason.
_BETAS = [
    # `thinking.block_binding` — see `_THINKING`.
    "thinking-binding-controls-2026-08-01",
    # `fallbacks: "default"` — see `_build_model`.
    "server-side-fallback-2026-07-01",
]


@dataclass
class SpeechwriterAgent:
    """Bundle of the compiled agent plus the handles needed to persist learned state."""

    agent: CompiledStateGraph
    store: BaseStore
    settings: Settings
    # The ceiling this agent actually resolved to. `settings.max_tokens` is only the
    # *override* and is usually None, so it can't answer "what is this running with?".
    # Defaulted so that adding it does not break anyone constructing the bundle directly.
    max_tokens: int | None = None
    # Owned by the bundle for the same reason `persist()` is: a consumer invoking
    # `bundle.agent` directly — the path the README documents — would otherwise get no
    # truncation signal at all, which is precisely what this warner exists to prevent.
    warner: TruncationWarner = field(default_factory=TruncationWarner)
    # Appended, not inserted — the rule `config.Settings` states, and this slot first broke.
    # A default is not sufficient on its own: a consumer constructing the bundle positionally
    # would have had their warner bound to *this* field instead, losing every truncation signal.
    #
    # What LangChain's profile says this model can emit at most, or None for an id it does not
    # profile. It took over this slot from `context_window` (which served the local path) in
    # place, rather than being slotted anywhere tidier, so positional order is unchanged.
    profiled_max_tokens: int | None = None
    # Appended, after `profiled_max_tokens`, for the reason that field's comment gives. Where
    # this process is sending traces, or None — carried on the bundle so both front ends can say
    # so before a turn is spent, the way they already print the ceiling. It reports what is *in
    # force*, not what this bundle's settings asked for: tracing is process-wide, so after one
    # build turned it on, every later bundle is traced too and says so.
    tracing: Tracing | None = None

    def persist(self) -> int:
        """Snapshot the learned speaker voice profiles to disk; returns the item count.

        Durability is owned by the bundle, not by the CLI: any consumer of the public
        API should call this when finished so cross-session memory is actually saved.
        """
        return save_store(self.store, self.settings)

    @property
    def ceiling_label(self) -> str:
        """The resolved output ceiling, rendered for whichever front end is asking.

        Not ``settings.max_tokens`` — that is only the *override*, and is None whenever the
        model's own profile is being trusted. Tested against None rather than truthiness so
        a ceiling of 0 is never reported as "model default" while it is actually in force.

        Lives on the bundle for the same reason ``persist()`` does: there is more than one
        UI, and a banner that re-derives this by hand is a banner that can quietly lie.
        """
        return f"{self.max_tokens:,}" if self.max_tokens is not None else "model default"

    @property
    def ceiling_exceeds_model(self) -> bool:
        """Whether the resolved ceiling is above what the model can actually emit.

        Only an explicit ``SPEECHWRITER_MAX_TOKENS`` can get here — tier 2 *is* the profiled
        figure — and the API rejects it at the first turn rather than clamping. Reported so the
        reader learns that before a turn is spent, and more likely after a model switch than at
        startup, because the override is global and outlives the model it was sized for.

        A property of its own rather than a suffix on :attr:`ceiling_label`: both front ends
        interpolate that label into a sentence telling the reader to *raise*
        ``SPEECHWRITER_MAX_TOKENS``, so folding this warning in produced advice that argued with
        itself. The label answers "what is the ceiling"; this answers "can it be honoured".
        """
        return (
            self.max_tokens is not None
            and self.profiled_max_tokens is not None
            and self.max_tokens > self.profiled_max_tokens
        )

    def turn_config(self, thread_id: str) -> RunnableConfig:
        """Build the config for one invocation: thread to resume + truncation detection.

        Prefer this over hand-writing ``{"configurable": {"thread_id": ...}}``. The
        callback propagates into subagent calls, so a critique clipped at the token
        ceiling is caught wherever it happens; a hand-built config reports nothing.
        """
        return {"configurable": {"thread_id": thread_id}, "callbacks": [self.warner]}


def _memory_namespace(_ctx: object) -> tuple[str, ...]:
    """Fixed Store namespace for persisted voice profiles.

    Passing an explicit namespace is required by deepagents (the implicit-namespace
    mode is deprecated and removed in 0.7); a single stable namespace also keeps the
    JSON snapshot in :mod:`speechwriter.memory` simple to reason about.
    """
    return ("speechwriter", "memories")


def _write_sandbox(settings: Settings) -> list[FilesystemPermission]:
    """Confine the agent's *write* tools to the workspace and memory paths.

    The FilesystemBackend is rooted at the repo so skills under ``/skills`` are
    readable, but that also exposes ``/src`` etc. to the write tools. Rather than
    trusting a prompt instruction, we enforce it: writes are allowed only under the
    drafts workspace and the memory route; everything else is denied. Reads stay open
    (no ``read`` rule), so skills and any reference material still load. Rules are
    first-match-wins with a default of allow, so the trailing deny is the backstop.
    """
    workspace = settings.workspace_vpath.rstrip("/")
    memories = settings.memories_vpath.rstrip("/")
    return [
        FilesystemPermission(
            operations=["write"],
            paths=[workspace, f"{workspace}/**", memories, f"{memories}/**"],
            mode="allow",
        ),
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
    ]


def _build_backend(settings: Settings, store: BaseStore) -> CompositeBackend:
    """Route the agent's single filesystem: ``/memories/`` to the Store, the rest to disk.

    Longest-prefix routing: ``/memories/`` is intercepted for persistent, cross-session
    storage; every other path (drafts, research notes) hits real disk under the repo.

    Built once as an instance, not as a ``backend(runtime)`` factory: deepagents 0.7
    removes both the callable-factory form of ``backend=`` and StoreBackend's ``runtime``
    argument (which 0.6 already ignores). The store is handed over explicitly rather than
    left to ``get_store()`` so this backend always resolves to the same object
    ``persist()`` snapshots, with or without a graph execution context.

    ``file_format`` is deliberately left at its default — pinning it to ``"v1"`` would
    make existing memory snapshots unreadable.

    Extracted from :func:`build_agent` for the same reason :func:`_write_sandbox` is a
    separate function: this is one of the consumers that must agree with
    :class:`~speechwriter.config.Settings` about the virtual paths, and a route buried in
    a local variable cannot be asserted against without building the whole graph.
    """
    return CompositeBackend(
        default=FilesystemBackend(root_dir=str(settings.project_root), virtual_mode=True),
        routes={settings.memories_vpath: StoreBackend(store=store, namespace=_memory_namespace)},
    )


def _chat_anthropic(settings: Settings, max_tokens: int | None) -> ChatAnthropic:
    """One ``ChatAnthropic`` with every per-call setting this agent sends.

    Built directly rather than through ``init_chat_model`` so each setting is a typed field the
    constructor validates — ``init_chat_model`` takes ``**kwargs: Any``, where a misspelled
    ``"thinkng"`` would be accepted and silently dropped.

    * ``streaming=True`` because profiled ceilings are 128k and the Anthropic SDK refuses a
      non-streaming request whose ``max_tokens`` could outlast its ten-minute timeout.
      ``invoke`` still returns one aggregated message, so nothing downstream changes.
    * ``fallbacks: "default"`` (via ``model_kwargs``, since the field is newer than the
      client's typed surface): when a safety classifier declines a turn, the API re-runs it on a
      fallback model inside the same call instead of returning an empty ``refusal``. It is the
      documented default for the 5.5 models. What it cannot rescue still arrives as
      ``stop_reason="refusal"``, which :class:`~speechwriter.observability.TruncationWarner`
      counts.
    * No ``temperature``, ``top_p`` or ``top_k`` — the 5.5 models reject non-default values
      with a 400. See the invariant in CLAUDE.md.
    * ``api_key`` omitted when unset rather than passed as ``None``: the client then falls back
      to its own empty default, still constructs, and fails only at the first call — which is
      what keeps ``build_agent`` offline for the test suite.
    """
    credentials: dict[str, Any] = (
        {"api_key": settings.anthropic_api_key} if settings.anthropic_api_key else {}
    )
    return ChatAnthropic(
        model=settings.model,
        max_tokens=max_tokens,
        effort=DEFAULT_EFFORT,
        thinking=_THINKING,
        betas=list(_BETAS),
        streaming=True,
        model_kwargs={"fallbacks": "default"},
        **credentials,
    )


def _build_model(settings: Settings) -> ChatAnthropic:
    """Build the chat client, settling its output-token ceiling in three tiers.

    1. An explicit ``SPEECHWRITER_MAX_TOKENS`` always wins.
    2. Otherwise, a ceiling LangChain resolved from the model's profile (128k for the 5.5 ids)
       is kept as-is.
    3. Only an id with *no* profile falls back to
       :data:`~speechwriter.config.DEFAULT_MAX_TOKENS`, with a warning.

    Tier 3 is the one that bites. ``ChatAnthropic`` gives an unprofiled id a ceiling of 4096,
    and adaptive thinking bills against the same budget — so a subagent can spend it all
    deliberating and emit no text, which deepagents forwards as an *empty* tool result with
    ``status="success"``. The failure is silent, and the orchestrator pays to retry it. It is
    not hypothetical: ``langchain-anthropic`` 1.6.1 did not profile the 5.5 ids, so the default
    model ran on exactly this path until the pin moved to 1.7.5.

    Constructing the client performs no network I/O, so ``build_agent`` stays offline.
    """
    if settings.max_tokens is not None:
        return _chat_anthropic(settings, settings.max_tokens)

    model = _chat_anthropic(settings, None)
    if (model.profile or {}).get("max_output_tokens"):
        return model

    logger.warning(
        "No output ceiling resolved for %r (no LangChain model profile) — it would otherwise "
        "inherit a 4096-token ceiling, which thinking alone can exhaust. Using max_tokens=%d "
        "instead; set SPEECHWRITER_MAX_TOKENS to override.",
        settings.model,
        DEFAULT_MAX_TOKENS,
    )
    return _chat_anthropic(settings, DEFAULT_MAX_TOKENS)


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is safe to run
    in tests — with or without ``ANTHROPIC_API_KEY``, which is only read at the first turn.
    """
    settings = settings or load_settings()
    store = load_store(settings)
    sandbox = _write_sandbox(settings)

    backend = _build_backend(settings, store)

    # Built, not named: a bare model string would inherit a 4096-token ceiling for any id
    # LangChain cannot profile, and would carry none of the thinking settings. See `_build_model`.
    model = _build_model(settings)

    # Here rather than in each front end so that every entry point — CLI, web UI, eval harness,
    # a library consumer — reports tracing alike. Reads the environment only: no client, no
    # socket, so the offline invariant holds with tracing on.
    tracing = current_tracing(settings)

    agent = create_deep_agent(
        model=model,
        # The orchestrator has no direct tools: research is delegated to a subagent so
        # its (potentially noisy) results never crowd the writing context.
        tools=[],
        system_prompt=orchestrator_prompt(settings),
        subagents=build_subagents(settings, permissions=sandbox),
        skills=[settings.skills_vpath],
        backend=backend,
        # Enforce the write-to-workspace-only sandbox rather than trusting the prompt.
        permissions=sandbox,
        store=store,
        checkpointer=MemorySaver(),
        name="speechwriter",
    )

    # Read with `getattr` rather than as ChatAnthropic attributes: `_build_model` is the seam the
    # prompt-vs-tools test swaps for a recording stub, which is a chat model without either.
    profile = getattr(model, "profile", None) or {}
    return SpeechwriterAgent(
        agent=agent,
        store=store,
        settings=settings,
        # Read off the *constructed* client rather than from `settings`, which carries only the
        # override: this is the figure a banner can promise before a turn is spent.
        max_tokens=getattr(model, "max_tokens", None),
        profiled_max_tokens=profile.get("max_output_tokens"),
        tracing=tracing,
    )
