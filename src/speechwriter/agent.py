"""The speechwriter agent factory — where every layer composes into one graph.

This is the single place that assembles the Deep Agent:

* **model**        — Anthropic Claude (configurable), with an explicit output-token
                     ceiling rather than one inherited from LangChain's profile table.
* **system_prompt**— the speechwriting method (see :mod:`speechwriter.prompts`).
* **subagents**    — ``researcher`` (Tavily) + ``style-critic`` (see :mod:`speechwriter.subagents`).
* **skills**       — the on-demand rhetoric library under ``/skills``.
* **backend**      — a ``CompositeBackend`` routing ``/memories/`` to a persistent
                     ``StoreBackend`` and everything else to real disk via ``FilesystemBackend``.
* **store**        — a JSON-snapshotted ``InMemoryStore`` for durable voice profiles.
* **checkpointer** — ``MemorySaver``, required so planning (``write_todos``) and any
                     human-in-the-loop interrupts have somewhere to persist per thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from speechwriter.config import Settings, load_settings
from speechwriter.memory import load_store, save_store
from speechwriter.prompts import orchestrator_prompt
from speechwriter.subagents import build_subagents

logger = logging.getLogger(__name__)


@dataclass
class SpeechwriterAgent:
    """Bundle of the compiled agent plus the handles needed to persist learned state."""

    agent: CompiledStateGraph
    store: BaseStore
    settings: Settings

    def persist(self) -> int:
        """Snapshot the learned speaker voice profiles to disk; returns the item count.

        Durability is owned by the bundle, not by the CLI: any consumer of the public
        API should call this when finished so cross-session memory is actually saved.
        """
        return save_store(self.store, self.settings)


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


def _build_model(settings: Settings) -> BaseChatModel:
    """Resolve the configured model id, pinning the output-token ceiling explicitly.

    Passing a bare model *string* to ``create_deep_agent`` lets ``init_chat_model`` take
    ``max_tokens`` from LangChain's model-profile table — which silently falls back to
    4096 for an id it does not recognise. Extended thinking bills against that same
    ceiling, so on an unrecognised id a subagent can spend its entire budget thinking and
    emit no text at all. deepagents forwards that as an *empty* tool result with
    ``status="success"`` (it walks back for the last message with text and finds none),
    so the failure is silent and the orchestrator pays to retry it.

    Pinning the ceiling here makes the budget independent of the profile table, and the
    warning turns a silently-degraded model id into something visible. Constructing the
    client performs no network I/O, so ``build_agent`` stays offline.
    """
    model = init_chat_model(settings.model, max_tokens=settings.max_tokens)
    if getattr(model, "profile", None) is None:
        logger.warning(
            "No LangChain model profile for %r — its limits are being guessed. Pinning "
            "max_tokens=%d. If output still comes back truncated, raise "
            "SPEECHWRITER_MAX_TOKENS or upgrade langchain.",
            settings.model,
            settings.max_tokens,
        )
    return model


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is
    safe to run in tests. An ``ANTHROPIC_API_KEY`` is only needed when the agent is
    actually invoked.
    """
    settings = settings or load_settings()
    store = load_store(settings)
    sandbox = _write_sandbox(settings)

    # Longest-prefix routing: /memories/ is intercepted for persistent, cross-session
    # storage; every other path (drafts, research notes) hits real disk under the repo.
    #
    # Built once as an instance, not as a `backend(runtime)` factory: deepagents 0.7
    # removes both the callable-factory form of `backend=` and StoreBackend's `runtime`
    # argument (which 0.6 already ignores). The store is handed over explicitly rather
    # than left to `get_store()` so this backend always resolves to the same object
    # `persist()` snapshots, with or without a graph execution context.
    #
    # `file_format` is deliberately left at its default — pinning it to "v1" would make
    # existing memory snapshots unreadable.
    backend = CompositeBackend(
        default=FilesystemBackend(root_dir=str(settings.project_root), virtual_mode=True),
        routes={settings.memories_vpath: StoreBackend(store=store, namespace=_memory_namespace)},
    )

    agent = create_deep_agent(
        # Built, not named: a bare model string would inherit a 4096-token ceiling for
        # any id LangChain cannot profile. See `_build_model`.
        model=_build_model(settings),
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

    return SpeechwriterAgent(agent=agent, store=store, settings=settings)
