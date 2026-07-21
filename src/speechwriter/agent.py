"""The speechwriter agent factory — where every layer composes into one graph.

This is the single place that assembles the Deep Agent:

* **model**        — Anthropic Claude (configurable).
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

from dataclasses import dataclass

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from speechwriter.config import Settings, load_settings
from speechwriter.memory import load_store
from speechwriter.prompts import orchestrator_prompt
from speechwriter.subagents import build_subagents


@dataclass
class SpeechwriterAgent:
    """Bundle of the compiled agent plus the handles the CLI needs to persist state."""

    agent: CompiledStateGraph
    store: BaseStore
    settings: Settings


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


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is
    safe to run in tests. An ``ANTHROPIC_API_KEY`` is only needed when the agent is
    actually invoked.
    """
    settings = settings or load_settings()
    store = load_store(settings)
    sandbox = _write_sandbox(settings)

    def backend(runtime: object) -> CompositeBackend:
        # Longest-prefix routing: /memories/ is intercepted for persistent, cross-session
        # storage; every other path (drafts, research notes) hits real disk under the repo.
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(settings.project_root), virtual_mode=True),
            routes={settings.memories_vpath: StoreBackend(runtime, namespace=_memory_namespace)},
        )

    agent = create_deep_agent(
        model=settings.model,
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
