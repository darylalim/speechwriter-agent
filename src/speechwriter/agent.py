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

from deepagents import create_deep_agent
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


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is
    safe to run in tests. An ``ANTHROPIC_API_KEY`` is only needed when the agent is
    actually invoked.
    """
    settings = settings or load_settings()
    store = load_store(settings)

    def backend(runtime: object) -> CompositeBackend:
        # Longest-prefix routing: /memories/ is intercepted for persistent, cross-session
        # storage; every other path (drafts, research notes) hits real disk under the repo.
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(settings.project_root), virtual_mode=True),
            routes={settings.memories_vpath: StoreBackend(runtime)},
        )

    agent = create_deep_agent(
        model=settings.model,
        # The orchestrator has no direct tools: research is delegated to a subagent so
        # its (potentially noisy) results never crowd the writing context.
        tools=[],
        system_prompt=orchestrator_prompt(settings),
        subagents=build_subagents(settings),
        skills=[settings.skills_vpath],
        backend=backend,
        store=store,
        checkpointer=MemorySaver(),
        name="speechwriter",
    )

    return SpeechwriterAgent(agent=agent, store=store, settings=settings)
