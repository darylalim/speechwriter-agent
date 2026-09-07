"""The speechwriter agent factory — where every layer composes into one graph.

This is the single place that assembles the Deep Agent:

* **model**        — Anthropic Claude by default, or any OpenAI-compatible endpoint via
                     ``SPEECHWRITER_BASE_URL``, with an explicit output-token ceiling
                     rather than one inherited from LangChain's profile table.
* **system_prompt**— the speechwriting method (see :mod:`speechwriter.prompts`).
* **subagents**    — ``researcher`` (Tavily) + ``style-critic`` (see :mod:`speechwriter.subagents`).
* **skills**       — the on-demand rhetoric library under ``/skills``.
* **backend**      — a ``CompositeBackend`` routing ``/memories/`` to a persistent
                     ``StoreBackend`` and everything else to real disk via ``FilesystemBackend``.
* **store**        — a JSON-snapshotted ``InMemoryStore`` for durable voice profiles.
* **checkpointer** — ``MemorySaver``, required so multi-turn conversation state and any
                     human-in-the-loop interrupts have somewhere to persist per thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TypedDict

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from speechwriter.config import (
    DEFAULT_LOCAL_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    Settings,
    load_settings,
)
from speechwriter.memory import load_store, save_store
from speechwriter.observability import TruncationWarner
from speechwriter.prompts import orchestrator_prompt
from speechwriter.subagents import build_subagents

logger = logging.getLogger(__name__)


class _ClientKwargs(TypedDict, total=False):
    """The ``init_chat_model`` arguments that select and shape a *local* client.

    Typed rather than left as ``dict[str, object]`` for the same reason
    :func:`~speechwriter.subagents.build_subagents` returns ``list[SubAgent]``: every key here
    is load-bearing and silently optional. ``init_chat_model`` takes ``**kwargs: Any``, so a
    misspelled ``"profiles"`` or ``"base_urls"`` would be accepted and forwarded into the
    client constructor's own ``**kwargs``, and the only symptom would be a local model quietly
    running against Anthropic or never compacting. Spelled out, ``ty`` rejects the typo.

    It also keeps the ``**client`` unpack matching ``init_chat_model``'s overloads: an
    inferred ``dict`` widens its value type to a union covering ``profile``'s nested mapping,
    which no longer satisfies the declared ``model_provider: str | None``.
    """

    model_provider: str
    base_url: str | None
    api_key: str
    profile: dict[str, int]


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
    # Appended, not inserted — the rule `config.Settings` states, and this field first broke.
    # A default is not sufficient on its own: a consumer constructing the bundle positionally
    # would have had their warner bound to *this* field instead, losing every truncation
    # signal and raising from `ceiling_label` on the first comparison.
    #
    # What the model itself says it can emit, when LangChain profiles it — kept beside the
    # resolved ceiling so a front end can say when an explicit override asks for *more* than
    # the model will accept. That pairing only became reachable when the model became
    # switchable: `SPEECHWRITER_MAX_TOKENS` is tier 1 and global, so an override set for one
    # model (128k, sized for Opus) silently follows a switch to another (Haiku, whose real
    # ceiling is 64k) and is rejected at the first turn, far from the switch that caused it.
    profiled_max_tokens: int | None = None

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
        """Whether the resolved ceiling asks for more output than the model will accept.

        Reachable only because the model became switchable: ``SPEECHWRITER_MAX_TOKENS`` is
        tier 1 and global, so an override sized for one model (128k, for Opus) follows a
        switch to another (Haiku, whose real ceiling is 64k) and is rejected at the first turn,
        far from the switch that caused it.

        A property of its own rather than a suffix on :attr:`ceiling_label`, which is where it
        started: both front ends interpolate that label into a sentence telling the reader to
        *raise* ``SPEECHWRITER_MAX_TOKENS``, so folding the warning in produced "raise
        SPEECHWRITER_MAX_TOKENS (currently 128,000 — above this model's 64,000)" — advice that
        contradicts itself. The label answers "what is the ceiling"; this answers "is it
        usable", and the two questions belong in different sentences.
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


def _build_model(settings: Settings) -> BaseChatModel:
    """Resolve the model id, settling its output-token ceiling in three tiers.

    1. An explicit ``SPEECHWRITER_MAX_TOKENS`` always wins.
    2. Otherwise a client that resolved **its own** ceiling keeps it — 64k-128k for current
       Claude models, from LangChain's profile table.
    3. Otherwise :data:`~speechwriter.config.DEFAULT_MAX_TOKENS`, because an id with no
       profile would silently inherit 4096.

    Tier 2 tests the **resolved ceiling**, not the presence of a profile. Those are the same
    question for ``ChatAnthropic`` and emphatically not for ``ChatOpenAI``: ``init_chat_model``
    applies a profile's ``max_tokens`` only on the Anthropic path, so a *profiled* id served
    over ``SPEECHWRITER_BASE_URL`` — ``gpt-4o`` on LM Studio or LiteLLM, say — comes back with
    ``max_tokens=None`` and no ceiling at all. Keying tier 2 on ``profile`` let exactly that
    case skip tier 3, which is the unbounded-thinking budget tier 3 exists to prevent.

    Tier 3 is the one that bites. Extended thinking bills against the same ceiling, so at
    4096 a subagent can spend its entire budget thinking and emit no text at all —
    deepagents forwards that as an *empty* tool result with ``status="success"`` (it walks
    back for the last message with text and finds none), so the failure is silent and the
    orchestrator pays to retry it.

    Tier 2 exists so that fallback never *lowers* a recognised model. Capping Opus at 32k
    when its profile says 128k would be the same mistake in the opposite direction:
    a blunt constant overriding better-informed knowledge.

    Constructing the client performs no network I/O, so ``build_agent`` stays offline.
    That still holds for a local endpoint: ``base_url`` is recorded on the client, never
    probed, so an unreachable server fails at the first turn rather than at build time.

    ``SPEECHWRITER_BASE_URL`` swaps the *client*, not the tiers. A locally served model has
    no LangChain profile, so it resolves through tier 3 — which is the wanted answer here
    rather than a fallback, hence the softer log level on that branch.
    """
    # An OpenAI-compatible endpoint (a local `mlx_lm.server`, vLLM, LM Studio) is selected by
    # URL, not by model id: "mlx-community/Qwen3.8-27B-4bit" carries no provider prefix for
    # `init_chat_model` to infer, so the provider is stated. Threaded through *every* tier
    # below rather than added to one branch — a ceiling path that omitted it would quietly
    # build an Anthropic client for a local model and fail at the first call.
    #
    # `profile` is not about the output ceiling — `init_chat_model` reads a profile's
    # `max_tokens` only on the Anthropic path, so this leaves the tiers below untouched and a
    # local model still resolves through tier 3. It is about *input*: deepagents sizes its
    # context-compaction trigger from the model's profile, and an unprofiled id — which every
    # locally served id is — gets a flat 170k-token trigger instead of a fraction of its real
    # window. No local server has a 170k window, so without this the conversation outgrows the
    # window and the server errors before compaction ever fires. See
    # `config.DEFAULT_LOCAL_CONTEXT_WINDOW`. Only `max_input_tokens` is carried: a
    # `max_output_tokens` key here would be inert, and would imply a ceiling mechanism that
    # does not exist on this path.
    client: _ClientKwargs = (
        {
            "model_provider": "openai",
            "base_url": settings.base_url,
            "api_key": settings.endpoint_api_key,
            "profile": {
                "max_input_tokens": settings.context_window or DEFAULT_LOCAL_CONTEXT_WINDOW
            },
        }
        if settings.uses_local_endpoint
        else {}
    )

    if settings.max_tokens is not None:
        return init_chat_model(settings.model, max_tokens=settings.max_tokens, **client)

    model = init_chat_model(settings.model, **client)
    # Both halves, guarding opposite failures. Without the profile check an *unprofiled*
    # ChatAnthropic keeps `max_tokens=4096` — LangChain's silent fallback, and the whole
    # reason tier 3 exists. Without the ceiling check a *profiled* ChatOpenAI keeps
    # `max_tokens=None`, because `init_chat_model` reads a profile's ceiling only on the
    # Anthropic path; that is worse still, being no ceiling at all. Neither test alone is
    # sufficient and each looks redundant until the other client is considered.
    if getattr(model, "profile", None) is not None and getattr(model, "max_tokens", None):
        return model

    if settings.uses_local_endpoint:
        # Expected, not a typo: LangChain profiles hosted ids, and a locally served one is
        # not in that table. It still takes the floor rather than ChatOpenAI's own `None`
        # (= "let the server decide"), because an unbounded ceiling on a *reasoning* model is
        # the same trap tier 3 exists for. Qwen3.8-27B defaults to `reasoning_effort: xhigh`
        # and will happily spend a thousand tokens deliberating before it writes a line.
        logger.info(
            "No output ceiling resolved for %r served at %s — expected for a local "
            "endpoint, and true even of a *profiled* id here, since init_chat_model "
            "applies a profile's max_tokens only on the Anthropic path. Pinning "
            "max_tokens=%d; set SPEECHWRITER_MAX_TOKENS to override.",
            settings.model,
            settings.base_url,
            DEFAULT_MAX_TOKENS,
        )
    else:
        logger.warning(
            "No output ceiling resolved for %r (no LangChain model profile) — it would "
            "otherwise inherit a 4096-token ceiling, which extended thinking can exhaust "
            "before any text is emitted. Using max_tokens=%d instead; set "
            "SPEECHWRITER_MAX_TOKENS to override.",
            settings.model,
            DEFAULT_MAX_TOKENS,
        )
    return init_chat_model(settings.model, max_tokens=DEFAULT_MAX_TOKENS, **client)


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is
    safe to run in tests. An ``ANTHROPIC_API_KEY`` is only needed when the agent is
    actually invoked.
    """
    settings = settings or load_settings()
    store = load_store(settings)
    sandbox = _write_sandbox(settings)

    backend = _build_backend(settings, store)

    # Built, not named: a bare model string would inherit a 4096-token ceiling for any id
    # LangChain cannot profile. See `_build_model`.
    model = _build_model(settings)

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

    profile = getattr(model, "profile", None)
    return SpeechwriterAgent(
        agent=agent,
        store=store,
        settings=settings,
        max_tokens=getattr(model, "max_tokens", None),
        # `.get` on a mapping we did not build: a profile is third-party data whose shape can
        # change on a dependency bump, and an absent key here should cost the warning, not the
        # build. None simply means "nothing to compare against".
        profiled_max_tokens=profile.get("max_output_tokens") if isinstance(profile, dict) else None,
    )
