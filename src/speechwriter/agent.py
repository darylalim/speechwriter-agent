"""The speechwriter agent factory — where every layer composes into one graph.

This is the single place that assembles the Deep Agent:

* **model**        — a locally served model, reached over the OpenAI-compatible endpoint
                     named by ``SPEECHWRITER_BASE_URL``, with an explicit output-token
                     ceiling rather than the client's unbounded default.
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


class _ClientKwargs(TypedDict):
    """The ``init_chat_model`` arguments that select and shape the client.

    Typed rather than left as ``dict[str, object]`` for the same reason
    :func:`~speechwriter.subagents.build_subagents` returns ``list[SubAgent]``: every key here
    is load-bearing and silently optional. ``init_chat_model`` takes ``**kwargs: Any``, so a
    misspelled ``"profiles"`` or ``"base_urls"`` would be accepted and forwarded into the
    client constructor's own ``**kwargs``, and the only symptom would be a model that never
    compacts, or one pointed at the wrong host. Spelled out, ``ty`` rejects the typo.

    **Total**, where it used to be ``total=False``. That laxity paid for one thing: the same
    record also had to describe the *empty* dict handed to the hosted path. There is no hosted
    path now, so every key is always supplied and the type can say so — which is what makes a
    dropped key a type error rather than a silent fallback.

    It also keeps the ``**client`` unpack matching ``init_chat_model``'s overloads: an
    inferred ``dict`` widens its value type to a union covering ``profile``'s nested mapping,
    which no longer satisfies the declared ``model_provider: str | None``.
    """

    model_provider: str
    base_url: str
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
    # The window the model is compacting for, kept beside the resolved ceiling because the two
    # share it: output and input come out of one budget on a local server, so an override
    # sized without reference to the window cannot be honoured. This replaced
    # `profiled_max_tokens`, which held what LangChain's table said a *hosted* model would
    # accept — a number that is now structurally always None, since `init_chat_model` fills a
    # profile's `max_output_tokens` only on the Anthropic path. A field that can only ever be
    # None is a comparison that can only ever be False, which is a warning that has quietly
    # stopped firing rather than one that has nothing to report.
    context_window: int = DEFAULT_LOCAL_CONTEXT_WINDOW

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
    def ceiling_crowds_context(self) -> bool:
        """Whether the resolved output ceiling leaves too little of the window for the prompt.

        The local analogue of the check this replaced. ``ceiling_exceeds_model`` compared the
        ceiling against what a *hosted* model advertised it would emit; served locally there is
        no such advertisement, but there is a harder constraint that the hosted path never had:
        **output and input share one window.** A ceiling of 32,000 against a 32,768-token window
        is not merely large, it leaves 768 tokens for the entire system prompt, the loaded
        skills and the draft under revision. vLLM rejects that outright at the first turn;
        others clamp it silently, which is worse, because the reader sees a short speech and no
        error.

        Half the window is the line. It is a judgement, not a measurement, and this is the
        argument for it: a revision turn's input — prompt, skills, the draft being revised — is
        routinely the same order of magnitude as its output, so a ceiling above half the window
        cannot be honoured alongside a realistic prompt. :data:`DEFAULT_MAX_TOKENS` sits
        comfortably under half of :data:`~speechwriter.config.DEFAULT_LOCAL_CONTEXT_WINDOW`, so
        the default configuration never trips it.

        A property of its own rather than a suffix on :attr:`ceiling_label`, which is where its
        predecessor started: both front ends interpolate that label into a sentence telling the
        reader to *raise* ``SPEECHWRITER_MAX_TOKENS``, so folding the warning in produced advice
        that argued with itself. The label answers "what is the ceiling"; this answers "can it
        be honoured", and the two questions belong in different sentences.
        """
        return self.max_tokens is not None and self.max_tokens > self.context_window // 2

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
    """Build the chat client, settling its output-token ceiling in two tiers.

    1. An explicit ``SPEECHWRITER_MAX_TOKENS`` always wins.
    2. Otherwise :data:`~speechwriter.config.DEFAULT_MAX_TOKENS`.

    **There used to be a middle tier, and it is gone by construction rather than by choice.**
    It kept a ceiling the client had resolved for itself — 64k-128k for a Claude id, out of
    LangChain's model-profile table. ``init_chat_model`` reads a profile's ``max_tokens`` only
    on the Anthropic path, so with that client removed the tier can never fire: ``ChatOpenAI``
    comes back with ``max_tokens=None`` whatever the id, including a *profiled* one such as
    ``gpt-4o`` behind LiteLLM. Left in place it would have been a branch that reads as live
    protection and is dead — the worst kind — so it is deleted rather than kept for symmetry.

    That also removes the reason the old tier 3 logged a warning. An unprofiled id used to mean
    a typo; now it is every id, so a line on every build would be noise. What reports the
    resolved ceiling is :attr:`SpeechwriterAgent.ceiling_label`, which both front ends already
    print before a turn is spent, and :class:`~speechwriter.observability.TruncationWarner`,
    which counts the responses that actually hit it.

    Tier 2 is the one that bites, and it is why the ceiling is pinned at all rather than left
    to the client's own ``None`` ("let the server decide"). Extended thinking bills against the
    same ceiling, so an unbounded reasoning model can spend a whole response deliberating and
    emit no text — deepagents forwards that as an *empty* tool result with ``status="success"``
    (it walks back for the last message with text and finds none), so the failure is silent and
    the orchestrator pays to retry it.

    Constructing the client performs no network I/O, so ``build_agent`` stays offline:
    ``base_url`` is recorded on the client, never probed, so an unreachable server fails at the
    first turn rather than at build time.
    """
    # The client is selected by URL, not by model id: "mlx-community/Qwen3.8-27B-4bit" carries
    # no provider prefix for `init_chat_model` to infer, so the provider is stated outright.
    #
    # `profile` is not about the output ceiling — `init_chat_model` reads a profile's
    # `max_tokens` only on the Anthropic path, so it does not feed the tiers below. It is about
    # *input*: deepagents sizes its context-compaction trigger from the model's profile, and an
    # id LangChain does not profile — which every locally served id is — gets a flat 170k-token
    # trigger instead of a fraction of its real window. No local server has a 170k window, so
    # without this the plan/draft/critique/revise rhythm outgrows the window and the server
    # errors before compaction ever fires. See `config.DEFAULT_LOCAL_CONTEXT_WINDOW`. Only
    # `max_input_tokens` is carried: a `max_output_tokens` key here is inert (measured), and
    # would imply a ceiling mechanism this path does not have.
    client: _ClientKwargs = {
        "model_provider": "openai",
        "base_url": settings.base_url,
        "api_key": settings.endpoint_api_key,
        "profile": {"max_input_tokens": settings.context_window or DEFAULT_LOCAL_CONTEXT_WINDOW},
    }

    ceiling = settings.max_tokens if settings.max_tokens is not None else DEFAULT_MAX_TOKENS
    return init_chat_model(settings.model, max_tokens=ceiling, **client)


def build_agent(settings: Settings | None = None) -> SpeechwriterAgent:
    """Assemble and compile the speechwriter Deep Agent.

    Constructing the agent does **not** call the model or the network, so this is safe to run
    in tests. The endpoint in ``settings.base_url`` is recorded on the client and never probed,
    so a server that is not running yet fails at the first turn rather than here.
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

    return SpeechwriterAgent(
        agent=agent,
        store=store,
        settings=settings,
        # Read off the *constructed* client rather than from `settings`, which carries only the
        # override: this is the figure a banner can promise before a turn is spent.
        max_tokens=getattr(model, "max_tokens", None),
        # The same value `_build_model` put in the client's profile, and it has to be resolved
        # the same way — `settings.context_window` is None for every choice that did not name
        # one, and reporting None as the window would silently disable `ceiling_crowds_context`.
        context_window=settings.context_window or DEFAULT_LOCAL_CONTEXT_WINDOW,
    )
