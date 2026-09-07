"""Runtime configuration for the speechwriter agent.

Everything the agent needs to know about *this machine* — which model to call,
which API keys are present, and where files live — is resolved here into a single
frozen :class:`Settings` object. Keeping this in one place means the agent,
the CLI, and the tests all agree on paths and never hard-code them.

Path model
----------
The agent's filesystem tools are backed by a ``FilesystemBackend`` rooted at
``PROJECT_ROOT`` (the repo). Inside that virtual root the agent sees:

* ``/skills/``     — the on-demand rhetoric skill library (read-only by convention)
* ``/workspace/``  — where drafts and research notes are written (real files on disk)
* ``/memories/``   — persistent, cross-session speaker voice profiles (routed to a Store)

``/memories/`` is intercepted by a ``CompositeBackend`` route *before* it reaches
disk, so it never appears as a real folder — it lives in the persistent Store.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ModelChoice(NamedTuple):
    """One selectable model: the id *and* where it is served from, as a single record.

    ``model`` and ``base_url`` travel together because they are not independent choices.
    :func:`~speechwriter.agent._build_model` keys the *client* off ``base_url``, never off
    the id — so a locally served id paired with ``base_url=None`` builds an **Anthropic**
    client and raises ``ValueError: Unable to infer model provider`` inside ``build_agent``,
    which in Streamlit is a page-level traceback (``get_bundle()`` runs at module scope).
    Pairing them here makes that combination unrepresentable rather than something every
    front end has to remember to validate.

    ``context_window`` is the local half's other half: see
    :data:`DEFAULT_LOCAL_CONTEXT_WINDOW`.
    """

    label: str
    model: str
    base_url: str | None = None
    context_window: int | None = None

    def is_current(self, settings: Settings) -> bool:
        """Whether ``settings`` is already running this choice.

        The identity of a choice is the pair, never the label — two entries naming the same
        model at the same endpoint are the same choice however they are captioned. Spelled out
        once because both front ends ask this to decide what to mark as current and whether a
        switch is a no-op, and a comparison that quietly dropped ``base_url`` would call a local
        model and its Anthropic namesake the same thing.
        """
        return self.model == settings.model and self.base_url == settings.base_url

    def applied_to(self, settings: Settings) -> Settings:
        """``settings`` with this choice in force, ready for ``build_agent``.

        All three fields move together, which is the point: ``base_url`` selects the client and
        ``context_window`` sizes compaction, so applying the model alone would leave a Claude id
        pointed at a local server, or a local model compacting for the previous one's window.
        """
        return dataclasses.replace(
            settings,
            model=self.model,
            base_url=self.base_url,
            context_window=self.context_window,
        )


# The workhorse model. Sonnet 5 is a strong writer at sensible cost; override with
# SPEECHWRITER_MODEL (e.g. "claude-opus-5" for the highest-quality drafting; no ceiling
# override is needed alongside it — LangChain profiles that id at its real 128k, so tier 2
# in agent.py keeps it. Only an id LangChain cannot profile falls to DEFAULT_MAX_TOKENS).
DEFAULT_MODEL = "claude-sonnet-5"

# Fallback output-token ceiling — used *only* when the model id has no LangChain profile.
#
# `init_chat_model` takes `max_tokens` from LangChain's model-profile table and falls back
# to 4096 for an id it does not recognise. Extended thinking bills against that same
# ceiling, so on an unrecognised id a subagent can spend the entire budget thinking and
# return *no text at all* — which deepagents forwards as an empty, `status="success"` tool
# result. 4096 is far too tight for that; 32k leaves comfortable room for a draft or
# critique plus thinking.
#
# A *profiled* model keeps its own, usually larger, ceiling (64k-128k) rather than being
# capped to this. See `agent._build_model` for the three-tier resolution.
DEFAULT_MAX_TOKENS = 32000

# How the agent files its output under `workspace_dir`, and the pace it writes for.
#
# Both were inline in `prompts.py` while the agent was the only party that cared. They are
# named here now because a second subsystem depends on them agreeing: `prompts.py`
# *instructs* the agent to write speeches at this pace into these folders, and
# `workspace.py` *reads* them back to estimate how long a saved draft runs. Two copies of
# "speeches" would drift into a browser that silently lists nothing.
SPEECHES_SUBDIR = "speeches"
RESEARCH_SUBDIR = "research"
WORDS_PER_MINUTE = 130

# Sent as the bearer token when `base_url` points somewhere that does not check one.
# `ChatOpenAI` requires *a* key at construction and raises without one, so an empty string
# is not an option; a local `mlx_lm.server` never reads it. Named rather than inlined so the
# value that shows up in a request log is greppable back to this comment.
LOCAL_API_KEY_PLACEHOLDER = "local"

# The context window assumed for a locally served model, and the reason `ModelChoice` carries
# one at all.
#
# deepagents sizes its context-compaction trigger from the model's LangChain profile: a
# profiled id compacts at a *fraction* of its real window, but an unprofiled one — which every
# locally served id is — gets a flat 170k-token trigger. No plausible local server has a
# 170k window, so the plan/draft/critique/revise rhythm outgrows the window and the server
# errors before compaction ever fires. `agent._build_model` therefore hands the local client a
# minimal profile built from this, so the trigger scales to the window the model actually has.
#
# 32768 is the conservative floor across the local servers this repo documents; a roster entry
# that knows better overrides it per model via `ModelChoice.context_window`. Deliberately not
# an environment variable: a `SPEECHWRITER_*` name is a contract with `.env.example`
# (`test_env_example_documents_every_setting`), and this is a property of a *chosen model*
# rather than a setting for the machine.
DEFAULT_LOCAL_CONTEXT_WINDOW = 32768

# The models the front ends offer. Anthropic ids only, on purpose — see `model_choices()`,
# which widens this with whatever pair the environment actually names.
#
# Every entry here must be an id LangChain profiles, so it keeps its own 64k-128k ceiling
# rather than falling to `DEFAULT_MAX_TOKENS`; that is not merely a convention but an
# assertion (`test_every_anthropic_model_choice_is_profiled_above_the_floor`). Three rather
# than all thirteen profiled ids: this is a writing tool, and the choice worth offering is
# quality-versus-cost, not a catalogue of dated snapshots.
MODEL_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice("Sonnet 5", DEFAULT_MODEL),
    ModelChoice("Opus 5", "claude-opus-5"),
    ModelChoice("Haiku 4.5", "claude-haiku-4-5"),
)

# Package dir is .../src/speechwriter ; the repo root is two levels up.
_PKG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # Virtual path where persistent speaker voice profiles live (routed to the Store).
    # A fixed convention, not per-instance config — hence a ClassVar, not a field.
    memories_vpath: ClassVar[str] = "/memories/"

    model: str
    anthropic_api_key: str | None
    tavily_api_key: str | None
    project_root: Path
    workspace_dir: Path
    skills_dir: Path
    store_path: Path
    max_research_results: int
    # Appended, not inserted: a new field in the middle silently shifts every positional
    # argument after it, so a caller constructing Settings by position would bind their
    # API key here. Explicit output-token override; None defers to the model's profile.
    max_tokens: int | None
    # Appended for the same reason, and *defaulted* for the reason `SpeechwriterAgent`
    # defaults its own added fields: `build_agent(settings)` is the documented library entry
    # point, so a consumer constructing Settings by hand would otherwise break on an upgrade
    # that only added an optional capability. An OpenAI-compatible endpoint to use *instead
    # of* Anthropic; None (the normal case) leaves the Anthropic path untouched.
    base_url: str | None = None
    openai_api_key: str | None = None
    # Appended and defaulted for the same reason as the two above — and, alone among these
    # fields, never read from the environment. `load_settings()` leaves it None; it is set
    # only by `dataclasses.replace` when a front end switches to a locally served model, and
    # is read only by `agent._build_model`. See `DEFAULT_LOCAL_CONTEXT_WINDOW` for what it
    # buys and why it is not a `SPEECHWRITER_*` knob.
    context_window: int | None = None

    # -- derived helpers -------------------------------------------------

    @property
    def research_enabled(self) -> bool:
        """Live web research is only possible when a Tavily key is present."""
        return bool(self.tavily_api_key)

    @property
    def uses_local_endpoint(self) -> bool:
        """Whether the model is served over an OpenAI-compatible URL rather than by Anthropic.

        One flag drives three coupled things, the same shape as ``research_enabled``:
        which client :func:`~speechwriter.agent._build_model` constructs, whether an
        ``ANTHROPIC_API_KEY`` is required at all, and what the front ends put on the banner.
        """
        return self.base_url is not None

    @property
    def model_credentials_present(self) -> bool:
        """Whether the configured model can actually be called.

        Both front ends gate on this rather than on ``anthropic_api_key`` directly: an
        Anthropic key is *irrelevant* when the model is served locally, and demanding one
        would refuse to start a configuration that works perfectly well. A local endpoint
        needs no credential of ours — reachability is a runtime concern, and probing it here
        would break the "building the agent touches no network" invariant.
        """
        return self.uses_local_endpoint or bool(self.anthropic_api_key)

    @property
    def endpoint_api_key(self) -> str:
        """The bearer token to send to ``base_url``, never empty.

        Resolved here rather than in :mod:`speechwriter.agent` so that reading credentials
        out of the environment stays this module's job — the same reason ``base_url`` is a
        field and not an ``os.environ`` lookup at the call site.
        """
        return self.openai_api_key or LOCAL_API_KEY_PLACEHOLDER

    def _vpath(self, path: Path) -> str:
        """Map a real path under ``project_root`` to the agent's virtual path."""
        rel = path.resolve().relative_to(self.project_root.resolve()).as_posix()
        return "/" + rel

    @property
    def skills_vpath(self) -> str:
        """Virtual dir the ``skills=`` param points at, e.g. ``/skills/``."""
        return self._vpath(self.skills_dir) + "/"

    @property
    def workspace_vpath(self) -> str:
        """Virtual dir the agent writes drafts under, e.g. ``/workspace``."""
        return self._vpath(self.workspace_dir)


def model_choices(*offered: Settings) -> tuple[ModelChoice, ...]:
    """The curated roster, widened so every configuration passed in stays selectable.

    This roster is deliberately never authoritative. Streamlit **silently** rewrites a
    ``session_state`` value that is not among a widget's options to option zero — no
    exception, no log — so a fixed list would take a reader who configured a local endpoint
    and retarget them onto the default Claude id, which on a machine with no Anthropic key
    flips both front ends into their "no credentials" state. Widening means the picker can
    only ever *add* to what the environment already says.

    It is also how locally served models reach the roster at all: rather than shipping
    hard-coded endpoints that are dead on any machine that has not started that particular
    server, the local entry is whatever ``SPEECHWRITER_BASE_URL`` and ``SPEECHWRITER_MODEL``
    name — correct by construction, because the operator configured it.
    :func:`speechwriter.endpoints.list_models` widens it further, on demand.

    **Variadic because one configuration is not enough, and passing only the live one is a
    bug.** Selecting a curated entry sets ``base_url`` to ``None``, so a roster derived from
    the *current* settings alone would drop the locally served entry the reader came from —
    the same silent-retarget failure this function exists to prevent, reached by a click
    instead of by a rerun, and unrecoverable without a restart. On a keyless machine it is
    worse than losing an option: picking a Claude id there also disables the chat input, so
    the one entry that still works has just been removed from the list. Callers therefore
    pass both the configuration the session *started* on and the one now in force.
    """
    choices = list(MODEL_CHOICES)
    seen = {(choice.model, choice.base_url) for choice in choices}
    for settings in offered:
        identity = (settings.model, settings.base_url)
        if identity in seen:
            continue
        seen.add(identity)
        choices.append(
            ModelChoice(settings.model, settings.model)
            if settings.base_url is None
            else local_choice(settings.model, settings.base_url, settings.context_window)
        )
    return tuple(choices)


def resolve_choice(choices: tuple[ModelChoice, ...], requested: str) -> ModelChoice | None:
    """Match a typed argument against a roster by position, label, or model id.

    Shared by the REPL's ``/model`` and the eval harness's ``--model`` so the two cannot
    disagree about what a reader may type — they had the same loop, byte for byte, and only
    one of them resolved against a roster containing the locally served entry.

    ``isdecimal``, not ``isdigit``: the latter is true for characters ``int()`` refuses ("²",
    "½"), so an index check built on it raises on a stray keystroke instead of answering.
    Every character ``isdecimal`` accepts, ``int`` parses.
    """
    if requested.isdecimal():
        index = int(requested)
        return choices[index - 1] if 1 <= index <= len(choices) else None
    wanted = requested.casefold()
    for choice in choices:
        if wanted in (choice.label.casefold(), choice.model.casefold()):
            return choice
    return None


def local_choice(model: str, base_url: str, context_window: int | None = None) -> ModelChoice:
    """A roster entry for ``model`` served at ``base_url``, labelled the one way.

    The label convention lives here, and in exactly one place, because equality decides
    correctness. ``ModelChoice`` is a ``NamedTuple``, so two entries for the same served model
    are equal only if their *labels* match too — and a front end that discovers models from a
    live endpoint has to produce entries indistinguishable from the ones
    :func:`model_choices` synthesises, or the same model appears twice in the picker and the
    selected one is not found among the options. Streamlit's response to that is to silently
    reset the selection to the first entry.

    The ``(local)`` suffix earns its place for the reason the CLI banner prints ``endpoint`` on
    a line of its own: "which model" and "served from where" fail differently, and an
    unsuffixed id would read as an Anthropic one.
    """
    return ModelChoice(f"{model} (local)", model, base_url, context_window)


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables and package layout.

    Recognised environment variables:

    * ``ANTHROPIC_API_KEY``  — required to actually run the agent (checked lazily).
    * ``TAVILY_API_KEY``     — enables the live-research subagent; optional.
    * ``SPEECHWRITER_MODEL`` — override the model id (default ``claude-sonnet-5``).
    * ``SPEECHWRITER_HOME``  — override the project root the agent operates in.
    * ``SPEECHWRITER_MAX_RESEARCH_RESULTS`` — Tavily results per query (default 5).
    * ``SPEECHWRITER_MAX_TOKENS`` — *override* the output-token ceiling per model call.
      Left unset, the model's own LangChain profile decides, falling back to
      ``DEFAULT_MAX_TOKENS`` only for an id that has no profile.
    * ``SPEECHWRITER_BASE_URL`` — point the agent at an OpenAI-compatible endpoint
      (a local ``mlx_lm.server``, vLLM, LM Studio, Ollama) instead of Anthropic. Unset
      is the normal case and changes nothing.
    * ``OPENAI_API_KEY`` — sent to that endpoint when one is set. Local servers ignore
      it, so it is optional and falls back to a placeholder; a hosted OpenAI-compatible
      service will need a real one.
    """
    project_root = Path(os.environ.get("SPEECHWRITER_HOME", _PKG_DIR.parents[1])).resolve()

    # Load the project's own .env (if present) so ANTHROPIC_API_KEY / TAVILY_API_KEY /
    # LANGSMITH_* are available without exporting them by hand. We point at the project
    # root explicitly rather than letting python-dotenv walk *up* the directory tree —
    # an upward walk can pull keys from an unrelated ancestor .env. Done here (not at
    # import) so `import speechwriter` has no side effects; real shell env wins.
    load_dotenv(project_root / ".env")

    # langsmith memoises env reads in an `lru_cache` on `get_env_var`, so the *first* read of
    # LANGSMITH_TRACING is the one that sticks for the life of the process. Anything that reads
    # tracing state before the line above — a module-level `Client()`, a `tracing_is_enabled()`
    # at import — permanently caches "off" for a value that only exists in the dotenv, with no
    # error to notice. Clearing the cache here makes the read *order* irrelevant for every entry
    # point (CLI, Streamlit, library consumers) instead of leaving an unwritten rule that only
    # `build_agent()` happens to satisfy. Imported inside the function so `import speechwriter`
    # keeps its lazy import surface.
    # `get_env_var` carries `@overload` stubs that shadow the `lru_cache` wrapper, so
    # `cache_clear` is invisible to a type checker but present at runtime. `getattr` states that
    # precisely, and doubles as the fallback for a langsmith that stops caching.
    try:
        from langsmith.utils import get_env_var
    except ImportError:  # pragma: no cover - langsmith ships with langchain-core
        pass
    else:
        cache_clear = getattr(get_env_var, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    workspace_dir = project_root / "workspace"
    skills_dir = project_root / "skills"
    store_path = project_root / ".speechwriter" / "memory-store.json"

    # Ensure the writable dirs exist so the first draft never fails on a missing folder.
    workspace_dir.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        model=os.environ.get("SPEECHWRITER_MODEL", DEFAULT_MODEL),
        max_tokens=_optional_int_env("SPEECHWRITER_MAX_TOKENS"),
        # `or None` rather than a bare `.get`: an exported-but-empty SPEECHWRITER_BASE_URL
        # is how a shell says "unset", and an empty string here would route every call to a
        # nonexistent endpoint while `uses_local_endpoint` still reported True.
        base_url=(os.environ.get("SPEECHWRITER_BASE_URL") or "").strip() or None,
        # Normalised the same way, and for the same reason: a blank or whitespace-only value
        # is how a shell says "unset", and left as-is it is *truthy* — so `endpoint_api_key`
        # would send "   " as the bearer token and a hosted endpoint would 401 far from the
        # typo, instead of falling back to the placeholder.
        openai_api_key=(os.environ.get("OPENAI_API_KEY") or "").strip() or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        project_root=project_root,
        workspace_dir=workspace_dir,
        skills_dir=skills_dir,
        store_path=store_path,
        max_research_results=_int_env("SPEECHWRITER_MAX_RESEARCH_RESULTS", 5),
    )


def _optional_int_env(name: str, *, minimum: int = 1) -> int | None:
    """Parse an int from the environment; ``None`` if unset, blank, invalid, or too small.

    ``None`` means *no opinion* — it leaves the caller free to treat an absent override
    differently from a supplied one, which is what makes deferring to a model's own
    profile possible.

    Out-of-range values are rejected rather than forwarded. Both callers hand the result
    to a client — an output-token ceiling and a result count — where a zero or negative
    would not fail at startup but at the first API call, with an opaque provider error
    far from the typo that caused it.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (not an integer).", name, raw)
        return None
    if value < minimum:
        logger.warning("Ignoring out-of-range %s=%r (minimum %d).", name, raw, minimum)
        return None
    return value


def _int_env(name: str, default: int) -> int:
    """Parse an int from the environment, falling back (with a warning) on bad input.

    A stray ``SPEECHWRITER_MAX_RESEARCH_RESULTS=ten`` should not crash startup with an
    opaque ``ValueError`` before the CLI can even render — and the operator should be able
    to see, from the log alone, which value actually took effect.
    """
    value = _optional_int_env(name)
    if value is None and (os.environ.get(name) or "").strip():
        logger.warning("Using default %s=%d.", name, default)
    return default if value is None else value
