"""Runtime configuration for the speechwriter agent.

Everything the agent needs to know about *this machine* — which model to call, which API
keys are present, and where files live — is resolved here into a single frozen
:class:`Settings` object. Keeping this in one place means the agent, the CLI, and the tests
all agree on paths and never hard-code them.

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
from typing import ClassVar, Literal, NamedTuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ModelChoice(NamedTuple):
    """One selectable model: a caption and the Claude model id it names.

    Every model this agent runs is a hosted Claude model reached through one client, so a
    choice is an id with a label for the picker. It used to carry an endpoint and a context
    window as well, while the agent ran on locally served models: those were halves of a pair
    that could be mispaired, and they left with the local path rather than staying behind as
    fields no entry sets.
    """

    label: str
    model: str

    def is_current(self, settings: Settings) -> bool:
        """Whether ``settings`` is already running this choice — by id, never by label."""
        return self.model == settings.model

    def applied_to(self, settings: Settings) -> Settings:
        """``settings`` with this choice in force, ready for ``build_agent``.

        Only the id moves. The credential stays: every entry is served by the same API with the
        same key, so there is no boundary for a switch to cross.
        """
        return dataclasses.replace(settings, model=self.model)


# The workhorse model. Sonnet 5.5 is a strong writer at sensible cost; override with
# SPEECHWRITER_MODEL (e.g. "claude-opus-5-5" for the highest-quality drafting). No ceiling
# override is needed alongside either: langchain-anthropic profiles both ids at their real 128k,
# so tier 2 in `agent._build_model` keeps it. Only an id LangChain cannot profile falls to
# DEFAULT_MAX_TOKENS — which is why `langchain-anthropic` is pinned at the first release that
# profiles the 5.5 ids (1.7.5; 1.6.1 fell back to 4096 for both).
DEFAULT_MODEL = "claude-sonnet-5-5"

# Fallback output-token ceiling — used *only* when the model id has no LangChain profile.
#
# `ChatAnthropic` takes `max_tokens` from LangChain's model-profile table and falls back to 4096
# for an id it does not recognise. Adaptive thinking bills against that same ceiling, so on an
# unrecognised id a subagent can spend the entire budget thinking and return *no text at all* —
# which deepagents forwards as an empty, `status="success"` tool result. 4096 is far too tight
# for that; 32k leaves comfortable room for a draft or critique plus thinking.
#
# Bounded from below by the longest speech anyone commissions: the largest committed eval
# example asks for 3250 words (~4.5k tokens) before the model thinks at all.
# `test_the_default_ceiling_clears_the_longest_commission` reads that floor out of
# `evals/datasets/`. A *profiled* model keeps its own, larger ceiling (128k for the 5.5 ids)
# rather than being capped to this. See `agent._build_model` for the three-tier resolution.
DEFAULT_MAX_TOKENS = 32000

# How hard the model thinks, sent as `output_config.effort` on every call.
#
# Set explicitly rather than left to the API because the default differs by model — `high` on
# Sonnet 5.5, `medium` on Opus 5.5 — so a model switch would otherwise silently change what
# every turn costs. `medium` is the documented starting point for multistep tool use, which is
# what the plan/draft/critique/revise rhythm is. Tune it against the eval suite, not by feel.
#
# Deliberately a constant, not a `SPEECHWRITER_*` knob: such a name is a contract with the
# example dotenv template (`test_env_example_documents_every_setting`), and this is a tuning
# decision for the repo rather than a setting for the machine.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
DEFAULT_EFFORT: Effort = "medium"

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

# The Phoenix project traces land in when `PHOENIX_PROJECT` does not name one — the name the
# LangSmith project had, so a reader switching backends finds the same project name waiting.
DEFAULT_PHOENIX_PROJECT = "speechwriter-agent"

# The curated roster both front ends offer. Every entry must accept `output_config.effort` and
# adaptive thinking, which `agent._build_model` sends on every call — so Haiku 4.5, which
# rejects both, is deliberately absent rather than an entry that 400s the moment it is picked.
# `test_every_model_choice_is_profiled_above_the_floor` keeps each one profiled.
MODEL_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice("Sonnet 5.5", DEFAULT_MODEL),
    ModelChoice("Opus 5.5", "claude-opus-5-5"),
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
    # Appended and defaulted, for the reason `SpeechwriterAgent` defaults its own added fields:
    # `build_agent(settings)` is the documented library entry point, and building the agent
    # never sends it — `ChatAnthropic` accepts a missing key at construction and fails only at
    # the first call, which is what keeps the offline suite offline.
    #
    # NOTE for anyone constructing Settings *positionally*: the local-endpoint fields
    # (`base_url`, `openai_api_key`, `context_window`, `endpoint_configured`) used to sit here,
    # and removing them shifted every field after them. That is a deliberate breaking change
    # rather than vestigial fields kept for compatibility — an endpoint this agent can no longer
    # reach reads as configuration in use.
    anthropic_api_key: str | None = None
    # Appended and defaulted, like the field above. Where traces go — see `speechwriter.tracing`.
    # There is deliberately **no default endpoint**: unset means tracing is off, because a
    # default would start shipping every draft to whatever holds a well-known port without
    # anyone having asked for traces at all. Naming the collector is the opt-in.
    phoenix_endpoint: str | None = None
    phoenix_project: str = DEFAULT_PHOENIX_PROJECT
    # Sent only to `phoenix_endpoint`, for a Phoenix running with authentication enabled.
    phoenix_api_key: str | None = None

    # -- derived helpers -------------------------------------------------

    @property
    def research_enabled(self) -> bool:
        """Live web research is only possible when a Tavily key is present."""
        return bool(self.tavily_api_key)

    @property
    def model_credentials_present(self) -> bool:
        """Whether a turn could be sent at all — the gate both front ends put before the input.

        A presence check, never a probe: validating the key would break the "building the agent
        touches no network" invariant. A key that is present but wrong fails at the first turn,
        with the provider's own 401, which names the problem better than anything checked here.
        """
        return bool(self.anthropic_api_key)

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

    :data:`MODEL_CHOICES` is authoritative; this only ever *adds* to it. The widening matters
    because Streamlit **silently** rewrites a ``session_state`` value that is not among a
    widget's options to option zero — no exception, no log — so a reader who set
    ``SPEECHWRITER_MODEL`` to an id the roster does not list (a pinned older Claude, say) would
    be moved onto Sonnet 5.5 by the first rerun. An off-roster id is offered under its own id
    as the label.

    **Variadic because one configuration is not enough, and passing only the live one is a
    bug.** Selecting any entry replaces ``model``, so a roster derived from the *current*
    settings alone would drop the off-roster id the reader came from — the same silent-retarget
    failure, reached by a click instead of by a rerun, and unrecoverable without a restart.
    Callers therefore pass both the configuration the session *started* on and the one now in
    force.
    """
    choices = list(MODEL_CHOICES)
    seen = {choice.model for choice in choices}
    for settings in offered:
        if settings.model in seen:
            continue
        seen.add(settings.model)
        choices.append(ModelChoice(settings.model, settings.model))
    return tuple(choices)


def resolve_choice(choices: tuple[ModelChoice, ...], requested: str) -> ModelChoice | None:
    """Match a typed argument against a roster by position, label, or model id.

    Shared by the REPL's ``/model`` and the eval harness's ``--model`` so the two cannot
    disagree about what a reader may type — they had the same loop, byte for byte, and only
    one of them resolved against a roster containing the configured entry.

    ``isdecimal``, not ``isdigit``: the latter is true for characters ``int()`` refuses ("²",
    "½"), so an index check built on it raises on a stray keystroke instead of answering.
    Every character ``isdecimal`` accepts, ``int`` parses.

    An ambiguous name resolves to ``None`` rather than to the first match, so the caller
    prints the roster and the reader picks by number — the one input never ambiguous.
    """
    if requested.isdecimal():
        index = int(requested)
        return choices[index - 1] if 1 <= index <= len(choices) else None
    matched = matching_choices(choices, requested)
    return matched[0] if len(matched) == 1 else None


def matching_choices(choices: tuple[ModelChoice, ...], requested: str) -> list[ModelChoice]:
    """Every entry ``requested`` names by label or model id — usually none, or one.

    Split out of :func:`resolve_choice` because ``None`` there answers two different questions
    and the eval harness has to tell them apart: it passes an unrecognised ``--model`` through
    verbatim as an id the operator means literally, which is right for "no such entry" and
    wrong for "two entries by that name".
    """
    wanted = requested.casefold()
    return [
        choice for choice in choices if wanted in (choice.label.casefold(), choice.model.casefold())
    ]


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables and package layout.

    Recognised environment variables:

    * ``ANTHROPIC_API_KEY``  — the Claude API key; required for any turn to run.
    * ``TAVILY_API_KEY``     — enables the live-research subagent; optional.
    * ``SPEECHWRITER_MODEL`` — the Claude model id (default ``claude-sonnet-5-5``).
    * ``SPEECHWRITER_HOME``  — override the project root the agent operates in.
    * ``SPEECHWRITER_MAX_RESEARCH_RESULTS`` — Tavily results per query (default 5).
    * ``SPEECHWRITER_MAX_TOKENS`` — *override* the output-token ceiling per model call.
      Left unset, the model's own LangChain profile decides, falling back to
      ``DEFAULT_MAX_TOKENS`` only for an id that has no profile.
    * ``PHOENIX_COLLECTOR_ENDPOINT`` — a self-hosted Phoenix to trace every turn to, e.g.
      ``http://localhost:6006``. Unset means no tracing; see :mod:`speechwriter.tracing`.
    * ``PHOENIX_PROJECT`` — the Phoenix project traces land in (default
      ``speechwriter-agent``). ``PHOENIX_PROJECT_NAME`` is read as an alias, as Phoenix does.
    * ``PHOENIX_API_KEY`` — sent to that collector only, for a Phoenix with auth enabled.
    """
    project_root = Path(os.environ.get("SPEECHWRITER_HOME", _PKG_DIR.parents[1])).resolve()

    # Load the project's own .env (if present) so ANTHROPIC_API_KEY / TAVILY_API_KEY /
    # PHOENIX_* are available without exporting them by hand. We point at the project
    # root explicitly rather than letting python-dotenv walk *up* the directory tree —
    # an upward walk can pull keys from an unrelated ancestor .env. Done here (not at
    # import) so `import speechwriter` has no side effects; real shell env wins.
    load_dotenv(project_root / ".env")

    # Runtime tracing is Phoenix's now (`speechwriter.tracing`), and so are the eval datasets and
    # experiments, but langsmith still arrives with langchain-core and still switches its own
    # tracer on from LANGSMITH_TRACING alone. So this stays: it keeps LangSmith's own reads of
    # the dotenv honest, for as long as a dotenv written for the old setup can still turn it on.
    #
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
        # Normalised: a blank or whitespace-only value is how a shell says "unset", and left
        # as-is it is *truthy* — so the gate would pass and the first turn would 401 far from
        # the typo.
        anthropic_api_key=_optional_env("ANTHROPIC_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        project_root=project_root,
        workspace_dir=workspace_dir,
        skills_dir=skills_dir,
        store_path=store_path,
        max_research_results=_int_env("SPEECHWRITER_MAX_RESEARCH_RESULTS", 5),
        phoenix_endpoint=_optional_env("PHOENIX_COLLECTOR_ENDPOINT"),
        phoenix_project=(
            _optional_env("PHOENIX_PROJECT")
            or _optional_env("PHOENIX_PROJECT_NAME")
            or DEFAULT_PHOENIX_PROJECT
        ),
        phoenix_api_key=_optional_env("PHOENIX_API_KEY"),
    )


def _optional_env(name: str) -> str | None:
    """A variable's stripped value, or ``None`` when it is unset *or blank*.

    Blank is how a dotenv copied from the example template says "unset" — ``KEY=`` with nothing
    after it — and left as-is an empty string is falsy in some places and a value in others: an
    empty ``PHOENIX_COLLECTOR_ENDPOINT`` would be "configured" to an endpoint no shape check can
    call, and whitespace in a key would be sent as the credential.
    """
    return (os.environ.get(name) or "").strip() or None


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
