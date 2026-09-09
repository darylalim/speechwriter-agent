"""Runtime configuration for the speechwriter agent.

Everything the agent needs to know about *this machine* — which model to call, which
endpoint serves it, which API keys are present, and where files live — is resolved here
into a single frozen :class:`Settings` object. Keeping this in one place means the agent,
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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

from dotenv import load_dotenv

from speechwriter import endpoints

logger = logging.getLogger(__name__)


# Where a locally served model answers when nothing is configured.
#
# `mlx_lm.server`'s own default port, which is also the one the README's local-model recipe
# uses. It is both the placeholder the endpoint field shows *and* — since this agent runs on
# locally served models only — the value `load_settings()` falls back to, so a fresh clone with
# no dotenv at all still names a coherent (model, endpoint) pair rather than a model with
# nowhere to send it. Defined above `ModelChoice` because a NamedTuple's defaults bind at class
# creation, not at call time.
DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8080/v1"


class ModelChoice(NamedTuple):
    """One selectable model: the id *and* where it is served from, as a single record.

    ``model`` and ``base_url`` travel together because they are not independent choices: an
    id names weights, and the same weights served by two machines are two different systems to
    talk to. Every model this agent can run is served over an OpenAI-compatible endpoint, so
    ``base_url`` is **not optional** — an entry carrying an id alone would be a model with
    nowhere to send it, and :func:`~speechwriter.agent._build_model` has no second client to
    fall back to. Pairing them here makes that combination unrepresentable rather than
    something every front end has to remember to validate.

    ``context_window`` is the third field of the same decision: see
    :data:`DEFAULT_LOCAL_CONTEXT_WINDOW`.
    """

    label: str
    model: str
    base_url: str = DEFAULT_LOCAL_ENDPOINT
    context_window: int | None = None

    def is_current(self, settings: Settings) -> bool:
        """Whether ``settings`` is already running this choice.

        The identity of a choice is the pair, never the label — two entries naming the same
        model at the same endpoint are the same choice however they are captioned. Spelled out
        once because both front ends ask this to decide what to mark as current and whether a
        switch is a no-op, and a comparison that quietly dropped ``base_url`` would call the
        same weights on a laptop and on a workstation the same thing.
        """
        return self.model == settings.model and self.base_url == settings.base_url

    def applied_to(self, settings: Settings) -> Settings:
        """``settings`` with this choice in force, ready for ``build_agent``.

        All three fields move together, which is the point: ``base_url`` selects the client and
        ``context_window`` sizes compaction, so applying the model alone would leave a Claude id
        pointed at a local server, or a local model compacting for the previous one's window.

        A **fourth** field moves with them, and it is the one that is easy to miss:
        ``openai_api_key`` is dropped unless this choice names the endpoint ``settings`` was
        configured with. :meth:`Settings.endpoint_api_key_for` draws that boundary for the
        ``/v1/models`` probe, and drawing it only there was a hole rather than a guard — the
        probe withheld the key while the *chat client* went on sending it to the same typed host
        on every turn, which is where it actually matters. Measured on the wire: one turn
        against a typed endpoint carried ``Authorization: Bearer <the reader's real key>``.

        ``settings`` must therefore be the **configured** pair, not the one now in force. Both
        callers pass it — ``webui.get_bundle`` applies to a fresh ``load_settings()`` and
        ``cli._switch_model`` to the ``configured`` it already holds — and applying to either is
        otherwise identical, since the only fields that differ are the three replaced here.
        """
        return dataclasses.replace(
            settings,
            model=self.model,
            base_url=self.base_url,
            context_window=self.context_window,
            openai_api_key=(
                settings.openai_api_key
                # No exemption, deliberately. This test used to carry an `is None` disjunct so a
                # *curated* (Anthropic) entry — which needs no endpoint credential, because that
                # client never reads one — kept the key through a detour away from the local
                # server and back. With every entry now a typed or configured endpoint, that
                # disjunct is a carve-out with no owner: it would hand the reader's real
                # `OPENAI_API_KEY` to any choice that happened to arrive without a `base_url`.
                if endpoints.same_origin(self.base_url, settings.base_url)
                else None
            ),
        )


# The workhorse model — and `DEFAULT_LOCAL_ENDPOINT` is the other half of the same decision,
# never to be defaulted apart from it. `_build_model` keys the client off the endpoint, so an
# id defaulted alone would be a model with nowhere to send it.
#
# This names weights, and naming them is not the same as having them. `mlx_lm.server` serves
# whatever is already in the Hugging Face cache and pulls nothing itself, so a fresh machine
# gets a server that starts, reports Ready, and 404s on the first turn — which is why the
# README's recipe now downloads this repo explicitly before starting one. Cheap to fix either
# way: point `SPEECHWRITER_MODEL` at what you actually serve, or ask the server what it *has* —
# the sidebar's **Detect models** button and the REPL's `/endpoint` both read `/v1/models`.
#
# Deliberately not a smaller, likelier-to-be-cached id: there is no id every machine has, so any
# choice is a guess, and this one at least matches the documented recipe end to end.
DEFAULT_MODEL = "mlx-community/Qwen3.8-27B-4bit"

# The output-token ceiling, used whenever `SPEECHWRITER_MAX_TOKENS` does not override it.
#
# Pinned rather than left to the client. `ChatOpenAI` defaults `max_tokens` to None — "let the
# server decide" — and an unbounded ceiling on a *reasoning* model is a real trap: Qwen3.8-27B
# defaults to `reasoning_effort: xhigh` and will happily spend a thousand tokens deliberating
# before it writes a line. Extended thinking bills against this same ceiling, so a subagent can
# exhaust the budget thinking and return *no text at all*, which deepagents forwards as an
# empty `status="success"` tool result.
#
# **Bounded from both sides, and the lower bound is the one that is easy to forget.**
#
# From above by `DEFAULT_LOCAL_CONTEXT_WINDOW`: output and input share one window on a local
# server. This was 32000 while the ceiling applied to 128k-window hosted models, which against a
# 32768-token window leaves 768 tokens for the entire prompt. vLLM rejects that outright; others
# clamp it silently, which is worse.
#
# From below by the longest speech anyone actually commissions. The ceiling is also the
# *thinking* budget, and a draft is not small: the largest committed eval example asks for 3250
# words (a 25-minute keynote at `WORDS_PER_MINUTE`), which is ~4.5k tokens of prose before
# Qwen3.8-27B spends anything on reasoning. A first pass at this constant chose 8192 by halving
# until it looked safe against the window alone, which is how a ceiling ends up truncating the
# one commission the datasets grade most heavily.
#
# 12288 clears that draft with room for a long deliberation and still leaves ~20k of window for
# the system prompt, the loaded skills and the draft under revision.
# `test_the_default_ceiling_fits_inside_the_assumed_context_window` pins both directions.
DEFAULT_MAX_TOKENS = 12288

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

# Empty on purpose, and it must stay a real (if empty) roster rather than be deleted.
#
# This held three hosted Anthropic ids while the agent had a hosted client to call them with.
# It cannot be repopulated with local ones: a hard-coded endpoint is a guess about which server
# the reader is running, and every guess that misses is a row in the picker that 404s when
# selected. What replaces it is `model_choices()`, which synthesises entries from the pair the
# environment actually names and from whatever a live `/v1/models` probe answers — correct by
# construction, because the operator configured the one and the server reported the other.
#
# Kept as a name because it is still the seed `model_choices()` builds on. Note where the
# non-empty guarantee moved to: `model_choices()` synthesises an entry for every configuration
# handed to it, and both front ends always hand it at least one, so the roster a picker sees is
# never empty even though this is. `streamlit_app.py` indexes it (`choices[0]`) and depends on
# exactly that — a caller passing no settings at all would hand it an empty list.
MODEL_CHOICES: tuple[ModelChoice, ...] = ()

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
    # The OpenAI-compatible endpoint the model is served from. Not optional and not nullable:
    # it selects the client, and there is no second client to fall back to — see `ModelChoice`.
    # Defaulted for the reason `SpeechwriterAgent` defaults its own added fields:
    # `build_agent(settings)` is the documented library entry point, so a consumer constructing
    # Settings by hand should not have to name a machine to get the documented one.
    #
    # NOTE for anyone constructing Settings *positionally*: `anthropic_api_key` used to sit
    # second, and removing it shifted every field after it by one. That is a deliberate
    # breaking change rather than a vestigial field kept for compatibility — a key this agent
    # can no longer send is worse than absent, because it reads as a credential in use.
    base_url: str = DEFAULT_LOCAL_ENDPOINT
    openai_api_key: str | None = None
    # Appended and defaulted for the same reason as the two above — and, alone among these
    # fields, never read from the environment. `load_settings()` leaves it None; it is set
    # only by `dataclasses.replace` when a front end switches to a locally served model, and
    # is read only by `agent._build_model`. See `DEFAULT_LOCAL_CONTEXT_WINDOW` for what it
    # buys and why it is not a `SPEECHWRITER_*` knob.
    context_window: int | None = None
    # Appended, like every field above it. Whether ``SPEECHWRITER_BASE_URL`` was actually *set*,
    # as opposed to defaulted to `DEFAULT_LOCAL_ENDPOINT` — a distinction that did not exist
    # while an unset value meant "no endpoint at all", and that guards a credential now that it
    # means "the documented local one". See `endpoint_api_key`.
    endpoint_configured: bool = False

    # -- derived helpers -------------------------------------------------

    @property
    def research_enabled(self) -> bool:
        """Live web research is only possible when a Tavily key is present."""
        return bool(self.tavily_api_key)

    @property
    def model_endpoint_usable(self) -> bool:
        """Whether the configured endpoint is one we could actually send a turn to.

        The gate both front ends put in front of the chat input, and it replaced
        ``model_credentials_present`` when the Anthropic client left: a locally served model
        needs no credential of ours, so "is a key present" stopped being a question with an
        answer, and a gate that is unconditionally true is a gate that has quietly stopped
        running.

        What is left is *shape*, and it is worth checking precisely because
        :func:`_configured_endpoint` deliberately does not rewrite what an operator wrote.
        :func:`~speechwriter.endpoints.usable_endpoint` rejects the two values that fail
        confusingly rather than loudly: a scheme outside ``http``/``https`` — ``file:///…``
        would otherwise have the model client and the ``/v1/models`` probe reading local disk —
        and a URL with no host, which raises inside ``urllib`` at *request construction*, far
        from the dotenv that caused it.

        Reachability is deliberately **not** part of this. Probing here would break the
        "building the agent touches no network" invariant; a server that is merely not running
        yet is what **Detect models** and ``/endpoint`` are for.
        """
        return endpoints.usable_endpoint(self.base_url) is not None

    @property
    def endpoint_api_key(self) -> str:
        """The bearer token to send to ``base_url``, never empty.

        Resolved here rather than in :mod:`speechwriter.agent` so that reading credentials
        out of the environment stays this module's job — the same reason ``base_url`` is a
        field and not an ``os.environ`` lookup at the call site.

        **A real key is sent only to an endpoint the operator actually named.** This became a
        live question when ``base_url`` gained a default: while unset meant "no endpoint", an
        ``OPENAI_API_KEY`` sitting in the environment for some hosted service was simply never
        read. Now unset means ``DEFAULT_LOCAL_ENDPOINT``, so without this guard a developer with
        that variable exported globally — an ordinary thing to have — would send their real key
        to whatever process happens to hold ``127.0.0.1:8080`` on every turn, having configured
        nothing. Loopback bounds the damage; it does not make it intended.

        A key with no named server is not a credential for *this* server, so it is withheld and
        the placeholder goes instead. Naming the endpoint is what opts in, which is the same
        rule :meth:`endpoint_api_key_for` and :meth:`ModelChoice.applied_to` already draw for a
        *typed* endpoint — one boundary, three places it has to hold.
        """
        if not self.endpoint_configured:
            return LOCAL_API_KEY_PLACEHOLDER
        return self.openai_api_key or LOCAL_API_KEY_PLACEHOLDER

    def endpoint_api_key_for(self, target: str) -> str | None:
        """The bearer to send when asking ``target`` what it serves — ``None`` unless it is
        the configured server.

        :func:`~speechwriter.endpoints.list_models` licenses forwarding the reader's key with
        "no new disclosure: the chat client already sends the very same credential to this very
        same host". That argument is exact, and it stops holding the moment the host is *typed*
        rather than configured: a reader with a real ``OPENAI_API_KEY`` for a hosted gateway who
        types a colleague's laptop address would hand that key to a machine the operator never
        named, over plaintext HTTP, on the very first request — before
        :class:`~speechwriter.endpoints._CredentialSafeRedirects`, which guards only the
        *second* hop, can see it.

        So the credential boundary is the same for both hops: the configured origin, and
        nothing else. The cost is a typed hosted endpoint answering 401, which the sidebar
        reports as "no credential sent" rather than as silence — different facts deserve
        different captions.
        """
        return self.endpoint_api_key if endpoints.same_origin(target, self.base_url) else None

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


def model_choices(
    *offered: Settings, detected: Iterable[ModelChoice] = ()
) -> tuple[ModelChoice, ...]:
    """The curated roster, widened so every configuration passed in stays selectable.

    :data:`MODEL_CHOICES` is empty, so this is not merely the authoritative roster — it is the
    *only* one. Every entry a front end offers is synthesised here from a configuration that
    exists: the pair ``SPEECHWRITER_BASE_URL`` and ``SPEECHWRITER_MODEL`` name, whatever
    :func:`speechwriter.endpoints.list_models` found at a live endpoint, and the pair currently
    in force. That is correct by construction — the operator configured the one and the server
    reported the other — where a shipped list of local endpoints would be a guess about which
    server the reader is running, dead on every machine that guessed wrong.

    The widening rule survives the empty seed and still matters: Streamlit **silently** rewrites
    a ``session_state`` value that is not among a widget's options to option zero — no
    exception, no log. So this may only ever *add* to what the environment already says.

    **Variadic because one configuration is not enough, and passing only the live one is a
    bug.** Selecting any entry replaces ``base_url`` with that entry's own, so a roster derived
    from
    the *current* settings alone would drop the locally served entry the reader came from —
    the same silent-retarget failure this function exists to prevent, reached by a click
    instead of by a rerun, and unrecoverable without a restart. On a keyless machine it is
    worse than losing an option: picking a Claude id there also disables the chat input, so
    the one entry that still works has just been removed from the list. Callers therefore
    pass both the configuration the session *started* on and the one now in force.

    ``detected`` carries whatever a live endpoint was asked for — see
    :func:`speechwriter.endpoints.list_models`. Two things about it are load-bearing.

    They arrive as whole :class:`ModelChoice` records, **never as bare ids**, because the
    endpoint they were found at is the half that cannot be re-derived later. A caller handing
    over ids and letting this function attach an endpoint would attach *some* endpoint — the
    configured one, or whatever a text field says now — and a pair naming the wrong server
    builds a client pointed at it. Detected as a pair, stored as a pair, offered as a pair.

    And they are merged **before** ``offered``, which is ordering rather than taste: ``offered``
    varies with what is *selected* and ``detected`` does not, so a detection placed after it
    would change position the moment it was picked — "picking one detected model reordered the
    rest of them", which the picker cannot survive, since ``index=choices.index(current)`` names
    a row number that must mean the same thing on the next render.
    """
    choices = list(MODEL_CHOICES)
    seen = {(choice.model, choice.base_url) for choice in choices}
    for choice in detected:
        if (choice.model, choice.base_url) in seen:
            continue
        seen.add((choice.model, choice.base_url))
        choices.append(choice)
    for settings in offered:
        identity = (settings.model, settings.base_url)
        if identity in seen:
            continue
        seen.add(identity)
        choices.append(local_choice(settings.model, settings.base_url, settings.context_window))
    return tuple(choices)


def resolve_choice(choices: tuple[ModelChoice, ...], requested: str) -> ModelChoice | None:
    """Match a typed argument against a roster by position, label, or model id.

    Shared by the REPL's ``/model`` and the eval harness's ``--model`` so the two cannot
    disagree about what a reader may type — they had the same loop, byte for byte, and only
    one of them resolved against a roster containing the locally served entry.

    ``isdecimal``, not ``isdigit``: the latter is true for characters ``int()`` refuses ("²",
    "½"), so an index check built on it raises on a stray keystroke instead of answering.
    Every character ``isdecimal`` accepts, ``int`` parses.

    **An ambiguous name is not an answer, so it resolves to ``None``.** One model id can be
    served by two machines — the same ``mlx-community/…`` weights on a laptop and a workstation,
    or an identical Ollama tag — and :func:`local_choice` labels both with the bare ``<id>``,
    because the label has to stay a pure function of the pair. Returning the first match
    silently pointed the agent at whichever server happened to be merged earlier. The caller
    prints the roster on ``None``, and that table carries ``base_url`` in a column of its own,
    so the reader can see the two apart and pick by number — the one input never ambiguous.
    """
    if requested.isdecimal():
        index = int(requested)
        return choices[index - 1] if 1 <= index <= len(choices) else None
    matched = matching_choices(choices, requested)
    return matched[0] if len(matched) == 1 else None


def matching_choices(choices: tuple[ModelChoice, ...], requested: str) -> list[ModelChoice]:
    """Every entry ``requested`` names by label or model id — usually none, or one.

    Split out of :func:`resolve_choice` because ``None`` there answers two different questions
    and one caller has to tell them apart. The eval harness passes an unrecognised ``--model``
    through verbatim as an id the operator means literally, which is right for "no such entry"
    and wrong for "two entries by that name": it would set ``SPEECHWRITER_MODEL`` while leaving
    ``SPEECHWRITER_BASE_URL`` alone, which is precisely the id-without-its-endpoint mispairing
    :class:`ModelChoice` exists to make unrepresentable.
    """
    wanted = requested.casefold()
    return [
        choice for choice in choices if wanted in (choice.label.casefold(), choice.model.casefold())
    ]


def local_choice(
    model: str, base_url: str = DEFAULT_LOCAL_ENDPOINT, context_window: int | None = None
) -> ModelChoice:
    """A roster entry for ``model`` served at ``base_url``, labelled the one way.

    The label convention lives here, and in exactly one place, because equality decides
    correctness. ``ModelChoice`` is a ``NamedTuple``, so two entries for the same served model
    are equal only if their *labels* match too — and a front end that discovers models from a
    live endpoint has to produce entries indistinguishable from the ones
    :func:`model_choices` synthesises, or the same model appears twice in the picker and the
    selected one is not found among the options. Streamlit's response to that is to silently
    reset the selection to the first entry.

    The label is now the bare id. It used to carry a ``(local)`` suffix, which earned its place
    only by contrast — an unsuffixed id sat beside three Anthropic ones and would have read as
    hosted. With every entry served over an endpoint the suffix distinguishes nothing, and it is
    not free: the picker truncates at ~34 characters, which ``mlx-community/Qwen3.8-27B-4bit``
    already reaches on its own. Where the model came from is still shown, on the line of its own
    that the CLI banner and the sidebar caption both give it — "which model" and "served from
    where" fail differently and belong in different sentences.
    """
    return ModelChoice(model, model, base_url, context_window)


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables and package layout.

    Recognised environment variables:

    * ``TAVILY_API_KEY``     — enables the live-research subagent; optional.
    * ``SPEECHWRITER_MODEL`` — the id your server serves (default
      ``mlx-community/Qwen3.8-27B-4bit``). Half of a pair — see ``SPEECHWRITER_BASE_URL``.
    * ``SPEECHWRITER_HOME``  — override the project root the agent operates in.
    * ``SPEECHWRITER_MAX_RESEARCH_RESULTS`` — Tavily results per query (default 5).
    * ``SPEECHWRITER_MAX_TOKENS`` — *override* the output-token ceiling per model call.
      Left unset, the model's own LangChain profile decides, falling back to
      ``DEFAULT_MAX_TOKENS`` only for an id that has no profile.
    * ``SPEECHWRITER_BASE_URL`` — the OpenAI-compatible endpoint serving the model (a local
      ``mlx_lm.server``, vLLM, LM Studio, Ollama). Unset falls back to
      :data:`DEFAULT_LOCAL_ENDPOINT`; there is no hosted path to fall back to instead.
    * ``OPENAI_API_KEY`` — sent to that endpoint when one is set. Local servers ignore
      it, so it is optional and falls back to a placeholder; a gateway in front of one
      (LiteLLM, a reverse proxy) may need a real one.
    """
    project_root = Path(os.environ.get("SPEECHWRITER_HOME", _PKG_DIR.parents[1])).resolve()

    # Load the project's own .env (if present) so TAVILY_API_KEY / OPENAI_API_KEY /
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

    configured_endpoint, endpoint_configured = _configured_endpoint()

    return Settings(
        model=os.environ.get("SPEECHWRITER_MODEL", DEFAULT_MODEL),
        max_tokens=_optional_int_env("SPEECHWRITER_MAX_TOKENS"),
        # Verbatim, deliberately — see `_configured_endpoint` for what normalising it broke.
        base_url=configured_endpoint,
        endpoint_configured=endpoint_configured,
        # Normalised the same way, and for the same reason: a blank or whitespace-only value
        # is how a shell says "unset", and left as-is it is *truthy* — so `endpoint_api_key`
        # would send "   " as the bearer token and a hosted endpoint would 401 far from the
        # typo, instead of falling back to the placeholder.
        openai_api_key=(os.environ.get("OPENAI_API_KEY") or "").strip() or None,
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        project_root=project_root,
        workspace_dir=workspace_dir,
        skills_dir=skills_dir,
        store_path=store_path,
        max_research_results=_int_env("SPEECHWRITER_MAX_RESEARCH_RESULTS", 5),
    )


def _configured_endpoint() -> tuple[str, bool]:
    """``SPEECHWRITER_BASE_URL`` and whether it was set — stripped, and nothing more.

    Returns the pair rather than the string because the two answers come from one read and a
    second ``os.environ`` lookup could disagree with the first. The flag is what
    :attr:`Settings.endpoint_api_key` gates a real credential on.

    It is named rather than inlined because what it *does not* do is the point, and a previous
    version got this wrong. It normalised the value — to stop one server appearing twice in the
    picker when a dotenv carried a trailing slash — and thereby rewrote endpoints that worked:
    an Azure deployment URL lost the ``?api-version=`` query it requires, and a LiteLLM or
    reverse-proxy front end serving the OpenAI API at the **root** gained a ``/v1`` that 404s.
    Both measured. ``build_agent`` never probes, so each failed at the first turn, far from the
    configuration that caused it, with no log line.

    A configured endpoint is an operator's deliberate string and every character of it may be
    load-bearing, so it reaches ``ChatOpenAI`` exactly as written. The duplicate that
    normalisation was reaching for is prevented on the other side instead:
    :func:`speechwriter.endpoints.usable_endpoint` reads a seeded value back *unchanged*, so
    detections against a configured server carry the same string the configured pair does, and
    dedup on ``(model, base_url)`` collapses them.

    The empty case is deliberate: an exported-but-blank ``SPEECHWRITER_BASE_URL`` is how a
    shell says "unset", and an empty string here would route every call to a nonexistent
    endpoint that no shape check could tell apart from a real one.

    Unset falls back to :data:`DEFAULT_LOCAL_ENDPOINT` rather than to ``None``. There is no
    second client to select by leaving this empty, so ``None`` could only ever mean "a model
    with nowhere to send it" — a state better made unrepresentable than reported. Note what
    this still does *not* do: an endpoint that is set but malformed is returned as written, for
    :attr:`Settings.model_endpoint_usable` to reject by name, because silently substituting the
    default for an operator's own string is how a typo becomes an unexplained wrong server.
    """
    raw = (os.environ.get("SPEECHWRITER_BASE_URL") or "").strip()
    return (raw or DEFAULT_LOCAL_ENDPOINT, bool(raw))


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
