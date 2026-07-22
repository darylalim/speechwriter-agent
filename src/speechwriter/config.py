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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# The workhorse model. Sonnet 5 is a strong writer at sensible cost; override with
# SPEECHWRITER_MODEL (e.g. "claude-opus-4-8" for the highest-quality drafting).
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

    # -- derived helpers -------------------------------------------------

    @property
    def research_enabled(self) -> bool:
        """Live web research is only possible when a Tavily key is present."""
        return bool(self.tavily_api_key)

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
    """
    project_root = Path(os.environ.get("SPEECHWRITER_HOME", _PKG_DIR.parents[1])).resolve()

    # Load the project's own .env (if present) so ANTHROPIC_API_KEY / TAVILY_API_KEY /
    # LANGSMITH_* are available without exporting them by hand. We point at the project
    # root explicitly rather than letting python-dotenv walk *up* the directory tree —
    # an upward walk can pull keys from an unrelated ancestor .env. Done here (not at
    # import) so `import speechwriter` has no side effects; real shell env wins.
    load_dotenv(project_root / ".env")

    workspace_dir = project_root / "workspace"
    skills_dir = project_root / "skills"
    store_path = project_root / ".speechwriter" / "memory-store.json"

    # Ensure the writable dirs exist so the first draft never fails on a missing folder.
    workspace_dir.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        model=os.environ.get("SPEECHWRITER_MODEL", DEFAULT_MODEL),
        max_tokens=_optional_int_env("SPEECHWRITER_MAX_TOKENS"),
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
