"""Specialized subagents the orchestrator can delegate to via the ``task`` tool.

Two roles, each chosen for *context isolation* — they run in a fresh window and hand
back only a distilled result, keeping the orchestrator's context clean for writing:

* ``researcher``   — live web research (only present when Tavily is configured).
* ``style-critic`` — a hard editorial pass on a draft.

Subagents are stateless and do **not** inherit the orchestrator's skills, so the
critic is given the skill library explicitly.
"""

from __future__ import annotations

from deepagents import FilesystemPermission, SubAgent

from speechwriter.config import Settings
from speechwriter.prompts import critic_prompt, researcher_prompt
from speechwriter.tools import build_research_tool


def build_subagents(
    settings: Settings, permissions: list[FilesystemPermission] | None = None
) -> list[SubAgent]:
    """Return the ``SubAgent`` dicts to pass to ``create_deep_agent(subagents=...)``.

    ``SubAgent`` is a ``TypedDict``, so annotating these dicts precisely (rather than
    ``dict[str, Any]``) makes the type checker reject a mistyped key — which matters
    here because a silent typo in ``skills`` or ``permissions`` would not fail loudly
    at runtime; the subagent would just quietly lose that capability.

    ``permissions`` is applied to every subagent so the write-sandbox is enforced for
    delegated work too — subagents run their own filesystem middleware and do not
    inherit the orchestrator's permissions.
    """
    subagents: list[SubAgent] = []

    research_tool = build_research_tool(settings)
    if research_tool is not None:
        subagents.append(
            {
                "name": "researcher",
                "description": (
                    "Runs live web research and returns a sourced brief of usable "
                    "facts, statistics, quotable lines, and angles. Delegate any claim "
                    "that needs verification or current context."
                ),
                "system_prompt": researcher_prompt(settings),
                "tools": [research_tool],
            }
        )

    subagents.append(
        {
            "name": "style-critic",
            "description": (
                "Reviews a speech draft for speakability, structure, rhetoric, "
                "audience fit, and authenticity, and returns specific line-level "
                "edits plus a score. Delegate a finished draft here before revising."
            ),
            "system_prompt": critic_prompt(settings),
            # Subagents don't inherit skills — hand the critic the library explicitly
            # so it can consult the same craft criteria the orchestrator uses.
            "skills": [settings.skills_vpath],
        }
    )

    if permissions:
        for subagent in subagents:
            subagent["permissions"] = permissions

    return subagents
