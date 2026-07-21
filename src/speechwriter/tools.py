"""Custom tools for the speechwriter agent.

Right now the only bespoke tool is live web research via Tavily. It lives on the
`researcher` subagent, not the orchestrator, so factual lookups run in an isolated
context and return only a distilled brief.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from speechwriter.config import Settings


def build_research_tool(settings: Settings) -> BaseTool | None:
    """Construct the Tavily web-search tool, or ``None`` if research is disabled.

    ``TavilySearch`` validates ``TAVILY_API_KEY`` at *construction* time and raises
    if it is missing — so we only build it when a key is actually present, letting the
    agent degrade gracefully to knowledge-only drafting instead of crashing.
    """
    if not settings.research_enabled:
        return None

    # Imported lazily so the package imports fine even if the extra isn't installed
    # or the key is absent (e.g. during offline tests).
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=settings.max_research_results,
        topic="general",
        search_depth="advanced",
    )
