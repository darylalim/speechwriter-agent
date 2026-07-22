"""speechwriter-agent: a speechwriter built with Deep Agents.

The public surface is intentionally small: import :func:`build_agent` to get a
configured Deep Agent, or run the package as a module (``python -m speechwriter``)
for the interactive CLI.

``build_agent`` is exposed lazily via module ``__getattr__`` so that a bare
``import speechwriter`` (or reading ``__version__``) does not eagerly import the
heavy agent stack (deepagents / langchain / langgraph).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["TruncationWarner", "build_agent"]
__version__ = "0.1.0"

if TYPE_CHECKING:
    from speechwriter.agent import build_agent
    from speechwriter.observability import TruncationWarner


def __getattr__(name: str):
    if name == "build_agent":
        from speechwriter.agent import build_agent

        return build_agent
    if name == "TruncationWarner":
        from speechwriter.observability import TruncationWarner

        return TruncationWarner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
