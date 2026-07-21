"""speechwriter-agent: a speechwriter built with Deep Agents.

The public surface is intentionally small: import :func:`build_agent` to get a
configured Deep Agent, or run the package as a module (``python -m speechwriter``)
for the interactive CLI.
"""

from speechwriter.agent import build_agent

__all__ = ["build_agent"]
__version__ = "0.1.0"
