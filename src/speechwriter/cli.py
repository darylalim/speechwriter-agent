"""Interactive terminal REPL for the speechwriter agent.

Run it with ``python -m speechwriter`` (or the ``speechwriter`` console script).
You type a commission ("Write a 3-minute wedding toast for my sister Ana…"); the
agent plans, optionally researches, drafts, self-critiques, revises, and saves the
speech to ``workspace/speeches/``. On exit, the speaker voice profiles it learned are
snapshotted to disk so the next session remembers them.

Rendering strategy: we stream the graph with ``stream_mode="values"`` (each step
yields the full message list) and print only messages we haven't shown yet, keyed by
message id. Tool calls and results render as dim one-liners; the agent's prose renders
as Markdown. This keeps the transcript readable without needing to know node names.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from speechwriter.agent import SpeechwriterAgent, build_agent
from speechwriter.memory import save_store

_EXIT_WORDS = {"exit", "quit", ":q", "q"}
_PREVIEW_LEN = 90


def _text_of(content: Any) -> str:
    """Flatten a message's content (str or list of Anthropic content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _truncate(text: str, length: int = _PREVIEW_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1] + "…"


def _render_message(console: Console, message: Any) -> None:
    """Render a single new message: tool calls, tool results, or assistant prose."""
    mtype = getattr(message, "type", None)

    if mtype == "ai" or isinstance(message, AIMessage):
        # Tool calls first — these are the agent's actions (plan, research, write, delegate).
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name", "tool")
            args = call.get("args", {})
            preview = _truncate(json.dumps(args, ensure_ascii=False, default=str))
            console.print(f"  [cyan]⚙  {name}[/]  [dim]{preview}[/]")
        # Then any prose the agent emitted.
        text = _text_of(getattr(message, "content", "")).strip()
        if text:
            console.print(
                Panel(
                    Markdown(text),
                    title="[bold]speechwriter[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
    elif mtype == "tool" or isinstance(message, ToolMessage):
        name = getattr(message, "name", "tool")
        status = getattr(message, "status", None)
        mark = "[red]✗[/]" if status == "error" else "[green]✓[/]"
        console.print(f"  {mark} [dim]{name}: {_truncate(_text_of(message.content))}[/]")


def _run_turn(
    console: Console,
    bundle: SpeechwriterAgent,
    user_text: str,
    config: dict[str, Any],
    seen_ids: set[str],
) -> None:
    """Stream one user turn through the agent, rendering new messages as they arrive."""
    payload = {"messages": [{"role": "user", "content": user_text}]}
    try:
        for chunk in bundle.agent.stream(payload, config=config, stream_mode="values"):
            for message in chunk.get("messages", []):
                mid = getattr(message, "id", None) or str(id(message))
                if mid in seen_ids or getattr(message, "type", None) in {"human", "system"}:
                    seen_ids.add(mid)
                    continue
                seen_ids.add(mid)
                _render_message(console, message)
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹  Cancelled this turn.[/] (Your session is still open.)")


def _banner(console: Console, bundle: SpeechwriterAgent) -> None:
    s = bundle.settings
    research = "[green]on (Tavily)[/]" if s.research_enabled else "[yellow]off[/]"
    console.print(
        Panel(
            f"[bold]✒  Speechwriter[/] — a Deep Agent that plans, researches, drafts, "
            f"critiques, and remembers.\n\n"
            f"[dim]model[/]      {s.model}\n"
            f"[dim]research[/]   {research}\n"
            f"[dim]speeches[/]   {s.workspace_dir / 'speeches'}\n"
            f"[dim]memory[/]     {s.store_path}\n\n"
            f"Describe your speech (speaker, audience, occasion, goal, length).\n"
            f"Type [bold]exit[/] to save memory and quit.",
            border_style="magenta",
            padding=(1, 2),
        )
    )


def main() -> None:
    console = Console()
    bundle = build_agent()

    if not bundle.settings.anthropic_api_key:
        console.print(
            Panel(
                "[bold red]No ANTHROPIC_API_KEY found.[/]\n\n"
                "Set it before running, e.g. add a line to a local [bold].env[/] file:\n"
                "  [dim]ANTHROPIC_API_KEY=sk-ant-...[/]\n"
                "and (for live research) [dim]TAVILY_API_KEY=tvly-...[/]",
                border_style="red",
                title="Setup needed",
                padding=(1, 2),
            )
        )
        raise SystemExit(1)

    _banner(console, bundle)

    # One thread for the whole session -> planning state and conversation persist across turns.
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4().hex[:8]}"}}
    seen_ids: set[str] = set()

    try:
        while True:
            try:
                user_text = console.input("\n[bold magenta]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text.lower() in _EXIT_WORDS:
                break
            console.print(Rule(style="dim"))
            _run_turn(console, bundle, user_text, config, seen_ids)
    finally:
        count = save_store(bundle.store, bundle.settings)
        console.print(f"\n[dim]💾 Saved {count} memory item(s) to {bundle.settings.store_path}.[/]")
        console.print("[bold magenta]Until next time. ✒[/]")


if __name__ == "__main__":
    main()
