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

import uuid
from typing import Any, Literal, NamedTuple

from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from speechwriter.agent import SpeechwriterAgent, build_agent
from speechwriter.config import ModelChoice, Settings, model_choices, resolve_choice
from speechwriter.transcript import clip, iter_events

_EXIT_WORDS = {"exit", "quit", ":q", "q"}
# The REPL's only other instruction. Slash-prefixed so it cannot collide with a commission —
# every other line typed here is a speech to write, and "model" alone is a plausible brief.
_MODEL_COMMAND = "/model"
_PREVIEW_LEN = 90


def _truncate(text: str, length: int = _PREVIEW_LEN) -> str:
    return clip(text, length)


def _render_message(console: Console, message: Any) -> None:
    """Render a single new message: tool calls, tool results, or assistant prose.

    Shares the message→event decode with the web UI via ``iter_events``; only the Rich
    formatting (dim one-liners, a bordered Markdown panel) is the CLI's own.
    """
    for event in iter_events(message):
        if event.kind == "prose":
            console.print(
                Panel(
                    Markdown(event.text),
                    title="[bold]speechwriter[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        elif event.kind == "call":
            console.print(f"  [cyan]⚙  {event.name}[/]  [dim]{_truncate(event.text)}[/]")
        else:
            mark = "[green]✓[/]" if event.ok else "[red]✗[/]"
            console.print(f"  {mark} [dim]{event.name}: {_truncate(event.text)}[/]")


def _run_turn(
    console: Console,
    bundle: SpeechwriterAgent,
    user_text: str,
    config: RunnableConfig,
    seen_ids: set[str],
) -> bool:
    """Stream one user turn, rendering new messages. Returns True if interrupted."""
    payload = {"messages": [{"role": "user", "content": user_text}]}
    try:
        for chunk in bundle.agent.stream(payload, config=config, stream_mode="values"):
            for message in chunk.get("messages", []):
                mid = getattr(message, "id", None) or str(id(message))
                skip = mid in seen_ids or getattr(message, "type", None) in {"human", "system"}
                seen_ids.add(mid)
                if not skip:
                    _render_message(console, message)
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹  Cancelled this turn.[/] (Your session is still open.)")
        return True
    return False


def _report_truncation(console: Console, bundle: SpeechwriterAgent) -> None:
    """Say out loud when the output-token ceiling clipped this turn.

    Nothing else surfaces it — a truncated draft or critique looks exactly like a finished
    one. Called from a ``finally`` so a turn that raises still reports what it saw.
    """
    count = bundle.warner.truncated
    if not count:
        return
    console.print(
        f"[yellow]⚠  {count} model response(s) hit the output-token ceiling and were "
        f"cut off.[/] [dim]Output above may be incomplete — raise SPEECHWRITER_MAX_TOKENS "
        f"(currently {bundle.ceiling_label}).[/]"
    )


def _banner(console: Console, bundle: SpeechwriterAgent) -> None:
    s = bundle.settings
    research = "[green]on (Tavily)[/]" if s.research_enabled else "[yellow]off[/]"
    ceiling = bundle.ceiling_label
    # Shown only when set, so the default Anthropic banner is unchanged. Worth a line of its
    # own: "which model" and "served from where" fail differently, and a local server that is
    # simply not running looks like a hung turn unless the banner said where it was pointed.
    endpoint = f"\n[dim]endpoint[/]   [green]local[/] {s.base_url}" if s.uses_local_endpoint else ""
    # On its own line rather than appended to the ceiling, because it is a different kind of
    # fact: the ceiling is a number, this is "that number will be refused". Only reachable
    # since the model became switchable — a global override outliving the model it was sized
    # for — and it fails at the first turn, so before one is taken is the only useful moment.
    if bundle.ceiling_exceeds_model:
        ceiling += (
            f" [yellow]— above this model's {bundle.profiled_max_tokens:,}; "
            f"unset SPEECHWRITER_MAX_TOKENS[/]"
        )
    console.print(
        Panel(
            f"[bold]✒  Speechwriter[/] — a Deep Agent that plans, researches, drafts, "
            f"critiques, and remembers.\n\n"
            f"[dim]model[/]      {s.model}{endpoint}\n"
            f"[dim]max tokens[/] {ceiling}\n"
            f"[dim]research[/]   {research}\n"
            f"[dim]speeches[/]   {s.workspace_dir / 'speeches'}\n"
            f"[dim]memory[/]     {s.store_path}\n\n"
            f"Describe your speech (speaker, audience, occasion, goal, length).\n"
            f"Type [bold]/model[/] to list or switch models, [bold]exit[/] to save and quit.",
            border_style="magenta",
            padding=(1, 2),
        )
    )


class Command(NamedTuple):
    """One line of input read as an instruction rather than as a commission."""

    name: Literal["exit", "model"]
    argument: str = ""


def _dispatch(user_text: str) -> Command | None:
    """Read a line as a command, or ``None`` when it is a speech to write.

    Pure on purpose — no console, no bundle, no environment — because it is the half of the
    command surface that is worth testing exhaustively and the half that is cheap to test.
    The exit words are absorbed here rather than left as a separate check in :func:`main`, so
    there is exactly one place that decides whether a line is a commission.
    """
    stripped = user_text.strip()
    lowered = stripped.lower()
    if lowered in _EXIT_WORDS:
        return Command("exit")
    # An exact match or a match followed by an argument — never a bare prefix, or a commission
    # beginning "/modelling the audience…" would be swallowed as a command.
    if lowered == _MODEL_COMMAND:
        return Command("model")
    if lowered.startswith(f"{_MODEL_COMMAND} "):
        return Command("model", stripped[len(_MODEL_COMMAND) :].strip())
    return None


def _roster(configured: Settings, bundle: SpeechwriterAgent) -> tuple[ModelChoice, ...]:
    """What ``/model`` offers: the curated entries, the configured pair, and the current one.

    ``configured`` is the pair the session *started* on, and passing it is what keeps a
    locally served model reachable: selecting a curated entry clears ``base_url``, so a roster
    built from the live bundle alone would drop the local entry and leave no way back to it.
    """
    return model_choices(configured, bundle.settings)


def _model_table(configured: Settings, bundle: SpeechwriterAgent) -> Table:
    """The roster, with the model currently in force marked."""
    settings = bundle.settings
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_column(style="dim")
    for index, choice in enumerate(_roster(configured, bundle), start=1):
        mark = "[bold magenta]›[/]" if choice.is_current(settings) else " "
        table.add_row(f"{mark} {index}", choice.label, choice.base_url or choice.model)
    return table


def _switch_model(
    console: Console, configured: Settings, bundle: SpeechwriterAgent, requested: str
) -> tuple[SpeechwriterAgent, bool]:
    """Rebuild on the requested model. Returns the bundle to use and whether it changed.

    With no argument this only lists, which is why the return is a pair rather than a bundle:
    the caller has to know whether to rotate the thread, and "printed a table" and "switched"
    must not look alike.

    ``persist()`` runs **before** ``build_agent``, and the order is the correctness argument:
    the rebuild rehydrates a brand-new Store from the on-disk snapshot, and ``save_store``
    rewrites that file wholesale rather than merging, so reversing these two lines writes the
    *new* bundle's freshly-loaded state over everything the old one learned this session.
    """
    choices = _roster(configured, bundle)
    if not requested:
        console.print(_model_table(configured, bundle))
        console.print("[dim]Switch with [bold]/model <number>[/] or [bold]/model <name>[/].[/]")
        return bundle, False

    chosen = resolve_choice(choices, requested)
    if chosen is None:
        console.print(f"[yellow]No model matches {requested!r}.[/]")
        console.print(_model_table(configured, bundle))
        return bundle, False

    if chosen.is_current(bundle.settings):
        console.print(f"[dim]Already running {chosen.label}.[/]")
        return bundle, False

    saved = bundle.persist()
    try:
        switched = build_agent(chosen.applied_to(bundle.settings))
    except Exception as exc:
        # Reachable from a typo, and not only in theory: `init_chat_model` raises at
        # *construction* for an id whose provider it cannot infer, and for an OpenAI-family id
        # with no key. Uncaught, that propagates out of the REPL loop and ends the session —
        # losing the conversation over a mistyped model name.
        console.print(f"[red]✗  Could not switch to {chosen.label}:[/] {type(exc).__name__}: {exc}")
        return bundle, False

    console.print(f"[dim]💾 Saved {saved} memory item(s) before switching.[/]")
    if not switched.settings.model_credentials_present:
        # The startup gate sits before the loop and cannot see this. Warn rather than refuse:
        # the operator may be about to set the key, and a switch that silently produced an
        # unusable agent would surface as an opaque auth error mid-draft instead.
        console.print(
            "[yellow]⚠  No ANTHROPIC_API_KEY is set, so this model cannot be called.[/] "
            "[dim]Set one, or switch back to a locally served model.[/]"
        )
    return switched, True


def main() -> None:
    console = Console()
    bundle = build_agent()

    # Not `anthropic_api_key` directly: a locally served model needs no key of ours, and
    # demanding one would refuse to start a configuration that runs fine.
    if not bundle.settings.model_credentials_present:
        console.print(
            Panel(
                "[bold red]No ANTHROPIC_API_KEY found.[/]\n\n"
                "Set it before running, e.g. add a line to a local [bold].env[/] file:\n"
                "  [dim]ANTHROPIC_API_KEY=sk-ant-...[/]\n"
                "and (for live research) [dim]TAVILY_API_KEY=tvly-...[/]\n\n"
                "Or run a local model instead — point the agent at any OpenAI-compatible\n"
                "server and no Anthropic key is needed:\n"
                "  [dim]SPEECHWRITER_BASE_URL=http://127.0.0.1:8080/v1[/]\n"
                "  [dim]SPEECHWRITER_MODEL=mlx-community/Qwen3.8-27B-4bit[/]",
                border_style="red",
                title="Setup needed",
                padding=(1, 2),
            )
        )
        raise SystemExit(1)

    _banner(console, bundle)

    # One thread for the whole session -> planning state and conversation persist across
    # turns. Rotated only after an interrupt, so we never resume a half-executed graph.
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    seen_ids: set[str] = set()
    # The pair the session started on, captured before any switch can clear `base_url`. Held
    # here rather than re-read, so `/model` keeps offering a locally served model after a
    # detour through a Claude one — otherwise the switch is a one-way trip off the local setup.
    configured = bundle.settings

    try:
        while True:
            try:
                user_text = console.input("\n[bold magenta]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            command = _dispatch(user_text)
            if command is not None:
                if command.name == "exit":
                    break
                # Rebinding `bundle` here is what makes the exit-time `persist()` in the
                # `finally` below save the *current* agent's store — which is safe only because
                # `_switch_model` already persisted the outgoing one.
                bundle, switched = _switch_model(console, configured, bundle, command.argument)
                if switched:
                    # The rebuild minted a fresh checkpointer, so the old thread names a
                    # checkpoint the new graph has never seen. Same rotation as after an
                    # interrupt, and for the same reason.
                    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                    seen_ids.clear()
                    _banner(console, bundle)
                    console.print("[dim]↻  Started a fresh thread; earlier context was dropped.[/]")
                continue
            console.print(Rule(style="dim"))
            bundle.warner.reset()
            # `turn_config` carries the truncation callback, which propagates into
            # subagent calls; a hand-built config would report nothing.
            config: RunnableConfig = bundle.turn_config(thread_id)
            try:
                interrupted = _run_turn(console, bundle, user_text, config, seen_ids)
            finally:
                # In a `finally` so a turn that raises still reports what it clipped.
                _report_truncation(console, bundle)
            if interrupted:
                thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                console.print("[dim]↻  Started a fresh thread; earlier context was dropped.[/]")
    finally:
        count = bundle.persist()
        console.print(f"\n[dim]💾 Saved {count} memory item(s) to {bundle.settings.store_path}.[/]")
        console.print("[bold magenta]Until next time. ✒[/]")


if __name__ == "__main__":
    main()
