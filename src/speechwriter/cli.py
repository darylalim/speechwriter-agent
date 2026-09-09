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

from speechwriter import endpoints
from speechwriter.agent import SpeechwriterAgent, build_agent
from speechwriter.config import (
    DEFAULT_LOCAL_ENDPOINT,
    ModelChoice,
    Settings,
    local_choice,
    model_choices,
    resolve_choice,
)
from speechwriter.transcript import clip, iter_events

_EXIT_WORDS = {"exit", "quit", ":q", "q"}
# The REPL's only other instruction. Slash-prefixed so it cannot collide with a commission —
# every other line typed here is a speech to write, and "model" alone is a plausible brief.
_MODEL_COMMAND = "/model"
# The second instruction, and slash-prefixed for the same reason: "endpoint" alone is a
# plausible word in a brief about infrastructure.
_ENDPOINT_COMMAND = "/endpoint"
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
            f"Type [bold]/model[/] to list or switch models, [bold]/endpoint <url>[/] to add a\n"
            f"local server's models to that list, [bold]exit[/] to save and quit.",
            border_style="magenta",
            padding=(1, 2),
        )
    )


class Command(NamedTuple):
    """One line of input read as an instruction rather than as a commission."""

    name: Literal["exit", "model", "endpoint"]
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
    for word, name in ((_MODEL_COMMAND, "model"), (_ENDPOINT_COMMAND, "endpoint")):
        if lowered == word:
            return Command(name)
        if lowered.startswith(f"{word} "):
            return Command(name, stripped[len(word) :].strip())
    return None


def _roster(
    configured: Settings, bundle: SpeechwriterAgent, detected: list[ModelChoice]
) -> tuple[ModelChoice, ...]:
    """What ``/model`` offers: the curated entries, anything ``/endpoint`` found, and both pairs.

    ``configured`` is the pair the session *started* on, and passing it is what keeps a
    locally served model reachable: selecting a curated entry clears ``base_url``, so a roster
    built from the live bundle alone would drop the local entry and leave no way back to it.

    The argument list is composed exactly as :func:`speechwriter.webui.available_choices`
    composes it, which is what keeps the two front ends offering the same rows in the same
    order. Both go through one :func:`~speechwriter.config.model_choices` call for that reason;
    an order assembled twice is an order that can drift.
    """
    return model_choices(configured, bundle.settings, detected=detected)


def _model_table(
    configured: Settings, bundle: SpeechwriterAgent, detected: list[ModelChoice]
) -> Table:
    """The roster, with the model currently in force marked."""
    settings = bundle.settings
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_column(style="dim")
    for index, choice in enumerate(_roster(configured, bundle, detected), start=1):
        mark = "[bold magenta]›[/]" if choice.is_current(settings) else " "
        table.add_row(f"{mark} {index}", choice.label, choice.base_url or choice.model)
    return table


def _set_endpoint(
    console: Console,
    configured: Settings,
    endpoint: str | None,
    detected: list[ModelChoice],
    requested: str,
) -> tuple[str | None, list[ModelChoice]]:
    """Point ``/endpoint`` at a server and ask what it serves. Returns the endpoint and its models.

    The REPL's half of the same capability the sidebar's endpoint field provides, and the
    terminal is where it matters most: :func:`main` used to *exit* when no model could be
    called, so the reader this feature exists for — a local server, no Anthropic key, nothing
    configured — could never reach the command that would fix it.

    With no argument this only reports, mirroring bare ``/model``. The models found are handed
    back rather than stored, because :func:`main` owns session state; they reach the picker
    through :func:`_roster`, already carrying the endpoint that answered.

    The branches that do not probe return ``detected`` **unchanged**, which is why it is a
    parameter rather than something this function invents. Returning ``[]`` from them read as
    tidy and was a bug: a bare ``/endpoint``, documented as only reporting, silently emptied the
    roster the reader had just built, and so did a typo — leaving them to re-probe a server that
    had never stopped answering. Only a real probe replaces the list.

    Nothing here can raise: :func:`~speechwriter.endpoints.normalize_endpoint` is total and
    :func:`~speechwriter.endpoints.list_models` promises a list. That is not tidiness — an
    exception propagates out of the REPL loop and ends the session over a keystroke, which is
    the failure :func:`~speechwriter.config.resolve_choice` is written to avoid as well.
    """
    if not requested:
        current = endpoint or "[dim]none[/]"
        console.print(f"[dim]endpoint[/]   {current}")
        console.print(
            f"[dim]Point at a server with [bold]/endpoint <url>[/] "
            f"(e.g. [bold]{DEFAULT_LOCAL_ENDPOINT}[/]).[/]"
        )
        return endpoint, detected

    target = endpoints.normalize_endpoint(requested)
    if target is None:
        console.print(f"[yellow]{requested!r} is not an HTTP endpoint.[/]")
        return endpoint, detected

    # Said before the probe, because a slow answer with no explanation reads as a hang — and
    # `mlx_lm.server` answers by walking the whole HuggingFace cache, which genuinely takes
    # seconds on a machine with a lot of models.
    console.print(f"[dim]… asking {target} what it serves[/]")
    found = endpoints.list_models(target, api_key=configured.endpoint_api_key_for(target))
    if not found:
        console.print(f"[yellow]{target} listed no models.[/] [dim]Is the server running?[/]")
        return target, []

    console.print(f"[dim]Found {len(found)} model(s) at {target}. Switch with [bold]/model[/].[/]")
    return target, [local_choice(model, target) for model in found]


def _switch_model(
    console: Console,
    configured: Settings,
    bundle: SpeechwriterAgent,
    requested: str,
    detected: list[ModelChoice],
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
    choices = _roster(configured, bundle, detected)
    if not requested:
        console.print(_model_table(configured, bundle, detected))
        console.print("[dim]Switch with [bold]/model <number>[/] or [bold]/model <name>[/].[/]")
        return bundle, False

    chosen = resolve_choice(choices, requested)
    if chosen is None:
        console.print(f"[yellow]No model matches {requested!r}.[/]")
        console.print(_model_table(configured, bundle, detected))
        return bundle, False

    if chosen.is_current(bundle.settings):
        console.print(f"[dim]Already running {chosen.label}.[/]")
        return bundle, False

    saved = bundle.persist()
    try:
        # Applied to `configured`, not to the live settings: `applied_to` decides whether
        # this endpoint may carry the reader's OPENAI_API_KEY by comparing against the
        # *configured* origin, and after one switch `bundle.settings.base_url` is already
        # the previous pick's. Otherwise identical — the only fields that differ are the
        # ones `applied_to` replaces.
        switched = build_agent(chosen.applied_to(configured))
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
    #
    # A warning rather than `SystemExit`, now that `/endpoint` exists. This gate used to end the
    # process here, and the panel it printed pointed at a local server as the way out — advice
    # the reader could act on only by editing a dotenv and starting again. The way out is a
    # command now, so exiting before the loop would refuse entry to the very reader the panel is
    # addressed to. Commissions are still refused below until a model can actually be called;
    # what is allowed through is the two commands that fix it.
    if not bundle.settings.model_credentials_present:
        console.print(
            Panel(
                "[bold yellow]No ANTHROPIC_API_KEY found.[/]\n\n"
                "Run a local model instead — point the agent at any OpenAI-compatible\n"
                "server and no Anthropic key is needed. Start one, then:\n"
                f"  [bold]/endpoint {DEFAULT_LOCAL_ENDPOINT}[/]\n"
                "  [bold]/model[/]  [dim]to pick from what it serves[/]\n\n"
                "Or set a key in a local dotenv file and restart:\n"
                "  [dim]ANTHROPIC_API_KEY=sk-ant-...[/]\n"
                "and (for live research) [dim]TAVILY_API_KEY=tvly-...[/]",
                border_style="yellow",
                title="No model configured yet",
                padding=(1, 2),
            )
        )

    _banner(console, bundle)

    # One thread for the whole session -> planning state and conversation persist across
    # turns. Rotated only after an interrupt, so we never resume a half-executed graph.
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    seen_ids: set[str] = set()
    # The pair the session started on, captured before any switch can clear `base_url`. Held
    # here rather than re-read, so `/model` keeps offering a locally served model after a
    # detour through a Claude one — otherwise the switch is a one-way trip off the local setup.
    configured = bundle.settings
    # This session's endpoint and whatever it last listed, the terminal's equivalent of the two
    # session-state keys the browser holds. Locals rather than module state for the reason
    # `_dispatch` is pure: everything the REPL remembers should be visible in one function.
    endpoint = configured.base_url
    detected: list[ModelChoice] = []

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
                if command.name == "endpoint":
                    # Deliberately not a switch: no persist, no rebuild, no thread rotation.
                    # Pointing at a server changes what `/model` may offer, never what is
                    # running — the model in force changes only when `/model` says so.
                    endpoint, detected = _set_endpoint(
                        console, configured, endpoint, detected, command.argument
                    )
                    continue
                # Rebinding `bundle` here is what makes the exit-time `persist()` in the
                # `finally` below save the *current* agent's store — which is safe only because
                # `_switch_model` already persisted the outgoing one.
                bundle, switched = _switch_model(
                    console, configured, bundle, command.argument, detected
                )
                if switched:
                    # The rebuild minted a fresh checkpointer, so the old thread names a
                    # checkpoint the new graph has never seen. Same rotation as after an
                    # interrupt, and for the same reason.
                    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
                    seen_ids.clear()
                    _banner(console, bundle)
                    console.print("[dim]↻  Started a fresh thread; earlier context was dropped.[/]")
                continue
            if not bundle.settings.model_credentials_present:
                # The startup panel no longer exits, so this is what stops a commission being
                # spent on a model that cannot be called. Refusing the turn rather than the
                # session is the whole point: `/endpoint` and `/model` are still reachable, and
                # they are what turns this branch off.
                console.print(
                    "[yellow]No model can be called yet.[/] [dim]Point at a local server with "
                    "[bold]/endpoint <url>[/], then choose one with [bold]/model[/].[/]"
                )
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
