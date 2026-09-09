"""Offline tests for the terminal REPL's command surface.

The CLI had no coverage at all until it grew a command. That was defensible while the only
non-brief input was an exit word — there was nothing to get wrong — but ``/model`` rebuilds the
agent, and it carries two failures that are silent rather than loud: rebuilding before
persisting drops everything learned this session (``save_store`` rewrites the snapshot
wholesale), and a banner that reports the *requested* model rather than the built one would
describe an agent that does not exist.

Same bargain as the other two suites: ``build_agent`` calls neither the model nor the network,
so a whole REPL session runs here for free. The scripted input is deliberately finite — running
past the end of it raises rather than blocking, so a dispatch bug fails CI instead of hanging it.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from speechwriter import cli, config
from speechwriter.agent import SpeechwriterAgent


@pytest.fixture
def repl(monkeypatch, tmp_path):
    """A REPL wired to a temp home and the documented default (model, endpoint) pair."""
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    # Both halves of the pair are cleared together, and this delenv now means the opposite of
    # what it used to: there is no hosted client left for it to keep selected, so it pins the
    # session on DEFAULT_MODEL served at DEFAULT_LOCAL_ENDPOINT. The two tests that need an
    # endpoint the session *cannot* call set it back themselves.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    # Tier 1 is global and is now the only thing that can move the ceiling at all — with one
    # client and two tiers, every locally served model resolves to DEFAULT_MAX_TOKENS — so an
    # exported override would change what the banner prints and whether the crowded-window
    # line fires.
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    # Not a credential this suite needs, but a real one in the developer's shell decides which
    # of the two endpoint captions the REPL prints. The one test about that boundary sets its
    # own.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def script(*lines: str) -> None:
        remaining = iter(lines)

        def fake_input(self, *args, **kwargs):
            try:
                return next(remaining)
            except StopIteration:
                raise AssertionError(
                    "the REPL asked for more input than the script provides — it did not treat "
                    "one of these lines as a command"
                ) from None

        monkeypatch.setattr(Console, "input", fake_input)

    return script


def _offer(monkeypatch, *choices: config.ModelChoice) -> None:
    """Put a fixed roster in front of ``/model``, carrying the windows a test needs.

    Every entry a session can reach on its own is synthesised from a configuration that
    exists — the pair the environment names, plus whatever ``/endpoint`` found — and none of
    those can carry a per-model ``context_window``, because ``Settings.context_window`` is
    deliberately never read from the environment. So a REPL cannot reach two models with
    *different* windows by itself, and the window is what has to do the work the resolved
    ceiling used to: with one client and two tiers, every local model resolves to the same
    ``DEFAULT_MAX_TOKENS``, so two bundles are no longer distinguishable by their ceilings.
    Handing :func:`cli._roster` its answer is what makes "this bundle was actually rebuilt"
    observable from the transcript.
    """
    monkeypatch.setattr(cli, "_roster", lambda *args, **kwargs: tuple(choices))


def test_dispatch_reads_commands_and_leaves_commissions_alone():
    # Pure, so it is worth being exhaustive about. The last two cases are the ones that matter:
    # a brief is not a command merely because it begins with the same letters, and the only way
    # to be sure is to require the space.
    assert cli._dispatch("exit") == cli.Command("exit")
    assert cli._dispatch("  QUIT  ") == cli.Command("exit")
    assert cli._dispatch(":q") == cli.Command("exit")
    assert cli._dispatch("/model") == cli.Command("model", "")
    assert cli._dispatch("/model 2") == cli.Command("model", "2")
    assert cli._dispatch("/MODEL Opus 5") == cli.Command("model", "Opus 5")

    assert cli._dispatch("/endpoint") == cli.Command("endpoint", "")
    assert cli._dispatch("/endpoint 127.0.0.1:8080") == cli.Command("endpoint", "127.0.0.1:8080")
    assert cli._dispatch("/ENDPOINT http://h/v1") == cli.Command("endpoint", "http://h/v1")

    assert cli._dispatch("Write a toast for Ana.") is None
    # Not a command: a commission may legitimately open with these letters, and swallowing it
    # would spend the turn printing a model table instead of writing the speech.
    assert cli._dispatch("/modelling the audience as sceptics") is None
    assert cli._dispatch("/endpoints of the argument are what matter") is None
    assert cli._dispatch("exit interviews are the subject of this speech") is None


def test_resolving_a_choice_never_raises_on_a_stray_argument():
    # Lives in `config` because the eval harness's --model resolves the same way; it was
    # duplicated in cli.py, and only one copy searched a roster containing the local entry.
    # `resolve_choice` runs on whatever the reader typed, and it must always *answer* — None is
    # a fine answer, an exception is not, because it propagates out of the REPL loop and ends the
    # session over a keystroke. The superscript is the one that caught this: "²".isdigit() is
    # True while int("²") raises, so `isdigit` was a crash waiting for a stray character.
    # "٣" is the other half — isdecimal() and int() agree on it, so it must still resolve.
    #
    # The roster is synthesised rather than read from `config.MODEL_CHOICES`, which is now the
    # empty tuple: indexing an empty roster answers None for *every* input, so a test built on
    # it would pass while checking nothing at all.
    roster = (
        config.local_choice("local/qwen", "http://127.0.0.1:8080/v1"),
        config.local_choice("local/granite", "http://127.0.0.1:8080/v1"),
        config.local_choice("local/mistral", "http://127.0.0.1:9000/v1"),
    )

    for stray in ("²", "½", "1²", "٣", "0", "-1", "99", "9" * 40, "", "  ", "🙂", "Opus"):
        config.resolve_choice(roster, stray)  # must not raise

    assert config.resolve_choice(roster, "²") is None
    assert config.resolve_choice(roster, "0") is None
    assert config.resolve_choice(roster, "99") is None
    assert config.resolve_choice(roster, "٣") == roster[2]
    assert config.resolve_choice(roster, "2") == roster[1]
    # One string covers both spellings the resolver accepts, because `local_choice` labels an
    # entry with the bare id — and case-folding still has to hold, since a reader picking by
    # name is copying one off the screen.
    assert config.resolve_choice(roster, "local/granite") == roster[1]
    assert config.resolve_choice(roster, "LOCAL/GRANITE") == roster[1]


def test_the_model_command_persists_before_it_rebuilds(repl, monkeypatch):
    # Order, not merely occurrence. `build_agent` rehydrates a brand-new Store from the on-disk
    # snapshot, so a rebuild that runs first throws away every voice profile learned this
    # session — and `save_store` then writes that emptier store back over the file. Nothing
    # raises; the memory is simply gone, which is why this is asserted rather than commented.
    _offer(
        monkeypatch,
        config.local_choice("local/qwen"),
        config.local_choice("local/granite", "http://127.0.0.1:9000/v1"),
    )
    calls: list[str] = []
    real_build = cli.build_agent

    def spy_build(settings=None):
        calls.append("build")
        return real_build(settings)

    monkeypatch.setattr(cli, "build_agent", spy_build)
    monkeypatch.setattr(SpeechwriterAgent, "persist", lambda self: calls.append("persist") or 0)

    repl("/model 2", "exit")
    cli.main()

    # Startup build; then the switch, which must save before it rebuilds; then the exit save.
    assert calls == ["build", "persist", "build", "persist"]


def test_the_banner_reports_the_model_the_bundle_actually_built(repl, monkeypatch, capsys):
    # The banner is re-printed after a switch, and it must describe the agent that now exists.
    # The discriminator used to be the resolved ceiling — Haiku's 64k against Sonnet's 128k —
    # and that is gone: every locally served model resolves through the same two tiers to the
    # same figure. `context_window` is what replaces it, and the banner renders it in the one
    # line that quotes it, so a `_switch_model` that returned the *old* bundle — or a banner
    # reading the requested choice rather than the built one — still shows up here and nowhere
    # else. The endpoint is the second half of the same check: it is the field that decides
    # which server a turn is sent to, and it moves with the model or not at all.
    crowded = config.local_choice("local/granite", "http://127.0.0.1:9000/v1", 4096)
    _offer(monkeypatch, config.local_choice("local/qwen"), crowded)

    repl("/model 2", "exit")
    cli.main()

    printed = capsys.readouterr().out
    before, _, after = printed.partition("local/granite")

    assert after, "the banner never named the model that was switched to"
    assert config.DEFAULT_MODEL in before
    assert "http://127.0.0.1:9000/v1" in after, "the banner kept the endpoint it started on"
    # 8,192 tokens of output against the 4,096-token window this choice declares — read off the
    # bundle that was built, not off the choice that was asked for.
    assert "4,096-token window" in after
    assert "token window" not in before, "the pair it started on was never crowded"


def test_an_unknown_model_leaves_the_session_on_the_one_it_had(repl, capsys, monkeypatch):
    # An unrecognised argument is rejected by name resolution, before anything is built or
    # saved: a typo must cost a line of output, never the session or the snapshot. The two
    # spies are the assertion — "printed a complaint" and "quietly rebuilt onto something
    # else" would otherwise look identical from the transcript.
    #
    # This does *not* reach `_switch_model`'s try/except, which guards a different failure:
    # `init_chat_model` raising at construction for a resolvable-but-unbuildable entry. Name
    # resolution rejects this argument first, so nothing is ever constructed here.
    rebuilt: list[str] = []
    real_build = cli.build_agent
    monkeypatch.setattr(
        cli, "build_agent", lambda settings=None: (rebuilt.append("x"), real_build(settings))[1]
    )

    repl("/model nonsense", "exit")
    cli.main()

    printed = capsys.readouterr().out
    assert "No model matches" in printed
    # One build: the startup one. A rejected argument must not mint a second agent.
    assert len(rebuilt) == 1
    # The table printed alongside the complaint still shows the pair the session is on — which
    # is the roster's only entry, since nothing has been detected and nothing is curated.
    assert config.DEFAULT_MODEL in printed


def test_an_unusable_endpoint_still_reaches_the_command_that_fixes_it(repl, monkeypatch, capsys):
    # `main` used to print a setup panel and `raise SystemExit(1)` when no model could be
    # called — and that panel's own advice was to run a local server. So the one reader it
    # addressed could act on it only by editing a dotenv and starting again, which is exactly
    # what `/endpoint` exists to remove. Exiting before the loop would leave the terminal half
    # of this feature unreachable for the reader it is for.
    #
    # "No model can be called" is no longer a question about a key — there is none to have —
    # but the state it named is still reachable, and by the likeliest dotenv mistake there is:
    # `_configured_endpoint` hands an operator's string back unrewritten, so an endpoint
    # written without a scheme arrives here verbatim and `usable_endpoint` rejects it. Typing
    # the very same text at `/endpoint` fixes it, because *that* path normalises.
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "127.0.0.1:8080")
    monkeypatch.setattr(
        cli.endpoints, "list_models", lambda url, **kw: ["local/qwen", "local/granite"]
    )
    repl("/endpoint 127.0.0.1:8080", "/model", "/model 2", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "Endpoint cannot be used" in out, out
    # Normalised on the way in, so what the reader typed reaches the server as a URL it answers.
    assert "http://127.0.0.1:8080/v1" in out
    # On the roster the *next* command reads, which is the whole point of handing the models
    # back to `main` rather than leaving them inside `_set_endpoint`.
    assert "local/granite" in out
    # And selectable by the number printed beside them: nothing is curated any more, so the
    # detections are merged *before* the configured pair and the second of them is 2. The
    # banner reprinted after the switch is where that is visible — partitioning on the save
    # line is what separates the table's rows from the choice they actually named.
    _, _, after_switch = out.partition("before switching")
    assert "local/granite" in after_switch, "`/model 2` did not name the second row it printed"
    # Picking one leaves a session that can run: the endpoint gets its own line, and the
    # post-switch warning that fires for a pair no turn could reach stays silent.
    assert "endpoint" in after_switch and "local http://127.0.0.1:8080/v1" in after_switch
    assert "is not a URL this can call" not in out, out


def test_a_commission_is_refused_while_no_model_can_be_called(repl, monkeypatch, capsys):
    # The other half of relaxing that gate. Letting the loop start must not let a brief through
    # to an endpoint that cannot be called — the turn would fail deep in the graph rather than
    # at the prompt, and `_run_turn` is where tokens start being spent. What makes an endpoint
    # uncallable is a shape question now rather than a credential one, but the gate is the
    # same one and it still has to refuse the *turn* rather than the session.
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "127.0.0.1:8080")
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_turn", lambda *a, **k: calls.append("turn") or False)
    repl("Write a toast for Ana.", "exit")

    cli.main()

    assert calls == [], "a commission ran with no callable model"
    assert "No model can be called yet" in capsys.readouterr().out


def test_pointing_at_a_server_is_not_a_model_switch(repl, monkeypatch, capsys):
    # `/endpoint` sits one branch away from `/model`, whose three steps (persist, rebuild,
    # rotate the thread) are load-bearing — so the tempting mistake is to run them here too.
    # Pointing at a server changes what may be *offered*; it must not drop the conversation.
    monkeypatch.setattr(cli.endpoints, "list_models", lambda url, **kw: ["local/qwen"])
    built: list[str] = []
    monkeypatch.setattr(cli, "build_agent", _counting(built))
    repl("/endpoint 127.0.0.1:8080", "exit")

    cli.main()

    assert built == ["initial"], f"pointing at a server rebuilt the agent: {built}"


def _counting(log: list[str]):
    """`build_agent`, wrapped so a test can count how many agents a session actually built."""
    real = cli.build_agent

    def wrapper(settings=None):
        log.append("initial" if settings is None else "rebuild")
        return real(settings) if settings is not None else real()

    return wrapper


def test_a_junk_endpoint_answers_rather_than_ending_the_session(repl, monkeypatch, capsys):
    # Same rule `resolve_choice` follows: whatever the reader typed, the REPL must *answer*.
    # An exception here propagates out of the loop and ends the session over a keystroke, and
    # `urlsplit` raises on an unclosed IPv6 bracket — which is one keystroke away now that an
    # endpoint is typed rather than configured.
    probes: list[str] = []
    monkeypatch.setattr(cli.endpoints, "list_models", lambda url, **kw: probes.append(url) or [])
    repl("/endpoint http://[::1", "/endpoint file:///etc", "/endpoint", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "is not an HTTP endpoint" in out
    # And neither reached the network: refusing before the probe is what makes the message
    # "that is not an endpoint" rather than "that endpoint answered nothing".
    assert probes == [], probes


def test_reporting_the_endpoint_does_not_discard_what_it_found(repl, monkeypatch, capsys):
    # `_set_endpoint` returned `[]` from both of its non-probing branches, which read as tidy and
    # was a bug: a bare `/endpoint`, documented as only reporting, silently emptied the roster
    # the reader had just built — and so did a typo, leaving them to re-probe a server that had
    # never stopped answering. Only a real probe may replace the list.
    probes: list[str] = []

    def listing(url, **kwargs):
        probes.append(url)
        return ["local/qwen"]

    monkeypatch.setattr(cli.endpoints, "list_models", listing)
    repl("/endpoint 127.0.0.1:8080", "/endpoint", "/endpoint nonsense://x", "/model", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "local/qwen" in out, "a bare or rejected /endpoint dropped the detections"
    # Neither of the two non-probing commands went near the network.
    assert probes == ["http://127.0.0.1:8080/v1"], probes


def test_a_withheld_credential_is_not_reported_as_a_dead_server(repl, monkeypatch, capsys):
    # `endpoint_api_key_for` withholds OPENAI_API_KEY from a host that is not the configured
    # one, by design — so a hosted gateway typed here answers 401 and `list_models` returns [].
    # Reported as "listed no models. Is the server running?" alone, that sends the reader to
    # debug a server answering perfectly correctly. The sidebar says so beside its own button;
    # the two front ends must not disagree about the same fact.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-gateway")
    monkeypatch.setattr(cli.endpoints, "list_models", lambda url, **kw: [])
    repl("/endpoint https://gateway.example.com/v1", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "listed no models" in out
    assert "No credential sent" in out, out
