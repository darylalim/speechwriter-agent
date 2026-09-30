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

# Profiled at 128k by LangChain, where an unprofiled id resolves to the 32k floor — so a banner
# quoting one figure or the other says which bundle was actually built.
_UNPROFILED = config.ModelChoice("Mystery", "claude-not-a-real-model-9")


@pytest.fixture
def repl(monkeypatch, tmp_path):
    """A REPL wired to a temp home, the default model, and a (dummy) key."""
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    # Tier 1 is global, so an exported override would change what the banner prints and
    # whether the over-the-model's-maximum clause fires.
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    # Dummy, and never sent: nothing here reaches `_run_turn` except through a spy. Set so the
    # credentials gate is open by default; the one test about the closed gate unsets it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")

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
    """Put a fixed roster in front of ``/model``.

    The curated roster's two entries are both profiled at the same 128k ceiling, so two bundles
    built from them are indistinguishable from the transcript by their ceilings. Handing
    :func:`cli._roster` an unprofiled entry beside one of them is what makes "this bundle was
    actually rebuilt" observable.
    """
    monkeypatch.setattr(cli, "_roster", lambda *args, **kwargs: tuple(choices))


def test_dispatch_reads_commands_and_leaves_commissions_alone():
    # Pure, so it is worth being exhaustive about. The last cases are the ones that matter: a
    # brief is not a command merely because it begins with the same letters, and the only way
    # to be sure is to require the space.
    assert cli._dispatch("exit") == cli.Command("exit")
    assert cli._dispatch("  QUIT  ") == cli.Command("exit")
    assert cli._dispatch(":q") == cli.Command("exit")
    assert cli._dispatch("/model") == cli.Command("model", "")
    assert cli._dispatch("/model 2") == cli.Command("model", "2")
    assert cli._dispatch("/MODEL Opus 5.5") == cli.Command("model", "Opus 5.5")

    assert cli._dispatch("Write a toast for Ana.") is None
    # Not a command: a commission may legitimately open with these letters, and swallowing it
    # would spend the turn printing a model table instead of writing the speech.
    assert cli._dispatch("/modelling the audience as sceptics") is None
    assert cli._dispatch("exit interviews are the subject of this speech") is None
    # The command that pointed at a local server is gone with the local path; its old spelling
    # is now just the opening of a brief.
    assert cli._dispatch("/endpoint http://127.0.0.1:8080/v1") is None


def test_resolving_a_choice_never_raises_on_a_stray_argument():
    # Lives in `config` because the eval harness's --model resolves the same way; it was
    # duplicated in cli.py, and only one copy searched a roster containing the configured entry.
    # `resolve_choice` runs on whatever the reader typed, and it must always *answer* — None is
    # a fine answer, an exception is not, because it propagates out of the REPL loop and ends the
    # session over a keystroke. The superscript is the one that caught this: "²".isdigit() is
    # True while int("²") raises, so `isdigit` was a crash waiting for a stray character.
    # "٣" is the other half — isdecimal() and int() agree on it, so it must still resolve.
    roster = (*config.MODEL_CHOICES, _UNPROFILED)
    assert len(roster) >= 3, "the index assertions below need three rows"

    for stray in ("²", "½", "1²", "٣", "0", "-1", "99", "9" * 40, "", "  ", "🙂", "Opus"):
        config.resolve_choice(roster, stray)  # must not raise

    assert config.resolve_choice(roster, "²") is None
    assert config.resolve_choice(roster, "0") is None
    assert config.resolve_choice(roster, "99") is None
    assert config.resolve_choice(roster, "٣") == roster[2]
    assert config.resolve_choice(roster, "2") == roster[1]
    # By label and by id, case-folded — a reader picking by name is copying one off the screen.
    assert config.resolve_choice(roster, "opus 5.5") == roster[1]
    assert config.resolve_choice(roster, "CLAUDE-OPUS-5-5") == roster[1]


def test_the_model_command_persists_before_it_rebuilds(repl, monkeypatch):
    # Order, not merely occurrence. `build_agent` rehydrates a brand-new Store from the on-disk
    # snapshot, so a rebuild that runs first throws away every voice profile learned this
    # session — and `save_store` then writes that emptier store back over the file. Nothing
    # raises; the memory is simply gone, which is why this is asserted rather than commented.
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
    # The ceiling is the discriminator: the default is profiled at 128k, the unprofiled entry
    # resolves to the 32k floor — so a `_switch_model` that returned the *old* bundle, or a
    # banner reading the requested choice rather than the built one, shows up here and nowhere
    # else.
    _offer(monkeypatch, config.MODEL_CHOICES[0], _UNPROFILED)

    repl("/model 2", "exit")
    cli.main()

    printed = capsys.readouterr().out
    before, _, after = printed.partition(_UNPROFILED.model)

    assert after, "the banner never named the model that was switched to"
    assert config.DEFAULT_MODEL in before
    assert "128,000" in before
    assert f"{config.DEFAULT_MAX_TOKENS:,}" in after, "the banner kept the ceiling it started on"
    assert "128,000" not in after


def test_an_unknown_model_leaves_the_session_on_the_one_it_had(repl, capsys, monkeypatch):
    # An unrecognised argument is rejected by name resolution, before anything is built or
    # saved: a typo must cost a line of output, never the session or the snapshot. The two
    # spies are the assertion — "printed a complaint" and "quietly rebuilt onto something
    # else" would otherwise look identical from the transcript.
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
    # The table printed alongside the complaint still shows the model the session is on.
    assert config.DEFAULT_MODEL in printed


def test_a_missing_key_warns_and_still_opens_the_session(repl, monkeypatch, capsys):
    # A warning, not `SystemExit`: the session opens so `/model` and `exit` still work, and the
    # zero-cost smoke test (`printf 'exit\n' | uv run speechwriter`) runs on a machine with no
    # key at all — CI's machine, for one.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    repl("/model", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "No ANTHROPIC_API_KEY found" in out, out
    assert config.DEFAULT_MODEL in out  # the banner and the table both still render


def test_a_commission_is_refused_while_no_model_can_be_called(repl, monkeypatch, capsys):
    # The other half of that warning. Letting the loop start must not let a brief through to a
    # call that can only 401 — the turn would fail deep in the graph rather than at the prompt,
    # and `_run_turn` is where tokens start being spent. Refused per turn, not per session.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_turn", lambda *a, **k: calls.append("turn") or False)
    repl("Write a toast for Ana.", "exit")

    cli.main()

    assert calls == [], "a commission ran with no callable model"
    assert "No model can be called yet" in capsys.readouterr().out


def test_a_refused_turn_is_reported_rather_than_passed_off_as_finished(repl, monkeypatch, capsys):
    # A classifier decline inside a subagent reaches the orchestrator as an empty success, so
    # the transcript alone looks like a turn that simply said less. The warner counts it; this is
    # the half that says so out loud, with the category a reader needs to tell a false positive
    # from a brief that genuinely strayed.
    def refused_turn(console, bundle, *args, **kwargs):
        bundle.warner.refused = 1
        bundle.warner.refusal_categories = ["general_harms"]
        return False

    monkeypatch.setattr(cli, "_run_turn", refused_turn)
    repl("Write a toast for Ana.", "exit")

    cli.main()

    out = capsys.readouterr().out
    assert "declined by a safety classifier (general_harms)" in out, out
