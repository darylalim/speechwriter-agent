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
    """A REPL wired to a temp home, a dummy key, and the default Anthropic model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn these tests red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    # Tier 1 is global, so an exported override would clamp every model to one figure and the
    # ceiling assertions below would stop discriminating between them.
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

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

    assert cli._dispatch("Write a toast for Ana.") is None
    # Not a command: a commission may legitimately open with these letters, and swallowing it
    # would spend the turn printing a model table instead of writing the speech.
    assert cli._dispatch("/modelling the audience as sceptics") is None
    assert cli._dispatch("exit interviews are the subject of this speech") is None


def test_resolving_a_choice_never_raises_on_a_stray_argument():
    # Lives in `config` because the eval harness's --model resolves the same way; it was
    # duplicated in cli.py, and only one copy searched a roster containing the local entry.
    # `resolve_choice` runs on whatever the reader typed, and it must always *answer* — None is
    # a fine answer, an exception is not, because it propagates out of the REPL loop and ends the
    # session over a keystroke. The superscript is the one that caught this: "²".isdigit() is
    # True while int("²") raises, so `isdigit` was a crash waiting for a stray character.
    # "٣" is the other half — isdecimal() and int() agree on it, so it must still resolve.
    roster = config.MODEL_CHOICES

    for stray in ("²", "½", "1²", "٣", "0", "-1", "99", "9" * 40, "", "  ", "🙂", "Opus"):
        config.resolve_choice(roster, stray)  # must not raise

    assert config.resolve_choice(roster, "²") is None
    assert config.resolve_choice(roster, "0") is None
    assert config.resolve_choice(roster, "99") is None
    assert config.resolve_choice(roster, "٣") == roster[2]
    assert config.resolve_choice(roster, "2") == roster[1]
    assert config.resolve_choice(roster, "opus 5") == roster[1]
    assert config.resolve_choice(roster, "CLAUDE-OPUS-5") == roster[1]


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

    repl("/model 3", "exit")
    cli.main()

    # Startup build; then the switch, which must save before it rebuilds; then the exit save.
    assert calls == ["build", "persist", "build", "persist"]


def test_the_banner_reports_the_model_the_bundle_actually_built(repl, capsys):
    # The banner is re-printed after a switch, and it must describe the agent that now exists.
    # Haiku 4.5 is the discriminating choice: it is the one roster entry whose ceiling differs
    # from the default's, so a `_switch_model` that returned the *old* bundle — or a banner
    # reading the requested id rather than the built one — shows up here and nowhere else.
    repl("/model 3", "exit")
    cli.main()

    printed = capsys.readouterr().out
    before, _, after = printed.partition("claude-haiku-4-5")

    assert after, "the banner never named the model that was switched to"
    assert "claude-sonnet-5" in before
    assert "128,000" in before
    assert "64,000" in after


def test_an_unknown_model_leaves_the_session_on_the_one_it_had(repl, capsys, monkeypatch):
    # An unrecognised argument is rejected by name resolution, before anything is built or
    # saved: a typo must cost a line of output, never the session or the snapshot. The two
    # spies are the assertion — "printed a complaint" and "quietly rebuilt onto something
    # else" would otherwise look identical from the transcript.
    #
    # This does *not* reach `_switch_model`'s try/except, which guards a different failure:
    # `init_chat_model` raising at construction for a resolvable-but-unbuildable entry. Only a
    # configured local pair can reach that, so it is not exercised from the curated roster.
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
    assert "claude-sonnet-5" in printed
