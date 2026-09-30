"""The eval datasets under ``evals/`` are gated here, so drift blocks like anything else.

``evals/validate_datasets.py`` recomputes every derived number in the 55 examples from
``config.WORDS_PER_MINUTE``, resolves their save paths through ``load_settings()``, and
checks every tool name, subagent name and skill slug against the live contract. Without
this test it was an advisory script nobody ran -- weaker than the hook CLAUDE.md deleted
in favour of tests, since it caught nothing on a push, a PR, or a hand edit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from speechwriter import config

REPO_ROOT = config._PKG_DIR.parents[1]


def test_eval_datasets_match_the_live_contract():
    # A subprocess, and not an in-process import, for the same reason
    # test_import_speechwriter_is_lazy uses one. The validator calls load_settings(), whose
    # dotenv load would read the project's real secrets file once SPEECHWRITER_HOME points at
    # the repo -- and that loader sets any variable not already present, so a real Tavily key
    # would land in os.environ for every test that ran afterwards. That is an order-dependent
    # flake, in a suite whose whole premise is that it runs offline with no keys. The child's
    # own environment is inherited and harmless: it validates JSON and exits, and nothing it
    # sets propagates back here.
    #
    # SPEECHWRITER_HOME is pinned to the real repo rather than a tmp_path -- the one place in
    # the suite where that is right, because the committed datasets are exactly what is under
    # test. It is still set explicitly, never inherited: a developer with the variable exported
    # would otherwise validate someone else's tree.
    env = os.environ | {"SPEECHWRITER_HOME": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "validate_datasets.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "evals/validate_datasets.py reported failures:\n" + result.stdout + result.stderr
    )


def _sync_module():
    """Import ``evals/sync_datasets.py`` by path.

    It is importable at all only because it has no module-level side effects -- unlike
    ``validate_datasets.py``, which runs its entire check at import and calls ``sys.exit``, and
    so can only ever be driven through a subprocess. Nothing reached here touches the wire:
    ``load_settings`` and the LangSmith client both live behind ``main()``, which is why these
    tests stay offline and free like the rest of the suite.
    """
    path = REPO_ROOT / "evals" / "sync_datasets.py"
    spec = importlib.util.spec_from_file_location("speechwriter_sync_datasets", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], and under `from __future__ import annotations` an
    # unregistered module makes that lookup return None and raise inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_roster_matches_the_validator():
    # sync_datasets.DESCRIPTIONS and validate_datasets.EXPECTED_COUNTS are two independent
    # literals naming the same four datasets, and nothing structural ties them: the validator
    # runs its whole check at import and exits, so the mirror cannot import the roster from it.
    # Same shape and same remedy as test_tool_pins_agree_wherever_they_are_declared.
    #
    # The failure is silent in the direction that matters. A fifth dataset added to
    # evals/datasets/ and to EXPECTED_COUNTS -- but not to DESCRIPTIONS -- validates clean,
    # prints "IN SYNC", and is simply never mirrored: the checker only looks at what it names.
    sync = _sync_module()
    source = (REPO_ROOT / "evals" / "validate_datasets.py").read_text(encoding="utf-8")
    block = re.search(r"EXPECTED_COUNTS\s*=\s*\{(.*?)\}", source, re.S)
    assert block, "EXPECTED_COUNTS not found in validate_datasets.py -- was it renamed?"
    validator_roster = set(re.findall(r'"([a-z_]+)"\s*:', block.group(1)))

    # Anti-vacuity: the comparison below is satisfied by two empty sets, so a regex that stopped
    # matching would turn this green while checking nothing. Pinned at 4 for the same reason
    # len(skill_dirs) == 4 is pinned.
    assert len(validator_roster) == 4, (
        f"parsed {sorted(validator_roster)} out of EXPECTED_COUNTS; expected 4 dataset names"
    )
    assert set(sync.DESCRIPTIONS) == validator_roster, (
        f"sync_datasets.DESCRIPTIONS names {sorted(sync.DESCRIPTIONS)} but "
        f"validate_datasets.EXPECTED_COUNTS names {sorted(validator_roster)}. A dataset in one "
        f"and not the other is validated but never mirrored, or mirrored but never validated."
    )
    for stem, description in sync.DESCRIPTIONS.items():
        # Not cosmetic: the trajectory description is the only place that tells a grader
        # expected_trajectory is a reference path, and LangSmith's client can set a description
        # only when it creates a dataset, so an empty one is fixed only by deleting and recreating.
        assert description.strip(), f"{stem}: empty description"


def test_sync_reads_every_committed_dataset():
    # load_local is the mirror's only reader and it rejects an example the diff could not key
    # on. Running it over the real files makes a missing metadata.id fail here, offline, rather
    # than at the wire. The 55 is pinned like EXPECTED_COUNTS, and for the same reason.
    sync = _sync_module()
    counts = {stem: len(sync.load_local(stem)) for stem in sync.DESCRIPTIONS}
    assert sum(counts.values()) == 55, f"expected 55 examples across four files, got {counts}"


def test_sync_diff_detects_every_kind_of_drift():
    # An edited output and a local-only example each turn the checker red, and a clean tree
    # turns it green. This pins that behaviour with no key and no network.
    sync = _sync_module()

    def loc(eid, n):
        return {"inputs": {"q": eid}, "outputs": {"n": n}, "metadata": {"id": eid}}

    def rem(eid, uuid, n):
        return {"uuid": uuid, "inputs": {"q": eid}, "outputs": {"n": n}, "metadata": {"id": eid}}

    plan = sync.diff_examples(
        local=[loc("keep", 1), loc("edited", 2), loc("added", 3)],
        remote=[
            rem("keep", "u1", 1),
            rem("edited", "u2", 99),
            rem("removed", "u3", 4),
            # No metadata.id: added through the LangSmith UI. A push-only mirror removes it,
            # which is what makes "the remote equals the repo" a fact rather than a hope.
            {"uuid": "u4", "inputs": {}, "outputs": {}, "metadata": {}},
        ],
    )
    assert [e["metadata"]["id"] for e in plan["create"]] == ["added"]
    assert [e["id"] for e in plan["update"]] == ["edited"]
    assert [e["fields"] for e in plan["update"]] == [["outputs"]], "changed field not named"
    # The update carries the server's own UUID, which is what `update_examples` needs.
    assert [e["uuid"] for e in plan["update"]] == ["u2"]
    assert sorted(e["uuid"] for e in plan["delete"]) == ["u3", "u4"]
    assert [e["id"] for e in plan["unchanged"]] == ["keep"]
    assert not sync.plan_is_clean(plan)
    assert sync.plan_is_clean(
        sync.diff_examples(local=[loc("keep", 1)], remote=[rem("keep", "u1", 1)])
    )


def test_sync_strips_the_split_langsmith_injects_but_refuses_a_local_one(tmp_path):
    # LangSmith writes dataset_split into every example's metadata. Comparing it would report
    # all 55 as drifted on a key no local file ever wrote -- the false positive as_record strips.
    # The local side is the opposite: declaring it claims a field the server overwrites, which
    # is a schema decision to make deliberately rather than absorb, so load_local refuses it.
    sync = _sync_module()

    class _Example:
        id = "u1"
        inputs = {"q": 1}
        outputs = {"n": 1}
        metadata = {"id": "keep", "dataset_split": ["base"]}

    record = sync.as_record(_Example())
    assert record["metadata"] == {"id": "keep"}, "dataset_split reached the comparison"
    assert sync.plan_is_clean(
        sync.diff_examples(
            local=[{"inputs": {"q": 1}, "outputs": {"n": 1}, "metadata": {"id": "keep"}}],
            remote=[record],
        )
    ), "a server-injected split read as drift"

    planted = tmp_path / "planted.json"
    planted.write_text(
        json.dumps(
            [{"inputs": {}, "outputs": {}, "metadata": {"id": "x", "dataset_split": ["a"]}}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_split"):
        sync.load_local("planted", path=planted)


def test_sync_refuses_a_dataset_it_could_not_key(tmp_path):
    # load_local is the mirror's only reader. A missing metadata.id could not be keyed at all. A
    # repeated one would pass the *check* silently: the diff keys local examples by id, so one
    # of the pair drops out of the plan and "IN SYNC" is printed over a file holding an example
    # the server never saw. Both fail here.
    sync = _sync_module()

    unkeyed = tmp_path / "unkeyed.json"
    unkeyed.write_text(
        json.dumps([{"inputs": {}, "outputs": {}, "metadata": {}}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata.id"):
        sync.load_local("unkeyed", path=unkeyed)

    twice = tmp_path / "twice.json"
    twice.write_text(
        json.dumps([{"inputs": {}, "outputs": {}, "metadata": {"id": "x"}}] * 2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="used twice"):
        sync.load_local("twice", path=twice)


def test_the_mirror_is_unavailable_without_a_key_never_in_sync(monkeypatch, tmp_path, capsys):
    # A check that has quietly stopped running must not look like one that is passing: no key
    # is exit 2, never 0, and it never reaches for the network to find out.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    sync = _sync_module()

    assert sync.main([]) == sync.EXIT_UNAVAILABLE
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def _harness_module():
    """Import ``evals/run_experiment.py`` by path — it has no module-level side effects."""
    path = REPO_ROOT / "evals" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("speechwriter_run_experiment", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_eval_judge_does_not_follow_the_model_under_test(monkeypatch, tmp_path):
    # `--model` moves the system under test; the judge must stay put. `grade()` builds its model
    # from `load_settings()`, which reads the same SPEECHWRITER_MODEL that `apply_model` sets —
    # so without the pinned settings, comparing two models grades each one with *itself*,
    # changing the instrument and the subject together and making the comparison meaningless.
    #
    # This exists because the bug shipped: the `judge` parameter was threaded into the
    # experiment-recording path and not into the plain live path, and nothing noticed.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5-5")

    harness = _harness_module()
    # Captured before the override, exactly where `main()` captures it.
    pinned = harness.configured_settings()

    # ...and now the system under test moves.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-sonnet-5-5")

    graded_with: list[str] = []

    def spy_build(settings):
        graded_with.append(settings.model)
        return object()

    monkeypatch.setattr("speechwriter.agent._build_model", spy_build)
    monkeypatch.setattr(harness, "score_example", lambda *a, **k: [])
    monkeypatch.setattr(harness, "judge_example", lambda *a, **k: [])

    harness.grade("final_response", harness.CANNED, {}, no_judge=False, judge=pinned)

    assert graded_with == ["claude-opus-5-5"], (
        "the judge followed the environment instead of the pinned settings, so every model "
        "would be graded by itself"
    )

    # And with no override in play the judge is the configured model, exactly as before.
    graded_with.clear()
    harness.grade("final_response", harness.CANNED, {}, no_judge=False)
    assert graded_with == ["claude-sonnet-5-5"]


def test_every_live_grading_path_pins_the_judge_when_the_model_is_overridden():
    # The mechanical half of the bug above: it was not that `grade()` ignored its argument, it
    # was that one of the two call sites never passed one. Read the source rather than the
    # behaviour, because the second path (`--langsmith`) needs a live workspace to exercise.
    source = (REPO_ROOT / "evals" / "run_experiment.py").read_text(encoding="utf-8")
    calls = re.findall(r"\b(run_one|run_langsmith)\((.*?)\)", source)
    invocations = [(name, args) for name, args in calls if "args." in args]

    assert {name for name, _ in invocations} == {"run_one", "run_langsmith"}, (
        "a live path is no longer called by these names — this test is watching the wrong ones"
    )
    for name, args in invocations:
        # Split and compare whole arguments rather than searching the text: `args.no_judge` is
        # passed to both paths and *contains* "judge", so a substring check passes even on the
        # unfixed source. It did — this assertion was vacuous on its first mutation run.
        passed = {argument.strip() for argument in args.split(",")}
        assert "judge" in passed, (
            f"{name}({args}) does not pass the pinned judge, so --model would make that path "
            f"grade every model with itself"
        )


def test_the_model_flag_accepts_the_label_the_front_ends_actually_show(monkeypatch, tmp_path):
    # `--model` is documented as taking "a roster label", and the roster a reader sees in
    # `/model` or the sidebar is `model_choices(...)`. Resolving against anything narrower once
    # passed the label through as a literal model id, which 404s at the first turn of every
    # graded example — long after the temp homes are set up.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5")

    harness = _harness_module()
    roster = config.model_choices(config.load_settings())

    # A curated label resolves to its id...
    harness.apply_model(config.MODEL_CHOICES[1].label)
    assert os.environ["SPEECHWRITER_MODEL"] == config.MODEL_CHOICES[1].model
    # ...and so does the off-roster entry the environment configured, labelled as it is shown.
    harness.apply_model(roster[-1].label)
    assert os.environ["SPEECHWRITER_MODEL"] == "claude-opus-5"

    # An id nothing on the roster matches is still passed through verbatim — the operator may
    # mean a model the environment has never named.
    harness.apply_model("claude-opus-4-8")
    assert os.environ["SPEECHWRITER_MODEL"] == "claude-opus-4-8"


def _evaluators_module():
    """Import ``evals/evaluators.py`` by path — pure scorers, no model and no wire."""
    path = REPO_ROOT / "evals" / "evaluators.py"
    spec = importlib.util.spec_from_file_location("speechwriter_evaluators", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], and under `from __future__ import annotations` an
    # unregistered module makes that lookup return None and raise inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_must_not_contain_split_follows_the_quoting_convention():
    # validate_datasets.py rejects a half-quoted entry precisely so this split is a mechanical
    # transform rather than a guess. If that convention ever loosens, this is what notices: a
    # bare entry routed to the literal branch would be searched for as a substring and always
    # pass, silently deleting a criterion.
    ev = _evaluators_module()
    literals, prose = ev.split_must_not_contain(
        ['"follow your passion"', "an advert for the speaker's firm", '"at the end of the day"']
    )
    assert literals == ["follow your passion", "at the end of the day"]
    assert prose == ["an advert for the speaker's firm"]

    # Every committed entry must land in exactly one branch, with nothing lost.
    fr = json.loads((REPO_ROOT / "evals" / "datasets" / "final_response.json").read_text("utf-8"))
    entries = [e for x in fr for e in x["outputs"]["must_not_contain"]]
    lit, pro = ev.split_must_not_contain(entries)
    assert len(lit) + len(pro) == len(entries) and entries, "entries lost in the split"


def test_order_constraints_parse_or_report_unscored_but_never_silently_pass():
    # The rule this module owns: a criterion it cannot read scores None, never 1.0. Three of the
    # twelve committed constraints have a prose side ("before any clarifying question about
    # Daryl's voice") and must land in the unscored tail — counting them as passes would inflate
    # every trajectory result by criteria nothing measured.
    ev = _evaluators_module()
    tr = json.loads((REPO_ROOT / "evals" / "datasets" / "trajectory.json").read_text("utf-8"))
    constraints = sorted({c for e in tr for c in (e["outputs"].get("order_constraints") or [])})
    assert len(constraints) >= 10, (
        f"only {len(constraints)} constraints found — did the shape change?"
    )

    empty = ev.RunRecord(text="", calls=())
    verdicts = {c: ev.check_order_constraint(empty, c) for c in constraints}
    parsed = [c for c, v in verdicts.items() if v.machine_scored]

    # Lower bound: the parser has not rotted against the shapes the datasets actually use.
    unparsed = [v.comment for v in verdicts.values() if not v.machine_scored]
    assert len(parsed) >= 9, (
        f"only {len(parsed)}/{len(constraints)} constraints parse; the shapes in the datasets "
        f"drifted from _SIDE. Unparsed: {unparsed}"
    )
    # Upper bound, and the half that matters. Without it the assertion above is satisfied by a
    # checker that scores EVERYTHING 1.0 — more constraints "parse", the test goes green, and the
    # rule this module exists to enforce is inverted. These three have a prose side; naming them
    # by substring keeps the test readable through small wording edits.
    must_be_unscored = (
        "before any clarifying question",
        "or the final overwriting write_file",
        "the revised draft is saved before",
    )
    for needle in must_be_unscored:
        matches = [c for c in constraints if needle in c]
        assert matches, f"no committed constraint contains {needle!r} — update this test"
        for c in matches:
            assert not verdicts[c].machine_scored, (
                f"a constraint with a prose side scored {verdicts[c].score!r} instead of None: "
                f"{c!r}. Silently passing what the parser cannot read inflates every trajectory "
                f"result with criteria nothing measured."
            )
    for v in verdicts.values():
        if not v.machine_scored:
            assert "not machine-checkable" in v.comment or "unparseable" in v.comment


def test_order_constraint_direction_is_actually_checked():
    # Mutation-proofed: reversing the run must flip the verdict, or the checker is asserting
    # only that both sides occurred and the word "before" is decoration.
    ev = _evaluators_module()
    write = ev.ToolCall("write_file", {"file_path": "/workspace/speeches/a.md"})
    critic = ev.ToolCall("task", {"subagent_type": "style-critic"})
    rule = "write_file(/workspace/speeches/) before task(style-critic)"
    assert ev.check_order_constraint(ev.RunRecord("", (write, critic)), rule).score == 1.0
    assert ev.check_order_constraint(ev.RunRecord("", (critic, write)), rule).score == 0.0
    # Vacuous when a side never occurs — the semantics every example declares.
    assert ev.check_order_constraint(ev.RunRecord("", (write,)), rule).score == 1.0


def test_trajectory_scorer_discriminates_a_bad_run_from_a_good_one():
    ev = _evaluators_module()
    tr = json.loads((REPO_ROOT / "evals" / "datasets" / "trajectory.json").read_text("utf-8"))
    example = next(e for e in tr if e["metadata"]["id"] == "ceremonial-wedding-toast-best-man")

    good = ev.RunRecord(
        "toast",
        (
            ev.ToolCall("read_file", {"file_path": "/skills/speech-structures/SKILL.md"}),
            ev.ToolCall("write_file", {"file_path": "/workspace/speeches/toast.md"}),
            ev.ToolCall("task", {"subagent_type": "style-critic"}),
        ),
    )
    bad = ev.RunRecord("toast", (ev.ToolCall("task", {"subagent_type": "researcher"}),))
    good_scores = ev.score_example("trajectory", good, example)
    bad_scores = ev.score_example("trajectory", bad, example)
    good_pass = sum(1 for s in good_scores if s.score == 1.0)
    bad_pass = sum(1 for s in bad_scores if s.score == 1.0)
    assert good_pass > bad_pass, (
        f"scorer cannot tell a good run from one that called the forbidden researcher and never "
        f"wrote a draft ({good_pass} vs {bad_pass})"
    )
    forbidden = next(s for s in bad_scores if s.key == "forbidden_subagents")
    assert forbidden.score == 0.0, "the forbidden researcher was not caught"


def test_word_count_scoring_uses_the_same_rule_the_browser_shows(monkeypatch, tmp_path):
    # Isolated: the scorer calls `load_settings()`, which on the real repo would read the
    # developer's dotenv into `os.environ` for the rest of the process (see CLAUDE.md).
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # spoken_words is workspace.py's, so a draft's header block and its [pause] cues are dropped
    # here exactly as they are in the web UI. A second implementation would let the eval and the
    # browser disagree about the same file — which is the drift config.py is single-sourced for.
    ev = _evaluators_module()
    from speechwriter.workspace import spoken_words

    draft = "---\ntitle: T\n---\n\n" + " ".join(["word"] * 130) + " [pause]"
    assert spoken_words(draft) == 130
    example = {
        "outputs": {"target_word_count": 130, "word_count_tolerance": 0.15, "saved_to": ""},
        "metadata": {},
    }
    scores = ev.score_example("final_response", ev.RunRecord(draft, ()), example)
    assert next(s for s in scores if s.key == "word_count").score == 1.0
    short = ev.score_example("final_response", ev.RunRecord("too short", ()), example)
    assert next(s for s in short if s.key == "word_count").score == 0.0


def test_coverage_never_counts_what_it_could_not_measure():
    ev = _evaluators_module()
    scores = [
        ev.Score("a", 1.0, ""),
        ev.Score("b", 0.0, ""),
        ev.Score("c", None, "prose"),
        ev.Score("d", None, "prose"),
    ]
    assert ev.coverage(scores) == 0.5
    assert [s.key for s in ev.unscored(scores)] == ["c", "d"]
    assert ev.coverage([]) == 0.0


def test_a_graded_runs_temp_home_can_build_the_agent_with_its_skills(tmp_path, monkeypatch):
    # run_experiment gives every graded run its own SPEECHWRITER_HOME so write-path criteria are
    # attributable -- but that variable overrides project_root, and skills_dir is
    # project_root / "skills", so a bare temp home runs the agent with the rhetoric library
    # absent. create_deep_agent only *logs* a missing skills tree, so the run would have produced
    # plausible numbers for a differently-configured agent.
    #
    # Copying the tree is the fix, and a symlink is NOT: Settings._vpath maps a real path to a
    # virtual one through `path.resolve().relative_to(project_root)`, and resolve() follows the
    # link back to the real repo, which is not under the temp root. skills_vpath then raises
    # inside orchestrator_prompt -- which is how the first live run died, on all four datasets,
    # before a single model call. Both halves are asserted below.
    #
    # Free and offline: build_agent() never touches the wire, which is the invariant that makes
    # this test possible at all.
    shutil.copytree(REPO_ROOT / "skills", tmp_path / "skills")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # No key is set, as in CI: `ChatAnthropic` constructs without one and fails only at the
    # first call, so build_agent() needs no credential to wire the graph up.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from speechwriter.agent import build_agent
    from speechwriter.prompts import orchestrator_prompt

    bundle = build_agent()
    prompt = orchestrator_prompt(bundle.settings)
    assert bundle.settings.skills_vpath == "/skills/"
    slugs = sorted(p.parent.name for p in bundle.settings.skills_dir.glob("*/SKILL.md"))
    assert len(slugs) == 4, f"the graded run would see {slugs}, not the four committed skills"
    assert "/skills/" in prompt

    # And the harness must COPY that tree, never link it. Asserted against the source rather
    # than by provoking the failure, so improving Settings._vpath does not fail this test for a
    # reason that is not a bug.
    source = (REPO_ROOT / "evals" / "run_experiment.py").read_text(encoding="utf-8")
    assert "copytree" in source, "run_experiment no longer copies the skills tree into the home"
    assert "symlink_to" not in source, (
        "run_experiment symlinks the skills tree. Settings._vpath maps a real path to a virtual "
        "one through path.resolve().relative_to(project_root), and resolve() follows the link "
        "back to the real repo -- outside the temp root -- so skills_vpath raises inside "
        "orchestrator_prompt. That killed the first live run on all four datasets before a "
        "single model call. Copy it."
    )


def _harness_module():
    """Import ``evals/run_experiment.py`` — module level is import-safe, the wire is behind main."""
    path = REPO_ROOT / "evals" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("speechwriter_run_experiment", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assistant_text_is_read_from_content_blocks_not_only_plain_strings():
    # AIMessage.content is a plain string for some responses and a list of content blocks for
    # others. Accepting only str is how the first live run scored a finished 1,560-word
    # commencement address as "the output is empty" -- while its own saved_to check confirmed
    # the file had been written. Thinking and tool_use blocks are deliberately not joined:
    # they are not what the speaker says.
    harness = _harness_module()

    class Msg:
        def __init__(self, content):
            self.content = content
            self.type = "ai"

    assert harness.message_text(Msg("plain string")) == "plain string"
    assert harness.message_text(Msg([{"type": "text", "text": "block text"}])) == "block text"
    assert (
        harness.message_text(
            Msg(
                [
                    {"type": "thinking", "thinking": "should not appear"},
                    {"type": "text", "text": "the speech"},
                    {"type": "tool_use", "name": "write_file", "input": {}},
                ]
            )
        )
        == "the speech"
    )
    assert harness.message_text(Msg(None)) == ""


def test_an_empty_output_cannot_bank_passes_on_absence_criteria(monkeypatch, tmp_path):
    # Isolated: the scorer calls `load_settings()`, which on the real repo would read the
    # developer's dotenv into `os.environ` for the rest of the process (see CLAUDE.md).
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # Every negative criterion in the suite -- "contains no advert", "invents no statistic",
    # "quotes nothing unsourced" -- is trivially satisfied by producing nothing. Before the
    # liveness precondition a run that returned no text scored 7/17 on this very example and
    # reported 85% coverage, so a harness bug read as a mediocre model rather than a broken run.
    ev = _evaluators_module()
    fr = json.loads((REPO_ROOT / "evals" / "datasets" / "final_response.json").read_text("utf-8"))
    example = next(
        e for e in fr if e["metadata"]["id"] == "ceremonial-commencement-first-job-not-the-verdict"
    )
    wrote_a_file = (ev.ToolCall("write_file", {"file_path": "/workspace/speeches/x.md"}),)

    empty = ev.RunRecord("", wrote_a_file)
    scores = ev.score_example("final_response", empty, example)
    scores += ev.judge_example(None, "final_response", empty, example)

    liveness = next(s for s in scores if s.key == "produced_output")
    assert liveness.score == 0.0, "an empty reply was not flagged"
    # The judge is never even called — a None model would raise if it were.
    assert all(
        not s.machine_scored
        for s in scores
        if s.key.startswith(("must_mention", "must_not_contain", "required_behaviors"))
    ), "an absence-based criterion scored a pass against an empty output"

    # And the same example with real text must still reach the deterministic absence check,
    # or the guard above would have disabled scoring wholesale.
    alive = ev.RunRecord("A speech about Kettleworth and the first job.", wrote_a_file)
    literal = next(
        s
        for s in ev.score_example("final_response", alive, example)
        if s.key == "must_not_contain_literal"
    )
    assert literal.machine_scored, "the literal check stopped running for non-empty output"


class _StubModel:
    """Minimal stand-in for a chat model: records prompts and methods, replays canned verdicts.

    ``method`` is keyword-only and **required**, which is the whole reason it appears here. The
    stub used to accept ``with_structured_output(schema)`` and ignore how the schema was asked
    for -- so every judge test passed while the suite was structurally blind to the one part of
    that call that fails on transport rather than on merit. See
    :func:`test_the_judge_asks_for_structured_output_a_local_server_can_answer`.
    """

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []
        self.methods: list[str] = []

    def with_structured_output(self, _schema, *, method):
        self.methods.append(method)
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self._verdicts.pop(0)


def test_the_judge_asks_for_structured_output_without_forcing_a_tool():
    # `langchain-anthropic` defaults `with_structured_output` to method="function_calling",
    # which binds the schema as a tool and *forces* the call. The 5.5 models reject forced tool
    # choice with a 400, so on the default path every judged example fails on the transport —
    # uniformly enough to read as "the judge disagrees" rather than "the judge never ran", the
    # same family of measurement bug this file already documents five of, all of which made the
    # agent look worse than it is. `json_schema` goes through `output_config.format` instead.
    #
    # The literal is pinned here rather than compared against the module's own constant, which
    # would pass for any value: what makes `json_schema` right is what the API accepts, not what
    # evaluators.py says about itself. It was `function_calling` while the judge ran on
    # `mlx_lm.server`, which is the other half of why it is pinned.
    ev = _evaluators_module()
    assert ev.JUDGE_STRUCTURED_OUTPUT_METHOD == "json_schema"

    # All three call sites, because `_structured` exists precisely so none of them can forget:
    # a site that did would not fail here, it would fail at the API, once, in whichever
    # criterion happened to use it.
    hits = _StubModel([{"verdict": "mention", "reason": "quoted in order to reject it"}])
    ev.judge_literal_hits(
        hits,
        ev.RunRecord("They will tell you to follow your passion. I will not.", ()),
        {"must_not_contain": ['"follow your passion"']},
    )

    questions = _StubModel([{"count": 0, "reason": "both are rhetorical, inside the draft"}])
    ev.judge_question_count(
        questions, ev.RunRecord("What is resilience? Is it endurance?", ()), {"max_questions": 0}
    )

    criteria = _StubModel(
        [{"verdicts": [{"index": 0, "satisfied": True, "applicable": True, "reason": "ok"}]}]
    )
    ev.judge_criteria(criteria, "must_cover", "a speech", ["names the occasion"])

    for stub in (hits, questions, criteria):
        assert stub.methods == ["json_schema"], (
            f"a judge call asked for structured output as {stub.methods!r}; function_calling "
            f"forces a tool call, which is a 400 on the 5.5 models"
        )


def test_the_real_judge_client_sends_no_forced_tool_choice(monkeypatch, tmp_path):
    # The stub above proves the helper *asks* for json_schema; this proves what that means on
    # the wire for the client the judge is actually built from. A langchain-anthropic that
    # started forcing a tool under json_schema too would pass the stub test and 400 every call.
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableSequence

    from speechwriter.agent import _build_model

    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    ev = _evaluators_module()
    model = _build_model(config.load_settings())
    schema = {
        "title": "Probe",
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    bound = ev._structured(model, schema)
    # `with_structured_output` returns model | parser; the bound model's kwargs are the payload
    # additions it would send.
    assert isinstance(bound, RunnableSequence)
    first = bound.first
    kwargs = getattr(first, "kwargs", {})
    payload = model._get_request_payload([HumanMessage("hi")], **kwargs)

    assert "tool_choice" not in payload, payload.get("tool_choice")
    assert payload.get("output_config", {}).get("format"), sorted(payload)


def test_a_banned_phrase_escalates_instead_of_failing_outright(monkeypatch, tmp_path):
    # Isolated: the scorer calls `load_settings()`, which on the real repo would read the
    # developer's dotenv into `os.environ` for the rest of the process (see CLAUDE.md).
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # A substring search cannot tell USE from MENTION, and the briefs themselves invite the
    # mention ("Don't tell them to follow your passion"), so a speech that quotes the cliche in
    # order to reject it was being failed for doing what was asked. The live run hit exactly
    # this: must_not_contain_literal FAILED on "follow your passion" while the judge passed
    # required_behaviors[3] saying the phrase "is never used as advice".
    #
    # Clean output stays fully deterministic and calls no model at all; only a hit escalates.
    ev = _evaluators_module()
    fr = json.loads((REPO_ROOT / "evals" / "datasets" / "final_response.json").read_text("utf-8"))
    example = next(
        e for e in fr if e["metadata"]["id"] == "ceremonial-commencement-first-job-not-the-verdict"
    )

    clean = ev.RunRecord("Kettleworth taught me one thing. The first job is not the verdict.", ())
    row = next(
        s
        for s in ev.score_example("final_response", clean, example)
        if s.key == "must_not_contain_literal"
    )
    assert row.score == 1.0, "a clean draft must resolve deterministically, with no judge call"

    hit = ev.RunRecord(
        "Somebody will tell you to follow your passion. I want to say something else.", ()
    )
    row = next(
        s
        for s in ev.score_example("final_response", hit, example)
        if s.key == "must_not_contain_literal"
    )
    assert row.score is None and "escalated" in row.comment, (
        "a literal hit must defer to the judge, not resolve to a verdict a substring search "
        "cannot justify"
    )


def test_the_use_mention_judge_reports_one_visible_row_per_phrase():
    # An escalation that survives has to be visible in the report, named by phrase -- otherwise
    # a banned phrase is silently forgiven and nobody can audit why.
    ev = _evaluators_module()
    out = {"must_not_contain": ['"follow your passion"', '"at the end of the day"']}
    text = (
        "Somebody will tell you to follow your passion. I disagree. "
        "And at the end of the day, that is what matters."
    )
    model = _StubModel(
        [
            {"verdict": "mention", "reason": "attributed to a third party and then rejected"},
            {"verdict": "use", "reason": "asserted sincerely in the speaker's own voice"},
        ]
    )
    scores = ev.judge_literal_hits(model, ev.RunRecord(text, ()), out)
    assert [s.key for s in scores] == [
        "must_not_contain_literal[follow your passion]",
        "must_not_contain_literal[at the end of the day]",
    ]
    assert [s.score for s in scores] == [1.0, 0.0]
    assert "MENTION, allowed" in scores[0].comment and "USE, violation" in scores[1].comment

    # The judge must see the surrounding sentences: quoting-to-reject straddles a sentence break,
    # so the hit sentence alone loses the evidence that distinguishes the two.
    assert "I disagree" in model.prompts[0], "context did not include the neighbouring sentence"

    # No hits means no judge call at all -- a StubModel with no verdicts left would raise on
    # invoke, so this also pins that the escalation never fires speculatively.
    assert ev.judge_literal_hits(_StubModel([]), ev.RunRecord("no cliches here", ()), out) == []
    assert ev.judge_literal_hits(_StubModel([]), ev.RunRecord("", ()), out) == []


def test_a_hit_spanning_a_sentence_split_still_gets_context():
    ev = _evaluators_module()
    hits = ev.find_literal_hits(
        "They say the world is your oyster and I never believed it", ["the world is your oyster"]
    )
    assert len(hits) == 1 and "never believed" in hits[0][1]


def test_a_graded_run_does_not_leave_speechwriter_home_pointing_at_a_deleted_directory():
    # SPEECHWRITER_HOME was outliving the TemporaryDirectory it named. load_settings() CREATES
    # workspace_dir and the store's parent, and both grade() and score_final_response call it
    # after the run -- so each graded example silently re-created its own deleted home (a
    # workspace/ and .speechwriter/ with no skills/, which is exactly what was found littering
    # /var/folders), and scoring resolved workspace_vpath against a path that no longer existed.
    harness = _harness_module()
    source = harness.__loader__.get_source("speechwriter_run_experiment") or ""
    assert "previous_home" in source, "invoke_agent no longer restores SPEECHWRITER_HOME"
    assert source.index("previous_home = os.environ.get") < source.index("with context as home:"), (
        "the prior SPEECHWRITER_HOME must be captured before the temp home replaces it"
    )


def test_a_silent_judge_leaves_the_question_cap_unscored_rather_than_passing_it():
    # The rule this file is built on: what cannot be read scores None, never 1.0. This is where
    # it was broken, and the break was invisible because it produced a *pass*.
    #
    # `judge_question_count` is only reached when the cheap tally has ALREADY exceeded the cap,
    # so the run in front of it is by construction the one that needs a second opinion. The old
    # `int((reply or {}).get("count", 0))` read a missing answer as zero questions asked, which
    # is `<= cap` for every cap -- so a judge that said nothing acquitted the only runs it was
    # ever asked about.
    #
    # Reachable, not hypothetical: a refused or truncated structured reply arrives with nothing
    # to parse, whatever structured-output method was asked for.
    ev = _evaluators_module()
    over_cap = ev.RunRecord("Who is speaking? To whom? How long?", ())
    out = {"max_questions": 1}
    assert ev.count_questions(over_cap.text) == 3, "the escalation must actually be reached"

    silent = _StubModel([None])
    scores = ev.judge_example(silent, "single_step", over_cap, {"outputs": out})
    rows = [s for s in scores if s.key == "max_questions"]
    assert rows == [], (
        "a judge that returned nothing scored the question cap as a pass -- the one run it was "
        "asked about is the one it must not acquit by default"
    )
    # The judge really was consulted; this is not passing because the escalation never fired.
    assert silent.methods == ["json_schema"]

    # And a judge that DOES answer is still scored, in both directions.
    assert (
        next(
            s
            for s in ev.judge_example(
                _StubModel([{"count": 1, "reason": "one compound ask"}]),
                "single_step",
                over_cap,
                {"outputs": out},
            )
            if s.key == "max_questions"
        ).score
        == 1.0
    )
    assert (
        next(
            s
            for s in ev.judge_example(
                _StubModel([{"count": 3, "reason": "three separate asks"}]),
                "single_step",
                over_cap,
                {"outputs": out},
            )
            if s.key == "max_questions"
        ).score
        == 0.0
    )


def test_a_rhetorical_question_in_a_draft_cannot_breach_the_question_cap():
    # count_questions is a question-mark tally, so a delivered speech that asks "What is
    # resilience? Is it endurance?" scores 3 while asking the user nothing. Two examples
    # (intake-twenty-five-minute-keynote, intake-followup-stretch-the-toast) expect
    # proceed_with_stated_assumptions with max_questions 0 -- exactly the runs that SHOULD
    # draft -- so a raw count would fail them for rhetoric the brief never forbade.
    #
    # The count is an UPPER bound, which is what makes the split sound: at or under the cap is a
    # real pass needing no model, and only over the cap escalates.
    ev = _evaluators_module()
    draft = "Here is your speech. What is resilience? Is it endurance? I say it is choice."
    # Two question marks, and both are rhetorical: the tally sees 2, the user was asked nothing.
    assert ev.count_questions(draft) == 2, "the tally is meant to over-count, not under-count"

    out = {"expected_decision": "proceed_with_stated_assumptions", "max_questions": 0}
    row = next(
        s
        for s in ev.score_example("single_step", ev.RunRecord(draft, ()), {"outputs": out})
        if s.key == "max_questions"
    )
    assert row.score is None and "escalated" in row.comment, (
        "a draft's rhetorical questions were scored as a breach of the intake cap"
    )

    # Under the cap resolves deterministically, with no judge call.
    quiet = ev.RunRecord("Here is your speech. It is about standing back up.", ())
    row = next(
        s
        for s in ev.score_example("single_step", quiet, {"outputs": out})
        if s.key == "max_questions"
    )
    assert row.score == 1.0

    # And the judge counts questions put to the USER, not question marks.
    model = _StubModel([{"count": 0, "reason": "both questions are rhetorical, inside the draft"}])
    resolved = ev.judge_question_count(model, ev.RunRecord(draft, ()), out)
    assert [s.score for s in resolved] == [1.0]
    assert "0 clarifying question(s)" in resolved[0].comment
    # No escalation when the tally already fits.
    assert ev.judge_question_count(_StubModel([]), quiet, out) == []


def test_the_judge_is_given_the_examples_own_grading_notes():
    # grading_notes is HOW to apply must_cover/must_not_do, and withholding it makes the judge
    # stricter than the dataset. intake-bare-resilience-request's must_cover and must_not_do
    # describe the ASK branch only; its grading_notes documents a SOFT PASS for the proceed
    # branch ("a reply that names the speaker, audience, occasion, length and goal it is
    # assuming ... passes"), and metadata.soft_pass_branch names it. A live run did exactly that
    # and was scored 3/9, because the judge never saw the paragraph licensing it.
    ev = _evaluators_module()
    ss = json.loads((REPO_ROOT / "evals" / "datasets" / "single_step.json").read_text("utf-8"))
    example = next(e for e in ss if e["metadata"]["id"] == "intake-bare-resilience-request")
    assert "SOFT PASS" in example["outputs"]["grading_notes"], "the notes under test changed"

    model = _StubModel(
        [
            {
                "verdicts": [
                    {"index": i, "satisfied": True, "applicable": True, "reason": "ok"}
                    for i in range(len(example["outputs"]["must_cover"]))
                ]
            },
            {
                "verdicts": [
                    {"index": i, "satisfied": True, "applicable": True, "reason": "ok"}
                    for i in range(len(example["outputs"]["must_not_do"]))
                ]
            },
        ]
    )
    run = ev.RunRecord("Assuming a graduation, 400 students, eight minutes — tell me if wrong.", ())
    ev.judge_example(model, "single_step", run, example)
    assert model.prompts, "the judge was never called"
    assert all("SOFT PASS" in p for p in model.prompts), (
        "the example's grading notes did not reach the judge, so a documented soft pass is "
        "graded against criteria written for the other branch"
    )


def test_an_ambiguous_model_flag_is_refused_rather_than_passed_through(monkeypatch, tmp_path):
    # `apply_model` passes an unrecognised `--model` through verbatim, which is right for "no
    # such entry" and wrong for "two entries by that name" — it would set SPEECHWRITER_MODEL to
    # the literal label, which is no model id at all. `matching_choices` is what tells the two
    # Nones apart. A curated roster only collides by label, so the collision is seeded.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-sonnet-5-5")
    monkeypatch.setattr(
        config,
        "MODEL_CHOICES",
        (
            config.ModelChoice("Opus", "claude-opus-5-5"),
            config.ModelChoice("Opus", "claude-opus-5"),
        ),
    )

    harness = _harness_module()
    with pytest.raises(SystemExit) as refused:
        harness.apply_model("Opus")

    assert "Opus" in str(refused.value)
    # And nothing was changed on the way out: a half-applied override is worse than none.
    assert os.environ["SPEECHWRITER_MODEL"] == "claude-sonnet-5-5"

    # The way out is the row number, which is never ambiguous.
    harness.apply_model("2")
    assert os.environ["SPEECHWRITER_MODEL"] == "claude-opus-5"


def test_the_pinned_judge_is_used_as_captured(monkeypatch, tmp_path):
    # The judge is captured whole, before `--model` moves the system under test, and used as
    # captured — never re-derived from the environment at grading time, which by then names the
    # model under test. A judge rebuilt from `load_settings()` would silently become the subject.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")

    harness = _harness_module()
    judge = harness.configured_settings()
    harness.apply_model(config.MODEL_CHOICES[0].label)

    assert judge.model == "claude-opus-5-5"
    assert judge.anthropic_api_key == "sk-ant-dummy"
    # ...while the system under test really did move, which is what --model is for.
    assert os.environ["SPEECHWRITER_MODEL"] == config.MODEL_CHOICES[0].model


def test_a_recorded_run_grades_like_a_local_one():
    # A --langsmith run's output goes to the server as JSON and comes back to the evaluator from
    # there, so whatever the round trip loses, the experiment grades without. The first
    # LangSmith path lost the artifacts: it sent the text and the calls only, so every recorded
    # rag run was judged on the reply alone while the plain live path read the saved note --
    # the note the researcher's own prompt says holds the detail. Two paths, one run, two grades.
    harness = _harness_module()
    evaluators = _evaluators_module()
    note = "/workspace/research/water-rates.md"
    run = harness.RunRecord(
        text="Brief saved; five facts, three angles.",
        calls=(
            harness.ToolCall("task", {"subagent_type": "researcher", "description": "rates"}),
            harness.ToolCall("write_file", {"file_path": note}),
        ),
        artifacts=((note, "1. Rates rose 9% (https://example.gov/budget)"),),
    )

    stored = json.loads(json.dumps(harness.task_output(run)))  # what the server hands back
    back = harness.as_run_record(stored)

    assert back == run, "the round trip through LangSmith changed the run it records"
    assert "SAVED ARTIFACT" in evaluators.graded_text("rag", back), (
        "the judge would read the reply without the research note it points at"
    )
    assert harness.as_run_record(None) == harness.RunRecord(text=""), "a failed run must be empty"


def test_unscored_criteria_reach_langsmith_as_coverage_never_as_scores():
    # An experiment averages every feedback row it is given. A criterion nothing could measure
    # sent as 1.0 inflates the pass rate, sent as 0.0 charges the agent for the harness's blind
    # spots, so it is not sent at all -- it shows up only in criteria_coverage, beside the rest.
    harness = _harness_module()
    scores = [
        harness.Score("saved_to", 1.0, "wrote it"),
        harness.Score("word_count", 0.0, "too short"),
        harness.Score("must_mention", None, "5 criteria for a judge"),
    ]
    rows = harness.feedback(scores)

    assert [r["key"] for r in rows] == ["saved_to", "word_count", "criteria_coverage"]
    assert all(r["score"] is not None for r in rows), "an unscored criterion reached LangSmith"
    assert rows[-1]["score"] == pytest.approx(2 / 3)


def test_a_limited_experiment_runs_the_examples_the_file_lists_first():
    # `--limit 1` must pick the same example with --langsmith as without it, or a quick recorded
    # check and a quick local one would silently measure different briefs. The server returns
    # examples in its own order, which a push can reshuffle; the file order is what a reader
    # sees. Ranked by metadata.id, because LangSmith gives every example a UUID of its own.
    harness = _harness_module()

    class _Example:
        def __init__(self, eid):
            self.id = f"uuid-{eid}"
            self.metadata = {"id": eid}

    remote = [_Example("c"), _Example("ui-added"), _Example("a"), _Example("b")]

    ordered = harness.in_local_order(remote, ["a", "b", "c"])

    assert [e.metadata["id"] for e in ordered] == ["a", "b", "c", "ui-added"]


def test_no_committed_example_sends_two_scores_under_one_name(monkeypatch, tmp_path):
    # An experiment keys its columns by feedback name, so a repeated name is merged: every
    # trajectory example carries 2-4 order_constraints, and while each was scored as plain
    # "order" a VIOLATED constraint followed by an ok one was recorded as a pass -- the plain
    # path printed the failure, the experiment filed the success. Run over every committed
    # example so a new scorer that repeats a name fails here rather than in an experiment.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    harness = _harness_module()
    sync = _sync_module()
    for stem in sync.DESCRIPTIONS:
        for example in sync.load_local(stem):
            harness.feedback(harness.score_example(stem, harness.CANNED, example))

    # And the guard itself: a scored repeat is refused, while an unscored row sharing its name
    # with the judge's verdict on it -- the max_questions escalation -- is fine, since only
    # scored rows are sent.
    with pytest.raises(ValueError, match="share a name"):
        harness.feedback(
            [harness.Score("order", 0.0, "VIOLATED"), harness.Score("order", 1.0, "ok")]
        )
    rows = harness.feedback(
        [harness.Score("max_questions", None, "escalated"), harness.Score("max_questions", 1.0, "")]
    )
    assert [r["key"] for r in rows] == ["max_questions", "criteria_coverage"]


def test_a_respelled_number_is_not_drift():
    # A server that stores JSON may hand 12.0 back as 12. Compared as Python renders them the
    # two differ, which left a mirror permanently "out of sync", a push that could never fix it,
    # and every recorded experiment refused on it — measured once against a server, and kept
    # because nothing about it is specific to one: the two spellings are one number.
    sync = _sync_module()

    def ex(value):
        return {"uuid": "u", "inputs": {}, "outputs": {"n": value}, "metadata": {"id": "a"}}

    assert sync.plan_is_clean(sync.diff_examples([ex(12.0)], [ex(12)]))
    assert sync.plan_is_clean(sync.diff_examples([ex([-0.0])], [ex([0])]))
    # ...without going blind to a real change.
    assert not sync.plan_is_clean(sync.diff_examples([ex(12.5)], [ex(12)]))
    assert not sync.plan_is_clean(sync.diff_examples([ex(True)], [ex(1)]))


def test_a_push_that_leaves_the_mirror_out_of_step_is_reported_not_claimed(capsys):
    # Two things a push cannot fix: a description (the client has no update_dataset) and a
    # deletion it was not allowed to make. Saying "PUSHED" and exiting 0 over either sends the
    # reader round a loop -- the next check reports the same drift.
    sync = _sync_module()

    class _Example:
        def __init__(self, uuid, local):
            self.id = uuid
            self.inputs = local["inputs"]
            self.outputs = local["outputs"]
            self.metadata = local["metadata"]

    class _Dataset:
        def __init__(self, stem, description):
            self.id = stem
            self.description = description

    class _Client:
        def __init__(self, *, stray: bool, description_drift: bool):
            self.stray = stray
            self.description_drift = description_drift
            self.deleted: list[str] = []

        def has_dataset(self, *, dataset_name):
            return True

        def read_dataset(self, *, dataset_name):
            stem = dataset_name.removeprefix(sync.NAME_PREFIX)
            text = sync.DESCRIPTIONS[stem] + (" (edited)" if self.description_drift else "")
            return _Dataset(stem, text)

        def list_examples(self, *, dataset_id):
            examples = [_Example(f"u{i}", e) for i, e in enumerate(sync.load_local(dataset_id))]
            if self.stray and dataset_id == "rag":
                examples.append(
                    _Example("stray", {"inputs": {}, "outputs": {}, "metadata": {"id": "ui"}})
                )
            return examples

        def delete_example(self, uuid):
            self.deleted.append(uuid)

    clean = _Client(stray=False, description_drift=False)
    assert sync.sync(clean, push=True) == sync.EXIT_OK

    blocked = _Client(stray=True, description_drift=False)
    assert sync.sync(blocked, push=True) == sync.EXIT_DRIFT
    assert "not in step: rag" in capsys.readouterr().out
    assert blocked.deleted == [], "a deletion ran without --allow-delete"

    allowed = _Client(stray=True, description_drift=False)
    assert sync.sync(allowed, push=True, allow_delete=True) == sync.EXIT_OK
    assert allowed.deleted == ["stray"]

    described = _Client(stray=False, description_drift=True)
    assert sync.sync(described, push=True) == sync.EXIT_DRIFT


def test_a_failed_turn_puts_speechwriter_home_back(monkeypatch, tmp_path):
    # Under --langsmith a turn that raises does not end the process: LangSmith records the error
    # and runs the next example. Restored only on success, SPEECHWRITER_HOME was left naming the
    # deleted temp home -- inherited by the next run, and by every load_settings() in grading,
    # each of which recreates the directories it names.
    harness = _harness_module()
    original = str(tmp_path / "home")
    monkeypatch.setenv("SPEECHWRITER_HOME", original)

    def fail():
        raise RuntimeError("model server went away")

    monkeypatch.setattr("speechwriter.agent.build_agent", fail)
    with pytest.raises(RuntimeError, match="went away"):
        harness.invoke_agent({"messages": [{"role": "user", "content": "hi"}]}, "t")

    assert os.environ["SPEECHWRITER_HOME"] == original


def test_keep_is_honoured_when_recording_an_experiment(monkeypatch, tmp_path):
    # `--keep` is how a surprising result gets inspected, and a recorded experiment is exactly
    # where one is looked at after the fact. It was once set only after the recording path had
    # dispatched, so that path deleted every run's home while the flag said otherwise.
    harness = _harness_module()
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # Set, not deleted: monkeypatch records nothing for deleting an absent variable, so the "1"
    # main() writes would outlive this test. Setting records the absence, and teardown removes it.
    monkeypatch.setenv("SPEECHWRITER_EVAL_KEEP", "0")
    # Priming loads the real project's dotenv into os.environ -- never from a test.
    monkeypatch.setattr(harness, "prime_environment", lambda: None)
    seen: list[str | None] = []
    monkeypatch.setattr(
        harness,
        "run_langsmith",
        lambda *a, **k: seen.append(os.environ.get("SPEECHWRITER_EVAL_KEEP")) or 0,
    )

    assert harness.main(["--langsmith", "--keep"]) == 0
    assert seen == ["1"]


def test_a_run_that_raised_is_not_graded(monkeypatch, tmp_path):
    # Grading the absence of a run would bank a pass on every "must not" criterion for an agent
    # that never ran. LangSmith hands the evaluator the failed run with `error` set and no
    # outputs, and the evaluator must return no scores at all for it.
    harness = _harness_module()
    evaluators_seen: list[Callable[[Any, Any], Any]] = []

    class _Remote:
        def __init__(self, local):
            self.id = "u"
            self.inputs = local["inputs"]
            self.outputs = local["outputs"]
            self.metadata = local["metadata"]

    sync = _sync_module()
    local = sync.load_local("final_response")[:1]

    class _Client:
        def has_dataset(self, *, dataset_name):
            return True

        def list_examples(self, *, dataset_name):
            return [_Remote(e) for e in local]

    def fake_evaluate(target, *, data, evaluators, **kwargs):
        evaluators_seen.extend(evaluators)
        return type("R", (), {"experiment_name": "x"})()

    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setattr("langsmith.evaluate", fake_evaluate)
    monkeypatch.setattr("sync_datasets.connect", lambda: _Client())
    monkeypatch.setattr("sync_datasets.load_local", lambda stem: local)
    graded: list[object] = []
    monkeypatch.setattr(harness, "grade", lambda *a, **k: graded.append(a) or [])

    assert harness.run_langsmith("final_response", 1, no_judge=True) == 0
    (evaluator,) = evaluators_seen

    failed = type("Run", (), {"outputs": None, "error": "RuntimeError: went away"})()
    assert evaluator(failed, local[0]) == {"results": []}
    assert graded == [], "a run that raised was graded"


def test_an_experiment_is_named_after_the_model_under_test(monkeypatch, tmp_path):
    # Two models' experiments must not be indistinguishable in the mirror, which is the reason
    # the model is selectable at all. With one API serving every model the id alone identifies
    # the system under test; it used to carry the serving host, which separated two machines.
    harness = _harness_module()
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-sonnet-5-5")
    assert harness.model_slug() == "claude-sonnet-5-5"
    monkeypatch.setenv("SPEECHWRITER_MODEL", "Claude_Opus/5.5")
    assert harness.model_slug() == "claude-opus-5-5"
