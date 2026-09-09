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
        # expected_trajectory is a reference path, and Client has no update_dataset, so an empty
        # one can be fixed only by deleting and recreating the dataset.
        assert description.strip(), f"{stem}: empty description"


def test_sync_reads_every_committed_dataset():
    # load_local is the mirror's only reader and it rejects an example the diff could not key
    # on. Running it over the real files makes a missing metadata.id fail here, offline, rather
    # than at the wire. The 55 is pinned like EXPECTED_COUNTS, and for the same reason.
    sync = _sync_module()
    counts = {stem: len(sync.load_local(stem)) for stem in sync.DESCRIPTIONS}
    assert sum(counts.values()) == 55, f"expected 55 examples across four files, got {counts}"


def test_sync_diff_detects_every_kind_of_drift():
    # Mutation-tested against the live account before being written down: an edited output and a
    # local-only example each turned the checker red, and a clean tree turned it green. This
    # pins that behaviour with no key and no network.
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

    unkeyed = tmp_path / "unkeyed.json"
    unkeyed.write_text(
        json.dumps([{"inputs": {}, "outputs": {}, "metadata": {}}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata.id"):
        sync.load_local("unkeyed", path=unkeyed)


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
    # so without the pinned pair, comparing two models grades each one with *itself*, changing
    # the instrument and the subject together and making the comparison meaningless.
    #
    # This exists because the bug shipped: the `judge` parameter was threaded into
    # `run_langsmith` and not into the plain live path, and nothing noticed.
    #
    # The whole *pair* is pinned, and asserted as a pair. With every model served over an
    # endpoint, two models under comparison are routinely two servers, so a judge that carried
    # the pinned id but read its endpoint from the environment would be built pointing at the
    # server the run had just moved to — the same instrument-follows-subject bug, one field down.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "judge-model")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:9999/v1")

    harness = _harness_module()
    # Captured before the override, exactly where `main()` captures it.
    pinned = harness.configured_settings()

    # ...and now the system under test moves, both halves of it.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "model-under-test")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    graded_with: list[tuple[str, str]] = []

    def spy_build(settings):
        graded_with.append((settings.model, settings.base_url))
        return object()

    monkeypatch.setattr("speechwriter.agent._build_model", spy_build)
    monkeypatch.setattr(harness, "score_example", lambda *a, **k: [])
    monkeypatch.setattr(harness, "judge_example", lambda *a, **k: [])

    harness.grade("final_response", harness.CANNED, {}, no_judge=False, judge=pinned)

    assert graded_with == [("judge-model", "http://127.0.0.1:9999/v1")], (
        "the judge followed the environment instead of the pinned pair, so every model "
        "would be graded by itself"
    )

    # And with no override in play the judge is the configured pair, exactly as before.
    graded_with.clear()
    harness.grade("final_response", harness.CANNED, {}, no_judge=False)
    assert graded_with == [("model-under-test", "http://127.0.0.1:8080/v1")]


def test_every_live_grading_path_pins_the_judge_when_the_model_is_overridden():
    # The mechanical half of the bug above: it was not that `grade()` ignored its argument, it
    # was that one of the two call sites never passed one. Read the source rather than the
    # behaviour, because the second path (`--langsmith`) needs a network to exercise.
    source = (REPO_ROOT / "evals" / "run_experiment.py").read_text(encoding="utf-8")
    calls = re.findall(r"\b(run_one|run_langsmith)\((.*?)\)", source)
    invocations = [(name, args) for name, args in calls if "args." in args]

    assert invocations, "neither live path is called — this test is watching the wrong names"
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
    # `/model` or the sidebar is `model_choices(...)`. That used to be the curated tuple plus a
    # locally served entry labelled "<id> (local)", and resolving against the curated tuple
    # alone passed that whole string through as a literal model id. The tuple is empty now, so
    # the same mistake is total rather than partial: a harness matching MODEL_CHOICES would
    # resolve *nothing*, and every `--model` would fall through to the pass-through branch.
    #
    # Which is why the endpoint is the assertion. Resolving a roster entry materialises the
    # whole pair into the environment; falling through sets the id and leaves the endpoint to
    # whatever the dotenv says next. Started absent, so the two are distinguishable — and that
    # is also the surviving half of the deleted test about `apply_model` never leaving the
    # endpoint for a later `load_settings()` to fill in.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

    harness = _harness_module()
    # Not a literal: the label a reader copies out of the picker is whatever `local_choice`
    # builds, which is the bare id now and was the suffixed one before.
    (entry,) = config.model_choices(config.load_settings())
    assert entry.label == "mlx-community/Qwen3.8-27B-4bit", (
        "the roster label changed shape; this test copies it the way a reader does"
    )

    harness.apply_model(entry.label)
    assert os.environ["SPEECHWRITER_MODEL"] == "mlx-community/Qwen3.8-27B-4bit"
    assert os.environ["SPEECHWRITER_BASE_URL"] == config.DEFAULT_LOCAL_ENDPOINT, (
        "the label resolved to an id without its endpoint, so the pair came apart"
    )

    # An id nothing on the roster matches is still passed through verbatim — the operator may
    # mean a model the environment has never named — and the endpoint is left alone.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    harness.apply_model("mlx-community/Llama-3.3-70B-Instruct-4bit")
    assert os.environ["SPEECHWRITER_MODEL"] == "mlx-community/Llama-3.3-70B-Instruct-4bit"
    assert "SPEECHWRITER_BASE_URL" not in os.environ


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


def test_word_count_scoring_uses_the_same_rule_the_browser_shows():
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
    # Pinned to the DEFAULT pair. An exported SPEECHWRITER_BASE_URL no longer swaps the client
    # -- there is only one -- but it still decides which server this build is aimed at, and a
    # developer driving their own would otherwise be testing a different configuration than CI.
    # No key is set: a locally served model is sent LOCAL_API_KEY_PLACEHOLDER, so build_agent()
    # needs no credential of ours to construct its client.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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


def test_an_empty_output_cannot_bank_passes_on_absence_criteria():
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


def test_the_judge_asks_for_structured_output_a_local_server_can_answer():
    # `langchain-openai` defaults `with_structured_output` to method="json_schema", which sends
    # `response_format: {"type": "json_schema", ...}`. That is a *server* feature, not an API
    # one, and `mlx_lm.server` -- the server DEFAULT_LOCAL_ENDPOINT names -- answers 400 to it.
    # While the judge was hosted this never showed. With every model served locally it fails
    # every judged example on the transport, uniformly enough to read as "the judge disagrees"
    # rather than "the judge never ran" -- the same family of measurement bug this file already
    # documents five of, all of which made the agent look worse than it is.
    #
    # The literal is pinned here rather than compared against the module's own constant, which
    # would pass for any value: what makes `function_calling` right is what the local servers
    # implement, not what evaluators.py says about itself.
    ev = _evaluators_module()
    assert ev.JUDGE_STRUCTURED_OUTPUT_METHOD == "function_calling"

    # All three call sites, because `_structured` exists precisely so none of them can forget:
    # a site that did would not fail here, it would fail at the endpoint, once, in whichever
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
        assert stub.methods == ["function_calling"], (
            f"a judge call asked for structured output as {stub.methods!r}; json_schema is a "
            f"400 on mlx_lm.server, so every judged example would fail on the transport"
        )


def test_a_banned_phrase_escalates_instead_of_failing_outright():
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
    # Reachable, not hypothetical: `with_structured_output(..., method="function_calling")`
    # returns None rather than raising when the model emits no tool call, which is the
    # local-server case JUDGE_STRUCTURED_OUTPUT_METHOD explicitly warns is not universal.
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
    assert silent.methods == ["function_calling"]

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


def test_an_ambiguous_model_flag_is_refused_rather_than_sent_to_the_wrong_server(
    monkeypatch, tmp_path
):
    # `apply_model` passes an unrecognised `--model` through verbatim, which is right for "no
    # such entry" and wrong for "two entries by that name" — it sets SPEECHWRITER_MODEL while
    # leaving SPEECHWRITER_BASE_URL alone, so the id goes to whichever server the environment
    # already named and 404s at the first turn of every graded example. That became reachable
    # when `resolve_choice` started answering None on ambiguity instead of silently returning
    # whichever entry was merged first.
    #
    # The collision used to be a curated Anthropic entry meeting its locally served namesake.
    # With the curated tuple empty it is the one CLAUDE.md calls a known ambiguity — **one model
    # id served by two machines** — and it is now *harder* to see, not easier: `local_choice`
    # labels both entries with the bare id, so the two rows are indistinguishable by name and
    # only their `base_url` column tells them apart.
    #
    # The second entry reaches a front end's roster by detection; `apply_model` builds its
    # roster from one `Settings` and so can only ever hold one, which is why the collision is
    # seeded through MODEL_CHOICES here. Everything downstream of the seed is the real thing:
    # `model_choices`' dedup, `local_choice`'s label, `matching_choices`, `resolve_choice`.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(
        config,
        "MODEL_CHOICES",
        (
            config.local_choice(
                "mlx-community/Qwen3.8-27B-4bit", "http://workstation.local:8080/v1"
            ),
        ),
    )

    harness = _harness_module()
    with pytest.raises(SystemExit) as refused:
        harness.apply_model("mlx-community/Qwen3.8-27B-4bit")

    assert "mlx-community/Qwen3.8-27B-4bit" in str(refused.value)
    # And nothing was changed on the way out: a half-applied override is worse than none.
    assert os.environ["SPEECHWRITER_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert os.environ["SPEECHWRITER_MODEL"] == "mlx-community/Qwen3.8-27B-4bit"

    # The way out is the row number, not the label: two servers serving one id have the same
    # label by construction, since `local_choice` must stay a pure function of the pair. Picked
    # by index, the *other* server is what gets applied — both halves of it.
    harness.apply_model("1")
    assert os.environ["SPEECHWRITER_MODEL"] == "mlx-community/Qwen3.8-27B-4bit"
    assert os.environ["SPEECHWRITER_BASE_URL"] == "http://workstation.local:8080/v1"


def test_the_pinned_judge_keeps_the_credential_its_endpoint_needs(monkeypatch, tmp_path):
    # The judge is captured before `--model` moves the system under test, and it used to be
    # captured as a ModelChoice and re-applied to a fresh `load_settings()` at grading time.
    # That broke when `applied_to` began moving the *credential* with the pair: by then
    # `apply_model` has blanked SPEECHWRITER_BASE_URL, so the judge's endpoint is compared
    # against None, judged a stranger, and stripped of the key — every judge call 401s, on the
    # one path that has a judge to pin. Capturing the settings whole sidesteps the question.
    # Two servers, because that is what makes `--model` move the *endpoint* at all: a roster
    # synthesised from one configuration names one pair, and an id it does not know is passed
    # through with the endpoint left alone. Seeded through MODEL_CHOICES for the same reason
    # the ambiguity test above seeds it — in a front end the second entry arrives by detection.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "gpt-4o")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-gateway")
    monkeypatch.setattr(
        config,
        "MODEL_CHOICES",
        (config.local_choice("mlx-community/Qwen3.8-27B-4bit", "http://127.0.0.1:8080/v1"),),
    )

    harness = _harness_module()
    judge = harness.configured_settings()
    harness.apply_model("mlx-community/Qwen3.8-27B-4bit")

    # The judge still names the pair that was in force, and still holds its bearer token.
    assert judge.model == "gpt-4o"
    assert judge.base_url == "https://gateway.example.com/v1"
    assert judge.endpoint_api_key == "sk-real-gateway", (
        "the pinned judge lost the credential its endpoint needs, so every judge call 401s"
    )
    # ...while the system under test really did move, both halves of it, which is what
    # --model is for.
    assert os.environ["SPEECHWRITER_MODEL"] == "mlx-community/Qwen3.8-27B-4bit"
    assert os.environ["SPEECHWRITER_BASE_URL"] == "http://127.0.0.1:8080/v1"

    # Anti-vacuity: the trap those assertions dodge is still live. A judge captured as a
    # ModelChoice and re-applied to the environment as it stands *now* is compared against the
    # server the run moved to, judged a stranger, and stripped of the key it needs.
    rederived = config.ModelChoice(judge.model, judge.model, judge.base_url).applied_to(
        config.load_settings()
    )
    assert rederived.openai_api_key is None, (
        "re-applying a captured pair no longer strips the credential, so this test is passing "
        "for a reason other than the one it was written for -- re-read its premise"
    )
