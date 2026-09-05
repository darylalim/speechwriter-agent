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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-offline")

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
    """Minimal stand-in for a chat model: records prompts, replays canned verdicts."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self._verdicts.pop(0)


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
