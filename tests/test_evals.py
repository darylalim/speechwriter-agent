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
