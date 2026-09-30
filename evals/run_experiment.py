"""Run the speechwriter against an eval dataset and score it.

Two modes, and the default is the cheap one. ``--dry-run`` scores a canned run with no model
call at all, which is how you check the wiring before spending anything; the live mode invokes
the real agent once per example. ``--langsmith`` additionally records the result as an
experiment against the mirrored dataset, so it shows up next to the examples it graded -- with
each run's full agent trace nested under it, since the agent's LangChain runs are traced into the
experiment.

**Each example runs in its own ``SPEECHWRITER_HOME``.** That is not tidiness. The agent writes
real files, and half the criteria are about *where* -- ``must_save_to``, ``required_write_paths``,
the sandbox probes. Sharing a workspace would let example N pass on a file example N-1 wrote.
The temp home also keeps the graded runs out of the real ``workspace/``.

**Writes are read back off disk, not only out of the message stream.** A subagent's tool calls
do not surface in the orchestrator's messages, so a researcher that correctly saved to
``/workspace/research/`` would look like it never wrote anything. The disk is the ground truth
for path criteria; the message stream is the ground truth for ordering. Both feed one
:class:`~evaluators.RunRecord`.

Scores come in two halves and are reported separately on purpose:
:func:`evaluators.score_example` is deterministic and free, :func:`evaluators.judge_example`
costs a model call per criterion list. ``--no-judge`` runs only the first half. Criteria that
neither half can reach are counted as *unscored* and printed, never folded into a pass rate --
a coverage number that quietly includes what it could not measure is the failure the dataset
audit went looking for.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only, so the heavy `speechwriter` import this module otherwise defers into
    # function bodies stays deferred; `from __future__ import annotations` makes it sufficient.
    from speechwriter.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluators import (  # noqa: E402  -- needs the sys.path line above
    RunRecord,
    Score,
    ToolCall,
    coverage,
    judge_example,
    score_example,
    unscored,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EV = Path(__file__).resolve().parent / "datasets"
DATASETS = ("final_response", "trajectory", "single_step", "rag")


def to_messages(inputs: dict[str, Any]) -> list[dict[str, str]]:
    """Both input shapes the datasets use: a message list, or a bare research question."""
    if isinstance(inputs.get("messages"), list):
        return [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in inputs["messages"]
        ]
    question = inputs.get("question", "")
    context = inputs.get("speech_context", "")
    return [{"role": "user", "content": f"{question}\n\nContext: {context}".strip()}]


def message_text(message: Any) -> str:
    """The assistant's prose, whichever shape the provider used.

    ``AIMessage.content`` is a plain string for some responses and a list of content blocks for
    others -- which is how the first live run scored a finished 1,560-word commencement address
    as "the output is empty" while its own saved_to check confirmed the file had been written.
    Only ``text`` blocks are joined: thinking and tool_use blocks are not what was said.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def extract(state: dict[str, Any], workspace: Path) -> RunRecord:
    """Turn a finished graph state plus the files it left behind into a RunRecord."""
    calls: list[ToolCall] = []
    text = ""
    for message in state.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(ToolCall(str(call.get("name", "")), dict(call.get("args") or {})))
        if getattr(message, "type", "") == "ai":
            spoken = message_text(message)
            if spoken.strip():
                text = spoken

    # Disk writes, mapped back to the virtual paths the criteria are written in. Contents are
    # captured too, not just paths: rag's sourcing criteria are about the saved research note,
    # which the returned brief only points at.
    artifacts: list[tuple[str, str]] = []
    for path in sorted(workspace.rglob("*.md")):
        virtual = "/workspace/" + path.relative_to(workspace).as_posix()
        if virtual not in [c.path for c in calls]:
            calls.append(ToolCall("write_file", {"file_path": virtual}))
        try:
            artifacts.append((virtual, path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return RunRecord(text=text, calls=tuple(calls), artifacts=tuple(artifacts))


def configured_settings() -> Settings:
    """The whole configuration in force right now — what the judge must be built from.

    Captured whole, before :func:`apply_model`, so the judge is built from exactly the
    configuration that was in force before the override — model and credential together —
    rather than re-derived from an environment ``--model`` has since changed.
    """
    from speechwriter.config import load_settings

    return load_settings()


def apply_model(requested: str) -> None:
    """Point this run at a chosen model, *before* ``prime_environment`` reads the dotenv.

    Works through ``os.environ`` rather than by passing a ``Settings`` down, because
    ``invoke_agent`` builds a fresh agent per example inside a temp home and reads the
    environment each time. ``load_dotenv`` never overrides an already-set variable, so a value
    put here survives the priming that follows.

    A roster label or id resolves to a :class:`~speechwriter.config.ModelChoice`; anything
    unrecognised is passed through verbatim as an id the operator means literally.
    """
    from speechwriter.config import (
        load_settings,
        matching_choices,
        model_choices,
        resolve_choice,
    )

    # Against the *full* roster, not just the curated tuple, so an off-roster SPEECHWRITER_MODEL
    # the reader copies out of `/model` or the sidebar resolves to itself.
    roster = model_choices(load_settings())
    choice = resolve_choice(roster, requested)
    if choice is None and len(matching_choices(roster, requested)) > 1:
        # `resolve_choice` answers None for "no such entry" *and* for "two entries by that
        # name", and the pass-through below is only right for the first.
        raise SystemExit(
            f"--model {requested!r} names more than one roster entry. Select by row number:\n"
            + "\n".join(
                f"  --model {roster.index(match) + 1}   {match.label}   {match.model}"
                for match in matching_choices(roster, requested)
            )
        )
    os.environ["SPEECHWRITER_MODEL"] = choice.model if choice is not None else requested


def model_slug() -> str:
    """A filename-safe tag for the model in force, for naming an experiment after it.

    Call only after :func:`prime_environment`. The model id alone identifies the system under
    test now that every model is served by one API; the host suffix it carried while models
    were served locally separated two *servers*, and there is only one.
    """
    from speechwriter.config import load_settings

    raw = load_settings().model.casefold()
    slug = "".join(char if char.isalnum() else "-" for char in raw)
    return "-".join(part for part in slug.split("-") if part)


def prime_environment() -> None:
    """Load the real project's settings once, before any ``SPEECHWRITER_HOME`` override.

    ``load_settings()`` derives ``project_root`` from ``SPEECHWRITER_HOME`` and loads *that*
    directory's dotenv, so a temp home would go looking for credentials that are not there and
    every run would fail at the first API call. Priming against the real root puts them in
    ``os.environ``, where ``load_dotenv``'s never-override rule then preserves them.
    """
    from speechwriter.config import load_settings

    os.environ.pop("SPEECHWRITER_HOME", None)
    load_settings()


def invoke_agent(inputs: dict[str, Any], thread_id: str) -> RunRecord:
    """One graded run of the real agent, in a workspace of its own."""
    from speechwriter.agent import build_agent

    keep = os.environ.get("SPEECHWRITER_EVAL_KEEP") == "1"
    context = (
        contextlib.nullcontext(tempfile.mkdtemp(prefix="speechwriter-eval-"))
        if keep
        else tempfile.TemporaryDirectory(prefix="speechwriter-eval-")
    )
    # Restored afterwards, because SPEECHWRITER_HOME otherwise outlives the directory it names:
    # grade() and score_final_response both call load_settings(), which CREATES workspace_dir and
    # the store's parent -- so every run was re-littering a home that had just been deleted, and
    # scoring resolved workspace_vpath against a path that no longer existed. Unsetting restores
    # the real project root, which is what the criteria are written against.
    previous_home = os.environ.get("SPEECHWRITER_HOME")
    with context as home:
        # skills_dir is project_root / "skills", and project_root IS the temp home -- so a bare
        # one silently runs the agent with the rhetoric library absent. That does not just fail
        # every required_skill_reads criterion, it changes the behaviour under test.
        #
        # Copied, not symlinked. Settings._vpath maps a real path to a virtual one with
        # `path.resolve().relative_to(project_root)`, and resolve() follows the link straight
        # back to the real repo -- which is not under the temp root, so skills_vpath raises
        # before a single model call. The tree is four SKILL.md files; a copy costs nothing.
        shutil.copytree(REPO_ROOT / "skills", Path(home) / "skills")
        os.environ["SPEECHWRITER_HOME"] = home
        # Restored in a `finally`, not on the way out: under --langsmith a turn that raises does
        # not end the process -- LangSmith records the error and moves on to the next example,
        # which would otherwise inherit (and later restore) a home that has been deleted.
        try:
            bundle = build_agent()
            state = bundle.agent.invoke(
                {"messages": to_messages(inputs)}, config=bundle.turn_config(thread_id)
            )
            record = extract(state, Path(home) / "workspace")
            if keep:
                print(f"   [kept] artifacts under {home}")
            return record
        finally:
            if previous_home is None:
                os.environ.pop("SPEECHWRITER_HOME", None)
            else:
                os.environ["SPEECHWRITER_HOME"] = previous_home


def grade(
    dataset: str,
    run: RunRecord,
    example: dict[str, Any],
    no_judge: bool,
    judge: Settings | None = None,
) -> list[Score]:
    """Score one run. ``judge`` pins the grading model when the agent's has been overridden.

    Without it the judge is whatever ``load_settings()`` reports — which is the agent's own
    model, since both read ``SPEECHWRITER_MODEL``. That is fine while there is one model, and
    wrong the moment ``--model`` exists: each model would be graded by itself, so a comparison
    between two of them would measure two different instruments as much as two writers.
    """
    scores = score_example(dataset, run, example)
    if not no_judge:
        from speechwriter.agent import _build_model
        from speechwriter.config import load_settings

        # The pinned judge is used *as captured*, never re-applied to the current environment,
        # which `apply_model` has since pointed at the model under test.
        settings = judge if judge is not None else load_settings()
        scores += judge_example(_build_model(settings), dataset, run, example)
    return scores


def run_one(
    example: dict[str, Any], no_judge: bool, judge: Settings | None = None
) -> tuple[RunRecord, list[Score]]:
    dataset = example["metadata"]["dataset_type"]
    run = invoke_agent(example["inputs"], example["metadata"]["id"])
    return run, grade(dataset, run, example, no_judge, judge)


def task_output(run: RunRecord) -> dict[str, Any]:
    """A :class:`RunRecord` as the JSON an experiment run stores. Inverse: :func:`as_run_record`.

    Artifacts travel too. The first LangSmith path sent only the text and the calls, so every
    experiment graded ``rag`` on the reply alone -- the researcher's own prompt says the detail
    lives in the saved note, and the note never reached the scorer. The plain live path had it
    all along; ``test_a_recorded_run_grades_like_a_local_one`` keeps the two equal.
    """
    return {
        "text": run.text,
        "calls": [{"name": c.name, "args": c.args} for c in run.calls],
        "artifacts": [{"path": path, "content": content} for path, content in run.artifacts],
    }


def as_run_record(output: Any) -> RunRecord:
    """Rebuild the :class:`RunRecord` a scorer needs from a stored run's output.

    The output has been through the server as JSON, so tuples come back as lists and nothing
    is guaranteed to be the shape it was sent in; anything unreadable reads as absent.
    """
    out = output if isinstance(output, dict) else {}
    return RunRecord(
        text=str(out.get("text") or ""),
        calls=tuple(
            ToolCall(str(c.get("name", "")), dict(c.get("args") or {}))
            for c in out.get("calls") or []
            if isinstance(c, dict)
        ),
        artifacts=tuple(
            (str(a.get("path", "")), str(a.get("content", "")))
            for a in out.get("artifacts") or []
            if isinstance(a, dict)
        ),
    )


def feedback(scores: list[Score]) -> list[dict[str, Any]]:
    """Scores as LangSmith evaluation results: one named feedback row per criterion.

    Unscored rows are left out of what LangSmith averages and reported as their own metric
    instead -- folding them in as 1.0 would inflate the pass rate with criteria nothing
    actually measured, and folding them in as 0.0 would charge the agent for the harness's
    blind spots.

    **Names must be unique, and a repeat is refused rather than sent.** An experiment's
    columns are keyed by feedback name, so two rows under one name are averaged into a single
    cell -- which is how two ``order`` constraints once turned a violation into a recorded
    pass. An unscored row may share its name with the judge's verdict on it (the
    ``max_questions`` escalation does exactly that); it is dropped above, so only *scored*
    names must differ.
    """
    rows: list[dict[str, Any]] = [
        {"key": s.key, "score": s.score, "comment": s.comment}
        for s in scores
        if s.score is not None
    ]
    names = [row["key"] for row in rows]
    repeated = sorted({str(n) for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"scores share a name, so an experiment would merge them: {repeated}")
    rows.append({"key": "criteria_coverage", "score": coverage(scores)})
    return rows


def _example_id(example: Any) -> str:
    """The dataset's own id for a LangSmith example — ``metadata.id``, never the server UUID."""
    return str((getattr(example, "metadata", None) or {}).get("id", ""))


def in_local_order(examples: list[Any], local_ids: list[str]) -> list[Any]:
    """Remote examples in the order the dataset file lists them.

    So ``--limit 1`` runs the same example with ``--langsmith`` as without it. The server's own
    order is not the file's, and a push can reshuffle it; the file is what a reader reads.
    Ranked by ``metadata.id`` because LangSmith assigns its own example UUIDs.
    """
    rank = {eid: i for i, eid in enumerate(local_ids)}
    return sorted(examples, key=lambda e: rank.get(_example_id(e), len(rank)))


def run_langsmith(dataset: str, limit: int, no_judge: bool, judge: Settings | None = None) -> int:
    """Record the run as a LangSmith experiment against the mirrored dataset.

    Capped by ``--limit`` on purpose: an experiment otherwise sweeps every example in the
    dataset, and each one is a full agent turn.

    **The mirror must be clean first.** An experiment is graded by the criteria on the server,
    and a local edit not yet pushed would have it score one version of an example while the
    file a reader opens says another -- a measurement bug of the kind this harness has hit five
    times, each making the agent look worse than it was. Refused, with the drift named.

    **Traces nest for free.** LangSmith runs each target inside a run of its own, and the
    agent's LangChain callbacks attach every model, tool and subagent call beneath it -- so each
    experiment row opens onto the whole trajectory that produced it. The judge's calls land
    under the evaluator the same way.
    """
    from langsmith import evaluate
    from sync_datasets import (
        as_record,
        canon,
        connect,
        describe,
        diff_examples,
        load_local,
        plan_is_clean,
        remote_name,
    )

    from speechwriter.config import load_settings

    settings = load_settings()
    client = connect()
    if client is None:
        print("--langsmith needs LANGSMITH_API_KEY set.", file=sys.stderr)
        return 2

    name = remote_name(dataset)
    if not client.has_dataset(dataset_name=name):
        print(f"{name!r} is not on LangSmith -- run evals/sync_datasets.py --push", file=sys.stderr)
        return 2
    remote = list(client.list_examples(dataset_name=name))
    local = load_local(dataset)
    plan = diff_examples(local, [as_record(e) for e in remote])
    if not plan_is_clean(plan):
        print("\n".join(describe(dataset, plan)), file=sys.stderr)
        print(
            f"{name!r} on LangSmith does not match evals/datasets/{dataset}.json -- run "
            f"evals/sync_datasets.py --push first, so the experiment grades what the file says.",
            file=sys.stderr,
        )
        return 2

    chosen = in_local_order(remote, [e["metadata"]["id"] for e in local])[:limit]
    # The target is handed only the example's inputs, so its id is recovered from them. Keyed on
    # the canonical rendering, which is what the mirror already compares on, so key order in a
    # round-tripped dict cannot miss the match.
    ids_by_inputs = {canon(e.inputs or {}): _example_id(e) for e in chosen}

    def run_agent(inputs: dict[str, Any]) -> dict[str, Any]:
        # Named for what it does, because LangSmith names each row's root run after it and the
        # agent's whole trace hangs off that run. The example id, not a slice of the input, is
        # the thread: it names the conversation exactly as the plain live path does.
        thread = ids_by_inputs.get(canon(inputs), "langsmith-run")
        return task_output(invoke_agent(dict(inputs), thread_id=thread))

    def speechwriter_criteria(run: Any, example: Any) -> dict[str, list[dict[str, Any]]]:
        output = getattr(run, "outputs", None)
        if getattr(run, "error", None) or output is None:
            # The target raised, and LangSmith has recorded that as the run's error. Grading
            # the absence would bank passes on every "must not" criterion for a run that never
            # ran.
            return {"results": []}
        record = {
            "outputs": dict(getattr(example, "outputs", None) or {}),
            "metadata": dict(getattr(example, "metadata", None) or {}),
        }
        scores = grade(dataset, as_run_record(output), record, no_judge, judge)
        report(dataset, record, scores)
        return {"results": feedback(scores)}

    graded_by = judge if judge is not None else settings
    results = evaluate(
        run_agent,
        data=chosen,
        evaluators=[speechwriter_criteria],
        client=client,
        # The model is part of the experiment's identity, not incidental to it: comparing two
        # models is the reason the model is selectable at all, and a name that said only the
        # dataset would file both runs under one indistinguishable prefix.
        experiment_prefix=f"speechwriter-{dataset}-{model_slug()}",
        metadata={
            "model": settings.model,
            "judge": None if no_judge else graded_by.model,
            "examples": [_example_id(e) for e in chosen],
        },
        # Sequential. `invoke_agent` repoints the process-wide SPEECHWRITER_HOME per example,
        # so concurrent targets would write into each other's homes. (LangSmith does not retry
        # a failed target, so a failed turn costs one run, not several.)
        max_concurrency=0,
    )
    print(f"\nrecorded experiment: {results.experiment_name}")
    return 0


CANNED = RunRecord(
    text="Friends — to Teodoro and Nkiru. [pause] Twenty seconds, and every one of them earned.",
    calls=(
        ToolCall("read_file", {"file_path": "/skills/audience-and-occasion/SKILL.md"}),
        ToolCall("write_file", {"file_path": "/workspace/speeches/auggie-toast.md"}),
        ToolCall("task", {"subagent_type": "style-critic"}),
        ToolCall("edit_file", {"file_path": "/workspace/speeches/auggie-toast.md"}),
    ),
)


def report(dataset: str, example: dict[str, Any], scores: list[Score]) -> dict[str, Any]:
    graded = [s for s in scores if s.machine_scored]
    passed = sum(1 for s in graded if s.score == 1.0)
    eid = example["metadata"]["id"]
    print(f"\n── {dataset} / {eid}")
    for s in scores:
        mark = " -- " if not s.machine_scored else (" ok " if s.score else "FAIL")
        print(f"   [{mark}] {s.key:<30} {s.comment[:78]}")
    print(f"   {passed}/{len(graded)} passed, {coverage(scores):.0%} of criteria measurable")
    return {
        "id": eid,
        "passed": passed,
        "graded": len(graded),
        "unscored": [s.key for s in unscored(scores)],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=DATASETS, default="trajectory")
    p.add_argument("--limit", type=int, default=1, help="examples to run (default 1)")
    p.add_argument("--dry-run", action="store_true", help="score a canned run; no model call")
    p.add_argument("--no-judge", action="store_true", help="deterministic scorers only")
    p.add_argument("--langsmith", action="store_true", help="record as a LangSmith experiment")
    p.add_argument("--keep", action="store_true", help="leave each run's temp home on disk")
    p.add_argument(
        "--model",
        help="model to grade: a roster label ('Opus 5.5'), a model id, or omit for the .env one",
    )
    args = p.parse_args(argv)

    if args.dry_run and (args.langsmith or args.model):
        # `--dry-run` scores a canned record and never calls a model, so both of these would be
        # silently ignored — and `--model` would still rewrite SPEECHWRITER_MODEL for the
        # process on its way to doing nothing. Rejected rather than dropped, so the flag that
        # was going to have no effect says so.
        other = "--langsmith" if args.langsmith else "--model"
        print(f"{other} and --dry-run are contradictory", file=sys.stderr)
        return 2

    # Primed once, here, for every live path. It pops SPEECHWRITER_HOME, reloads the dotenv and
    # clears the langsmith env cache — an ordering this module is otherwise careful about, so
    # doing it twice is worth avoiding even where it is harmless.
    if not args.dry_run:
        prime_environment()

    # `--model` moves the system under test; the judge must not move with it. `grade()` builds
    # its model from `load_settings()`, which reads the same SPEECHWRITER_MODEL, so the pair in
    # force is captured *after* priming and *before* the override, then pinned for grading.
    # Otherwise comparing two models would grade each one with itself, changing the instrument
    # and the subject together.
    judge: Settings | None = None
    if args.model:
        judge = configured_settings()
        apply_model(args.model)

    # Before the dispatch, so both live paths honour it: a recorded experiment is exactly where a
    # surprising result gets inspected after the fact.
    if args.keep:
        os.environ["SPEECHWRITER_EVAL_KEEP"] = "1"

    if args.langsmith:
        return run_langsmith(args.dataset, args.limit, args.no_judge, judge)

    examples = json.loads((EV / f"{args.dataset}.json").read_text(encoding="utf-8"))[: args.limit]
    print(
        f"{args.dataset}: {len(examples)} example(s), "
        f"{'DRY RUN — no model calls' if args.dry_run else 'LIVE — this costs tokens'}"
    )

    rows = []
    for example in examples:
        if args.dry_run:
            scores = score_example(args.dataset, CANNED, example)
        else:
            _, scores = run_one(example, args.no_judge, judge)
        rows.append(report(args.dataset, example, scores))

    total_p = sum(r["passed"] for r in rows)
    total_g = sum(r["graded"] for r in rows)
    print(
        f"\nTOTAL {total_p}/{total_g} machine-scored criteria passed across {len(rows)} example(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
