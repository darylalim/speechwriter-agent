"""Run the speechwriter against an eval dataset and score it.

Two modes, and the default is the cheap one. ``--dry-run`` scores a canned run with no model
call at all, which is how you check the wiring before spending anything; the live mode invokes
the real agent once per example. ``--langsmith`` additionally records the result as an
experiment against the mirrored dataset, so it shows up next to the examples it graded.

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
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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
NAME_PREFIX = "Speechwriter: "


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

    # Disk writes, mapped back to the virtual paths the criteria are written in.
    for path in sorted(workspace.rglob("*.md")):
        virtual = "/workspace/" + path.relative_to(workspace).as_posix()
        if virtual not in [c.path for c in calls]:
            calls.append(ToolCall("write_file", {"file_path": virtual}))
    return RunRecord(text=text, calls=tuple(calls))


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

    with tempfile.TemporaryDirectory(prefix="speechwriter-eval-") as home:
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
        bundle = build_agent()
        state = bundle.agent.invoke(
            {"messages": to_messages(inputs)}, config=bundle.turn_config(thread_id)
        )
        return extract(state, Path(home) / "workspace")


def grade(dataset: str, run: RunRecord, example: dict[str, Any], no_judge: bool) -> list[Score]:
    scores = score_example(dataset, run, example)
    if not no_judge:
        from speechwriter.agent import _build_model
        from speechwriter.config import load_settings

        scores += judge_example(_build_model(load_settings()), dataset, run, example)
    return scores


def run_one(example: dict[str, Any], no_judge: bool) -> tuple[RunRecord, list[Score]]:
    dataset = example["metadata"]["dataset_type"]
    run = invoke_agent(example["inputs"], example["metadata"]["id"])
    return run, grade(dataset, run, example, no_judge)


def run_langsmith(dataset: str, limit: int, no_judge: bool) -> int:
    """Record the run as a LangSmith experiment against the mirrored dataset.

    Capped by ``--limit`` on purpose: ``evaluate`` would otherwise sweep every example in the
    dataset, and each one is a full agent turn.
    """
    from langsmith import Client, evaluate
    from langsmith.evaluation import EvaluationResult, EvaluationResults
    from langsmith.schemas import Example, Run

    from speechwriter.config import load_settings

    load_settings()
    client = Client()
    name = f"{NAME_PREFIX}{dataset}"
    examples = list(client.list_examples(dataset_name=name))[:limit]
    if not examples:
        print(f"no examples found in {name!r} -- is the mirror pushed?", file=sys.stderr)
        return 2

    def target(inputs: dict) -> dict:
        run = invoke_agent(inputs, thread_id=str(inputs)[:64])
        return {"text": run.text, "calls": [{"name": c.name, "args": c.args} for c in run.calls]}

    def scorer(run: Run, example: Example | None) -> EvaluationResults:
        outputs = run.outputs or {}
        record = RunRecord(
            text=str(outputs.get("text", "")),
            calls=tuple(
                ToolCall(str(c.get("name", "")), dict(c.get("args") or {}))
                for c in outputs.get("calls") or []
            ),
        )
        payload: dict[str, Any] = {
            "outputs": (example.outputs if example else None) or {},
            "metadata": (example.metadata if example else None) or {},
        }
        scores = grade(dataset, record, payload, no_judge)
        # Unscored rows are dropped from the feedback LangSmith averages and reported as their
        # own metric instead -- folding them in as 1.0 would inflate the pass rate with criteria
        # nothing actually measured.
        rows = [
            EvaluationResult(key=s.key, score=s.score, comment=s.comment)
            for s in scores
            if s.machine_scored
        ]
        rows.append(EvaluationResult(key="criteria_coverage", score=coverage(scores)))
        return {"results": rows}

    results = evaluate(
        target,
        data=examples,
        evaluators=[scorer],
        experiment_prefix=f"speechwriter-{dataset}",
        max_concurrency=2,
    )
    print(f"\nrecorded experiment: {getattr(results, 'experiment_name', '(see LangSmith)')}")
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
    args = p.parse_args(argv)

    if args.langsmith:
        if args.dry_run:
            print("--langsmith and --dry-run are contradictory", file=sys.stderr)
            return 2
        prime_environment()
        return run_langsmith(args.dataset, args.limit, args.no_judge)

    if not args.dry_run:
        prime_environment()

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
            _, scores = run_one(example, args.no_judge)
        rows.append(report(args.dataset, example, scores))

    total_p = sum(r["passed"] for r in rows)
    total_g = sum(r["graded"] for r in rows)
    print(
        f"\nTOTAL {total_p}/{total_g} machine-scored criteria passed across {len(rows)} example(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
