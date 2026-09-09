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
    from speechwriter.config import ModelChoice, Settings

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


def configured_choice() -> ModelChoice:
    """The model/endpoint pair the environment currently names, as one record."""
    from speechwriter.config import ModelChoice, load_settings

    settings = load_settings()
    return ModelChoice(settings.model, settings.model, settings.base_url, settings.context_window)


def configured_settings() -> Settings:
    """The whole configuration in force right now — what the judge must be built from.

    A :class:`~speechwriter.config.ModelChoice` is not enough, and used to be what was captured
    here. ``ModelChoice.applied_to`` moves the *credential* along with the pair now, dropping
    ``openai_api_key`` unless the choice names the endpoint the settings were configured with —
    so applying the captured judge to a ``load_settings()`` read *after* :func:`apply_model` has
    blanked ``SPEECHWRITER_BASE_URL`` compares the judge's endpoint against ``None``, decides it
    is a stranger, and sends the placeholder bearer instead of the real key. Every judge call
    then 401s, and only when ``--model`` is used, which is the one path that has a judge to pin.

    Capturing the settings whole sidesteps the question: the judge is built from exactly the
    configuration that was in force before the override, credential included.
    """
    from speechwriter.config import load_settings

    return load_settings()


def apply_model(requested: str) -> None:
    """Point this run at a chosen model, *before* ``prime_environment`` reads the dotenv.

    Works through ``os.environ`` rather than by passing a ``Settings`` down, because
    ``invoke_agent`` builds a fresh agent per example inside a temp home and reads the
    environment each time. ``load_dotenv`` never overrides an already-set variable, so a value
    put here survives the priming that follows.

    A roster label or id resolves to a whole :class:`~speechwriter.config.ModelChoice`, and the
    endpoint moves with it. That half is not cosmetic: ``base_url`` — never the id — is what
    selects the client, so choosing a Claude entry while ``SPEECHWRITER_BASE_URL`` happens to be
    exported would otherwise send that id to a local server. Anything unrecognised is passed
    through verbatim as an id the operator means literally, endpoint untouched.
    """
    from speechwriter.config import (
        load_settings,
        matching_choices,
        model_choices,
        resolve_choice,
    )

    # Against the *full* roster, not just the curated tuple: the label a reader copies out of
    # `/model` or the sidebar for a local model is "<id> (local)", and matching only
    # MODEL_CHOICES passed that whole string through as a model id — so the server 404s at the
    # first turn of every graded example, long after the temp homes are set up.
    roster = model_choices(load_settings())
    choice = resolve_choice(roster, requested)
    if choice is None and len(matching_choices(roster, requested)) > 1:
        # `resolve_choice` answers None for "no such entry" *and* for "two entries by that
        # name", and the pass-through below is only right for the first. One id served by two
        # machines makes the second reachable — the same weights on a laptop and a workstation,
        # or an identical Ollama tag — and passing it through would set the id while leaving
        # SPEECHWRITER_BASE_URL pointed at whichever server it already named.
        # The remedy is the ROW NUMBER, not the label. `local_choice` labels an entry with the
        # bare model id, so two servers serving one id have identical labels and
        # `matching_choices` matches on either — telling the reader to "use the label" would
        # print the same string twice and name no way out. The endpoint is the only thing that
        # differs, so it is what the message shows, beside the index that selects it.
        raise SystemExit(
            f"--model {requested!r} names more than one roster entry — the same id on more "
            f"than one server. Select by row number instead:\n"
            + "\n".join(
                f"  --model {roster.index(match) + 1}   {match.model}   {match.base_url}"
                for match in matching_choices(roster, requested)
            )
        )
    if choice is not None:
        os.environ["SPEECHWRITER_MODEL"] = choice.model
        # Set to empty, never popped. Every later `load_settings()` — in `prime_environment`,
        # in `run_langsmith`, and in the `build_agent()` of every graded example — reloads
        # the dotenv, and `load_dotenv` only skips keys already present in `os.environ`, so
        # a popped variable comes straight back. An empty one stays: it is *present*, so the
        # dotenv will not override it.
        #
        # No `or ""` guard: a choice always carries an endpoint now, because `ModelChoice` makes
        # an id without one unrepresentable, so this is always a real URL. It is still *set*
        # rather than left alone — the whole point of `--model` taking a pair is that both
        # halves move together. (A blank value would no longer mean what it used to, either:
        # `load_settings` reads `"" -> DEFAULT_LOCAL_ENDPOINT` now, not `"" -> None`.)
        os.environ["SPEECHWRITER_BASE_URL"] = choice.base_url
        return
    os.environ["SPEECHWRITER_MODEL"] = requested


def model_slug() -> str:
    """A filename-safe tag for the model in force, for naming an experiment after it.

    Call only after :func:`prime_environment`. **The host is part of the tag**, and it carries
    exactly the weight the old ``-local`` suffix did. That suffix separated a model served
    locally from the same id served by Anthropic; with no hosted path left it would be true of
    every run and separate nothing, while the distinction it existed to make — same id, two
    different systems under test — is now precisely the one between two *servers*. An
    experiment name that could not tell them apart would silently pool their results.

    The cost is a noisier name on the common single-server setup. That is the right side to err
    on: a name that is ugly is a nuisance, a name that merges two experiments is a wrong number.
    """
    from urllib.parse import urlsplit

    from speechwriter.config import load_settings

    settings = load_settings()
    host = urlsplit(settings.base_url).netloc
    raw = f"{settings.model}-{host}".casefold()
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
        bundle = build_agent()
        state = bundle.agent.invoke(
            {"messages": to_messages(inputs)}, config=bundle.turn_config(thread_id)
        )
        record = extract(state, Path(home) / "workspace")
        if keep:
            print(f"   [kept] artifacts under {home}")
        if previous_home is None:
            os.environ.pop("SPEECHWRITER_HOME", None)
        else:
            os.environ["SPEECHWRITER_HOME"] = previous_home
        return record


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

        # The pinned judge is used *as captured*, never re-applied to the current environment:
        # `apply_model` has since blanked SPEECHWRITER_BASE_URL, and `applied_to` would read
        # that as "this endpoint is a stranger" and withhold the endpoint credential.
        settings = judge if judge is not None else load_settings()
        scores += judge_example(_build_model(settings), dataset, run, example)
    return scores


def run_one(
    example: dict[str, Any], no_judge: bool, judge: Settings | None = None
) -> tuple[RunRecord, list[Score]]:
    dataset = example["metadata"]["dataset_type"]
    run = invoke_agent(example["inputs"], example["metadata"]["id"])
    return run, grade(dataset, run, example, no_judge, judge)


def run_langsmith(dataset: str, limit: int, no_judge: bool, judge: Settings | None = None) -> int:
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
        scores = grade(dataset, record, payload, no_judge, judge)
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
        # The model is part of the experiment's identity, not incidental to it: comparing two
        # models is the reason the model is selectable at all, and a prefix that named only the
        # dataset would file both runs under one indistinguishable name in the mirror.
        experiment_prefix=f"speechwriter-{dataset}-{model_slug()}",
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
    p.add_argument("--keep", action="store_true", help="leave each run's temp home on disk")
    p.add_argument(
        "--model",
        help="model to grade: a roster label ('Opus 5'), a model id, or omit for the .env one",
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

    if args.langsmith:
        return run_langsmith(args.dataset, args.limit, args.no_judge, judge)

    if args.keep:
        os.environ["SPEECHWRITER_EVAL_KEEP"] = "1"

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
