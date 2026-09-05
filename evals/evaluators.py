"""Scoring one graded run against the criteria in ``evals/datasets/*.json``.

Split deliberately in two. Everything here is **pure**: it takes a :class:`RunRecord` -- what
the agent did, already extracted from the message stream -- plus an example's ``outputs`` and
``metadata``, and returns :class:`Score` rows. No model, no network, no LangSmith. That is what
lets the scorers be tested offline against synthetic runs, and it is the same split that keeps
the rest of this repo's suite free.

Three rules the datasets state about themselves, encoded here rather than re-derived:

**``expected_trajectory`` is a reference path, not an assertion.** The dataset description on
LangSmith says so outright: "exact sequence matching fails legitimately-correct runs." It is
reported as a similarity signal and never scored. The pass/fail axes are ``required_tools``,
``required_subagents``, the write-path fields and ``order_constraints``.

**``required_write_paths`` entries are prefixes.** Every trajectory example carries
``path_semantics`` saying so ("a write matches if its path starts with the entry"), and
``order_constraint_semantics`` says each ordering entry is "a pairwise 'A before B' check,
vacuously satisfied when either side does not occur in the run."

**``must_not_contain`` mixes two kinds in one list, and the quoting tells them apart.** An entry
wrapped in double quotes is a literal substring; a bare entry is prose for a judge.
``validate_datasets.py`` locks that convention -- it rejects a half-quoted entry precisely so the
split stays a mechanical transform -- and :func:`split_must_not_contain` is the transform.

The one rule that is this module's own: **a criterion that cannot be machine-checked scores
``None``, never ``1.0``.** An order constraint whose right-hand side is prose ("before any
clarifying question about Daryl's voice") is not a passing constraint, it is an unscored one, and
:func:`unscored` counts them so a report can say how much of an example was actually measured.
Silently passing what it cannot read is the exact vacuity failure the dataset audit hunted for.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from speechwriter.config import SPEECHES_SUBDIR, WORDS_PER_MINUTE, load_settings
from speechwriter.workspace import spoken_words

BOUND_TOOLS = ("ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task")

# Tool argument keys deepagents uses. Kept in one place: a rename in the harness should break
# every scorer at once and loudly, not silently zero one axis.
PATH_KEYS = ("file_path", "path")
SUBAGENT_KEY = "subagent_type"


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        for key in PATH_KEYS:
            value = self.args.get(key)
            if isinstance(value, str):
                return value
        return ""

    @property
    def subagent(self) -> str:
        value = self.args.get(SUBAGENT_KEY)
        return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class RunRecord:
    """What one graded run produced: the final text, and every tool call in order."""

    text: str
    calls: tuple[ToolCall, ...] = ()

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    @property
    def subagents(self) -> list[str]:
        return [c.subagent for c in self.calls if c.name == "task" and c.subagent]

    def paths(self, *tools: str) -> list[str]:
        return [c.path for c in self.calls if c.name in tools and c.path]

    @property
    def writes(self) -> list[str]:
        return self.paths("write_file", "edit_file")

    @property
    def reads(self) -> list[str]:
        return self.paths("read_file")


@dataclass(frozen=True)
class Score:
    """One graded axis. ``score=None`` means *not machine-checkable*, which is not a pass."""

    key: str
    score: float | None
    comment: str

    @property
    def machine_scored(self) -> bool:
        return self.score is not None


def unscored(scores: list[Score]) -> list[Score]:
    """The rows a report must surface separately, so coverage is never overstated."""
    return [s for s in scores if not s.machine_scored]


def coverage(scores: list[Score]) -> float:
    """Fraction of criteria that were actually measured. 1.0 only if nothing fell through."""
    return (len(scores) - len(unscored(scores))) / len(scores) if scores else 0.0


# --- must_not_contain: the quoting convention validate_datasets.py locks ---


def split_must_not_contain(entries: list[str]) -> tuple[list[str], list[str]]:
    """``(literal substrings, judge prose)``.

    An entry fully wrapped in double quotes is a literal to search for; a bare entry is prose.
    The validator rejects a half-quoted entry, so this transform is total.
    """
    literals, prose = [], []
    for raw in entries:
        text = str(raw).strip()
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            literals.append(text[1:-1])
        else:
            prose.append(text)
    return literals, prose


# --- order constraints ---

# "at least one read_file(/skills/**/SKILL.md)" -> tool=read_file arg=/skills/**/SKILL.md.
# Anything that does not match this shape is prose, and prose is unscored rather than passed.
# The parenthesised argument is optional, so a bare "the first edit_file" is checkable while
# "the final edit_file (or the final overwriting write_file) on the same speech path" is not --
# the anchors reject it, and it lands in the unscored tail where it belongs.
_SIDE = re.compile(
    r"^(?:at least one|the first|the final|the|any|a)?\s*"
    r"(?P<tool>" + "|".join(BOUND_TOOLS) + r")"
    r"(?:\((?P<arg>[^)]*)\))?$"
)


def _side_matches(call: ToolCall, tool: str, arg: str | None) -> bool:
    if call.name != tool:
        return False
    if not arg:
        return True
    if tool == "task":
        return call.subagent == arg
    if any(ch in arg for ch in "*?"):
        return fnmatch.fnmatch(call.path, arg)
    return call.path.startswith(arg)


def check_order_constraint(run: RunRecord, constraint: str) -> Score:
    """Pairwise 'A before B', vacuously satisfied when either side never occurs.

    A trailing ``, if ...`` condition needs no special handling: the condition always names the
    circumstance under which one side occurs at all, and an absent side is already vacuous.
    """
    body = re.split(r",\s*if\s+", constraint, maxsplit=1)[0]
    parts = re.split(r"\s+before\s+", body, maxsplit=1)
    if len(parts) != 2:
        return Score("order", None, f"unparseable (no 'before'): {constraint!r}")
    left, right = (_SIDE.match(p.strip()) for p in parts)
    if left is None or right is None:
        side = "left" if left is None else "right"
        return Score("order", None, f"{side} side is prose, not machine-checkable: {constraint!r}")

    first_a = next(
        (i for i, c in enumerate(run.calls) if _side_matches(c, **left.groupdict())), None
    )
    last_b = next(
        (
            i
            for i in range(len(run.calls) - 1, -1, -1)
            if _side_matches(run.calls[i], **right.groupdict())
        ),
        None,
    )
    if first_a is None or last_b is None:
        return Score("order", 1.0, f"vacuous (a side never occurs): {constraint}")
    ok = first_a < last_b
    return Score("order", float(ok), f"{'ok' if ok else 'VIOLATED'}: {constraint}")


# --- per-dataset scorers ---


def _prefix_hits(paths: list[str], prefixes: list[str]) -> list[str]:
    return [p for p in prefixes if not any(w.startswith(p) for w in paths)]


def score_trajectory(run: RunRecord, out: dict[str, Any], meta: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    called, subs = set(run.tool_names), set(run.subagents)

    missing = [t for t in out.get("required_tools") or [] if t.split("(")[0].strip() not in called]
    scores.append(
        Score(
            "required_tools", float(not missing), f"missing {missing}" if missing else "all present"
        )
    )

    need = [s for s in out.get("required_subagents") or [] if s not in subs]
    scores.append(
        Score("required_subagents", float(not need), f"missing {need}" if need else "all present")
    )
    banned = [s for s in out.get("forbidden_subagents") or [] if s in subs]
    scores.append(
        Score(
            "forbidden_subagents",
            float(not banned),
            f"called {banned}" if banned else "none called",
        )
    )

    unwritten = _prefix_hits(run.writes, out.get("required_write_paths") or [])
    scores.append(
        Score(
            "required_write_paths",
            float(not unwritten),
            f"never written: {unwritten}" if unwritten else "ok",
        )
    )
    for key in ("forbidden_write_paths", "forbidden_write_attempt_paths"):
        hit = [
            p for p in out.get(key) or [] if any(w.startswith(p.rstrip("*")) for w in run.writes)
        ]
        scores.append(Score(key, float(not hit), f"wrote {hit}" if hit else "none"))

    unread = _prefix_hits(run.reads, out.get("required_skill_reads") or [])
    scores.append(
        Score(
            "required_skill_reads", float(not unread), f"never read: {unread}" if unread else "ok"
        )
    )
    any_of = out.get("required_skill_read_any_of") or []
    if any_of:
        hit = any(any(r.startswith(p) for r in run.reads) for p in any_of)
        scores.append(Score("required_skill_read_any_of", float(hit), f"any of {any_of}: {hit}"))

    for constraint in out.get("order_constraints") or []:
        scores.append(check_order_constraint(run, constraint))

    # Reported, never scored -- the dataset description is explicit that exact matching fails
    # legitimately-correct runs. Kept as a signal a human can eyeball in the experiment table.
    expected = out.get("expected_trajectory") or []
    overlap = len(set(t.split("(")[0] for t in expected) & called)
    scores.append(
        Score(
            "expected_trajectory_similarity",
            None,
            f"reference {expected} vs actual {run.tool_names} ({overlap} distinct tools shared)",
        )
    )
    return scores


def score_final_response(run: RunRecord, out: dict[str, Any], meta: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    target, tol = out.get("target_word_count"), out.get("word_count_tolerance")
    words = spoken_words(run.text)
    if isinstance(target, (int, float)) and isinstance(tol, (int, float)) and target:
        lo, hi = target * (1 - tol), target * (1 + tol)
        ok = lo <= words <= hi
        scores.append(
            Score(
                "word_count",
                float(ok),
                f"{words} words vs {target} +/-{tol:.0%} ({lo:.0f}-{hi:.0f}), "
                f"~{words / WORDS_PER_MINUTE:.2f}min at {WORDS_PER_MINUTE}wpm",
            )
        )

    saved = out.get("saved_to", "")
    speeches = f"{load_settings().workspace_vpath}/{SPEECHES_SUBDIR}/"
    # The directory and the .md suffix are strict; the slug is indicative, per grading_notes.
    hit = [w for w in run.writes if w.startswith(speeches) and w.endswith(".md")]
    scores.append(
        Score("saved_to", float(bool(hit)), f"wrote {hit or 'nothing'} (reference {saved!r})")
    )

    literals, prose = split_must_not_contain(out.get("must_not_contain") or [])
    found = [lit for lit in literals if lit.lower() in run.text.lower()]
    scores.append(
        Score(
            "must_not_contain_literal",
            float(not found),
            f"found {found}" if found else f"clean ({len(literals)} literals)",
        )
    )
    for key, items in (
        ("must_mention", out.get("must_mention") or []),
        ("must_not_contain_prose", prose),
        ("required_behaviors", out.get("required_behaviors") or []),
    ):
        if items:
            scores.append(Score(key, None, f"{len(items)} criteria for a judge"))
    return scores


def score_single_step(run: RunRecord, out: dict[str, Any], meta: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    asked = count_questions(run.text)
    cap = out.get("max_questions")
    if isinstance(cap, int):
        ok = asked <= cap
        scores.append(Score("max_questions", float(ok), f"asked ~{asked}, cap {cap}"))
    decision = out.get("expected_decision")
    branch = "either branch passes" if decision == "either_acceptable" else "judge the branch"
    scores.append(
        Score(
            "expected_decision",
            None,
            f"reference {decision!r}; ~{asked} questions asked ({branch})",
        )
    )
    for key in ("must_cover", "must_not_do"):
        if out.get(key):
            scores.append(Score(key, None, f"{len(out[key])} criteria for a judge"))
    return scores


def score_rag(run: RunRecord, out: dict[str, Any], meta: dict[str, Any]) -> list[Score]:
    scores: list[Score] = []
    must = out.get("must_save_to", "")
    hit = [w for w in run.writes if w.startswith(must.rstrip("/") if must else "\0")]
    scores.append(
        Score("must_save_to", float(bool(hit)), f"wrote {hit or 'nothing'} (under {must!r})")
    )
    for key in ("required_source_properties", "must_not_do", "required_fact_types"):
        if out.get(key):
            scores.append(Score(key, None, f"{len(out[key])} criteria for a judge"))
    for key in (
        "min_facts",
        "max_facts",
        "min_distinct_sources",
        "angles_range",
        "must_return_angles",
    ):
        if key in out:
            scores.append(
                Score(key, None, f"needs the note parsed by a judge; reference {out[key]!r}")
            )
    return scores


# A question mark ends a question; a bare "?" inside a quotation does not start a new one. Crude
# on purpose -- it is a *cap* check, and the branch itself is judged, so an off-by-one here
# cannot flip a correct run to failing on its own.
_QUESTION = re.compile(r"\?[\s\"')\]]*(?:\n|$|[A-Z])")


def count_questions(text: str) -> int:
    return len(_QUESTION.findall(text)) or text.count("?")


SCORERS = {
    "final_response": score_final_response,
    "trajectory": score_trajectory,
    "single_step": score_single_step,
    "rag": score_rag,
}


def score_example(dataset: str, run: RunRecord, example: dict[str, Any]) -> list[Score]:
    """Every machine-checkable criterion for one example, plus an explicit unscored tail."""
    return SCORERS[dataset](run, example.get("outputs") or {}, example.get("metadata") or {})


# --- the judge half: prose criteria a substring check cannot reach ---

JUDGE_SYSTEM = """You grade one speechwriting output against one criterion at a time.

You are not the speechwriter and you are not improving anything. For each criterion return a
verdict on whether the OUTPUT satisfies it, judged only on what the output actually says.

Rules:
- Judge the criterion as written. Do not substitute a stricter or looser one.
- A `must_mention` criterion is satisfied by substance, not by exact wording: a paraphrase that
  a listener would recognise as the same fact passes.
- A `must_not_contain` criterion is VIOLATED if the output does the thing described. Absence is
  the passing state, so default to satisfied when the output simply never goes there.
- A `required_behaviors` criterion often offers alternatives joined by EITHER/OR. Any one branch
  satisfies it.
- If a criterion is not applicable to the branch the output took (for example a word-count
  criterion when the output correctly asked a clarifying question and stopped), return
  applicable=false rather than guessing.
"""

JUDGE_SCHEMA = {
    "title": "CriterionVerdicts",
    "description": "One verdict per criterion, in the order given.",
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "satisfied": {"type": "boolean"},
                    "applicable": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "satisfied", "applicable", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


def judge_criteria(model: Any, key: str, output_text: str, criteria: list[str]) -> list[Score]:
    """Grade one list of prose criteria, one :class:`Score` per criterion.

    ``model`` is any LangChain chat model; it is passed in rather than constructed so the caller
    owns the ceiling resolution (``agent._build_model``) and so this module still imports with no
    key present. An inapplicable criterion scores ``None`` -- unscored, not passed -- for the same
    reason the order-constraint parser does.
    """
    if not criteria:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
    prompt = (
        f"CRITERION LIST ({key}), {len(criteria)} items:\n{numbered}\n\n"
        f"OUTPUT UNDER TEST:\n<<<\n{output_text}\n>>>\n\n"
        f"Return exactly {len(criteria)} verdicts, one per index."
    )
    reply = model.with_structured_output(JUDGE_SCHEMA).invoke(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    )
    by_index = {v["index"]: v for v in (reply or {}).get("verdicts", [])}
    scores = []
    for i, criterion in enumerate(criteria):
        v = by_index.get(i)
        if v is None:
            scores.append(
                Score(f"{key}[{i}]", None, f"judge returned no verdict: {criterion[:70]}")
            )
        elif not v.get("applicable", True):
            scores.append(
                Score(f"{key}[{i}]", None, f"n/a for this branch: {v.get('reason', '')[:80]}")
            )
        else:
            scores.append(
                Score(f"{key}[{i}]", float(bool(v["satisfied"])), v.get("reason", "")[:160])
            )
    return scores


# Which prose fields each dataset hands the judge, and whether satisfying them means the output
# DOES the thing or AVOIDS it. The negative lists are graded the same way -- the criterion text
# already describes the forbidden behaviour, and JUDGE_SYSTEM tells the judge absence passes.
JUDGED_FIELDS = {
    "final_response": ("must_mention", "required_behaviors"),
    "single_step": ("must_cover", "must_not_do"),
    "rag": ("required_source_properties", "must_not_do", "required_fact_types"),
    "trajectory": (),
}


def judge_example(model: Any, dataset: str, run: RunRecord, example: dict[str, Any]) -> list[Score]:
    """Every prose criterion for one example. Pairs with :func:`score_example`."""
    out = example.get("outputs") or {}
    scores: list[Score] = []
    for key in JUDGED_FIELDS[dataset]:
        scores.extend(judge_criteria(model, key, run.text, list(out.get(key) or [])))
    if dataset == "final_response":
        _, prose = split_must_not_contain(out.get("must_not_contain") or [])
        scores.extend(judge_criteria(model, "must_not_contain", run.text, prose))
    return scores
