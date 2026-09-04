"""Deterministic validation of the four eval datasets under ``evals/datasets``.

Trusts nothing an agent reported: every derived number is recomputed here, and every
path, tool name, subagent name and skill slug is checked against the live contract
rather than against a copy of it. The virtual paths come from :func:`load_settings`
for the reason CLAUDE.md gives about its four consumers -- a validator that hard-codes
``/workspace/`` would keep passing after a rename while the sandbox started denying
the writes these datasets tell the agent to make.

Run it with ``uv run python evals/validate_datasets.py`` (it imports the package, so a
bare ``python`` will not resolve ``speechwriter.config``). Exit 1 on any hard failure.
"""

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from speechwriter.config import (
    RESEARCH_SUBDIR,
    SPEECHES_SUBDIR,
    WORDS_PER_MINUTE,
    load_settings,
)

ROOT = Path(__file__).resolve().parent.parent
EV = Path(__file__).resolve().parent / "datasets"

settings = load_settings()
WORKSPACE = settings.workspace_vpath  # "/workspace", deliberately no trailing slash
MEMORIES = settings.memories_vpath  # "/memories/", deliberately with one
SKILLS_V = settings.skills_vpath  # "/skills/", deliberately with one
SPEECHES = f"{WORKSPACE}/{SPEECHES_SUBDIR}/"
RESEARCH = f"{WORKSPACE}/{RESEARCH_SUBDIR}/"
WRITE_ALLOWED = (f"{WORKSPACE}/", MEMORIES)

BOUND_TOOLS = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task"}
SUBAGENTS = {"researcher", "style-critic"}
SKILLS = {p.parent.name for p in ROOT.glob("skills/*/SKILL.md")}

# Hard-coded like ``test_all_skills_have_valid_frontmatter``'s ``len(skill_dirs) == 4``:
# without a floor, a dataset truncated to ``[]`` by a bad merge validates clean.
EXPECTED_COUNTS = {"final_response": 16, "trajectory": 13, "single_step": 14, "rag": 12}

fails: list[str] = []
warns: list[str] = []


def check(cond: object, msg: str, hard: bool = True) -> None:
    if not cond:
        (fails if hard else warns).append(msg)


def is_int(x: object) -> bool:
    """``bool`` subclasses ``int``, so ``isinstance`` would accept JSON ``true``."""
    return type(x) is int


def is_num(x: object) -> bool:
    return type(x) in (int, float)


def in_sandbox(p: str) -> bool:
    """Mirrors ``agent._write_sandbox()``: writes are allowed under workspace + memories."""
    return p.startswith(WRITE_ALLOWED)


def skill_slug(p: str) -> str | None:
    m = re.search(rf"{re.escape(SKILLS_V)}([a-z0-9-]+)/", p) or re.fullmatch(r"([a-z0-9-]+)", p)
    return m.group(1) if m else None


def check_skill_refs(label: str, o: dict[str, Any], *keys: str) -> None:
    for k in keys:
        for p in o.get(k) or []:
            check(skill_slug(str(p)) in SKILLS, f"{label}: {k} names unknown skill {p!r}")


print(
    f"contract: WPM={WORDS_PER_MINUTE} workspace={WORKSPACE!r} memories={MEMORIES!r} "
    f"skills={SKILLS_V!r}\n  speeches={SPEECHES!r} research={RESEARCH!r} "
    f"slugs={sorted(SKILLS)}\n"
)

# The trailing-slash asymmetry is load-bearing (CLAUDE.md); a normalising edit yields "//".
for label, p in (("speeches", SPEECHES), ("research", RESEARCH), ("memories", MEMORIES)):
    check("//" not in p, f"derived path {label}={p!r} contains '//' -- trailing-slash drift")

# --- load: fixed roster, so a renamed or vanished file fails instead of being skipped ---
found = {p.stem for p in EV.glob("*.json")}
check(found == set(EXPECTED_COUNTS), f"dataset files {sorted(found)} != {sorted(EXPECTED_COUNTS)}")

data: dict[str, list[dict[str, Any]]] = {}
for name in EXPECTED_COUNTS:
    p = EV / f"{name}.json"
    if not p.exists():
        fails.append(f"{name}: {p} is missing")
        data[name] = []
        continue
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fails.append(f"{name}: unreadable ({type(exc).__name__}: {exc})")
        data[name] = []
        continue
    check(isinstance(loaded, list), f"{name}: top level is {type(loaded).__name__}, not a list")
    data[name] = loaded if isinstance(loaded, list) else []

for name, want in EXPECTED_COUNTS.items():
    got = len(data[name])
    check(
        got == want, f"{name}: {got} examples, expected {want} (update EXPECTED_COUNTS if intended)"
    )

# --- universal structure; malformed examples are recorded, then excluded from later passes ---
ok: dict[str, list[dict[str, Any]]] = {}
for name, exs in data.items():
    ids, good = [], []
    for i, e in enumerate(exs):
        loc = f"{name}[{i}]"
        if not isinstance(e, dict):
            fails.append(f"{loc}: not an object")
            continue
        shaped = True
        for k in ("inputs", "outputs", "metadata"):
            if not isinstance(e.get(k), dict):
                fails.append(f"{loc}: '{k}' missing or not an object")
                shaped = False
        if not shaped:
            continue
        mid = e["metadata"].get("id")
        check(bool(mid) and isinstance(mid, str), f"{loc}: metadata.id missing or not a string")
        ids.append(mid)
        dt = e["metadata"].get("dataset_type")
        # Hard, not advisory: dataset_type selects the evaluator, so a mis-filed example is
        # graded by the wrong one and scores vacuously rather than failing.
        check(dt == name, f"{loc} ({mid}): dataset_type={dt!r} != {name!r}")
        good.append(e)
    dupes = {x for x in ids if ids.count(x) > 1}
    check(not dupes, f"{name}: duplicate ids {sorted(dupes)}")
    ok[name] = good


def label_of(name: str, e: dict[str, Any]) -> str:
    return f"{name} {e['metadata'].get('id')}"


try:
    # --- every derived word count, recomputed from the imported constant ---
    for e in ok["final_response"]:
        lab, o, m = label_of("final_response", e), e["outputs"], e["metadata"]
        tm = m.get("target_minutes")
        check(is_num(tm), f"{lab}: target_minutes={tm!r}")
        if is_num(tm) and "target_word_count" in o:
            want = round(tm * WORDS_PER_MINUTE)
            check(
                o["target_word_count"] == want,
                f"{lab}: target_word_count={o['target_word_count']}, "
                f"{tm}min*{WORDS_PER_MINUTE}={want}",
            )
        tol = o.get("word_count_tolerance")
        check(is_num(tol) and 0 < tol <= 1, f"{lab}: word_count_tolerance={tol!r}")
        st = o.get("saved_to", "")
        check(
            isinstance(st, str) and st.startswith(SPEECHES) and st.endswith(".md"),
            f"{lab}: saved_to={st!r} not under {SPEECHES}",
        )
        for k in ("must_mention", "must_not_contain", "required_behaviors"):
            check(isinstance(o.get(k), list), f"{lab}: {k} is not a list")
        check(bool(o.get("grading_notes")), f"{lab}: empty grading_notes")

    # The same 130-wpm figure appears in trajectory and rag metadata; recompute it there too,
    # or a WORDS_PER_MINUTE change fails only final_response and leaves the rest silently stale.
    for name in ("trajectory", "rag", "single_step"):
        for e in ok[name]:
            lab, m = label_of(name, e), e["metadata"]
            if "approx_expected_words_at_130wpm" not in m:
                continue
            tm = m.get("target_minutes")
            check(is_num(tm), f"{lab}: approx_expected_words_at_130wpm without target_minutes")
            if is_num(tm):
                want = round(tm * WORDS_PER_MINUTE)
                got = m["approx_expected_words_at_130wpm"]
                check(
                    got == want,
                    f"{lab}: approx_expected_words_at_130wpm={got}, "
                    f"{tm}min*{WORDS_PER_MINUTE}={want}",
                )

    # --- trajectory: tools, subagents, every path field, every skill field ---
    for e in ok["trajectory"]:
        lab, o, m = label_of("trajectory", e), e["outputs"], e["metadata"]
        for k in ("expected_trajectory", "required_tools"):
            for t in o.get(k) or []:
                base = re.split(r"[(\[]", str(t))[0].strip()
                check(base in BOUND_TOOLS, f"{lab}: {k} names unbound tool {t!r}")
        for k in ("required_subagents", "forbidden_subagents"):
            for s in o.get(k) or []:
                check(s in SUBAGENTS, f"{lab}: {k} names unknown subagent {s!r}")
        # Paths the run must write: inside the sandbox, or the harness cannot satisfy them.
        for k in ("required_write_paths", "subagent_write_paths"):
            for p in o.get(k) or []:
                check(in_sandbox(str(p)), f"{lab}: {k} outside the write sandbox: {p!r}")
        # Paths the run must NOT write: outside it, or the example forbids a legal write.
        for k in ("forbidden_write_paths", "forbidden_write_attempt_paths"):
            for p in o.get(k) or []:
                check(
                    not in_sandbox(str(p).rstrip("*")),
                    f"{lab}: {k} forbids {p!r}, which the sandbox allows",
                )
        check_skill_refs(
            lab, o, "required_skill_reads", "expected_skill_reads", "required_skill_read_any_of"
        )
        check(bool(o.get("tolerance_notes")), f"{lab}: no tolerance_notes")
        if m.get("tavily_enabled") is False:
            check(
                "researcher" in (o.get("forbidden_subagents") or []),
                f"{lab}: tavily disabled but researcher is not forbidden",
            )

    # --- single_step ---
    DECISIONS = {
        "ask_clarifying_questions",
        "proceed_with_stated_assumptions",
        "either_acceptable",
        "write_memory",
        "skip_memory",
    }
    for e in ok["single_step"]:
        lab, o = label_of("single_step", e), e["outputs"]
        check(o.get("expected_decision") in DECISIONS, f"{lab}: bad expected_decision")
        msgs = e["inputs"].get("messages")
        check(isinstance(msgs, list) and bool(msgs), f"{lab}: inputs.messages missing or empty")
        for msg in msgs or []:
            check(
                isinstance(msg, dict) and msg.get("role") in {"user", "assistant", "system"},
                f"{lab}: bad message role {(msg or {}).get('role')!r}",
            )
        mp = e["metadata"].get("expected_memory_path")
        if mp:
            check(str(mp).startswith(MEMORIES), f"{lab}: expected_memory_path={mp!r}")

    # --- rag ---
    for e in ok["rag"]:
        lab, o, m = label_of("rag", e), e["outputs"], e["metadata"]
        check("question" in e["inputs"], f"{lab}: no inputs.question")
        ms = o.get("must_save_to", "")
        dirform = ms == RESEARCH
        pathform = isinstance(ms, str) and ms.startswith(RESEARCH) and ms.endswith(".md")
        check(dirform or pathform, f"{lab}: must_save_to={ms!r} not under {RESEARCH}")
        if dirform:
            # Directory form is the deliberate anti-over-constraint choice: the researcher
            # invents its own slug, so pinning a filename would fail a correct run.
            check(
                o.get("slug_should_name_topic") is True,
                f"{lab}: directory-form must_save_to without slug_should_name_topic",
            )
        lo, hi = o.get("min_facts"), o.get("max_facts")
        check(is_int(lo) and is_int(hi) and lo <= hi, f"{lab}: fact bounds {lo!r}-{hi!r}")
        nds = o.get("min_distinct_sources")
        check(is_int(nds) and nds >= 1, f"{lab}: min_distinct_sources={nds!r}")
        # Biconditional, so the reverse mistake is caught too: only the documented-negative
        # example may drop to a single source, and it must say so in metadata.
        check(
            (nds == 1) == (m.get("null_result_expected") is True),
            f"{lab}: min_distinct_sources={nds!r} disagrees with "
            f"metadata.null_result_expected={m.get('null_result_expected')!r}",
        )
        ar = o.get("angles_range")
        check(
            isinstance(ar, list)
            and len(ar) == 2
            and is_int(ar[0])
            and is_int(ar[1])
            and ar[0] <= ar[1],
            f"{lab}: angles_range={ar!r}",
        )
        check(o.get("must_return_angles") in (2, 3), f"{lab}: must_return_angles")

    # --- fabrication scan: a URL anywhere in an example is a fabricated-source risk ---
    # Scans inputs and metadata as well as outputs: a URL pasted into a store_seed fixture
    # or a speech_context is ground truth the agent is scored against just the same.
    URL = re.compile(r"https?://")
    for name, exs in ok.items():
        for e in exs:
            for section in ("inputs", "outputs", "metadata"):
                if URL.search(json.dumps(e.get(section, {}))):
                    fails.append(f"{label_of(name, e)}: {section} contains a URL")

    # --- no example may name a tool the model cannot call ---
    for name, exs in ok.items():
        for e in exs:
            check("write_todos" not in json.dumps(e), f"{label_of(name, e)}: mentions write_todos")

except Exception:  # noqa: BLE001 -- the report must survive any bug in the checks above
    fails.append("validator raised while checking:\n" + traceback.format_exc())

# --- report ---
print(f"{'DATASET':<18}{'N':>4}{'OK':>5}  ids unique")
for name in EXPECTED_COUNTS:
    n, g = len(data[name]), len(ok.get(name, []))
    unique = "yes" if len({x["metadata"].get("id") for x in ok.get(name, [])}) == g else "NO"
    print(f"{name:<18}{n:>4}{g:>5}  {unique}")
print(f"\ntotal examples: {sum(len(v) for v in data.values())}")
print(f"\nHARD FAILURES: {len(fails)}")
for f in fails:
    print("  x", f)
print(f"\nWARNINGS: {len(warns)}")
for w in warns:
    print("  !", w)
sys.exit(1 if fails else 0)
