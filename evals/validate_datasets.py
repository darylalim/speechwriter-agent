"""Deterministic validation of the four eval datasets. Trusts nothing an agent reported."""

import json
import re
import sys
from pathlib import Path

from speechwriter.config import RESEARCH_SUBDIR, SPEECHES_SUBDIR, WORDS_PER_MINUTE

ROOT = Path(__file__).resolve().parent.parent
EV = Path(__file__).resolve().parent / "datasets"
BOUND_TOOLS = {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task"}
SKILLS = {p.parent.name for p in ROOT.glob("skills/*/SKILL.md")}
SUBAGENTS = {"researcher", "style-critic"}
fails, warns = [], []


def check(cond, msg, hard=True):
    if not cond:
        (fails if hard else warns).append(msg)


print(
    f"contract: WPM={WORDS_PER_MINUTE} speeches={SPEECHES_SUBDIR} "
    f"research={RESEARCH_SUBDIR} skills={sorted(SKILLS)}\n"
)

data = {p.stem: json.loads(p.read_text()) for p in sorted(EV.glob("*.json"))}

# --- universal structure ---
all_ids = {}
for name, exs in data.items():
    ids = []
    for i, e in enumerate(exs):
        loc = f"{name}[{i}]"
        check(isinstance(e, dict), f"{loc}: not an object")
        for k in ("inputs", "outputs", "metadata"):
            check(k in e, f"{loc}: missing '{k}'")
        mid = (e.get("metadata") or {}).get("id")
        check(bool(mid), f"{loc}: metadata.id missing")
        ids.append(mid)
        dt = (e.get("metadata") or {}).get("dataset_type")
        check(dt == name, f"{loc} ({mid}): dataset_type={dt!r} != {name!r}", hard=False)
    dupes = {x for x in ids if ids.count(x) > 1}
    check(not dupes, f"{name}: duplicate ids {dupes}")
    all_ids[name] = ids

# --- final_response: recompute every word count ---
for e in data["final_response"]:
    mid = e["metadata"]["id"]
    o = e["outputs"]
    m = e["metadata"]
    tm = m.get("target_minutes")
    check(tm is not None, f"final_response {mid}: no target_minutes")
    if tm is not None and "target_word_count" in o:
        want = round(tm * WORDS_PER_MINUTE)
        got = o["target_word_count"]
        check(
            got == want,
            f"final_response {mid}: target_word_count={got}, {tm}min*{WORDS_PER_MINUTE}={want}",
        )
    tol = o.get("word_count_tolerance")
    check(
        isinstance(tol, (int, float)) and 0 < tol <= 1, f"final_response {mid}: tolerance {tol!r}"
    )
    st = o.get("saved_to", "")
    check(
        st.startswith(f"/workspace/{SPEECHES_SUBDIR}/") and st.endswith(".md"),
        f"final_response {mid}: saved_to={st!r}",
    )
    for k in ("must_mention", "must_not_contain", "required_behaviors"):
        check(isinstance(o.get(k), list), f"final_response {mid}: {k} not a list")
    check(bool(o.get("grading_notes")), f"final_response {mid}: empty grading_notes")

# --- trajectory: tool names, paths, subagents ---
for e in data["trajectory"]:
    mid = e["metadata"]["id"]
    o = e["outputs"]
    for t in o.get("expected_trajectory", []):
        base = re.split(r"[(\[]", str(t))[0].strip()
        check(base in BOUND_TOOLS, f"trajectory {mid}: unbound tool {t!r}")
    for t in o.get("required_tools", []):
        base = re.split(r"[(\[]", str(t))[0].strip()
        check(base in BOUND_TOOLS, f"trajectory {mid}: unbound required_tool {t!r}")
    for s in o.get("required_subagents", []) + o.get("forbidden_subagents", []):
        check(s in SUBAGENTS, f"trajectory {mid}: unknown subagent {s!r}")
    for p in o.get("required_write_paths", []):
        check(
            p.startswith("/workspace/") or p.startswith("/memories/"),
            f"trajectory {mid}: required_write_path outside sandbox: {p!r}",
        )
    for p in o.get("required_skill_reads", []):
        mm = re.search(r"/skills/([a-z-]+)/", p) or re.fullmatch(r"([a-z-]+)", p)
        slug = mm.group(1) if mm else None
        check(slug in SKILLS, f"trajectory {mid}: unknown skill {p!r}")
    check(bool(o.get("tolerance_notes")), f"trajectory {mid}: no tolerance_notes")
    if e["metadata"].get("tavily_enabled") is False:
        check(
            "researcher" in o.get("forbidden_subagents", []),
            f"trajectory {mid}: tavily disabled but researcher not forbidden",
        )

# --- single_step ---
DEC = {
    "ask_clarifying_questions",
    "proceed_with_stated_assumptions",
    "either_acceptable",
    "write_memory",
    "skip_memory",
}
for e in data["single_step"]:
    mid = e["metadata"]["id"]
    o = e["outputs"]
    check(
        o.get("expected_decision") in DEC,
        f"single_step {mid}: bad decision {o.get('expected_decision')!r}",
    )
    check(
        isinstance(e["inputs"].get("messages"), list) and e["inputs"]["messages"],
        f"single_step {mid}: inputs.messages missing/empty",
    )
    for msg in e["inputs"]["messages"]:
        check(
            msg.get("role") in {"user", "assistant", "system"},
            f"single_step {mid}: bad role {msg.get('role')!r}",
        )

# --- rag ---
for e in data["rag"]:
    mid = e["metadata"]["id"]
    o = e["outputs"]
    check("question" in e["inputs"], f"rag {mid}: no inputs.question")
    ms = o.get("must_save_to", "")
    dirform = ms == f"/workspace/{RESEARCH_SUBDIR}/"
    pathform = ms.startswith(f"/workspace/{RESEARCH_SUBDIR}/") and ms.endswith(".md")
    check(dirform or pathform, f"rag {mid}: must_save_to={ms!r}")
    # directory form is the deliberate anti-over-constraint choice; it must carry the slug flag
    if dirform:
        check(
            o.get("slug_should_name_topic") is True,
            f"rag {mid}: directory-form must_save_to without slug_should_name_topic",
        )
    lo, hi = o.get("min_facts"), o.get("max_facts")
    check(
        isinstance(lo, int) and isinstance(hi, int) and lo <= hi,
        f"rag {mid}: fact bounds {lo}-{hi}",
    )
    nds = o.get("min_distinct_sources")
    check(isinstance(nds, int) and nds >= 1, f"rag {mid}: min_distinct_sources={nds!r}")
    # only the documented-negative example may drop to a single source
    if nds == 1:
        check(
            e["metadata"].get("null_result_expected") is True,
            f"rag {mid}: min_distinct_sources=1 but metadata.null_result_expected is not true",
        )
    ar = o.get("angles_range")
    check(
        isinstance(ar, list) and len(ar) == 2 and ar[0] <= ar[1], f"rag {mid}: angles_range={ar!r}"
    )
    check(
        o.get("must_return_angles") in (2, 3),
        f"rag {mid}: must_return_angles={o.get('must_return_angles')!r}",
    )

# --- fabrication scan: URLs and hard numbers asserted in EXPECTED OUTPUTS ---
URL = re.compile(r"https?://")
for name, exs in data.items():
    for e in exs:
        blob = json.dumps(e.get("outputs", {}))
        if URL.search(blob):
            fails.append(
                f"{name} {e['metadata']['id']}: expected output contains a URL (fabrication risk)"
            )

# --- banned tool mention anywhere ---
for name, exs in data.items():
    if "write_todos" in json.dumps(exs):
        fails.append(f"{name}: mentions write_todos (not a bound tool)")

# --- report ---
print(f"{'DATASET':<18}{'N':>4}  ids unique")
for n, exs in data.items():
    print(f"{n:<18}{len(exs):>4}  {'yes' if len(set(all_ids[n])) == len(exs) else 'NO'}")
print(f"\ntotal examples: {sum(len(v) for v in data.values())}")
print(f"\nHARD FAILURES: {len(fails)}")
for f in fails:
    print("  x", f)
print(f"\nWARNINGS: {len(warns)}")
for w in warns:
    print("  !", w)
sys.exit(1 if fails else 0)
