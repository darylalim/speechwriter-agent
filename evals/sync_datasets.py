"""Mirror ``evals/datasets/*.json`` into LangSmith, or verify that the mirror is intact.

The four datasets were already on LangSmith and the repo had no way to know it: nothing here
could tell an in-sync remote from a stale one, so the only way to answer the question was to
re-derive it by hand. ``--check`` (the default) answers it and writes nothing; ``--push`` makes
the remote match local exactly. Local is always the source of truth -- this is a *push-only*
mirror, so an edit made in the LangSmith UI is overwritten rather than merged back.

Two traps are encoded here because both cost a wrong answer the first time:

**Never diff against ``langsmith dataset export``.** That CLI path drops ``metadata`` entirely,
so a comparison built on it reports all 55 examples as having lost the field that carries their
grading semantics -- ``path_semantics``, ``max_clarifying_questions``,
``grade_tool_calls_after_message_index``. The server holds it fine; only the export omits it.
Everything below reads through ``Client.list_examples``, which returns the server's own record.

**LangSmith injects ``dataset_split`` into every example's metadata.** It is server-side
bookkeeping, not ours, and a naive equality check flags all 55 as drifted on a key no local file
ever wrote. ``SERVER_INJECTED`` is the single place that knowledge lives.

``DESCRIPTIONS`` is declared here rather than only on the server, and it is not decoration: the
trajectory one tells a grader that ``expected_trajectory`` is a *reference* path and that exact
sequence matching fails legitimately-correct runs. Held only in LangSmith, that guidance dies
the first time a dataset is deleted and recreated. Note the asymmetry it has to live with --
``Client`` exposes no ``update_dataset``, so a description can be *set* at creation and only
*reported* thereafter.

Run it with ``uv run python evals/sync_datasets.py`` (it imports the package, so a bare
``python`` will not resolve ``speechwriter.config``). Unlike ``validate_datasets.py`` this one
needs ``LANGSMITH_API_KEY`` and the network, which is why it is not a pytest gate: the suite's
premise is that it runs offline and free. The pure diff below is tested; the wire is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from speechwriter.config import load_settings

EV = Path(__file__).resolve().parent / "datasets"

# The remote name is derived, never typed twice: a dataset renamed in the UI reads here as
# "absent", which --push then recreates rather than silently forking a second copy.
NAME_PREFIX = "Speechwriter: "

# Metadata keys LangSmith writes itself. Stripped from the remote side before comparing, and
# *rejected* on the local side -- a local file that declared one would be asserting ownership of
# a field the server overwrites, which is a schema decision to make deliberately, not to absorb.
SERVER_INJECTED = frozenset({"dataset_split"})

# The roster, and the descriptions that are only otherwise on the server. Must agree with
# validate_datasets.EXPECTED_COUNTS, which cannot be imported from (that module runs its whole
# check at import and calls sys.exit), so test_sync_roster_matches_the_validator ties them.
DESCRIPTIONS: dict[str, str] = {
    "final_response": (
        "Speechwriter agent, end-to-end. Brief in, finished speech out. Grades word count "
        "against the 130 wpm contract in config.WORDS_PER_MINUTE, must-mention/must-not-contain "
        "items, required behaviours, and the /workspace/speeches/ save path. 20s-25min length "
        "range."
    ),
    "trajectory": (
        "Speechwriter agent execution path. Grades tool usage against the bound deepagents set, "
        "researcher/style-critic delegation, write-sandbox paths, and ordering. "
        "expected_trajectory is a REFERENCE path -- score with "
        "required_tools/required_subagents/order_constraints and read outputs.tolerance_notes; "
        "exact sequence matching fails legitimately-correct runs."
    ),
    "single_step": (
        "Speechwriter agent single decision points: intake discipline (ask vs proceed, both can "
        "be correct -- see expected_decision 'either_acceptable') and memory-write discipline "
        "(only durable speaker-level facts belong in /memories/)."
    ),
    "rag": (
        "Speechwriter researcher subagent. Grades sourcing discipline, not answers: provenance "
        "per claim, distinct domains, the /workspace/research/ save path, 5-10 facts + 2-3 "
        "angles. No expected answers, URLs or facts are asserted anywhere -- every criterion is "
        "a property of a good research brief."
    ),
}

EXIT_OK = 0
EXIT_DRIFT = 1
# Distinct from EXIT_DRIFT on purpose, and never 0. A missing key or an unreachable API must not
# read as "in sync" -- the same reasoning CLAUDE.md gives for the hooks exiting 1 rather than 0
# when tooling is absent: a check that has quietly stopped running looks exactly like one that
# is passing.
EXIT_UNAVAILABLE = 2


# --- pure: no client, no network, so the diff is testable offline like the rest of the suite ---


def remote_name(stem: str) -> str:
    return f"{NAME_PREFIX}{stem}"


def canon(value: object) -> str:
    """Order-insensitive rendering, so a reshuffled JSON key never reads as drift."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def strip_injected(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (metadata or {}).items() if k not in SERVER_INJECTED}


def load_local(stem: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Read one dataset file, rejecting anything the diff below could not key on.

    ``path`` overrides the location so the rejection branches are reachable from a test
    without planting a fifth file in ``evals/datasets/``, which the validator's fixed roster
    would then fail on.
    """
    raw = json.loads((path or EV / f"{stem}.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{stem}.json: top level is {type(raw).__name__}, not a list")
    checked: list[dict[str, Any]] = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(f"{stem}[{i}]: not an object")
        metadata = e.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{stem}[{i}]: metadata missing or not an object")
        eid = metadata.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError(f"{stem}[{i}]: metadata.id missing -- the mirror keys on it")
        clash = SERVER_INJECTED & set(metadata)
        if clash:
            raise ValueError(
                f"{stem} ({eid}): metadata declares {sorted(clash)}, which LangSmith owns and "
                f"overwrites. Decide the schema explicitly rather than shipping a field the "
                f"server will silently replace."
            )
        # Rebuilt rather than appended: `isinstance(e, dict)` narrows only to
        # dict[Unknown, Unknown], and dict is invariant in its key type. JSON object keys
        # are strings by definition, so this states that rather than widening to Any.
        checked.append({str(k): v for k, v in e.items()})
    return checked


def as_record(example: Any) -> dict[str, Any]:
    """Flatten a LangSmith ``Example`` into the plain shape ``diff_examples`` compares.

    Kept separate from the diff so the diff needs no SDK object and no wire.
    """
    return {
        "uuid": str(example.id),
        "inputs": example.inputs or {},
        "outputs": example.outputs or {},
        "metadata": strip_injected(example.metadata),
    }


def diff_examples(
    local: list[dict[str, Any]], remote: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Plan the mirror, keyed on ``metadata.id``.

    A remote example with no ``metadata.id``, or with one already claimed, is an orphan: added
    through the UI, or duplicated. A push-only mirror removes it, which is exactly the property
    that makes "the remote equals the repo" true rather than aspirational.
    """
    by_local = {e["metadata"]["id"]: e for e in local}
    by_remote: dict[str, dict[str, Any]] = {}
    orphans: list[dict[str, Any]] = []
    for r in remote:
        rid = r["metadata"].get("id")
        if not isinstance(rid, str) or not rid or rid in by_remote:
            orphans.append(r)
        else:
            by_remote[rid] = r

    create = [by_local[i] for i in sorted(set(by_local) - set(by_remote))]
    delete = orphans + [by_remote[i] for i in sorted(set(by_remote) - set(by_local))]
    update: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for i in sorted(set(by_local) & set(by_remote)):
        loc, rem = by_local[i], by_remote[i]
        fields = [f for f in ("inputs", "outputs", "metadata") if canon(loc[f]) != canon(rem[f])]
        entry = {"id": i, "uuid": rem["uuid"], "local": loc, "fields": fields}
        (update if fields else unchanged).append(entry)
    return {"create": create, "update": update, "delete": delete, "unchanged": unchanged}


def plan_is_clean(plan: dict[str, list[dict[str, Any]]]) -> bool:
    return not (plan["create"] or plan["update"] or plan["delete"])


def describe(stem: str, plan: dict[str, list[dict[str, Any]]]) -> list[str]:
    """One block of human-readable lines per dataset. Named ids, never just counts."""
    lines = [
        f"{stem:<16} unchanged={len(plan['unchanged']):>3} create={len(plan['create']):>3} "
        f"update={len(plan['update']):>3} delete={len(plan['delete']):>3}"
    ]
    for e in plan["create"]:
        lines.append(f"    + {e['metadata']['id']}")
    for e in plan["update"]:
        lines.append(f"    ~ {e['id']}  ({', '.join(e['fields'])})")
    for e in plan["delete"]:
        rid = e["metadata"].get("id") or "<no metadata.id -- UI-added or duplicate>"
        lines.append(f"    - {rid}")
    return lines


# --- the wire ---


def sync(push: bool, allow_delete: bool) -> int:
    from langsmith import Client

    client = Client()
    drifted, deletions_blocked = [], []

    for stem, description in DESCRIPTIONS.items():
        local = load_local(stem)
        name = remote_name(stem)

        if not client.has_dataset(dataset_name=name):
            if not push:
                print(f"{stem:<16} ABSENT on LangSmith ({len(local)} local examples)")
                drifted.append(stem)
                continue
            # Creation is the only moment a description can be set: Client has no
            # update_dataset, so getting it wrong here is a delete-and-recreate to fix.
            client.create_dataset(dataset_name=name, description=description)
            print(f"{stem:<16} created dataset {name!r}")

        dataset = client.read_dataset(dataset_name=name)
        remote = [as_record(e) for e in client.list_examples(dataset_id=dataset.id)]
        plan = diff_examples(local, remote)
        print("\n".join(describe(stem, plan)))

        if (dataset.description or "") != description:
            # Reported, never repaired: no update_dataset to call. Loud rather than silent,
            # because the trajectory description carries grading semantics a scorer needs.
            print(f"    ! description differs from DESCRIPTIONS[{stem!r}] -- edit it in the UI")
            drifted.append(stem)

        if plan_is_clean(plan):
            continue
        if not push:
            drifted.append(stem)
            continue

        if plan["create"]:
            client.create_examples(
                dataset_id=dataset.id,
                examples=[
                    {"inputs": e["inputs"], "outputs": e["outputs"], "metadata": e["metadata"]}
                    for e in plan["create"]
                ],
            )
        if plan["update"]:
            client.update_examples(
                dataset_id=dataset.id,
                updates=[
                    {
                        "id": e["uuid"],
                        "inputs": e["local"]["inputs"],
                        "outputs": e["local"]["outputs"],
                        "metadata": e["local"]["metadata"],
                    }
                    for e in plan["update"]
                ],
            )
        if plan["delete"]:
            # Irreversible, and the plausible cause is a convention change that unkeyed every
            # example at once -- so deleting is opt-in even inside --push.
            if not allow_delete:
                deletions_blocked.append(stem)
                print(f"    ! {len(plan['delete'])} deletion(s) skipped; pass --allow-delete")
            else:
                for e in plan["delete"]:
                    client.delete_example(e["uuid"])
        print(f"{stem:<16} pushed")

    if deletions_blocked:
        print(f"\nDELETIONS SKIPPED in: {', '.join(deletions_blocked)} (re-run --allow-delete)")
        return EXIT_DRIFT
    if push:
        print("\nPUSHED. Re-run without --push to confirm the mirror is clean.")
        return EXIT_OK
    if drifted:
        print(f"\nDRIFT in: {', '.join(sorted(set(drifted)))}. Re-run with --push to fix.")
        return EXIT_DRIFT
    print(
        f"\nIN SYNC: {sum(len(load_local(s)) for s in DESCRIPTIONS)} examples across "
        f"{len(DESCRIPTIONS)} datasets match LangSmith exactly."
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--push",
        action="store_true",
        help="make LangSmith match local (default is a read-only check)",
    )
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="with --push, also remove remote examples local no longer has",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    del settings  # loaded for its dotenv side effect: LANGSMITH_API_KEY reaches os.environ

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set -- cannot check the mirror.", file=sys.stderr)
        return EXIT_UNAVAILABLE

    try:
        return sync(push=args.push, allow_delete=args.allow_delete)
    except Exception as exc:  # noqa: BLE001 -- any wire failure is "unknown", not "in sync"
        print(f"sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
