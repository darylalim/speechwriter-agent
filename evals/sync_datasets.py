"""Mirror ``evals/datasets/*.json`` into LangSmith, or verify that the mirror is intact.

``--check`` (the default) answers whether the four remote datasets match the repo and writes
nothing; ``--push`` makes them match. Local is always the source of truth -- this is a
*push-only* mirror, so an edit made in the LangSmith UI is overwritten rather than merged back.

The workspace is the one tracing already uses: ``LANGSMITH_API_KEY`` (and ``LANGSMITH_ENDPOINT``
for a self-hosted LangSmith), read through ``load_settings()``. One set of variables answers
"where is LangSmith" for traces, datasets and experiments alike, so they cannot drift apart.

Three traps are encoded here because each cost a wrong answer the first time this mirror
existed, before the eval harness spent a while on Phoenix and came back:

**Never diff against ``langsmith dataset export``.** That CLI path drops ``metadata`` entirely,
so a comparison built on it reports all 55 examples as having lost the field that carries their
grading semantics -- ``path_semantics``, ``max_clarifying_questions``,
``grade_tool_calls_after_message_index``. The server holds it fine; only the export omits it.
Everything below reads through ``Client.list_examples``, which returns the server's own record.

**LangSmith injects ``dataset_split`` into every example's metadata.** It is server-side
bookkeeping, not ours, and a naive equality check flags all 55 as drifted on a key no local file
ever wrote. ``SERVER_INJECTED`` is the single place that knowledge lives.

**A delete is irreversible.** The plausible cause of a plan full of deletions is a convention
change that unkeyed every example at once, so deleting is opt-in even inside ``--push``:
``--allow-delete``.

``DESCRIPTIONS`` is declared here rather than only on the server, and it is not decoration: the
trajectory one tells a grader that ``expected_trajectory`` is a *reference* path and that exact
sequence matching fails legitimately-correct runs. ``Client`` exposes no ``update_dataset``, so a
description can be *set* at creation and only *reported* thereafter.

Run it with ``uv run python evals/sync_datasets.py`` (it imports the package, so a bare
``python`` will not resolve ``speechwriter.config``). Unlike ``validate_datasets.py`` this one
needs ``LANGSMITH_API_KEY`` and the network, which is why it is not a pytest gate: the suite's
premise is that it runs offline and free. The pure diff below is tested; the wire is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speechwriter.config import load_settings

if TYPE_CHECKING:
    from langsmith import Client

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
# Distinct from EXIT_DRIFT on purpose, and never 0. A missing key or an unreachable server must
# not read as "in sync" -- the same reasoning CLAUDE.md gives for the hooks exiting 1 rather
# than 0 when tooling is absent: a check that has quietly stopped running looks exactly like one
# that is passing.
EXIT_UNAVAILABLE = 2


# --- pure: no client, no network, so the diff is testable offline like the rest of the suite ---


def remote_name(stem: str) -> str:
    return f"{NAME_PREFIX}{stem}"


def _integral_floats_as_ints(value: object) -> object:
    """Numbers as a JSON round trip may respell them: an integral float is the integer it equals.

    A server that stores JSON may hand ``12.0`` back as ``12`` (or the reverse), and compared as
    Python renders them the two differ — a check that reported drift forever and a push that
    could never clear it. Learned against Phoenix, whose RFC 8785 hashing does exactly this, and
    kept because nothing about the fold is Phoenix-specific: the two spellings are one number.
    ``-0.0`` folds to ``0`` the same way.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _integral_floats_as_ints(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_integral_floats_as_ints(v) for v in value]
    return value


def canon(value: object) -> str:
    """Order-insensitive rendering, so a reshuffled JSON key never reads as drift -- and
    number-insensitive, so a respelled number never does either."""
    return json.dumps(_integral_floats_as_ints(value), sort_keys=True, ensure_ascii=False)


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
    seen: set[str] = set()
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(f"{stem}[{i}]: not an object")
        metadata = e.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{stem}[{i}]: metadata missing or not an object")
        eid = metadata.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError(f"{stem}[{i}]: metadata.id missing -- the mirror keys on it")
        if eid in seen:
            # The diff keys local examples by id, so one of the pair would drop out of the plan
            # and "IN SYNC" could be printed over a file holding an example the server has never
            # seen.
            raise ValueError(f"{stem}[{i}]: metadata.id {eid!r} is used twice")
        seen.add(eid)
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

    LangSmith assigns its own example UUIDs, so ``metadata.id`` is the only key both sides
    share; the UUID is carried alongside so an update or delete can name the server's record.
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


def connect() -> Client | None:
    """A LangSmith client for the workspace ``LANGSMITH_API_KEY`` names, or ``None`` without one.

    Call after :func:`~speechwriter.config.load_settings`, which loads the dotenv and clears
    langsmith's env caches — the client reads its key and endpoint through them.
    """
    from langsmith import Client, utils

    if not utils.get_env_var("API_KEY"):
        return None
    return Client()


def sync(client: Client, push: bool, allow_delete: bool = False) -> int:
    drifted: list[str] = []
    # What a push cannot repair. Kept apart from `drifted` because it decides --push's exit
    # code: "PUSHED" and 0 over a dataset still out of step would send the reader round a loop.
    unrepaired: list[str] = []

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
            unrepaired.append(stem)

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
            if not allow_delete:
                unrepaired.append(stem)
                print(f"    ! {len(plan['delete'])} deletion(s) skipped; pass --allow-delete")
            else:
                for e in plan["delete"]:
                    client.delete_example(e["uuid"])
        print(f"{stem:<16} pushed")

    if push and unrepaired:
        print(
            f"\nPUSHED, but not in step: {', '.join(sorted(set(unrepaired)))}. "
            f"See the ! lines above."
        )
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
        help="with --push, also delete remote examples local does not have (irreversible)",
    )
    args = parser.parse_args(argv)
    if args.allow_delete and not args.push:
        parser.error("--allow-delete only means something with --push")

    try:
        load_settings()  # the dotenv, and langsmith's caches cleared, before the client reads
        client = connect()
        if client is None:
            print(
                "LANGSMITH_API_KEY is not set -- cannot check the mirror.",
                file=sys.stderr,
            )
            return EXIT_UNAVAILABLE
        return sync(client, push=args.push, allow_delete=args.allow_delete)
    except Exception as exc:  # noqa: BLE001 -- any wire failure is "unknown", not "in sync"
        print(f"sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
