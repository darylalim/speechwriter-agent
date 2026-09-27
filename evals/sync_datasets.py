"""Mirror ``evals/datasets/*.json`` into Phoenix, or verify that the mirror is intact.

``--check`` (the default) answers whether the four remote datasets match the repo and writes
nothing; ``--push`` makes them match. Local is always the source of truth -- this is a
*push-only* mirror, so an edit made in the Phoenix UI is overwritten rather than merged back.

The Phoenix is the one tracing already names: ``PHOENIX_COLLECTOR_ENDPOINT`` (and
``PHOENIX_API_KEY`` for a Phoenix with auth on), read through ``load_settings()``. One variable
answers "where is Phoenix" for traces, datasets and experiments alike, so they cannot drift
onto two different servers.

Three properties of Phoenix's dataset API shape the code, each measured against a running
server rather than read off the docs:

**A push is one upload, and the upload *is* the dataset.** ``create_dataset`` on an existing
name writes a new version containing exactly the examples sent -- edited ones updated in place,
new ones added, and any left out dropped. An identical upload writes no version at all. So the
plan below is a *report*, not a list of calls to replay: whatever it says, the push is the
whole local file, sent once.

**Nothing is ever deleted.** A dropped example is gone from the latest version and still in the
previous one (``get_dataset(version_id=...)``). The LangSmith mirror gated deletions behind
``--allow-delete`` because a LangSmith delete was irreversible; here the reason is gone, so the
flag went with it.

**The example id is ours.** Each upload carries ``metadata.id`` as the example's own ``id``, and
Phoenix keeps it -- which is what lets the diff key on the id the server reports and predict
what the upload will do. An upload without it would get server-generated ids, and every later
check would read as 55 deletions plus 55 creations. :func:`upload_payload` is the one place the
two are tied, and ``test_a_pushed_dataset_reads_back_clean`` holds it.

``DESCRIPTIONS`` is declared here rather than only on the server, and it is not decoration: the
trajectory one tells a grader that ``expected_trajectory`` is a *reference* path and that exact
sequence matching fails legitimately-correct runs. Phoenix reads a description only when it
*creates* a dataset -- a later upload ignores it (measured), and the REST API has no route to
change one -- so a description can be set once and only *reported* thereafter.

Run it with ``uv run python evals/sync_datasets.py`` (it imports the package, so a bare
``python`` will not resolve ``speechwriter.config``). Unlike ``validate_datasets.py`` this one
needs a running Phoenix, which is why it is not a pytest gate: the suite's premise is that it
runs offline and free. The pure diff below is tested; the wire is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from speechwriter.config import Settings, load_settings
from speechwriter.tracing import server_url

if TYPE_CHECKING:
    from phoenix.client import Client

EV = Path(__file__).resolve().parent / "datasets"

# The remote name is derived, never typed twice: a dataset renamed in the UI reads here as
# "absent", which --push then recreates rather than silently forking a second copy.
NAME_PREFIX = "Speechwriter: "

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
# Distinct from EXIT_DRIFT on purpose, and never 0. An unset endpoint or an unreachable server
# must not read as "in sync" -- the same reasoning CLAUDE.md gives for the hooks exiting 1 rather
# than 0 when tooling is absent: a check that has quietly stopped running looks exactly like one
# that is passing.
EXIT_UNAVAILABLE = 2


# --- pure: no client, no network, so the diff is testable offline like the rest of the suite ---


def remote_name(stem: str) -> str:
    return f"{NAME_PREFIX}{stem}"


def _as_phoenix_hashes(value: object) -> object:
    """Numbers as Phoenix compares them: an integral float is the integer it equals.

    The server decides "unchanged" by hashing each example's canonical JSON (RFC 8785), which
    writes ``12.0`` as ``12`` -- so an edit from one to the other is, to Phoenix, no edit at
    all, and the upload writes no version. Compared as Python renders them they differ, which
    left a check that reported drift forever and a push that could never clear it (measured).
    ``-0.0`` folds to ``0`` the same way.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _as_phoenix_hashes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_phoenix_hashes(v) for v in value]
    return value


def canon(value: object) -> str:
    """Order-insensitive rendering, so a reshuffled JSON key never reads as drift -- and
    number-insensitive the way the server is, so a respelled number never does either."""
    return json.dumps(_as_phoenix_hashes(value), sort_keys=True, ensure_ascii=False)


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
            # The id becomes the Phoenix example id. A push would fail loudly (Phoenix rejects
            # an upload repeating one); the *check* would not -- the diff keys local examples by
            # id, so one of the pair drops out of the plan and "IN SYNC" can be printed over a
            # file holding an example the server has never seen.
            raise ValueError(f"{stem}[{i}]: metadata.id {eid!r} is used twice")
        seen.add(eid)
        # Rebuilt rather than appended: `isinstance(e, dict)` narrows only to
        # dict[Unknown, Unknown], and dict is invariant in its key type. JSON object keys
        # are strings by definition, so this states that rather than widening to Any.
        checked.append({str(k): v for k, v in e.items()})
    return checked


def upload_payload(local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Local examples in the shape Phoenix's upload takes, ``metadata.id`` as the example id.

    Two renames, both Phoenix's spelling rather than a choice: the datasets say ``inputs`` and
    ``outputs`` (LangSmith's words, and the validator's), Phoenix says ``input`` and ``output``.
    """
    return [
        {
            "id": e["metadata"]["id"],
            "input": e["inputs"],
            "output": e["outputs"],
            "metadata": e["metadata"],
        }
        for e in local
    ]


def as_record(example: Any) -> dict[str, Any]:
    """Flatten a Phoenix example into the plain shape ``diff_examples`` compares.

    Kept separate from the diff so the diff needs no SDK object and no wire. Keyed on the
    example's own ``id``, not ``metadata.id``: the id is what an upload matches on, so it is
    what predicts what a push will do. A metadata id edited in the UI is then an ordinary
    ``metadata`` drift on the example it belongs to, which the push repairs.
    """
    return {
        "id": str(example["id"]),
        "inputs": example.get("input") or {},
        "outputs": example.get("output") or {},
        "metadata": example.get("metadata") or {},
    }


def diff_examples(
    local: list[dict[str, Any]], remote: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Plan the mirror, keyed on the example id.

    A remote example with an id local does not have -- added through the UI, or removed from
    the repo -- is a deletion: the push drops it from the latest version, which is exactly the
    property that makes "the remote equals the repo" true rather than aspirational.
    """
    by_local = {e["metadata"]["id"]: e for e in local}
    by_remote = {r["id"]: r for r in remote}

    create = [by_local[i] for i in sorted(set(by_local) - set(by_remote))]
    delete = [by_remote[i] for i in sorted(set(by_remote) - set(by_local))]
    update: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for i in sorted(set(by_local) & set(by_remote)):
        loc, rem = by_local[i], by_remote[i]
        fields = [f for f in ("inputs", "outputs", "metadata") if canon(loc[f]) != canon(rem[f])]
        entry = {"id": i, "local": loc, "fields": fields}
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
        lines.append(f"    - {e['id']}")
    return lines


# --- the wire ---


def connect(settings: Settings) -> Client | None:
    """A Phoenix client for the server ``PHOENIX_COLLECTOR_ENDPOINT`` names, or ``None``.

    Built only from :class:`~speechwriter.config.Settings`, by handing the client a transport
    of its own. Given none, it builds one from *its* environment: it merges any
    ``PHOENIX_CLIENT_HEADERS`` found in the shell into every request -- and into the experiment
    tracer's span exports -- and walks *up* from the working directory for a ``.env.phoenix`` to
    take headers and a key from, the upward walk ``load_settings()`` refuses for the project's
    own dotenv. A Phoenix Cloud key exported for another project would then go to this server
    on every call. Given a transport, it reads nothing: the bearer below is the only credential
    that leaves, and only for the server the operator named -- the rule ``tracing.py`` follows.
    """
    if settings.phoenix_endpoint is None:
        return None
    base_url = server_url(settings.phoenix_endpoint)
    if base_url is None:
        return None
    import httpx
    from phoenix.client import Client

    headers = (
        {"Authorization": f"Bearer {settings.phoenix_api_key}"} if settings.phoenix_api_key else {}
    )
    # The client's own defaults, restated because a supplied transport replaces them.
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    return Client(http_client=httpx.Client(base_url=base_url, headers=headers, timeout=timeout))


def remote_names(client: Client) -> set[str]:
    """Every dataset name on the server. ``get_dataset`` answers a missing name with a bare
    ``ValueError``, indistinguishable from any other, so existence is asked this way instead."""
    return {d["name"] for d in client.datasets.list()}


def sync(client: Client, push: bool) -> int:
    drifted: list[str] = []
    # What a push cannot repair. Kept apart from `drifted` because it decides --push's exit
    # code: "PUSHED" and 0 over a dataset still out of step would send the reader round a loop.
    unrepaired: list[str] = []
    present = remote_names(client)

    for stem, description in DESCRIPTIONS.items():
        local = load_local(stem)
        name = remote_name(stem)

        if name not in present:
            print(f"{stem:<16} ABSENT on Phoenix ({len(local)} local examples)")
            if not push:
                drifted.append(stem)
                continue
            # Creation is the only moment a description can be set: a later upload ignores it,
            # so getting it wrong here is a delete-and-recreate to fix.
            client.datasets.create_dataset(
                name=name, examples=upload_payload(local), dataset_description=description
            )
            print(f"{stem:<16} created dataset {name!r}")
            continue

        dataset = client.datasets.get_dataset(dataset=name)
        plan = diff_examples(local, [as_record(e) for e in dataset.examples])
        print("\n".join(describe(stem, plan)))

        if (dataset.description or "") != description:
            # Reported, never repaired: nothing on the wire can change it. Loud rather than
            # silent, because the trajectory description carries grading semantics a scorer needs.
            print(f"    ! description differs from DESCRIPTIONS[{stem!r}] -- edit it in the UI")
            drifted.append(stem)
            unrepaired.append(stem)

        if plan_is_clean(plan):
            continue
        if not push:
            drifted.append(stem)
            continue

        # The whole file, once: the upload replaces the example set, so this one call is the
        # create, the update and the delete together, written as one new version.
        updated = client.datasets.create_dataset(name=name, examples=upload_payload(local))
        if updated.version_id == dataset.version_id:
            # The server judged the upload identical, so the drift reported above is one this
            # checker sees and Phoenix does not -- a backstop behind canon(), for whatever
            # difference it has not learned to ignore yet. A push cannot clear it.
            print(
                "    ! Phoenix wrote no new version -- it sees this file as unchanged. The "
                "check and the server disagree about what counts as a change; see canon()."
            )
            unrepaired.append(stem)
            continue
        print(f"{stem:<16} pushed as version {updated.version_id}")

    if push and unrepaired:
        print(
            f"\nPUSHED, but a push cannot repair: {', '.join(sorted(set(unrepaired)))}. "
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
        f"{len(DESCRIPTIONS)} datasets match Phoenix exactly."
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--push",
        action="store_true",
        help="make Phoenix match local (default is a read-only check)",
    )
    args = parser.parse_args(argv)

    # Everything inside the `try`, connecting included: the client is a *dev* dependency, and an
    # ImportError here would otherwise exit 1 -- which this script reserves for "drift".
    try:
        client = connect(load_settings())
        if client is None:
            print(
                "PHOENIX_COLLECTOR_ENDPOINT is not set to an http(s) URL -- cannot check the "
                "mirror. Start Phoenix (`uvx --from arize-phoenix phoenix serve`) and set it to "
                "http://localhost:6006.",
                file=sys.stderr,
            )
            return EXIT_UNAVAILABLE
        return sync(client, push=args.push)
    except Exception as exc:  # noqa: BLE001 -- any wire failure is "unknown", not "in sync"
        print(f"sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
