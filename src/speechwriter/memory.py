"""Durable long-term memory for speaker voice profiles.

The Deep Agents ``StoreBackend`` gives us cross-*thread* persistence, but the only
``Store`` implementation installed locally is ``InMemoryStore`` — which evaporates
when the process exits. That would silently defeat the whole point of "remember how
this speaker sounds across sessions".

So we make the in-memory Store *durable* the simple way: snapshot every item to a
JSON file on save, and rehydrate it on startup. No database required. Swap in
``PostgresStore`` here if you ever deploy this as a service.

Two correctness rules this module takes seriously, because the whole feature is
about *not losing data*:

* **Exhaust pagination.** ``Store.search`` and ``Store.list_namespaces`` default to
  small limits (10 and 100) and silently truncate. We always loop with an explicit
  page size until a short page, so every item is captured.
* **Never clobber.** If an existing snapshot can't be read or parsed, we move it
  aside (``.corrupt``) before starting empty, so a subsequent save can't overwrite
  recoverable data — and we never crash on a malformed file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from langgraph.store.base import Item
from langgraph.store.memory import InMemoryStore

from speechwriter.config import Settings

logger = logging.getLogger(__name__)

# Page size for exhausting the Store's paginated APIs. Large enough that most
# real workloads finish in one page, but we still loop to be correct.
_PAGE = 1000
_REQUIRED_FIELDS = frozenset({"namespace", "key", "value"})


def load_store(settings: Settings) -> InMemoryStore:
    """Create an ``InMemoryStore`` and rehydrate it from the on-disk snapshot.

    Guarantees: never raises on a malformed/unreadable snapshot, and never lets a
    later ``save_store`` silently discard data. An unusable existing file is
    quarantined (renamed to ``*.corrupt``) and startup proceeds with an empty store.
    """
    store = InMemoryStore()
    path = settings.store_path
    if not path.exists():
        return store

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("snapshot root must be a JSON array")
        for rec in records:
            if not isinstance(rec, dict) or not _REQUIRED_FIELDS <= rec.keys():
                raise ValueError("snapshot record missing namespace/key/value")
            store.put(tuple(rec["namespace"]), rec["key"], rec["value"])
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        _quarantine(path, exc)
        # Discard any partial puts so we never present a half-loaded store.
        return InMemoryStore()
    return store


def save_store(store: InMemoryStore, settings: Settings) -> int:
    """Snapshot every item in the Store to the JSON file. Returns the item count.

    Walks every namespace and every item via exhaustive pagination, dedupes by
    ``(namespace, key)``, and writes atomically so a crash mid-write can't corrupt
    the snapshot.
    """
    seen: dict[tuple[tuple[str, ...], str], object] = {}
    for namespace in _all_namespaces(store):
        for item in _all_items(store, namespace):
            seen[(tuple(item.namespace), item.key)] = item.value

    records = [
        {"namespace": list(ns), "key": key, "value": value} for (ns, key), value in seen.items()
    ]

    settings.store_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(settings.store_path, records)
    return len(records)


def _all_namespaces(store: InMemoryStore) -> list[tuple[str, ...]]:
    """Every namespace, paging past the default list_namespaces limit of 100."""
    out: list[tuple[str, ...]] = []
    offset = 0
    while True:
        page = store.list_namespaces(limit=_PAGE, offset=offset)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        if len(page) < _PAGE:
            break
    return out


def _all_items(store: InMemoryStore, namespace: tuple[str, ...]) -> list[Item]:
    """Every item in a namespace, paging past the default search limit of 10."""
    out: list[Item] = []
    offset = 0
    while True:
        page = store.search(namespace, limit=_PAGE, offset=offset)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        if len(page) < _PAGE:
            break
    return out


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON to a temp file and atomically replace, so a partial write never wins."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _quarantine(path: Path, exc: Exception) -> None:
    """Move an unreadable/corrupt snapshot aside so it is never overwritten.

    If we cannot even move it aside, re-raise: failing loudly is safer than
    silently returning an empty store that a later save would clobber onto real data.
    """
    backup = path.with_name(path.name + ".corrupt")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.corrupt.{counter}")
        counter += 1
    try:
        os.replace(path, backup)
    except OSError:
        logger.error("Could not read or quarantine memory snapshot %s (%s)", path, exc)
        raise
    logger.warning("Unreadable memory snapshot %s (%s); moved to %s", path, exc, backup)
