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
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from speechwriter.config import Settings

logger = logging.getLogger(__name__)

# Page size for exhausting the Store's paginated APIs. Large enough that most
# real workloads finish in one page, but we still loop to be correct.
_PAGE = 1000
_REQUIRED_FIELDS = frozenset({"namespace", "key", "value"})
_T = TypeVar("_T")


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


def save_store(store: BaseStore, settings: Settings) -> int:
    """Snapshot every item in the Store to the JSON file. Returns the item count.

    Walks every namespace and every item via exhaustive pagination, dedupes by
    ``(namespace, key)``, and writes atomically so a crash mid-write can't corrupt
    the snapshot.
    """
    seen: dict[tuple[tuple[str, ...], str], object] = {}
    for item in all_items(store):
        seen[(tuple(item.namespace), item.key)] = item.value

    records = [
        {"namespace": list(ns), "key": key, "value": value} for (ns, key), value in seen.items()
    ]

    settings.store_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(settings.store_path, records)
    return len(records)


def all_items(store: BaseStore) -> list[Any]:
    """Every item in every namespace, with pagination exhausted.

    Public because snapshotting is no longer the only exhaustive read: the web UI lists
    learned voice profiles straight from the live Store. That read has to honor the same
    invariant ``save_store`` does — a second, hand-rolled walk would quietly stop at the
    Store's default ``search`` limit of 10 and show a partial memory as if it were whole.
    """
    items: list[Any] = []
    for namespace in _all_namespaces(store):
        items.extend(_all_items(store, namespace))
    return items


def _paginate(fetch: Callable[[int, int], list[_T]]) -> list[_T]:
    """Exhaust a paginated Store API, fetching (limit, offset) pages until a short one.

    Both ``search`` and ``list_namespaces`` default to small limits that silently
    truncate; this is the single place that exhaustive-read invariant is enforced.
    """
    out: list[_T] = []
    offset = 0
    while True:
        page = fetch(_PAGE, offset)
        out.extend(page)
        offset += len(page)
        if len(page) < _PAGE:
            return out


def _all_namespaces(store: BaseStore) -> list[tuple[str, ...]]:
    """Every namespace, paging past the default list_namespaces limit of 100."""
    return _paginate(lambda limit, offset: store.list_namespaces(limit=limit, offset=offset))


def _all_items(store: BaseStore, namespace: tuple[str, ...]) -> list[Any]:
    """Every item in a namespace, paging past the default search limit of 10."""
    return _paginate(lambda limit, offset: store.search(namespace, limit=limit, offset=offset))


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
