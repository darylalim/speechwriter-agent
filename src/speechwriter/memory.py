"""Durable long-term memory for speaker voice profiles.

The Deep Agents ``StoreBackend`` gives us cross-*thread* persistence, but the only
``Store`` implementation installed locally is ``InMemoryStore`` — which evaporates
when the process exits. That would silently defeat the whole point of "remember how
this speaker sounds across sessions".

So we make the in-memory Store *durable* the simple way: snapshot every item to a
JSON file on save, and rehydrate it on startup. No database required. Swap in
``PostgresStore`` here if you ever deploy this as a service.
"""

from __future__ import annotations

import json

from langgraph.store.memory import InMemoryStore

from speechwriter.config import Settings


def load_store(settings: Settings) -> InMemoryStore:
    """Create an ``InMemoryStore`` and rehydrate it from the on-disk snapshot."""
    store = InMemoryStore()
    path = settings.store_path
    if not path.exists():
        return store

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt snapshot should never block startup — start fresh.
        return store

    for rec in records:
        store.put(tuple(rec["namespace"]), rec["key"], rec["value"])
    return store


def save_store(store: InMemoryStore, settings: Settings) -> int:
    """Snapshot every item in the Store to the JSON file. Returns the item count.

    We walk every namespace that holds data and dedupe by ``(namespace, key)`` so
    prefix-overlapping searches can't double-count an item.
    """
    seen: dict[tuple[tuple[str, ...], str], object] = {}
    for namespace in store.list_namespaces():
        for item in store.search(namespace):
            seen[(tuple(item.namespace), item.key)] = item.value

    records = [
        {"namespace": list(ns), "key": key, "value": value} for (ns, key), value in seen.items()
    ]

    settings.store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.store_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(records)
