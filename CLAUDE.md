# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A speechwriter agent built on **Deep Agents** (`deepagents` on LangChain/LangGraph). The project mostly *configures* the harness rather than implementing agent machinery: planning (`write_todos`), filesystem tools, subagent delegation, and skills all come from `create_deep_agent`. The code supplies the model, the prompts, the backend routing, the permission sandbox, and durable memory.

## Commands

```bash
uv sync                                  # install into .venv from uv.lock
uv run speechwriter                      # interactive REPL (also: uv run python -m speechwriter)
uv run pytest                            # full suite — offline, no API key or network
uv run pytest tests/test_build.py::test_write_sandbox_confines_writes   # single test
uvx ruff check . && uvx ruff format .    # lint + format (line-length 100)
uvx ty check                             # type check
```

**When working with Python here, invoke the relevant Astral skill first** — `/astral:uv` for dependencies and environments, `/astral:ruff` for lint and format, `/astral:ty` for type checking — so the current best practices are followed rather than guessed at. They also encode the right invocation form: `uv run` for anything that must import the project's dependencies (e.g. `pytest`), `uvx` for standalone tools (`ruff`, `ty`).

All three gates are **clean**: `uv run pytest` (11 passed), `uvx ruff check .`, and `uvx ty check` (0 diagnostics). Keep them that way. Prefer typing something precisely over widening it to `Any`; if a suppression is genuinely unavoidable, use a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.

## Architecture

Everything composes in one place: `build_agent()` in `src/speechwriter/agent.py`. Read that file first; the rest of the package feeds it.

### The virtual filesystem is the central design

The agent sees a `CompositeBackend` that routes by path prefix. Virtual paths are **not** real paths — `FilesystemBackend` is rooted at `project_root` with `virtual_mode=True`.

```
/skills/     → FilesystemBackend   read-only reference (the rhetoric library)
/workspace/  → FilesystemBackend   real .md files on disk (drafts, research notes)
/memories/   → StoreBackend        intercepted before disk; never a real folder
```

### Paths are single-sourced in `config.Settings` — and three subsystems must agree

`memories_vpath` is a `ClassVar` (`/memories/`); `skills_vpath` and `workspace_vpath` are properties derived from real dirs via `Settings._vpath()`. Three independent consumers depend on them agreeing, and nothing enforces the agreement:

1. **Backend routes** — `agent.py:backend()` routes `memories_vpath` to the Store.
2. **Sandbox rules** — `agent.py:_write_sandbox()` allows writes only under workspace + memories.
3. **Prompt text** — `prompts.py` interpolates all three paths into the system prompts.

Change a path and you must propagate it through all three, or the agent will be *instructed* to write somewhere the sandbox *denies*.

### The write sandbox is enforced, not prompted

`_write_sandbox()` returns first-match-wins `FilesystemPermission` rules: allow `write` under workspace + memories, then deny `write` on `/**` as the backstop. Reads are left open so skills and reference material still load. The same rules are applied **twice** — to `create_deep_agent(permissions=...)` *and* to every subagent — because subagents run their own filesystem middleware and inherit nothing.

### Subagents inherit nothing

`build_subagents()` must hand each subagent everything it needs explicitly:
- **Skills** — the `style-critic` gets `"skills": [settings.skills_vpath]` because subagent skill sets are not inherited.
- **Permissions** — passed in from `build_agent` as the `permissions=` argument.

Subagents are also stateless across `task` calls; the orchestrator prompt says so, and any new subagent must be given complete self-contained instructions per call.

Because "inherits nothing" makes every key load-bearing, `build_subagents()` returns `list[SubAgent]` — deepagents' `TypedDict`, not `dict[str, Any]`. That is deliberate: a typo like `"skill":` for `"skills":` would *not* fail at runtime, the `style-critic` would just silently lose the rhetoric library. Typed, `ty` rejects the unknown key. Keep the precise type when adding a subagent.

### Research is capability-gated, and it changes the agent's shape

One env var flips two coupled behaviors. Without `TAVILY_API_KEY`, `build_research_tool()` returns `None`, so:
- the `researcher` subagent is **absent from the subagent list entirely**, and
- `orchestrator_prompt()` swaps in a variant instructing the agent to flag unverified claims with `[VERIFY]`.

`TavilySearch` validates its key at *construction* time, which is why the tool is built conditionally and imported lazily. Both branches have tests.

### Memory: JSON snapshot, and persistence is the bundle's job

`StoreBackend` gives cross-thread persistence, but the only local `Store` is `InMemoryStore`, which dies with the process. `memory.py` snapshots it to `.speechwriter/memory-store.json` and rehydrates on startup. Swap `PostgresStore` in here to make it multi-user.

Durability is owned by `SpeechwriterAgent.persist()`, **not** the CLI — the CLI just calls it in a `finally`. Library consumers must call `bundle.persist()` themselves or learned voice profiles are lost.

Two correctness rules in `memory.py`, both with regression tests — preserve them:
- **Exhaust pagination.** `Store.search` and `list_namespaces` default to limits of 10 and 100 and silently truncate. `_paginate()` is the single place this invariant lives.
- **Never clobber.** An unreadable or wrong-shaped snapshot is renamed `*.corrupt` before starting empty, so a later save can't overwrite recoverable data.

## Invariants to preserve

- **Building the agent must not call the model or the network.** This is what makes the entire test suite free and offline. Anything that would make `build_agent()` hit the wire belongs behind a lazy path.
- **`import speechwriter` must stay lazy.** `__init__.py` exposes `build_agent` via module `__getattr__` so the heavy `deepagents`/`langchain` stack isn't imported eagerly. `test_import_speechwriter_is_lazy` spawns a subprocess to assert this — don't add a top-level import of `agent.py` to `__init__.py`.
- **`load_dotenv` targets `project_root / ".env"` explicitly**, never an upward walk (an ancestor `.env` could leak unrelated keys), and it is called inside `load_settings()` so `import speechwriter` has no side effects. Real shell env wins over `.env`.
- **The Store namespace is explicit** (`_memory_namespace` → `("speechwriter", "memories")`). deepagents' implicit-namespace mode is deprecated and removed in 0.7; the dependency is pinned `>=0.6.12,<0.8`.

## Skills

Each `skills/<slug>/SKILL.md` is loaded on demand by the agent (progressive disclosure — the description tells the agent when to read the body). `test_all_skills_have_valid_frontmatter` enforces the contract:

- YAML frontmatter with `name` **matching the directory slug** and a non-empty `description`.
- Body sections: `## Overview`, `## When to Use`, `## Instructions`, `## Pitfalls`.
- The test hard-codes `len(skill_dirs) == 4` — **adding or removing a skill requires updating that assertion.**

## Gotchas

- `tests/test_build.py` imports two things that aren't declared dependencies or public API: `yaml` (pyyaml arrives transitively via langchain) and `deepagents.middleware.filesystem._check_fs_permission` (private). Either can break on a dependency bump; the sandbox test is the likely casualty.
- `workspace/` and `.speechwriter/` are gitignored runtime output — `load_settings()` creates them on startup, so a missing folder never fails the first draft.
- The CLI rotates `thread_id` after a `KeyboardInterrupt` so it never resumes a half-executed graph; that intentionally drops prior conversation context.
- The orchestrator is given **no direct tools** (`tools=[]`). Research is delegated so noisy search results never crowd the writing context. Add new capabilities as subagents unless the orchestrator genuinely needs them inline.
