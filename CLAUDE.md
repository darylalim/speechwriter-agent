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
printf 'exit\n' | uv run speechwriter    # zero-cost smoke test: builds, banners, persists, exits — no model calls
printf 'BRIEF\nexit\n' | uv run speechwriter   # drive one real turn non-interactively (costs tokens)
```

**When working with Python here, invoke the relevant Astral skill first** — `/astral:uv` for dependencies and environments, `/astral:ruff` for lint and format, `/astral:ty` for type checking — so the current best practices are followed rather than guessed at. They also encode the right invocation form: `uv run` for anything that must import the project's dependencies (e.g. `pytest`), `uvx` for standalone tools (`ruff`, `ty`).

All three gates are **clean** and must stay that way: `uv run pytest`, `uvx ruff check .`, and `uvx ty check`. (No pass-count is quoted here on purpose — it went stale twice in one sitting, and `pytest` reports the number better than a doc can.) Prefer typing something precisely over widening it to `Any`; if a suppression is genuinely unavoidable, use a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.

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

1. **Backend routes** — the `CompositeBackend` built in `agent.py:build_agent()` routes `memories_vpath` to the Store.
2. **Sandbox rules** — `agent.py:_write_sandbox()` allows writes only under workspace + memories.
3. **Prompt text** — `prompts.py` interpolates all three paths into the system prompts.

Change a path and you must propagate it through all three, or the agent will be *instructed* to write somewhere the sandbox *denies*.

Likewise `SPEECHES_SUBDIR` / `RESEARCH_SUBDIR` / `WORDS_PER_MINUTE` are single-sourced in `config.py`: `prompts.py` *tells* the agent to file drafts under those folders at that pace, and `workspace.py` *reads* them back to list drafts and estimate spoken length. Re-hardcoding either side yields a browser that silently lists nothing — `test_prompt_points_the_agent_at_the_folder_the_browser_reads` guards the seam.

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

### There are two front ends over one bundle

`cli.py` (Rich REPL) and the Streamlit app (`streamlit_app.py` → `app_pages/write.py` + `app_pages/browse.py`, glued by `webui.py`) are both thin views over the same `SpeechwriterAgent`. This is *why* durability (`persist()`), observability (`turn_config()`), and the resolved-ceiling label (`ceiling_label`) live on the bundle, not in either UI — a fact asserted above for the first two; the web UI is the second consumer that makes it load-bearing rather than hypothetical.
- **Build once.** `webui.get_bundle()` is `@st.cache_resource`; Streamlit reruns the whole script per interaction, so rebuilding would mint a fresh `InMemoryStore` each time and silently drop every voice profile learned this session.
- **Persist per turn, not on exit.** A closed browser tab runs no teardown, so `write.py` calls `bundle.persist()` after each turn (the CLI's `finally` has no analog).
- **A turn is recorded as data (`webui.Turn`), then replayed.** Live render and replay share `_render_event`, so they cannot drift. `_new_events` dedupes on message id because `stream_mode="values"` replays the whole message list every step.
- **The message→event decode is single-sourced in `transcript.py`** (`Event`, `iter_events`, `clip`) — a streamlit-free module both `cli.py` and `webui.py` consume, so the two front ends agree on *what* a message means; each keeps only its own formatting. `Event.text` is raw; the renderer clips/escapes. Keep `transcript.py` free of any Streamlit/Rich import.
- **A cancelled turn rotates the thread.** `submit_mode="stop"` raises a `BaseException` past `run_turn`'s `except Exception`, so a stopped turn never records or rotates. `run_turn` flags `_PENDING` while streaming; `_rotate_if_interrupted()` (first line of the next `run_turn`) rotates `thread_id` if the flag survived — the web analog of the CLI's post-interrupt rotation, so a half-executed graph is never resumed.
- **The browse page reads through `webui.documents(dir)`, which is `@st.cache_data`-cached** keyed on each file's name+mtime — so the expensive read+parse re-runs only when the folder changes, not on every rerun. `workspace.py` stays UI-free (no Streamlit import); the caching lives in `webui.py`.
- The page is `app_pages/browse.py`, deliberately **not** `workspace.py`, so it never collides with the `speechwriter.workspace` module or the `workspace/` data dir.

Two reader gotchas the web UI exposed:
- **`memory.all_items(store)` is the public exhaustive read** — the web UI lists profiles from the live Store through it, not the JSON snapshot. A hand-rolled `store.search(...)` stops at the default limit of 10 and shows a partial memory as whole.
- **`workspace.py` strips `---` front matter before rendering or counting.** The agent fences its header block with `---`; in CommonMark a `---` line right after a paragraph makes it a setext H2, so raw `st.markdown` renders the header as one run-on heading and its words inflate the spoken-length estimate. Bracketed cues (`[pause]`) are dropped from the word count too — they are delivered, not spoken.

## Invariants to preserve

- **Building the agent must not call the model or the network.** This is what makes the entire test suite free and offline. Anything that would make `build_agent()` hit the wire belongs behind a lazy path.
- **The model's output ceiling is resolved in three tiers, never inherited blindly.** `build_agent` passes a *constructed* model (`agent.py:_build_model`), never a bare id string, because `init_chat_model` takes `max_tokens` from LangChain's model-profile table and silently falls back to **4096** for an id it cannot profile — and `claude-sonnet-5`, the default, is currently unprofiled while its recognised siblings get 64k–128k. Extended thinking bills against that same ceiling, so an unprofiled id lets a subagent spend its entire budget thinking and emit no text, which deepagents forwards as an *empty* `status="success"` tool result (it walks back for the last message with text and finds none). Resolution order: **explicit `SPEECHWRITER_MAX_TOKENS` → the model's own profile → `DEFAULT_MAX_TOKENS` (32k)**. Tier 2 is load-bearing in the other direction — a flat constant would *cap* Opus at 32k when its profile says 128k, which is the same mistake inverted. `settings.max_tokens` is only the override and is normally `None`; the resolved value lives on `SpeechwriterAgent.max_tokens`. `SPEECHWRITER_MAX_TOKENS` is validated `>= 1` at load time — a 0 or negative ceiling is accepted by `init_chat_model` and only fails at the first API call, far from the typo. Three signals guard this, all tested: `_build_model` warns on an unprofiled id; `TruncationWarner` (`observability.py`) counts responses that actually hit the ceiling, matching `stop_reason` *and* `finish_reason` across providers since `SPEECHWRITER_MODEL` is free-form; and the bundle owns that warner, handing it out via `SpeechwriterAgent.turn_config()`. **Attach observability through `turn_config()`, not by hand-building `{"configurable": {...}}`** — same reasoning as `persist()`: the CLI is one of two entry points, and the README documents the other. Don't revert to `model=settings.model`.
- **`import speechwriter` must stay lazy.** `__init__.py` exposes `build_agent` and `TruncationWarner` via module `__getattr__` so the heavy `deepagents`/`langchain` stack isn't imported eagerly. `test_import_speechwriter_is_lazy` spawns a subprocess to assert this — don't add a top-level import of `agent.py` to `__init__.py`.
- **`load_dotenv` targets `project_root / ".env"` explicitly**, never an upward walk (an ancestor `.env` could leak unrelated keys), and it is called inside `load_settings()` so `import speechwriter` has no side effects. Real shell env wins over `.env`.
- **The Store namespace is explicit** (`_memory_namespace` → `("speechwriter", "memories")`). deepagents' implicit-namespace mode is deprecated and removed in 0.7; the dependency is pinned `>=0.6.12,<0.8`.

## Automation (`.claude/`)

There is no CI, no pre-commit, and no git hooks here — the checked-in `.claude/` config is the only thing enforcing the gates and invariants above. Three hooks, wired in `.claude/settings.json`:

| Hook | Event | Behavior |
|---|---|---|
| `hooks/ruff-ty-gate.sh` | `PostToolUse` on `Edit\|Write` | Runs `uvx ruff format` + `ruff check --fix` on the edited `.py`, then re-checks with `ruff check` and `ty check`. **Exit 2** on remaining diagnostics. ~0.6s. |
| `hooks/invariant-hints.sh` | `PostToolUse` on `Edit\|Write` | **Advisory only.** On a `skills/` edit, compares the directory count against the hard-coded assertion in `tests/test_build.py` and reports drift. On a `config.py` edit, restates the path-agreement invariant. |
| `hooks/pytest-gate.sh` | `Stop` | Runs `uv run pytest` if the working tree is dirty under `src/ tests/ skills/ pyproject.toml uv.lock`. **Exit 2** on failure. |

Design rules to preserve if you touch these:

- **Blocking vs advisory is deliberate.** Lint and type failures are objectively wrong and mechanically fixable, so those hooks exit 2. "You added a skill" is a fork in the road, not an error — `invariant-hints.sh` only emits `additionalContext` on stdout and always exits 0. Blocking on advisory signal is how a hook gets deleted.
- **The Stop gate must stay cheap in the common case.** It short-circuits on `git status --porcelain` in ~20ms and only then pays the ~1.2s suite. `--untracked-files=all` is load-bearing: `git diff` sees only *tracked* files, so a brand-new `skills/<slug>/SKILL.md` — exactly what trips the count assertion — would slip past untested.
- **Path matching lives in the scripts, not in settings.json.** The hook `if:` field is real, but its patterns are working-directory-relative and the leading-slash form is underspecified; `if: "Edit(/skills/**)"` can silently match nothing. Filtering inside the script is explicit and testable — each script parses `tool_input.file_path` from stdin and can be exercised directly:

  ```bash
  echo '{"tool_input":{"file_path":"'$PWD'/src/speechwriter/config.py"}}' | .claude/hooks/invariant-hints.sh
  echo '{"stop_hook_active":false}' | .claude/hooks/pytest-gate.sh; echo $?
  ```

- **Fail open on missing tooling, closed on real failures.** A missing `uv`/`git`/`jq` exits 0 with a note; only an actual test or lint failure exits 2. A "command not found" must never masquerade as a broken gate.
- `pytest-gate.sh` honors `stop_hook_active` so it can never re-block a turn it already blocked.
- **Add an import and its first use in the same edit.** `ruff-ty-gate.sh` runs `ruff check --fix`, which deletes a just-added import as unused (F401) before you have written the code that needs it — silently reverting your edit. Writing the usage first works too, at the cost of one blocking failure.
- **Mutation-test a new test before trusting it.** Break the code it covers, confirm it fails, restore. Nothing else here would catch a test that passes vacuously.

Note for `invariant-hints.sh`: `README.md`'s routing table hard-codes `/skills/`, `/workspace/`, `/memories/` too, making it a fourth (documentation-level) consumer of the paths beyond the three code subsystems listed above. The virtual paths are also deliberately *asymmetric* — `skills_vpath` and `memories_vpath` carry a trailing slash, `workspace_vpath` does not, because `prompts.py` renders `{workspace_vpath}/speeches/<slug>.md`. Normalizing them for consistency silently yields `/workspace//speeches/`.

## Skills

Each `skills/<slug>/SKILL.md` is loaded on demand by the agent (progressive disclosure — the description tells the agent when to read the body). `test_all_skills_have_valid_frontmatter` enforces the contract:

- YAML frontmatter with `name` **matching the directory slug** and a non-empty `description`.
- Body sections: `## Overview`, `## When to Use`, `## Instructions`, `## Pitfalls`.
- The test hard-codes `len(skill_dirs) == 4` — **adding or removing a skill requires updating that assertion.**

## Web UI (Streamlit)

- Added with `uv add streamlit` (main dep, in `uv.lock`) — not the pip line the Streamlit skill's discovery script prints, which `uv sync` would undo.
- `.streamlit/config.toml` holds the theme **and** `server.address = "localhost"`: Streamlit otherwise binds all interfaces and prints an External URL, exposing an unauthenticated, budget-spending agent. `.streamlit/secrets.toml` is gitignored; config is read from `.env` via `load_settings()`, never `st.secrets`.
- `tests/test_webapp.py` renders both pages headlessly with `streamlit.testing.v1.AppTest` — still offline/free. Call `st.cache_resource.clear()` before each `AppTest` run or a cached bundle pins the wrong `SPEECHWRITER_HOME`.

## Gotchas

- `tests/test_build.py` imports two things that aren't declared dependencies or public API: `yaml` (pyyaml arrives transitively via langchain) and `deepagents.middleware.filesystem._check_fs_permission` (private). Either can break on a dependency bump; the sandbox test is the likely casualty.
- `workspace/` and `.speechwriter/` are gitignored runtime output — `load_settings()` creates them on startup, so a missing folder never fails the first draft.
- The CLI rotates `thread_id` after a `KeyboardInterrupt` so it never resumes a half-executed graph; that intentionally drops prior conversation context.
- The orchestrator is given **no direct tools** (`tools=[]`). Research is delegated so noisy search results never crowd the writing context. Add new capabilities as subagents unless the orchestrator genuinely needs them inline.
- `langsmith.utils.get_env_var` is `lru_cache`d, so anything that reads tracing state before `load_settings()` calls `load_dotenv` permanently caches "tracing off". `build_agent()` calls `load_settings()` first — that ordering is what makes `LANGSMITH_TRACING` in `.env` work at all.
- The agent revises `workspace/speeches/<slug>.md` **in place**; copy it aside first if you want to diff a re-run against the previous draft.
