# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A speechwriter agent built on **Deep Agents** (`deepagents` on LangChain/LangGraph). The project mostly *configures* the harness rather than implementing agent machinery: filesystem tools, subagent delegation, and skills all come from `create_deep_agent`. The code supplies the model, the prompts, the backend routing, the permission sandbox, and durable memory.

## Commands

```bash
uv sync                                  # install into .venv from uv.lock
uv run speechwriter                      # interactive REPL (also: uv run python -m speechwriter)
uv run streamlit run streamlit_app.py    # the web UI (same bundle, second front end)
uv run pytest                            # full suite — offline, no API key or network
uv run pytest tests/test_build.py::test_write_sandbox_confines_writes   # single test
uvx ruff@0.16.0 check .                  # lint (line-length 100)
uvx ruff@0.16.0 format .                 # format
uvx ty@0.0.63 check                      # type check
printf 'exit\n' | uv run speechwriter    # zero-cost smoke test: builds, banners, persists, exits — no model calls
printf 'BRIEF\nexit\n' | uv run speechwriter   # drive one real turn non-interactively (costs tokens)
```

**When working with Python here, invoke the relevant Astral skill first** — `/astral:uv` for dependencies and environments, `/astral:ruff` for lint and format, `/astral:ty` for type checking — so the current best practices are followed rather than guessed at. They also encode the right invocation form: `uv run` for anything that must import the project's dependencies (e.g. `pytest`), `uvx` for standalone tools (`ruff`, `ty`).

### Environment

`load_settings()` reads six variables, and its docstring (`config.py`) is the roster — `ANTHROPIC_API_KEY` (lazy; the CLI panics with a setup panel, `build_agent()` does not), `TAVILY_API_KEY`, `SPEECHWRITER_MODEL`, `SPEECHWRITER_MAX_TOKENS`, `SPEECHWRITER_MAX_RESEARCH_RESULTS`, and `SPEECHWRITER_HOME`. Both integer knobs go through `_optional_int_env`, which *warns and falls back* rather than raising, so a typo'd value costs a log line, not a crash.

**`SPEECHWRITER_HOME` overrides `project_root`, and everything else derives from it** — workspace, skills, store path, and which dotenv file gets loaded. There is no `tests/conftest.py`; it is the suite's *only* isolation mechanism and all ~38 test call sites set it by hand. A new test that forgets it reads and writes the real repo. Note the trade-off it carries: `load_settings()` creates `workspace_dir` and the store's parent but **never `skills_dir`**, and `create_deep_agent` only logs when a skills tree is missing — so every test pointing `SPEECHWRITER_HOME` at a `tmp_path` runs with the rhetoric library silently absent. Nothing in the suite exercises real skill loading; `test_all_skills_have_valid_frontmatter` reads the files off disk instead.

All three gates are **clean** and must stay that way: `uv run pytest`, `uvx ruff@0.16.0 check .`, and `uvx ty@0.0.63 check`. **Type the pins.** Unpinned, `uvx` resolves whatever Astral shipped today, and a checker one version off CI can demand a suppression that CI then rejects as unused — a state no source satisfies (see the pin rule under Automation). (No pass-count is quoted here on purpose — it went stale twice in one sitting, and `pytest` reports the number better than a doc can.) Prefer typing something precisely over widening it to `Any`; if a suppression is genuinely unavoidable, use a rule-specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.

## Architecture

Everything composes in one place: `build_agent()` in `src/speechwriter/agent.py`. Read that file first; the rest of the package feeds it.

### The virtual filesystem is the central design

The agent sees a `CompositeBackend` that routes by path prefix. Virtual paths are **not** real paths — `FilesystemBackend` is rooted at `project_root` with `virtual_mode=True`.

```
/skills/     → FilesystemBackend   read-only reference (the rhetoric library)
/workspace/  → FilesystemBackend   real .md files on disk (drafts, research notes)
/memories/   → StoreBackend        intercepted before disk; never a real folder
```

### Paths are single-sourced in `config.Settings` — and four consumers must agree

`memories_vpath` is a `ClassVar` (`/memories/`); `skills_vpath` and `workspace_vpath` are properties derived from real dirs via `Settings._vpath()`. Four independent consumers depend on them agreeing:

1. **Backend routes** — `agent.py:_build_backend()` routes `memories_vpath` to the Store. *(`test_backend_routes_memories_to_the_store_and_everything_else_to_disk`)*
2. **Sandbox rules** — `agent.py:_write_sandbox()` allows writes only under workspace + memories. *(`test_write_sandbox_confines_writes`)*
3. **Prompt text** — `prompts.py` interpolates all three paths into the system prompts. *(`test_every_virtual_path_reaches_the_prompts`)*
4. **README's routing table** — documentation, and the first thing a reader meets. *(`test_readme_routing_table_matches_the_configured_paths`)*

Change a path and you must propagate it through all four, or the agent will be *instructed* to write somewhere the sandbox *denies*.

This used to be enforced by nothing but an advisory Claude Code hook that restated the rule on a `config.py` edit. It is now four tests, which is strictly stronger: they run in CI and for contributors not using Claude Code, and they *block*. Two things they encode deliberately. The backend consumer is tested through `_build_backend()` — extracted from `build_agent()` for exactly this reason, since a route buried in a local cannot be asserted against without compiling the graph — and it asserts the **behaviour** (a `/memories/` write reaches the Store and never becomes a real folder), not just the route key, because that is the property the design exists for. And a *consistent* rename must keep passing: the prompt interpolates from `Settings`, so only hard-coded drift should fail. That is why README, which spells the paths out as prose, is the consumer that catches a rename.

The trailing-slash asymmetry is deliberate and load-bearing: `skills_vpath` and `memories_vpath` carry a trailing slash, `workspace_vpath` does not, because `prompts.py` renders `{workspace_vpath}/speeches/<slug>.md`. Normalizing them for tidiness silently yields `/workspace//speeches/` — which is why the prompt test asserts no rendered prompt contains `//`.

Likewise `SPEECHES_SUBDIR` / `RESEARCH_SUBDIR` / `WORDS_PER_MINUTE` are single-sourced in `config.py`: `prompts.py` *tells* the agent to file drafts under those folders at that pace, and `workspace.py` *reads* them back to list drafts and estimate spoken length. Re-hardcoding either side yields a browser that silently lists nothing — `test_prompts_point_the_agent_at_the_folders_the_browser_reads` guards the folder seam (both halves: `browse.py` lists `research_dir()` as well as `speeches_dir()`, and the researcher subagent that writes there is steered by its own prompt, so that prompt gets its own case), and `test_spoken_length_uses_the_pace_the_prompt_prescribes` parses the pace back out of the rendered prompt rather than reading `WORDS_PER_MINUTE` on both sides. Both were mutation-tested: hard-code the folder or the pace into `prompts.py` and they fail. Note `ruff` will *not* save you here — re-typing one constant leaves the others imported, so F401 never fires.

### The write sandbox is enforced, not prompted

`_write_sandbox()` returns first-match-wins `FilesystemPermission` rules: allow `write` under workspace + memories, then deny `write` on `/**` as the backstop. Reads are left open so skills and reference material still load. The same rules are applied **twice** — to `create_deep_agent(permissions=...)` *and* to every subagent — because each subagent is built with its own `FilesystemMiddleware`. Passing them explicitly is belt-and-braces rather than the only thing between the critic and the repo: `graph.py`'s `spec.get("permissions", permissions)` means an *omitted* key falls back to the parent's rules (true across the whole pinned range, not just 0.7). But the fallback is all-or-nothing — an explicit list **replaces** the parent's entirely — so if you pass one, keep it complete. Skills are the key that genuinely inherits nothing.

### Subagents inherit nothing

`build_subagents()` must hand each subagent everything it needs explicitly:
- **Skills** — the `style-critic` gets `"skills": [settings.skills_vpath]` because subagent skill sets are not inherited.
- **Permissions** — passed in from `build_agent` as the `permissions=` argument. (These *would* be inherited if omitted — see the sandbox section above. Skills and cross-call state are the two that genuinely are not.)

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
- **The model's output ceiling is resolved in three tiers, never inherited blindly.** `build_agent` passes a *constructed* model (`agent.py:_build_model`), never a bare id string, because `init_chat_model` takes `max_tokens` from LangChain's model-profile table and silently falls back to **4096** for an id it cannot profile. As of the current pin every shipped Claude id *is* profiled (`claude-sonnet-5`/`claude-opus-5`/`claude-opus-4-8` at 128k, `claude-sonnet-4-5`/`claude-haiku-4-5` at 64k), so the default resolves through **tier 2** and `ceiling_label` reads `128,000` — but a typo'd or newly-released id still lands on 4096, which is the whole point. Extended thinking bills against that same ceiling, so an unprofiled id lets a subagent spend its entire budget thinking and emit no text, which deepagents forwards as an *empty* `status="success"` tool result (it walks back for the last message with text and finds none). Resolution order: **explicit `SPEECHWRITER_MAX_TOKENS` → the model's own profile → `DEFAULT_MAX_TOKENS` (32k)**. Tier 2 is load-bearing in the other direction — a flat constant would *cap* Opus at 32k when its profile says 128k, which is the same mistake inverted. `settings.max_tokens` is only the override and is normally `None`; the resolved value lives on `SpeechwriterAgent.max_tokens`. `SPEECHWRITER_MAX_TOKENS` is validated `>= 1` at load time — a 0 or negative ceiling is accepted by `init_chat_model` and only fails at the first API call, far from the typo. Four signals guard this, all tested: `_build_model` warns on an unprofiled id; `test_default_model_still_resolves_through_tier_two` fails if LangChain stops profiling `DEFAULT_MODEL` or profiles it below the floor; `TruncationWarner` (`observability.py`) counts responses that actually hit the ceiling, matching `stop_reason` *and* `finish_reason` across providers since `SPEECHWRITER_MODEL` is free-form; and the bundle owns that warner, handing it out via `SpeechwriterAgent.turn_config()`. That second signal covers a gap `test_ceiling_resolution_is_three_tier` structurally cannot see: it proves tier 2 through `claude-opus-4-8`, so the *default* could lose its profile and drop 4× to the floor with every other assertion still green. It is one of only two assertions in the suite about a **dependency's** own behaviour rather than our code, deliberately (the other is the langsmith cache canary under Gotchas) — that table is the input this bullet rests on, it moves on langchain-anthropic's schedule, and it has already moved once, which is how this bullet went stale before. A failure there is a prompt to re-read this bullet and decide whether to pin `SPEECHWRITER_MAX_TOKENS`, not a bug to fix. **Attach observability through `turn_config()`, not by hand-building `{"configurable": {...}}`** — same reasoning as `persist()`: the CLI is one of two entry points, and the README documents the other. Don't revert to `model=settings.model`.
- **`temperature`, `top_p` and `top_k` must never reach the request payload.** Current Claude models reject them outright with a 400, and `build_agent()` never touches the wire — so nothing else here would notice a `langchain-anthropic` bump that began sending one by default; every *real* turn would fail instead, far from the upgrade that caused it. `test_payload_omits_parameters_current_models_reject` asserts it offline against `ChatAnthropic._get_request_payload([])` for three model ids across both ceiling branches. Don't add a sampling parameter to `_build_model` to "make output more consistent".
- **`import speechwriter` must stay lazy.** `__init__.py` exposes `build_agent` and `TruncationWarner` via module `__getattr__` so the heavy `deepagents`/`langchain` stack isn't imported eagerly. `test_import_speechwriter_is_lazy` spawns a subprocess to assert this — don't add a top-level import of `agent.py` to `__init__.py`.
- **`load_dotenv` targets `project_root / ".env"` explicitly**, never an upward walk (an ancestor `.env` could leak unrelated keys), and it is called inside `load_settings()` so `import speechwriter` has no side effects. Real shell env wins over `.env`.
- **The Store namespace is explicit** (`_memory_namespace` → `("speechwriter", "memories")`). deepagents' implicit-namespace mode is deprecated and removed in 0.7; the dependency is pinned `>=0.6.12,<0.8`.
- **The prompts may only name capabilities that actually exist.** `prompts.py` told the model to use its "planning tool (`write_todos`)" from the day it was written, and `create_deep_agent` has never bound one. Nothing structural could see it — `ty` checks Python and not prose, `ruff` checks syntax — so `test_prompts_only_advertise_tools_the_model_can_call` compares the rendered prompts against the tool surface: every backticked lowercase identifier must be a bound tool, a subagent reachable via `task`, **a tool owned by one of those subagents**, or an explicit entry in `_NON_TOOL_BACKTICKS`. That third source is load-bearing, not decorative: `researcher_prompt` advertises `tavily_search`, which belongs to the `researcher` and is never bound to the orchestrator's model — which is also why the test sets `TAVILY_API_KEY`, since without it the researcher is absent and its tool leaves the vocabulary with it. That allowlist is empty **on purpose** — a new backticked word should force a choice (bind the tool, or declare the word prose), not pass silently. Two details are load-bearing if you refactor it. The vocabulary is captured from what is bound to the **model**, not from the compiled graph's ToolNode (`agent.nodes["tools"].bound.tools_by_name`): the latter is a superset carrying `execute`, which middleware strips before the model sees it, so asserting against it would permit precisely the bug. And `bind_tools` fires when the graph *steps*, not when it is built, so the capture runs one turn against a recording stub — no network, so the offline invariant above still holds. The trailing `assert "task" in advertised` is an anti-vacuity canary: every other assertion is satisfied by an empty match set, so a pattern that stops matching would turn the test green while checking nothing. **When it fails, the fix is usually the prompt, not the test** — a new tool or subagent widens the vocabulary automatically, a new backticked prose word does not.

## Automation (`.claude/` and `.github/`)

The gates and invariants above are guarded by **two layers**, and the split is the point. There is no pre-commit and no git hooks.

1. **`.claude/` hooks — the inner loop.** Sub-second, fire on Claude Code's own events, and speak *to the model*: an exit-2 stderr lands in context so the diagnostic is fixed while the edit's intent is still live. They see the uncommitted tree, but only for edits Claude Code itself made.
2. **`.github/workflows/ci.yml` — the independent check.** Runs the same three gates on push to `main` and on every PR, in a clean clone with no `.env`, matrixed over Python 3.11/3.12/3.13 (~25s for the whole run). It covers what hooks structurally cannot: hand edits, contributors not using Claude Code, and fresh-environment-only breakage (the private third-party symbols noted under Gotchas). It also exercises the offline invariant — no API key is set anywhere in the workflow, so the day `build_agent()` starts needing the wire, CI goes red instead of quietly billing.

   **It reports; it does not block.** No branch protection rule requires these checks, so a red run is advisory — a merge or a push to `main` still lands. Making it a real gate is a repo-settings change, not a file: require the four checks (`ruff check` and `pytest + ty (py3.11|3.12|3.13)`) on `main`. Until then, don't describe CI as if it enforces anything. (The one exception is `release.yml` below, whose gate re-run does block a publish.)

Neither layer subsumes the other. The hooks are fast and land *in the model's context*, but only for edits Claude Code made and are bypassed by editing a file any other way. CI sees every commit in a clean environment, but out of band and with no channel back to the model — and, today, without the authority to stop anything. Three hooks, wired in `.claude/settings.json` — **all three blocking**; there is no advisory hook, by design (see below):

| Hook | Event | Behavior |
|---|---|---|
| `hooks/env-guard.sh` | `PreToolUse` on `Bash` | Blocks any Bash command referencing a `.env*` file or `secrets.toml`. **Exit 2** to deny. Measured 0.03s. |
| `hooks/ruff-ty-gate.sh` | `PostToolUse` on `Edit\|Write` | Runs `uvx ruff format` + `ruff check --fix` on the edited `.py`, then re-checks with `ruff check` and `ty check` — all at the **same pins as CI** (see below). **Exit 2** on remaining diagnostics. Measured 0.55s. |
| `hooks/pytest-gate.sh` | `Stop` | Runs `uv run pytest` if the working tree is dirty under `src/ tests/ skills/ .streamlit/ pyproject.toml uv.lock`. **Exit 2** on failure. Measured 0.02s clean / 3.0s dirty. |

Design rules to preserve if you touch these:

- **A hook must catch something no test can.** This is the rule that shrank the set. An advisory `invariant-hints.sh` used to restate the path invariant on a `config.py` edit and the skill-count assertion on a `skills/` edit — but the skill branch duplicated a gate that already *blocks* (`pytest-gate.sh` watches `skills` with `--untracked-files=all` precisely so a new `SKILL.md` trips `len(skill_dirs) == 4`), and the path branch duplicated the CLAUDE.md section above, which is already in the model's context at session start. Both were replaced by tests. **Before adding an advisory hook, check whether it is a test you have not written yet** — a hint is a suggestion that only fires for edits Claude Code made; a test blocks, runs in CI, and works for everyone.
- **The Stop gate must stay cheap in the common case.** It short-circuits on `git status --porcelain` in ~20ms and only then pays the ~3s suite (measured; `pytest` itself is ~2.5s and `uv` startup the rest). It fires whenever the tree is dirty under the watched paths, not when *this turn* touched them — a Stop hook has no reliable per-turn diff — so once `src/` has uncommitted work every turn pays it. At ~3s that is the intended trade. `--untracked-files=all` is load-bearing: `git diff` sees only *tracked* files, so a brand-new `skills/<slug>/SKILL.md` — exactly what trips the count assertion — would slip past untested.
- **Path matching lives in the scripts, not in settings.json.** The hook `if:` field is real, but its patterns are working-directory-relative and the leading-slash form is underspecified; `if: "Edit(/skills/**)"` can silently match nothing. Filtering inside the script is explicit and testable — each script parses the stdin payload for the key its own event actually carries (`ruff-ty-gate.sh` reads `tool_input.file_path`, `env-guard.sh` reads `tool_input.command`, and `pytest-gate.sh` reads only `stop_hook_active`, taking its paths from `git` rather than the payload), so each can be exercised directly — note the three invocations below are three different payload shapes for that reason:

  ```bash
  echo '{"tool_input":{"file_path":"'$PWD'/src/speechwriter/agent.py"}}' | .claude/hooks/ruff-ty-gate.sh; echo $?
  echo '{"stop_hook_active":false}' | .claude/hooks/pytest-gate.sh; echo $?
  echo '{"tool_input":{"command":"cat .env"}}' | .claude/hooks/env-guard.sh; echo $?   # want 2
  ```

- **`ruff-ty-gate.sh` pins ruff and ty to the same versions CI does, and a test keeps the three sites in step.** Floating locally while CI pins is not merely noisier or quieter — it admits source states that satisfy *neither*: a type-checker suppression comment is required by a version that cannot resolve a symbol and rejected as an unused-ignore by a version that can, so the gate goes green in the model's context and red on push. That is not hypothetical; it is what `a304a89` fixed. The pins live in four independent sites with nothing structural tying them — `ci.yml`, `release.yml`, the hook, and the commands in **Commands** above — so `test_tool_pins_agree_wherever_they_are_declared` asserts they agree, the same shape and the same remedy as `test_package_version_matches_pyproject`. **Bump all four together**; the test names every site and both versions when they drift, so it tells you where to look. It reads two spellings, which is why the doc can be a site at all: the named form (`TY_VERSION: "0.0.63"`, `TY_VERSION="0.0.63"`) and the invocation form (`uvx ty@0.0.63`). Two knock-ons worth knowing: pinning costs nothing at runtime (`uvx` caches per version — measured 0.55s, slightly *faster* than the floating gate it replaced), and the hook's failure banners now name the exact pinned command (`[uvx ty@0.0.63 check]`) so what it reports is what you can re-run.
- **Missing tooling is non-blocking but *visible* — exit 1, not 0.** A missing `uv`/`jq`/`python3` must never masquerade as a lint or test failure, so it never exits 2. But it must not exit 0 either: stderr on exit 0 is swallowed, so a gate that has quietly stopped running looks exactly like a gate that is passing. Exit 1 is non-blocking *and* surfaces the note. Both quality gates follow this.
- **`env-guard.sh` inverts that rule, deliberately.** It is the one hook guarding credentials rather than code quality, so it **fails closed**: no `python3`, or an unparseable payload, and it denies the command. A credential guard that silently switches itself off when a parser is missing is worse than no guard, because you believe you have one. Its scope mirrors the tool-level `Read(./.env.*)` deny rule exactly — `.env.example` included, though it is committed and secret-free — so the two cannot disagree; narrowing it is one line.
- **`permissions.deny` is tool-scoped, and Bash routes around it.** `Read(./.env)` stops the `Read` tool and nothing else; `cat .env` was never covered by any rule, and auto mode makes reading files via `cat`/`sed`/`grep` the *default* path. That gap is what `env-guard.sh` exists to close — remember it whenever you add a `deny` entry expecting it to be airtight.
- `pytest-gate.sh` honors `stop_hook_active` so it can never re-block a turn it already blocked.
- **Add an import and its first use in the same edit.** `ruff-ty-gate.sh` runs `ruff check --fix`, which deletes a just-added import as unused (F401) before you have written the code that needs it — silently reverting your edit. Writing the usage first works too, at the cost of one blocking failure.
- **Mutation-test a new test before trusting it.** Break the code it covers, confirm it fails, restore. Nothing else here would catch a test that passes vacuously.

Writing a `.env` value is therefore a **user action, not a Claude action**: ask the user to make the edit, then verify indirectly (assert a variable is non-empty, run the offline suite) without printing anything. The guard fires on the literal token in *any* Bash command, prose included — a heredoc writing documentation about it is blocked too. Use Write/Edit for that, as the hook's own message says.

**And that scope collides with a test.** `test_env_example_documents_every_setting` is a fifth consumer of `config.py`: it extracts every `SPEECHWRITER_[A-Z_]+` token from the source and asserts each appears in the example template. But that template matches the deny rules and `env-guard.sh`'s pattern alike — deliberately, so the guard and the rules cannot disagree. So **adding a `SPEECHWRITER_*` knob to `config.py` turns the Stop gate red in a way Claude structurally cannot clear**: it must ask the user to add the documenting line. Expect that, rather than rediscovering it as a mystery failure.

### Releases are automatic, and the version lives in two files

`.github/workflows/release.yml` tags and publishes a GitHub Release, unattended, when
`[project] version` in `pyproject.toml` changes on `main`. There is no draft step — it goes live.

That is only safe because of three guards, each load-bearing:

- **The trigger is `paths: ["pyproject.toml"]`, which also fires on every `uv add`.** So a
  `check` job reads the version and stops unless `v$VERSION` is a tag that does not exist yet; a
  dependency bump lands as `check ✓ / release ⊘ skipped`, not a spurious release. That job
  deliberately runs no `uv sync` and never sets up uv — the no-op is the common path and should
  cost seconds. It reads the version with `tomllib` on the runner's own `python3` (ubuntu-24.04
  ships 3.12), and every failure mode there is loud: a missing key or a bad parse fails the step
  rather than tagging something wrong.
- **The `release` job re-runs all three gates before tagging.** `ci.yml` runs the full matrix on
  the same commit, but nothing orders the two workflows, so this re-checks on the 3.11 floor
  rather than tagging a commit whose verdict has not landed. **This is the only place in the repo
  where a failing gate actually stops something** — the counterpoint to "it reports; it does not
  block" above.
- **The version is two independent literals** — `[project] version` in `pyproject.toml` and
  `__version__` in `src/speechwriter/__init__.py`. Nothing structural ties them: hatchling builds
  from the first, `import speechwriter` reports the second.
  `test_package_version_matches_pyproject` is what makes an unattended release safe, precisely
  because it runs inside that gate re-run — a half-done bump fails the job instead of publishing a
  release whose installed package still reports the old version. **Bump both, in one commit.**

Two mechanics worth not re-deriving. `gh release create` creates the tag itself at
`--target $GITHUB_SHA`, so there is no `git tag`/`git push` step and no bot git identity to
configure. And `permissions` is asymmetric on purpose: `release.yml` needs `contents: write` for
that call, `ci.yml` stays `contents: read`.

## Skills

Each `skills/<slug>/SKILL.md` is loaded on demand by the agent (progressive disclosure — the description tells the agent when to read the body). `test_all_skills_have_valid_frontmatter` enforces the contract:

- YAML frontmatter with `name` **matching the directory slug** and a non-empty `description`.
- Body sections: `## Overview`, `## When to Use`, `## Instructions`, `## Pitfalls`.
- The test hard-codes `len(skill_dirs) == 4` — **adding or removing a skill requires updating that assertion.**
- `test_orchestrator_prompt_names_every_skill` additionally requires each slug to appear in the parenthetical list in step 4 of `orchestrator_prompt()`. Progressive disclosure cuts both ways: the agent only reads a `SKILL.md` if it knows the skill exists, so a skill absent from that list is dead weight on disk. Slugs are matched in their prose form (`delivery-and-cadence` → "delivery & cadence"); if a new skill does not fit that shape, name it in the prompt however reads best and widen the normalisation in the test.

## Web UI (Streamlit)

- Added with `uv add streamlit` (main dep, in `uv.lock`) — not the pip line the Streamlit skill's discovery script prints, which `uv sync` would undo.
- `.streamlit/config.toml` holds the theme **and** `server.address = "localhost"`: Streamlit otherwise binds all interfaces and prints an External URL, exposing an unauthenticated, budget-spending agent. `.streamlit/secrets.toml` is gitignored; config is read from `.env` via `load_settings()`, never `st.secrets`.
- `tests/test_webapp.py` renders both pages headlessly with `streamlit.testing.v1.AppTest` — still offline/free. Call `st.cache_resource.clear()` before each `AppTest` run or a cached bundle pins the wrong `SPEECHWRITER_HOME`.

## Gotchas

- **`tests/test_build.py` reaches into three private third-party symbols**, any of which can vanish on a dependency bump: `deepagents.middleware.filesystem._check_fs_permission` (the sandbox test, the likely casualty), `agent.nodes["tools"].bound.tools_by_name` (the prompt-vs-tool test), and `ChatAnthropic._get_request_payload` (the payload test). That is deliberate — a rename breaks them loudly, which is the failure mode we want. It also imports `yaml`, which is **no longer** transitive-only: `pyyaml>=6.0` is declared in `[dependency-groups] dev` as of `d4265dd`.
- **`[tool.ruff.format] exclude = ["*.md"]` is load-bearing — don't drop it.** Ruff 0.16 formats Python code blocks inside `.md`, which would make the formatter the only part of the toolchain that reaches into docs: `ruff check` skips Markdown entirely (`No Python files found`), and `hooks/ruff-ty-gate.sh` filters to `*.py`/`*.pyi`. Without the exclude, the documented `uvx ruff@0.16.0 format .` silently rewrites the hand-typeset fences in `README.md` — and in this file, whose **Commands** block is column-aligned by hand — so an unrelated commit picks up a docs diff. With it, the formatter sees 18 Python files and `.` is safe for the hook, CI, and the command line alike.
- `workspace/` and `.speechwriter/` are gitignored runtime output — `load_settings()` creates them on startup, so a missing folder never fails the first draft.
- The CLI rotates `thread_id` after a `KeyboardInterrupt` so it never resumes a half-executed graph; that intentionally drops prior conversation context.
- The orchestrator is given **no direct tools** (`tools=[]`). Research is delegated so noisy search results never crowd the writing context. Add new capabilities as subagents unless the orchestrator genuinely needs them inline.
- **There is no planning tool, and that is not an oversight.** `create_deep_agent` binds exactly `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, `task` — no `write_todos`. deepagents 0.6 had no `TodoListMiddleware` at all; 0.7 installs it only for the OpenAI-Codex harness profile. The staged rhythm lives in `prompts.py` instead. If you ever want the tool, it is `create_deep_agent(middleware=[TodoListMiddleware(system_prompt="")])` from `langchain.agents.middleware` (the empty `system_prompt` trims LangChain's default prose, which otherwise duplicates the tool's own schema description) — but do not re-add a *claim* that planning is automatic without binding it, or the system prompt goes back to advertising a tool the model cannot call — `test_prompts_only_advertise_tools_the_model_can_call` fails if you do.
- **`langsmith.utils.get_env_var` is `lru_cache`d, so `load_settings()` clears that cache right after `load_dotenv`.** The first read of `LANGSMITH_TRACING` is the one that sticks for the process, so anything touching tracing state at import — a module-level `Client()`, a `tracing_is_enabled()` — would otherwise cache "off" *permanently* for a value that exists only in `.env`, with nothing raised. Tracing just never happens while every setting still reads as correct, which is indistinguishable from a project whose traces aged out of retention. This used to be an unwritten ordering rule that only `build_agent()` happened to satisfy; clearing the cache makes the order irrelevant for **all three entry points** (CLI, Streamlit, library consumers), which is why the fix lives in `config.py` and not in `agent.py`. `test_load_settings_reopens_the_langsmith_env_cache` guards it, and its first half is an anti-vacuity canary: it asserts *directly* on `get_env_var` rather than through `tracing_is_enabled()`, which returns early on three context-var paths before it ever consults the cache — a future default there would let the test pass while exercising nothing. That canary is the suite's **second** assertion about a dependency's own behaviour rather than our code (see the ceiling bullet's `test_default_model_still_resolves_through_tier_two`); if it fails, langsmith stopped caching and the `cache_clear()` is obsolete — re-read this bullet, don't "fix" the test. Note `cache_clear` is invisible to `ty` (`@overload` stubs shadow the `lru_cache` wrapper): `config.py` reaches it through `getattr` and degrades quietly, the test binds it directly and fails loudly, and that asymmetry is deliberate.
- **Store keys are not one shape.** A write through the backend keeps its leading slash — `/memories/mayor.md` lands as key `/mayor.md` — while tests and any code seeding the Store directly use the bare `mayor.md`. Both coexist in practice, and `browse.py`'s `_humanize` title-cases the raw key, so a genuine agent-written profile renders as **"/Mayor"** in the Memory expander. Strip the slash at the presentation layer if you care; don't "fix" it in the backend.
- The agent revises `workspace/speeches/<slug>.md` **in place**; copy it aside first if you want to diff a re-run against the previous draft.
