# Speechwriter Agent

A speechwriter built with **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** (LangChain + LangGraph). You describe the speaker, audience, occasion, and goal; the agent researches facts, drafts a speech written *for the ear*, critiques its own draft, revises, and **remembers how each speaker sounds** across sessions.

```
you › Write a 4-minute wedding toast. Speaker: David, best man. Audience: 80 guests,
      mixed ages. Couple: Ana & Priya, met hiking. Warm, a little funny, no clichés.

  ⚙  read_file         {"file_path":"/skills/audience-and-occasion/SKILL.md","limit":1000}
  ⚙  task              {"subagent_type":"style-critic","description":"Critique toast draft"}
  ✓  task: Verdict 8/10. Tighten the open; the hiking callback lands. …
  ⚙  write_file        {"file_path":"/workspace/speeches/ana-priya-toast.md", …}
╭─ speechwriter ───────────────────────────────────────────────────────────────╮
│  Here's the toast — about 3 minutes 40 at a relaxed pace. …                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Why Deep Agents?

A speech is a **long-horizon** task: intake → research → outline → draft → critique → revise. That maps cleanly onto what the Deep Agents harness provides, so this project mostly *configures* capabilities rather than implementing them:

| Speechwriting need | Deep Agents primitive | Where it lives |
|---|---|---|
| Break the commission into stages | The staged operating rhythm in the system prompt — *prompted, not tooled* | [`prompts.py`](src/speechwriter/prompts.py) |
| Keep draft versions & research notes | Filesystem tools + `FilesystemBackend` | [`agent.py`](src/speechwriter/agent.py) |
| Look up facts without polluting the writing context | `researcher` **subagent** (Tavily) | [`subagents.py`](src/speechwriter/subagents.py) |
| A hard editorial pass | `style-critic` **subagent** | [`subagents.py`](src/speechwriter/subagents.py) |
| Rhetoric/structure know-how, loaded on demand | **Skills** (`SKILL.md`) | [`skills/`](skills/) |
| Remember a speaker's voice across sessions | **`StoreBackend`** via `CompositeBackend` | [`agent.py`](src/speechwriter/agent.py) + [`memory.py`](src/speechwriter/memory.py) |

### The memory architecture (the interesting part)

The agent's filesystem is a **`CompositeBackend`** that routes by path prefix:

```
/skills/…      ─▶ FilesystemBackend  (read-only reference; the rhetoric library)
/workspace/…   ─▶ FilesystemBackend  (real .md files on disk — you can open them)
/memories/…    ─▶ StoreBackend       (persistent, cross-session speaker voice profiles)
```

`/memories/` is intercepted *before* it reaches disk and sent to a LangGraph `Store`. Because the only local `Store` is `InMemoryStore` (which dies with the process), [`memory.py`](src/speechwriter/memory.py) **snapshots it to `.speechwriter/memory-store.json`** on exit and rehydrates it on startup — so "remember how the Mayor likes to sound" actually survives to next week. Swap in `PostgresStore` there to make it multi-user.

Two tiers of knowledge, kept deliberately separate:
- **Principles = code.** How to write *any* speech lives in the system prompt ([`prompts.py`](src/speechwriter/prompts.py)) and the skills.
- **A speaker's voice = memory.** What's specific to *this* speaker lives in `/memories/<speaker>.md` and persists.

---

## Setup

Requires **Python ≥ 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install into .venv from the lockfile
cp .env.example .env         # then fill in your keys
```

Every turn is sent to **Claude**, so one key is required; the rest are optional:

```ini
ANTHROPIC_API_KEY=sk-ant-...     # required — the model that writes
TAVILY_API_KEY=tvly-...          # optional — enables live web research
DEEPGRAM_API_KEY=...             # optional — enables measuring a draft's spoken length
LANGSMITH_TRACING=true           # optional — trace every turn to LangSmith
LANGSMITH_API_KEY=lsv2_...       #   (with the key it uploads with)
```

Without a Tavily key the agent still works; it writes from its own knowledge and marks anything it can't verify with `[VERIFY]`. With one, a `researcher` subagent pulls current, sourced facts.

Without an Anthropic key both front ends still open — so you can see what is configured — but they refuse to run a commission and say what to set.

---

## Usage

The agent has two front ends over the **same** bundle — the same graph, the same workspace, the same persistent memory. Use whichever suits the moment.

### Terminal (REPL)

```bash
uv run speechwriter          # or:  uv run python -m speechwriter
```

Then just talk to it. Give it as much of the brief as you can — the agent will ask for anything essential it's missing:

> Draft a 12-minute commencement address for a state university. Speaker is a first-gen
> founder. One big idea: "usefulness beats prestige." Warm, story-driven, one good laugh.

- Finished speeches are saved to `workspace/speeches/` as Markdown.
- Research notes land in `workspace/research/`.
- Type `exit` (or `Ctrl-D`) to quit — voice-profile memory is snapshotted on the way out.

### Browser (Streamlit)

```bash
uv run streamlit run streamlit_app.py
```

A two-page web app reading the same configuration as the CLI — no separate setup, and the only
thing it lets you change without a restart is the model:

- **Write** — commission a speech and watch the agent plan, research, draft, and self-critique in a live activity log; each finished turn is snapshotted immediately (a closed tab runs no shutdown hook, so waiting until exit would usually mean never).
- **Workspace** — browse saved drafts (with a spoken-length estimate, and a **Measure** button that synthesizes the draft for a real one), research notes, and the voice profiles the agent has learned, read straight from the live Store.

It binds to `localhost` only by default; the app spends your API budget and reads and writes your workspace, so it is not meant to face the network. Override with `--server.address` if you genuinely intend to share it.

### Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Checked for presence before a turn is spent; a wrong key fails at the first turn with Anthropic's own 401. |
| `SPEECHWRITER_MODEL` | `claude-sonnet-5-5` | The Claude model to *start* on. Both front ends can switch mid-session; see [Switching models](#switching-models). |
| `SPEECHWRITER_MAX_TOKENS` | model's own | Overrides the output-token ceiling. Left unset, a model LangChain profiles keeps its real maximum (128k for the 5.5 models); an id it does not profile gets a 32,000 floor rather than LangChain's silent 4,096, which adaptive thinking alone can exhaust. |
| `SPEECHWRITER_MAX_RESEARCH_RESULTS` | `5` | Tavily results per query. |
| `SPEECHWRITER_HOME` | repo root | Root dir the agent reads/writes under. |
| `LANGSMITH_TRACING` | — | `true` traces every turn to [LangSmith](https://docs.smith.langchain.com). See [Tracing with LangSmith](#tracing-with-langsmith). |
| `LANGSMITH_API_KEY` | — | The key LangSmith uploads with. Also what the eval mirror and experiments use. |
| `LANGSMITH_PROJECT` | `speechwriter-agent` | The project traces land in. Defaulted here, rather than LangSmith's shared `default`. |
| `LANGSMITH_ENDPOINT` | hosted API | Only for a self-hosted LangSmith. |
| `DEEPGRAM_API_KEY` | — | Enables the **Measure** button. See [Measuring spoken length for real](#measuring-spoken-length-for-real). |

Every call also carries settings that are fixed in code rather than knobs, because each one is
load-bearing for the 5.5 models: adaptive thinking with summarized display, `effort: medium`,
thinking blocks that are *dropped* rather than rejected when context compaction rewrites history,
and Anthropic's server-side refusal fallback. No `temperature`, `top_p` or `top_k` is ever sent —
the 5.5 models reject them.

### Switching models

`SPEECHWRITER_MODEL` sets the model the session *starts* on. Both front ends can change it
without a restart — the sidebar dropdown in the browser, `/model` in the terminal:

```
› /model
 › 1  Sonnet 5.5   claude-sonnet-5-5
   2  Opus 5.5     claude-opus-5-5
Switch with /model <number> or /model <name>.
```

Sonnet 5.5 is the default: a strong writer at a sensible price. Opus 5.5 is the one to reach for
when the draft matters most. A `SPEECHWRITER_MODEL` that is not on the list — a pinned older
Claude, say — gets a row of its own and stays in the list after you switch away, so the trip is
never one-way.

Two things follow from how the switch works, and they are the same in both front ends.

- **It is a rebuild, not a setting.** The output ceiling is resolved from the constructed
  client, so it has to be. Learned voice profiles are snapshotted *first* and rehydrated by the
  rebuild, so they survive; the conversation does not — the new agent has a new checkpointer and
  cannot resume the old thread, so the thread is rotated and the transcript starts fresh.
- **`SPEECHWRITER_MAX_TOKENS` is global and wins over every model.** An override sized for one
  model follows you to the next, and a ceiling above what a model can emit is rejected at the
  first turn. The ceiling line says so before you spend one:
  `Output ceiling — 200,000 — above this model's 128,000-token maximum`.

### Tracing with LangSmith

Every turn can be traced to [LangSmith](https://docs.smith.langchain.com): each model call, tool
call and subagent run becomes a run in the trace, so you can see what the orchestrator
delegated, what the `researcher` searched for, and what the `style-critic` said — nested the way
it actually happened.

```ini
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=speechwriter-agent   # optional; this is the default
```

LangChain does the tracing itself — there is no exporter here to configure. Both front ends say
where traces are going before a turn is spent (the banner's `traces` line, the sidebar's
**Traces** caption), and every entry point is traced alike: the REPL, the web UI, the eval
harness, and `build_agent()` used as a library. Three things worth knowing:

- **One conversation is one LangSmith thread.** Threads are keyed by the agent's `thread_id`, so
  a conversation reads turn by turn, and a new one starts exactly when the agent does (after
  `Ctrl-C` in the terminal, or **New conversation** in the browser).
- **Tracing switched on without a key is called out, not hidden.** The label says uploads will
  be rejected, instead of a warning per batch far from the cause.
- **The REPL waits for the last turn's runs at `exit`.** The tracer uploads on a background
  thread, so a short session would otherwise end with its final turn still queued.

### Evals in LangSmith

`evals/datasets/` holds 55 graded examples across four datasets. The LangSmith workspace that
receives traces keeps a mirror of them and records experiments against it:

```bash
uv run python evals/sync_datasets.py          # is the LangSmith mirror in sync? (read-only)
uv run python evals/sync_datasets.py --push   # make it so
uv run python evals/run_experiment.py --langsmith --dataset trajectory --limit 1   # costs tokens
```

The files in the repo are the source of truth. A push creates and updates examples; it deletes
one only with `--allow-delete`, because a LangSmith delete cannot be undone. An experiment
refuses to run against a stale mirror, so it always grades against what the file says. Each
row opens onto the agent's full trace — every model call, tool call and subagent — and each
criterion is its own feedback column, with anything no scorer could measure reported as
`criteria_coverage` rather than counted as a pass.

### Measuring spoken length for real

`WORDS_PER_MINUTE` is one constant standing in for pace, and it cannot know that one draft is
dense with long words while another is short and punchy. With `DEEPGRAM_API_KEY` set, the
Workspace page's **Measure** button synthesizes the draft with
[Deepgram Aura-2](https://developers.deepgram.com/docs/tts-models) and reports the real duration
next to the estimate — and plays it back, since hearing a draft is the fastest way to catch
what a "speakability" critique can only infer. Without the key the button is disabled and says
what to set.

It is a button rather than something the page computes on load because it is billed per
character: a three-minute speech is a couple of requests. Each distinct draft is measured once
and cached, and a revised draft measures again.

**Read the two numbers as different things, not as right-and-wrong.** A TTS voice reads faster
than a speaker on a stage and does not stop for laughter, applause, or breath — a `[pause]` cue
adds no silence to the measurement. 130 wpm may well be the better guide to *time on stage*; the
measurement is the better guide to *time to say the words*. The gap between them is the
interesting part.

---

## Using it as a library

The agent is a plain compiled LangGraph graph:

```python
from speechwriter import build_agent

bundle = build_agent()          # agent, store, settings, max_tokens, warner
result = bundle.agent.invoke(
    {"messages": [{"role": "user", "content": "Write a 2-minute retirement toast for Sam."}]},
    config=bundle.turn_config("demo"),   # thread id + truncation detection
)
print(result["messages"][-1].content)

if bundle.warner.truncated:     # nothing else reports this — a clipped draft or
    print("raise SPEECHWRITER_MAX_TOKENS")   # critique looks exactly like a finished one

bundle.persist()   # snapshot learned voice profiles so the next run remembers them
```

---

## Project layout

```
src/speechwriter/
├── config.py      Settings: model, keys, virtual paths (single source of truth)
├── prompts.py     Orchestrator + researcher + critic system prompts
├── tools.py       Lazy Tavily research tool (degrades gracefully with no key)
├── subagents.py   researcher + style-critic SubAgent definitions
├── memory.py      Persistent Store: JSON snapshot load/save + exhaustive read
├── tracing.py     Reports (and flushes) the LangSmith tracing LangChain does itself
├── agent.py       build_agent() — composes every layer into one graph
├── cli.py         Rich streaming REPL
├── workspace.py   UI-free reader: drafts, research notes, voice profiles, Deepgram timing
└── webui.py       Streamlit glue: stream a turn, record it, replay it
streamlit_app.py   Web entry point (router) + app_pages/ (Write, Workspace)
skills/            On-demand rhetoric library (SKILL.md, progressive disclosure)
├── rhetorical-devices/     delivery-and-cadence/
├── speech-structures/      audience-and-occasion/
tests/             Offline tests — build the graph, toggle research, round-trip memory,
                   render both pages headlessly (all without the model or network)
evals/             Eval datasets, pure scorers, and the LangSmith mirror + experiment harness
```

## Development

```bash
uv run pytest                # offline: no API key or network needed
uvx ruff check . && uvx ruff format .
uvx ty check
```

The tests construct the full agent graph *without* calling the model or the network, so they run for free in CI — and they assert the research subagent appears only with a Tavily key, memory survives a save/load round-trip, and every `SKILL.md` is well-formed.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs all three gates on every push to `main` and every PR: `uv sync --locked` (so a `pyproject.toml` edit with a stale lockfile turns the run red), then `pytest` and `ty check` across Python 3.11/3.12/3.13, plus `ruff check` once — about 25s end to end. **No API key is configured in the workflow** — that's deliberate, and it turns "building the agent never touches the network" from a claim in the docs into something CI would fail on. The checks report status; they aren't wired to branch protection, so nothing is blocked on them.

---

## License

Released under the [MIT License](LICENSE) — © 2026 Daryl Lim.
