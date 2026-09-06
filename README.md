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

Set your keys in `.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-...     # required
TAVILY_API_KEY=tvly-...          # optional — enables live web research
```

Without a Tavily key the agent still works; it writes from its own knowledge and marks anything it can't verify with `[VERIFY]`. With one, a `researcher` subagent pulls current, sourced facts.

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

A two-page web app reading the same `.env` — no separate configuration:

- **Write** — commission a speech and watch the agent plan, research, draft, and self-critique in a live activity log; each finished turn is snapshotted immediately (a closed tab runs no shutdown hook, so waiting until exit would usually mean never).
- **Workspace** — browse saved drafts (with a spoken-length estimate), research notes, and the voice profiles the agent has learned, read straight from the live Store.

It binds to `localhost` only by default; the agent spends your API budget and reads your workspace, so it is not meant to face the network. Override with `--server.address` if you genuinely intend to share it.

### Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `SPEECHWRITER_MODEL` | `claude-sonnet-5` | Any Claude model id — `claude-opus-5` for top quality, with no ceiling override needed alongside it (LangChain profiles it at its real 128k). |
| `SPEECHWRITER_MAX_TOKENS` | model's own profile | Overrides the output-token ceiling. Unset, a model LangChain can profile keeps its own ceiling — as of the pinned `langchain-anthropic` every shipped Claude id is profiled at 64k–128k, so the default resolves to **128000**. An id it *cannot* profile (a typo, or one newer than the pin) would silently inherit 4096, so it gets 32000 instead plus a warning. Extended thinking bills against the same ceiling, which is why 4096 is not enough. |
| `SPEECHWRITER_BASE_URL` | — | Point the agent at an OpenAI-compatible endpoint instead of Anthropic — a local `mlx_lm.server`, vLLM, LM Studio, Ollama. Set it and no `ANTHROPIC_API_KEY` is required. See [Running a local model](#running-a-local-model). |
| `OPENAI_API_KEY` | — | Sent to that endpoint. Local servers ignore it, so it is optional; a hosted OpenAI-compatible service will need a real one. |
| `SPEECHWRITER_MAX_RESEARCH_RESULTS` | `5` | Tavily results per query. |
| `SPEECHWRITER_HOME` | repo root | Root dir the agent reads/writes under. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | — | Optional [LangSmith](https://docs.langchain.com/langsmith/home) tracing. |

### Running a local model

The agent can run entirely on your machine — no API key, no per-token cost, nothing leaving
the laptop. Any OpenAI-compatible server works; on Apple Silicon, [MLX](https://github.com/ml-explore/mlx-lm) is the fastest path:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.8-27B-4bit --port 8080
```

Then point the agent at it:

```ini
SPEECHWRITER_BASE_URL=http://127.0.0.1:8080/v1
SPEECHWRITER_MODEL=mlx-community/Qwen3.8-27B-4bit
```

`SPEECHWRITER_BASE_URL` selects the *client*, not just the id — a local model name carries no
provider prefix for LangChain to infer, so the endpoint is what makes the choice unambiguous.
Nothing else changes: the same graph, subagents, skills, sandbox, and memory.

Two things worth knowing:

- **The ceiling resolves through tier 3.** A locally served id has no LangChain profile, so it
  takes the 32000-token floor. That is the wanted answer here, not a fallback — `ChatOpenAI`'s
  own default is "let the server decide", which on a reasoning model is an unbounded thinking
  budget.
- **Reasoning effort is worth tuning.** Qwen3.8's chat template defaults to
  `reasoning_effort: xhigh`, which spends ~1400 tokens deliberating before it writes a line.
  For prose, `low` is both faster and better; pass it via the server's
  `chat_template_args`.

Sizing, on 32GB unified memory: the 4-bit 27B weighs 15GB on disk and peaks at ~15.5GB
resident, generating ~21 tok/s on an M2 Max — comfortably inside the ~24GB macOS allows the
GPU by default, with headroom left for the KV cache.

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
├── agent.py       build_agent() — composes every layer into one graph
├── cli.py         Rich streaming REPL
├── workspace.py   UI-free reader: drafts, research notes, voice profiles
└── webui.py       Streamlit glue: stream a turn, record it, replay it
streamlit_app.py   Web entry point (router) + app_pages/ (Write, Workspace)
skills/            On-demand rhetoric library (SKILL.md, progressive disclosure)
├── rhetorical-devices/     delivery-and-cadence/
├── speech-structures/      audience-and-occasion/
tests/             Offline tests — build the graph, toggle research, round-trip memory,
                   render both pages headlessly (all without the model or network)
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
