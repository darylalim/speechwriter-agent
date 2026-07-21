# speechwriter-agent

A speechwriter built with **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** (LangChain + LangGraph). You describe the speaker, audience, occasion, and goal; the agent plans the work, researches facts, drafts a speech written *for the ear*, critiques its own draft, revises, and **remembers how each speaker sounds** across sessions.

```
you › Write a 4-minute wedding toast. Speaker: David, best man. Audience: 80 guests,
      mixed ages. Couple: Ana & Priya, met hiking. Warm, a little funny, no clichés.

  ⚙  write_todos       [{"content":"Intake + recall David's voice","status":"in_progress"}, …]
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

A speech is a **long-horizon** task: intake → research → outline → draft → critique → revise. That maps cleanly onto what the Deep Agents harness provides out of the box, so this project mostly *configures* capabilities rather than implementing them:

| Speechwriting need | Deep Agents primitive | Where it lives |
|---|---|---|
| Break the commission into stages | `write_todos` planning (built in) | automatic |
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

```bash
uv run speechwriter          # or:  uv run python -m speechwriter
```

Then just talk to it. Give it as much of the brief as you can — the agent will ask for anything essential it's missing:

> Draft a 12-minute commencement address for a state university. Speaker is a first-gen
> founder. One big idea: "usefulness beats prestige." Warm, story-driven, one good laugh.

- Finished speeches are saved to `workspace/speeches/` as Markdown.
- Research notes land in `workspace/research/`.
- Type `exit` (or `Ctrl-D`) to quit — voice-profile memory is snapshotted on the way out.

### Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `SPEECHWRITER_MODEL` | `claude-sonnet-5` | Any Claude model id (`claude-opus-4-8` for top quality). |
| `SPEECHWRITER_MAX_RESEARCH_RESULTS` | `5` | Tavily results per query. |
| `SPEECHWRITER_HOME` | repo root | Root dir the agent reads/writes under. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | — | Optional [LangSmith](https://docs.langchain.com/langsmith/home) tracing. |

---

## Using it as a library

The agent is a plain compiled LangGraph graph:

```python
from speechwriter import build_agent

bundle = build_agent()                       # SpeechwriterAgent(agent, store, settings)
config = {"configurable": {"thread_id": "demo"}}
result = bundle.agent.invoke(
    {"messages": [{"role": "user", "content": "Write a 2-minute retirement toast for Sam."}]},
    config=config,
)
print(result["messages"][-1].content)
```

---

## Project layout

```
src/speechwriter/
├── config.py      Settings: model, keys, virtual paths (single source of truth)
├── prompts.py     Orchestrator + researcher + critic system prompts
├── tools.py       Lazy Tavily research tool (degrades gracefully with no key)
├── subagents.py   researcher + style-critic SubAgent definitions
├── memory.py      Persistent Store: JSON snapshot load/save
├── agent.py       build_agent() — composes every layer into one graph
└── cli.py         Rich streaming REPL
skills/            On-demand rhetoric library (SKILL.md, progressive disclosure)
├── rhetorical-devices/     delivery-and-cadence/
├── speech-structures/      audience-and-occasion/
tests/             Offline tests — build the graph, toggle research, round-trip memory
```

## Development

```bash
uv run pytest                # offline: no API key or network needed
uvx ruff check . && uvx ruff format .
```

The tests construct the full agent graph *without* calling the model or the network, so they run for free in CI — and they assert the research subagent appears only with a Tavily key, memory survives a save/load round-trip, and every `SKILL.md` is well-formed.
