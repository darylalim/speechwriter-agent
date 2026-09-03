"""Offline tests: everything here runs without an API key or network.

Constructing a Deep Agent does not call the model, so we can assert the whole graph
wires up, the research subagent toggles on the Tavily key, memory survives a
save/load round-trip, and every SKILL.md is well-formed — all in CI, for free.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tomllib
import uuid

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

import speechwriter
from speechwriter import config, memory, prompts
from speechwriter.agent import _build_backend, _build_model, _write_sandbox, build_agent
from speechwriter.config import load_settings
from speechwriter.observability import TruncationWarner
from speechwriter.subagents import build_subagents


def test_agent_builds_without_research(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    settings = load_settings()
    assert settings.research_enabled is False
    assert [sa["name"] for sa in build_subagents(settings)] == ["style-critic"]

    bundle = build_agent(settings)
    assert bundle.agent.__class__.__name__ == "CompiledStateGraph"
    assert bundle.settings.model == "claude-sonnet-5"


def test_research_subagent_appears_with_tavily(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    settings = load_settings()
    assert settings.research_enabled is True
    assert [sa["name"] for sa in build_subagents(settings)] == ["researcher", "style-critic"]


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    assert load_settings().model == "claude-opus-4-8"


def test_memory_snapshot_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    store = memory.load_store(settings)  # starts empty
    store.put(("voice_profiles",), "mayor.md", {"content": "warm, plainspoken"})
    assert memory.save_store(store, settings) == 1
    assert settings.store_path.exists()

    reloaded = memory.load_store(settings)
    item = reloaded.get(("voice_profiles",), "mayor.md")
    assert item is not None
    assert item.value == {"content": "warm, plainspoken"}


def test_memory_roundtrip_beyond_search_limit(monkeypatch, tmp_path):
    # Regression: save_store must page past the Store's default search limit (10) and
    # list_namespaces limit (100), or profiles beyond those bounds are silently lost.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    store = memory.load_store(settings)
    for i in range(25):
        store.put(("speechwriter", "memories"), f"speaker-{i:02d}.md", {"content": f"v{i}"})
    assert memory.save_store(store, settings) == 25

    reloaded = memory.load_store(settings)
    got = memory._all_items(reloaded, ("speechwriter", "memories"))
    assert len(got) == 25
    assert {item.value["content"] for item in got} == {f"v{i}" for i in range(25)}


def test_corrupt_snapshot_is_quarantined_not_clobbered(monkeypatch, tmp_path):
    # Invalid JSON: must not crash, must move the bad file aside (never overwrite it).
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    settings.store_path.write_text("{not valid json", encoding="utf-8")

    store = memory.load_store(settings)
    assert list(store.list_namespaces()) == []
    assert not settings.store_path.exists()  # moved aside
    backup = settings.store_path.with_name(settings.store_path.name + ".corrupt")
    assert backup.exists() and backup.read_text(encoding="utf-8") == "{not valid json"


def test_wrong_shape_snapshot_is_quarantined(monkeypatch, tmp_path):
    # Valid JSON but wrong shape (object, not list of records): must degrade, not crash.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    settings.store_path.write_text(json.dumps({"oops": "not a list"}), encoding="utf-8")

    store = memory.load_store(settings)
    assert list(store.list_namespaces()) == []
    assert settings.store_path.with_name(settings.store_path.name + ".corrupt").exists()


def test_bad_int_env_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MAX_RESEARCH_RESULTS", "ten")
    assert load_settings().max_research_results == 5  # default, no crash


def test_max_tokens_env_is_an_optional_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    assert load_settings().max_tokens is None  # unset: defer to the model's own profile

    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "8000")
    assert load_settings().max_tokens == 8000

    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "loads")
    assert load_settings().max_tokens is None  # bad value, no crash


def test_max_tokens_rejects_out_of_range_values(monkeypatch, tmp_path):
    # A zero or negative ceiling is accepted by init_chat_model without complaint and only
    # fails at the first API call, with an opaque provider error far from the typo that
    # caused it — so it must be rejected at load time, not forwarded to the client.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    for bad in ("0", "-5"):
        monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", bad)
        assert load_settings().max_tokens is None, f"{bad} must not reach the model"
        assert getattr(_build_model(load_settings()), "max_tokens", None) != int(bad)


def test_ceiling_resolution_is_three_tier(monkeypatch, tmp_path):
    # Regression, both directions. A bare model string lets init_chat_model take max_tokens
    # from LangChain's profile table, which silently falls back to 4096 for an id it cannot
    # profile — and extended thinking bills against that same ceiling, so a subagent can
    # spend the whole budget thinking and emit no text, which deepagents forwards as an
    # empty status="success" task result. But a blunt constant must not *lower* a model
    # LangChain does know: capping Opus at 32k would be the same mistake inverted.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    # Tier 2: a profiled model keeps its own, larger ceiling.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-4-8")
    assert getattr(_build_model(load_settings()), "max_tokens", 0) > config.DEFAULT_MAX_TOKENS

    # Tier 3: an unprofiled id gets our floor, never init_chat_model's 4096.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")
    resolved = getattr(_build_model(load_settings()), "max_tokens", None)
    assert resolved == config.DEFAULT_MAX_TOKENS

    # Tier 1: an explicit override beats both.
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "4242")
    for model_id in ("claude-opus-4-8", "claude-not-a-real-model-9"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        assert getattr(_build_model(load_settings()), "max_tokens", None) == 4242


def test_unprofiled_model_id_warns(monkeypatch, tmp_path, caplog):
    # A model id LangChain cannot profile must not degrade silently. Uses a fabricated id
    # so the test keeps meaning once the real ids gain profiles.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")

    with caplog.at_level(logging.WARNING, logger="speechwriter.agent"):
        _build_model(load_settings())

    assert "model profile" in caplog.text


def test_default_model_still_resolves_through_tier_two(monkeypatch, tmp_path):
    # A tripwire on someone else's data, deliberately. `test_ceiling_resolution_is_three_tier`
    # proves tier 2 works, but it proves it through `claude-opus-4-8` — so the day LangChain
    # stops profiling DEFAULT_MODEL, every other assertion in this file still passes while the
    # default configuration quietly drops to the 32k floor. Nothing would surface it: falling
    # back is *correct* behaviour, just four times smaller, and `ceiling_label` is the only
    # place it shows.
    #
    # This is the one assertion here about a third-party table rather than about our own code,
    # which is the point: that table is the input the whole ceiling story rests on, it moves on
    # langchain-anthropic's schedule rather than ours, and it has already moved once — the id
    # was unprofiled when the three tiers were designed, which is why the notes describing them
    # went stale. A failure is not a bug. It is a prompt to re-read the ceiling notes in
    # CLAUDE.md and config.py, and to decide whether to pin SPEECHWRITER_MAX_TOKENS.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    model = _build_model(load_settings())

    assert getattr(model, "profile", None) is not None, (
        f"LangChain no longer profiles {config.DEFAULT_MODEL}, so the default configuration "
        f"now resolves through tier 3 to the {config.DEFAULT_MAX_TOKENS}-token floor."
    )
    # Not implied by the line above: a profile *below* the floor would keep tier 2 and leave
    # the default running under the ceiling an unprofiled id would have been given.
    resolved = getattr(model, "max_tokens", 0)
    assert resolved > config.DEFAULT_MAX_TOKENS, (
        f"{config.DEFAULT_MODEL} profiles at {resolved}, at or below the "
        f"{config.DEFAULT_MAX_TOKENS} floor — re-check the figure CLAUDE.md quotes."
    )


def test_payload_omits_parameters_current_models_reject(monkeypatch, tmp_path):
    # temperature/top_p/top_k are rejected outright (400) on claude-opus-5, claude-sonnet-5,
    # and claude-opus-4-8. `build_agent()` never touches the wire — that is what makes this
    # suite free — so nothing else here would notice a langchain-anthropic bump that began
    # sending one by default; instead every real turn would fail, far from the upgrade that
    # caused it. `_get_request_payload` builds the dict offline, so the seam is assertable for
    # free. It is private, like the other LangChain/deepagents internals this file reaches
    # into: a rename breaks this test loudly, which is the failure mode we want.
    rejected = {"temperature", "top_p", "top_k"}
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    for model_id in (config.DEFAULT_MODEL, "claude-opus-5", "claude-opus-4-8"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        # Every ceiling branch, since a stray default could be injected on either call:
        # None exercises the profile/floor path, the override exercises tier 1.
        for override in (None, "128000"):
            if override is None:
                monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
            else:
                monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", override)
            model = _build_model(load_settings())
            # Narrows `BaseChatModel` for the type checker, and pins the client type while
            # we are here: `settings.model` is free-form, so that is worth asserting too.
            assert isinstance(model, ChatAnthropic)
            payload = model._get_request_payload([])
            assert rejected.isdisjoint(payload), f"{model_id} (override={override}): {payload}"


def test_truncation_warner_counts_ceiling_stops():
    # A response cut off at the token ceiling is reported only via stop_reason; nothing
    # raises, so a clipped critique otherwise looks exactly like a finished one.
    warner = TruncationWarner()

    def response(stop_reason: str) -> LLMResult:
        message = AIMessage(content="...", response_metadata={"stop_reason": stop_reason})
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    warner.on_llm_end(response("end_turn"), run_id=uuid.uuid4())
    assert warner.truncated == 0

    warner.on_llm_end(response("max_tokens"), run_id=uuid.uuid4())
    assert warner.truncated == 1

    warner.reset()
    assert warner.truncated == 0


def test_truncation_warner_is_provider_agnostic():
    # SPEECHWRITER_MODEL is free-form and init_chat_model infers the provider from it, so
    # matching only Anthropic's `stop_reason` would silently switch detection off for any
    # other provider — reinstating the exact bug this warner exists to catch.
    warner = TruncationWarner()

    def response(metadata: dict[str, str]) -> LLMResult:
        message = AIMessage(content="...", response_metadata=metadata)
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    warner.on_llm_end(response({"finish_reason": "length"}), run_id=uuid.uuid4())
    warner.on_llm_end(response({"stop_reason": "MAX_TOKENS"}), run_id=uuid.uuid4())
    assert warner.truncated == 2  # OpenAI-style, and case-insensitive (Gemini shouts)

    warner.on_llm_end(response({"finish_reason": "stop"}), run_id=uuid.uuid4())
    assert warner.truncated == 2  # a normal completion must not count


def test_bundle_owns_the_truncation_warner(monkeypatch, tmp_path):
    # Observability belongs to the bundle for the same reason persist() does: a consumer
    # invoking bundle.agent directly — the path the README documents — must not silently
    # lose truncation reporting just because the CLI is not involved.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    bundle = build_agent()

    config = bundle.turn_config("thread-1")
    assert config["configurable"]["thread_id"] == "thread-1"

    callbacks = config["callbacks"]
    assert isinstance(callbacks, list)  # narrows the RunnableConfig union
    assert bundle.warner in callbacks


def test_write_sandbox_confines_writes(monkeypatch, tmp_path):
    from deepagents.middleware.filesystem import _check_fs_permission

    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    rules = _write_sandbox(load_settings())

    assert _check_fs_permission(rules, "write", "/workspace/speeches/t.md") == "allow"
    assert _check_fs_permission(rules, "write", "/memories/mayor.md") == "allow"
    assert _check_fs_permission(rules, "write", "/src/speechwriter/agent.py") == "deny"
    assert _check_fs_permission(rules, "write", "/pyproject.toml") == "deny"
    # Reads stay open so skills and reference material still load.
    assert _check_fs_permission(rules, "read", "/src/speechwriter/agent.py") == "allow"


def test_import_speechwriter_is_lazy():
    # `import speechwriter` must not pull in the heavy agent stack (deepagents).
    script = (
        "import sys, speechwriter\n"
        "assert 'deepagents' not in sys.modules, 'deepagents imported eagerly'\n"
        "_ = speechwriter.build_agent\n"  # now triggers the lazy import
        "assert 'deepagents' in sys.modules, 'lazy build_agent did not import'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_env_example_documents_every_setting():
    # `.env.example` is the template users copy to `.env`, so a knob added to config.py but
    # never documented there is invisible to anyone setting the project up. Nothing else
    # keeps the pair in sync — the README table is maintained separately and has drifted
    # before. Presence anywhere in the file counts: `.env.example` deliberately ships
    # optional settings commented out.
    config_src = (config._PKG_DIR / "config.py").read_text(encoding="utf-8")
    documented = (config._PKG_DIR.parents[1] / ".env.example").read_text(encoding="utf-8")

    # Whole-word matching on both sides, then a set difference. A plain substring test
    # would report SPEECHWRITER_MAX_TOKENS as documented when `.env.example` mentions only
    # SPEECHWRITER_MAX_TOKENS_EXTRA — a false pass on exactly the drift this test exists
    # to catch. (The regex still sees names mentioned only in prose; that errs toward
    # demanding documentation, which is the safe direction.)
    names = re.compile(r"\bSPEECHWRITER_[A-Z_]+\b")
    read_by_config = set(names.findall(config_src))
    assert read_by_config, "expected config.py to reference at least one SPEECHWRITER_* var"

    missing = sorted(read_by_config - set(names.findall(documented)))
    assert not missing, f".env.example does not document: {', '.join(missing)}"


def test_all_skills_have_valid_frontmatter():
    skills_dir = config._PKG_DIR.parents[1] / "skills"
    skill_dirs = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    assert len(skill_dirs) == 4

    required_sections = ["## Overview", "## When to Use", "## Instructions", "## Pitfalls"]
    for d in skill_dirs:
        text = (d / "SKILL.md").read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        assert match, f"{d.name} is missing a YAML frontmatter block"

        meta = yaml.safe_load(match.group(1))
        assert meta["name"] == d.name, f"{d.name} frontmatter name must match its slug"
        assert meta.get("description"), f"{d.name} needs a description"

        body = text[match.end() :]
        for section in required_sections:
            assert section in body, f"{d.name} is missing '{section}'"


# Backticked tokens shaped like an identifier: lowercase, no slash, dot, angle bracket or
# space — so virtual paths (`/memories/`), filename placeholders (`<slug>.md`) and markers
# (`[VERIFY]`) fall out, and only tool- and subagent-shaped names survive.
_BACKTICKED_IDENT = re.compile(r"`([a-z][a-z0-9_-]*)`")

# Backticked identifiers in the prompts that are deliberately NOT capabilities. Empty today,
# and that is the point: a new backticked word forces a conscious choice — bind the tool, or
# declare the word prose. Silence is exactly what let `write_todos` sit in the orchestrator
# prompt from the day it was written.
_NON_TOOL_BACKTICKS: frozenset[str] = frozenset()


def _model_bound_tools(monkeypatch, settings) -> set[str]:
    """Tool names the orchestrator's model is actually offered, captured without the wire.

    `bind_tools` fires when the graph *steps*, not when it is built, so this runs a single
    turn against a stub that records the tool list and then ends the turn. Nothing reaches
    the network, so the offline invariant holds.

    The compiled graph's own `nodes["tools"].tools_by_name` would be cheaper and needs no
    stub, but it is a superset: it carries `execute`, which middleware strips before the
    model ever sees it. Asserting against that list would let a prompt advertise a tool the
    model cannot call — precisely the bug this guards.
    """
    captured: set[str] = set()

    class _Recorder(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "recorder"

        def bind_tools(self, tools, **kwargs):
            for tool in tools:
                name = getattr(tool, "name", None)
                captured.add(name if name else tool["name"])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            # No tool calls, so the turn ends after this one step.
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr("speechwriter.agent._build_model", lambda _s: _Recorder())
    bundle = build_agent(settings)
    bundle.agent.invoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        config={"configurable": {"thread_id": "tool-surface"}},
    )
    assert captured, "capture stub never ran — bind_tools was not called"
    return captured


def test_prompts_only_advertise_tools_the_model_can_call(monkeypatch, tmp_path):
    # The orchestrator prompt spent its whole life telling the model to use its "planning
    # tool (`write_todos`)" — a tool `create_deep_agent` has never bound. Nothing caught it:
    # `ty` checks Python and not prose, `ruff` checks syntax, and no test compared the
    # rendered prompt against the tool surface. This is the prompt<->tools twin of
    # test_prompt_points_the_agent_at_the_folder_the_browser_reads, which guards
    # prompt<->workspace for the same reason: two sources of truth, nothing enforcing them.
    #
    # A miss here is invisible at runtime too. The model is told it has a capability, then
    # either emits a call that comes back an error or silently drops the instruction — and
    # `status="success"` on the surrounding turn either way.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dummy")  # both subagents present
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()

    subagents = build_subagents(settings)
    # Everything a prompt may legitimately name: tools bound to the orchestrator, the
    # subagents reachable via `task`, and each subagent's own explicit tools. Subagents run
    # the same filesystem middleware, so the orchestrator's file tools cover them too.
    vocabulary = (
        _model_bound_tools(monkeypatch, settings)
        | {sa["name"] for sa in subagents}
        | {tool.name for sa in subagents for tool in sa.get("tools", [])}
        | _NON_TOOL_BACKTICKS
    )
    assert "task" in vocabulary, f"expected the delegation tool in {sorted(vocabulary)}"

    advertised: set[str] = set()
    for label, text in (
        ("orchestrator", prompts.orchestrator_prompt(settings)),
        ("researcher", prompts.researcher_prompt(settings)),
        ("style-critic", prompts.critic_prompt(settings)),
    ):
        named = set(_BACKTICKED_IDENT.findall(text))
        advertised |= named
        unknown = sorted(named - vocabulary)
        assert not unknown, (
            f"{label} prompt advertises {unknown}, which is neither a bound tool, a "
            f"subagent, nor listed in _NON_TOOL_BACKTICKS — bind it, rename it, or "
            f"declare it prose."
        )

    # Anti-vacuity canary. Every assertion above is satisfied by an empty match set, so a
    # broken pattern or a prompt rewrite that drops backticks would turn this test green
    # while checking nothing. The orchestrator names its delegation tool, so `task` is the
    # one identifier that must survive extraction.
    assert "task" in advertised, (
        f"extracted {sorted(advertised)} from the prompts — expected `task`. The pattern "
        f"has stopped matching, so this test is no longer checking anything."
    )


# --- The path-agreement invariant -------------------------------------------------------
#
# `config.Settings` is the single source for the three virtual paths, and four consumers
# must agree with it: the backend routes, the write sandbox, the prompt text, and README's
# routing table. Nothing structural enforced that agreement, so it lived as an advisory
# Claude Code hook that restated CLAUDE.md at edit time. A hint is a suggestion; these are
# a gate, and unlike the hook they also run in CI and for contributors not using Claude
# Code. `test_write_sandbox_confines_writes` already covers the sandbox consumer.


def test_backend_routes_memories_to_the_store_and_everything_else_to_disk(monkeypatch, tmp_path):
    # Consumer 1 of the path invariant. The route lives in `_build_backend`, and the whole
    # point of `/memories/` is that it is intercepted *before* disk — so this asserts the
    # behaviour, not just the route key. A route that drifted from `settings.memories_vpath`
    # would silently fall through to the default FilesystemBackend and start writing voice
    # profiles into a real `memories/` folder that no snapshot ever persists.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    store = memory.load_store(settings)
    backend = _build_backend(settings, store)

    assert set(backend.routes) == {settings.memories_vpath}, (
        f"backend routes {sorted(backend.routes)} but Settings says "
        f"{settings.memories_vpath!r} — propagate the path change into agent.py."
    )

    backend.write(f"{settings.memories_vpath}mayor.md", "prefers short sentences")
    backend.write(f"{settings.workspace_vpath}/{config.SPEECHES_SUBDIR}/toast.md", "# Toast")

    # Intercepted: it reached the Store and never became a real directory.
    assert [item.key for item in memory.all_items(store)] == ["/mayor.md"]
    assert not (settings.project_root / "memories").exists(), (
        "/memories/ fell through to the FilesystemBackend and hit real disk"
    )
    # Not intercepted: drafts are real files the user can open.
    draft = settings.workspace_dir / config.SPEECHES_SUBDIR / "toast.md"
    assert draft.read_text(encoding="utf-8") == "# Toast"


def test_every_virtual_path_reaches_the_prompts(monkeypatch, tmp_path):
    # Consumer 3. `test_prompt_points_the_agent_at_the_folder_the_browser_reads` covers the
    # workspace path because the browser reads it back; the other two had no such second
    # reader, so a renamed skills or memories directory would leave the agent instructed to
    # read and write somewhere that no longer exists — and the sandbox would deny the write.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    rendered = {
        "orchestrator": prompts.orchestrator_prompt(settings),
        "researcher": prompts.researcher_prompt(settings),
        "style-critic": prompts.critic_prompt(settings),
    }
    orchestrator = rendered["orchestrator"]

    for label, vpath in (
        ("memories", settings.memories_vpath),
        ("skills", settings.skills_vpath),
        ("workspace", settings.workspace_vpath),
    ):
        assert vpath in orchestrator, (
            f"the orchestrator prompt never names {label}_vpath ({vpath!r}); config.py is "
            f"the single source, so a path change must be propagated into prompts.py."
        )

    # The trailing-slash asymmetry is deliberate, and normalising it for tidiness is the
    # documented way to break this: prompts.py renders `{workspace_vpath}/speeches/`, so a
    # trailing slash on workspace_vpath silently yields `/workspace//speeches/`.
    assert settings.memories_vpath.endswith("/")
    assert settings.skills_vpath.endswith("/")
    assert not settings.workspace_vpath.endswith("/"), (
        "workspace_vpath must not carry a trailing slash — prompts.py appends its own."
    )
    for label, text in rendered.items():
        assert "//" not in text, f"{label} prompt contains a doubled slash: {text!r}"


def test_readme_routing_table_matches_the_configured_paths(monkeypatch, tmp_path):
    # Consumer 4, and the one nothing else could ever catch: README's routing table
    # hard-codes all three virtual paths as prose. It is the first thing a reader meets, so
    # a stale table misdescribes the central design decision of the project.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    readme = (config._PKG_DIR.parents[1] / "README.md").read_text(encoding="utf-8")

    for label, vpath in (
        ("memories", settings.memories_vpath),
        ("skills", settings.skills_vpath),
        ("workspace", settings.workspace_vpath),
    ):
        assert vpath in readme, (
            f"README.md's routing table does not mention {label}_vpath ({vpath!r}) — it is "
            f"an undeclared fourth consumer of config.Settings and has gone stale."
        )


def test_orchestrator_prompt_names_every_skill():
    # Not a path consumer, but the same class of drift and the other half of what the
    # advisory hook used to say. Skills are progressive-disclosure: the agent only reads a
    # SKILL.md if it knows the skill exists, and step 4 of the operating rhythm is the only
    # place the library is enumerated. A skill absent from that list is dead weight on disk.
    #
    # Slugs are matched in their prose form, which is the convention the prompt already
    # uses: `delivery-and-cadence` is written "delivery & cadence". If a new skill does not
    # fit that shape, name it in the prompt however reads best and widen this normalisation.
    skill_dirs = sorted(
        p.name for p in (config._PKG_DIR.parents[1] / "skills").iterdir() if p.is_dir()
    )
    assert skill_dirs, "no skills found — this test would otherwise pass vacuously"

    text = prompts.orchestrator_prompt(load_settings())
    for slug in skill_dirs:
        label = slug.replace("-and-", " & ").replace("-", " ")
        assert label in text, (
            f"skills/{slug}/ exists but the orchestrator prompt never mentions {label!r}, "
            f"so the agent will never know to load it. Add it to the parenthetical list in "
            f"step 4 of orchestrator_prompt()."
        )


def test_package_version_matches_pyproject():
    # The version is two independent literals — `[project] version` in pyproject.toml and
    # `__version__` in src/speechwriter/__init__.py — and nothing structural ties them:
    # hatchling builds from the first, `import speechwriter` reports the second.
    #
    # .github/workflows/release.yml watches the pyproject one and tags + publishes a GitHub
    # Release unattended the moment it changes, so a half-done bump would ship a release
    # whose installed package still reports the previous version. This test is what makes
    # running that workflow without a human safe: it runs inside the release job, before
    # anything is tagged, and a mismatch stops the release rather than publishing it.
    root = config._PKG_DIR.parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]

    assert declared == speechwriter.__version__, (
        f"pyproject.toml declares version {declared!r} but speechwriter.__version__ is "
        f"{speechwriter.__version__!r}. Bump both together — release.yml would otherwise tag "
        f"v{declared} for a build that reports {speechwriter.__version__!r}."
    )


def test_load_settings_reopens_the_langsmith_env_cache(monkeypatch, tmp_path):
    # langsmith memoises env reads in an `lru_cache` on `get_env_var`, so the first read of
    # LANGSMITH_TRACING sticks for the life of the process. A value that exists only in the
    # dotenv is therefore invisible to anything that read tracing state earlier — permanently,
    # and with nothing raised. Tracing simply never happens while every setting still looks
    # correct, which is indistinguishable from a LangSmith project whose traces aged out.
    #
    # `load_settings()` clears that cache right after `load_dotenv`, which is what makes the
    # read *order* irrelevant. That placement is the point: the CLI, the Streamlit app and any
    # library consumer all route through `load_settings()`, so none of them can reintroduce the
    # hazard by importing something that touches langsmith at module scope.
    from langsmith.utils import get_env_var

    # `@overload` stubs on get_env_var shadow the lru_cache wrapper, so whether a type checker
    # can see cache_clear depends on the checker's version: ty 0.0.78 reports
    # unresolved-attribute, while ty 0.0.63 — the pin in ci.yml — resolves it and then flags the
    # suppression itself as an unused ignore. A direct access needs a `ty: ignore` that is
    # correct under exactly one of them; going through getattr needs none and agrees with both.
    # The assert keeps the loud failure a bare attribute access would have given, and says more.
    cache_clear = getattr(get_env_var, "cache_clear", None)
    assert cache_clear is not None, (
        "langsmith.utils.get_env_var no longer exposes cache_clear, so the call in "
        "load_settings() is now a silent no-op and tracing config can go missing again."
    )

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("LANGSMITH_TRACING=true\n", encoding="utf-8")

    monkeypatch.setenv("SPEECHWRITER_HOME", str(home))
    # Recorded so monkeypatch's undo also removes whatever load_dotenv sets below; real shell
    # env wins over the dotenv, so an inherited value would otherwise decide this test.
    for var in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(var, raising=False)

    try:
        # Canary: the caching hazard is still live, so the cache_clear() is still load-bearing.
        # Asserted against `get_env_var` itself rather than through `tracing_is_enabled()`,
        # which returns early on three context-var paths before it ever consults the cache — a
        # future default there would let this test pass while exercising nothing.
        cache_clear()
        assert get_env_var("TRACING", default="") == ""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        assert get_env_var("TRACING", default="") == "", (
            "langsmith no longer caches get_env_var, so the cache_clear() in load_settings() "
            "guards nothing. Re-read the comment there and decide whether to drop it — this is "
            "an assertion about a dependency's behaviour, not a bug to fix here."
        )
        monkeypatch.delenv("LANGSMITH_TRACING")

        # The invariant: a read that lands before the dotenv must not outlive load_settings().
        cache_clear()
        assert get_env_var("TRACING", default="") == ""  # poisoned, exactly as an early import
        load_settings()

        assert get_env_var("TRACING", default="") == "true", (
            "load_settings() left a stale 'tracing off' cached even though the dotenv sets "
            "LANGSMITH_TRACING=true. Its cache_clear() after load_dotenv is what repairs an "
            "early read — without it every turn runs untraced and the LangSmith project stays "
            "silently empty."
        )
    finally:
        # Never leak this test's env into the cache the rest of the suite reads.
        cache_clear()
