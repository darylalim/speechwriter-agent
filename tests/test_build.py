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
import uuid

import yaml
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from speechwriter import config, memory
from speechwriter.agent import _build_model, _write_sandbox, build_agent
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
