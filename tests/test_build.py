"""Offline tests: everything here runs without an API key or network.

Constructing a Deep Agent does not call the model, so we can assert the whole graph
wires up, the research subagent toggles on the Tavily key, memory survives a
save/load round-trip, and every SKILL.md is well-formed — all in CI, for free.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import yaml

from speechwriter import config, memory
from speechwriter.agent import _write_sandbox, build_agent
from speechwriter.config import load_settings
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
