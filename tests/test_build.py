"""Offline tests: everything here runs without an API key or network.

Constructing a Deep Agent does not call the model, so we can assert the whole graph
wires up, the research subagent toggles on the Tavily key, memory survives a
save/load round-trip, and every SKILL.md is well-formed — all in CI, for free.
"""

from __future__ import annotations

import re

import yaml

from speechwriter import config, memory
from speechwriter.agent import build_agent
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


def test_corrupt_snapshot_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    settings = load_settings()
    settings.store_path.write_text("{not valid json", encoding="utf-8")
    # Should degrade to an empty store rather than raising.
    store = memory.load_store(settings)
    assert list(store.list_namespaces()) == []


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
