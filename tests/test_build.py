"""Offline tests: everything here runs without an API key or network.

Constructing a Deep Agent does not call the model, so we can assert the whole graph
wires up, the research subagent toggles on the Tavily key, memory survives a
save/load round-trip, and every SKILL.md is well-formed — all in CI, for free.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_openai import ChatOpenAI

import speechwriter
from speechwriter import config, endpoints, memory, prompts
from speechwriter.agent import (
    SpeechwriterAgent,
    _build_backend,
    _build_model,
    _write_sandbox,
    build_agent,
)
from speechwriter.config import load_settings
from speechwriter.observability import TruncationWarner
from speechwriter.subagents import build_subagents


def test_agent_builds_without_research(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # SPEECHWRITER_BASE_URL swaps the client for an OpenAI one; a developer who exported
    # it to drive the local model would otherwise silently run this against ChatOpenAI.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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
    # SPEECHWRITER_BASE_URL swaps the client for an OpenAI one; a developer who exported
    # it to drive the local model would otherwise silently run this against ChatOpenAI.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
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
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn this test red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn this test red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn this test red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn this test red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

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


def test_local_endpoint_builds_an_openai_client(monkeypatch, tmp_path):
    # SPEECHWRITER_BASE_URL is the whole switch: it selects the *client*, because a locally
    # served id like "mlx-community/Qwen3.8-27B-4bit" carries no provider prefix that
    # `init_chat_model` could infer. Asserted through `_build_model` rather than on Settings
    # so a branch that forgot to thread the kwargs through a ceiling tier is caught here.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")

    settings = load_settings()
    assert settings.uses_local_endpoint is True
    # The point of the gate: no Anthropic key, yet the configuration is runnable.
    assert settings.anthropic_api_key is None
    assert settings.model_credentials_present is True

    model = _build_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://127.0.0.1:8080/v1"
    assert model.model_name == "mlx-community/Qwen3.8-27B-4bit"


def test_local_endpoint_keeps_the_three_tier_ceiling(monkeypatch, tmp_path):
    # The client swap must not bypass the ceiling logic. A local model has no LangChain
    # profile, so it lands on tier 3 -- and that matters more here than on the Anthropic
    # path: ChatOpenAI's own default is `max_tokens=None`, i.e. "let the server decide",
    # which on a reasoning model that defaults to a high effort level is an unbounded
    # thinking budget rather than a merely small one.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")

    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    assert getattr(_build_model(load_settings()), "max_tokens", None) == config.DEFAULT_MAX_TOKENS

    # Tier 1 still wins, and still reaches the OpenAI client rather than silently
    # constructing an Anthropic one on the override branch.
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "4242")
    overridden = _build_model(load_settings())
    assert isinstance(overridden, ChatOpenAI)
    assert overridden.max_tokens == 4242


def test_local_endpoint_payload_omits_rejected_parameters(monkeypatch, tmp_path):
    # The sibling of test_payload_omits_parameters_current_models_reject, for the second
    # client. Same reasoning, and the same reason it cannot be folded into that test: the
    # assertion there pins `isinstance(model, ChatAnthropic)`, which is exactly what this
    # configuration must *not* be.
    rejected = {"temperature", "top_p", "top_k"}
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")

    for override in (None, "128000"):
        if override is None:
            monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
        else:
            monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", override)
        model = _build_model(load_settings())
        assert isinstance(model, ChatOpenAI)
        assert rejected.isdisjoint(model._get_request_payload([]))


def test_blank_base_url_is_treated_as_unset(monkeypatch, tmp_path):
    # `export SPEECHWRITER_BASE_URL=` is how a shell says "unset". Read with a bare
    # `os.environ.get` that empty string is truthy enough to set the field, and every call
    # would be routed at a nonexistent endpoint while the banner reported a local model.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)

    for blank in ("", "   "):
        monkeypatch.setenv("SPEECHWRITER_BASE_URL", blank)
        settings = load_settings()
        assert settings.base_url is None
        assert settings.uses_local_endpoint is False
        assert isinstance(_build_model(settings), ChatAnthropic)


def test_model_credentials_gate_refuses_only_when_nothing_is_configured(monkeypatch, tmp_path):
    # The CLI raises SystemExit on this and the web UI disables its chat input, so a false
    # negative silently bricks a working setup and a false positive defers the failure to
    # the first turn. Both front ends read this one property; assert it directly.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    assert load_settings().model_credentials_present is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    assert load_settings().model_credentials_present is True


def test_local_endpoint_api_key_falls_back_to_a_placeholder(monkeypatch, tmp_path):
    # ChatOpenAI raises without *a* key, and a local server never reads one -- so the
    # placeholder is what makes the no-credentials-at-all case work at all. A real
    # OPENAI_API_KEY must still win, for a hosted OpenAI-compatible endpoint.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_settings().endpoint_api_key == config.LOCAL_API_KEY_PLACEHOLDER

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert load_settings().endpoint_api_key == "sk-real"


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
    # SPEECHWRITER_BASE_URL swaps the client for an OpenAI one; a developer who exported
    # it to drive the local model would otherwise silently run this against ChatOpenAI.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
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


def test_tool_pins_agree_wherever_they_are_declared():
    # ruff and ty versions are four independent literals — ci.yml, release.yml,
    # .claude/hooks/ruff-ty-gate.sh, and the commands CLAUDE.md tells a human to type — and
    # nothing structural ties them. The same shape as test_package_version_matches_pyproject,
    # and guarded the same way, because the failure modes are all silent.
    #
    # Hook drifting from CI: a type-checker suppression comment is *required* by a checker that
    # cannot resolve a symbol and *rejected* as an unused-ignore by one that can, so a hook and a
    # CI on different versions admit source states that satisfy neither. The gate goes green in
    # the model's context and red on push, with no version named in either message. (Spelling
    # that directive out here would itself be parsed as one — hence the paraphrase.)
    #
    # release.yml drifting from ci.yml: that workflow tags and publishes a GitHub Release
    # unattended off its own gate re-run, which CLAUDE.md calls the one place a failing gate
    # actually stops something. A stale pin there is a release hazard, not a lint annoyance.
    root = config._PKG_DIR.parents[1]
    sites = (
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        ".claude/hooks/ruff-ty-gate.sh",
        "CLAUDE.md",
    )
    # Two spellings, which is what lets a docs file be a site at all. The named form covers the
    # YAML (`TY_VERSION: "0.0.63"`) and shell (`TY_VERSION="0.0.63"`) declarations; the
    # invocation form covers the commands CLAUDE.md tells a human to type (`uvx ty@0.0.63`).
    # Neither matches `uvx ruff@"$RUFF_VERSION"` — the version must start with a digit — so the
    # hook's use sites are read from its declaration, not from themselves.
    named = re.compile(r'\b(RUFF|TY)_VERSION\b\s*[:=]\s*"?([0-9][0-9A-Za-z.\-]*)"?')
    invoked = re.compile(r"\buvx\s+(ruff|ty)@\"?([0-9][0-9A-Za-z.\-]*)\"?")

    found = {}
    for rel in sites:
        text = (root / rel).read_text(encoding="utf-8")
        pins = {}
        for tool, version in named.findall(text) + [
            (t.upper(), v) for t, v in invoked.findall(text)
        ]:
            pins.setdefault(tool, set()).add(version)

        # Anti-vacuity: every assertion below is satisfied by an empty match set, so a site that
        # stopped pinning — or a pattern that stopped matching — would turn this green while
        # checking nothing.
        assert set(pins) == {"RUFF", "TY"}, (
            f"{rel} pins {sorted(pins) or 'nothing'}; expected both ruff and ty. An unpinned "
            f"site can disagree with the others in ways no source state satisfies."
        )
        for tool, versions in pins.items():
            assert len(versions) == 1, (
                f"{rel} names {tool} at {sorted(versions)} — it disagrees with itself, so at "
                f"least one mention was missed when the pin was bumped."
            )
        found[rel] = {tool: next(iter(versions)) for tool, versions in pins.items()}

    for tool in ("RUFF", "TY"):
        declared = {rel: pins[tool] for rel, pins in found.items()}
        assert len(set(declared.values())) == 1, (
            f"{tool} pin disagrees across sites: {declared}. Bump all four together."
        )


def test_a_profiled_id_over_a_local_endpoint_still_gets_a_ceiling(monkeypatch, tmp_path):
    # Regression. Tier 2 used to ask "is there a profile?", which is the same question as
    # "was a ceiling resolved?" for ChatAnthropic and emphatically not for ChatOpenAI:
    # init_chat_model applies a profile's max_tokens only on the Anthropic path. So a
    # *profiled* id served over SPEECHWRITER_BASE_URL -- `gpt-4o` on LM Studio, LiteLLM or a
    # hosted OpenAI-compatible service, all of which the README names -- skipped tier 3 and
    # came back with max_tokens=None: no ceiling at all, which is the unbounded thinking
    # budget tier 3 exists to prevent, and strictly worse than the 4096 it guards against.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "gpt-4o")

    model = _build_model(load_settings())

    assert isinstance(model, ChatOpenAI)
    assert getattr(model, "profile", None) is not None, (
        "gpt-4o is no longer profiled, so this test no longer exercises the branch it guards"
    )
    assert model.max_tokens == config.DEFAULT_MAX_TOKENS


def test_an_unprofiled_anthropic_id_does_not_keep_langchains_4096(monkeypatch, tmp_path):
    # The other half of the same condition, and the reason it cannot be simplified to a bare
    # max_tokens check: ChatAnthropic *always* carries a max_tokens, and for an unprofiled id
    # that value is LangChain's silent 4096 fallback -- the original trap. Asserting the
    # resolved ceiling alone would happily accept it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")

    assert getattr(_build_model(load_settings()), "max_tokens", None) == config.DEFAULT_MAX_TOKENS


def test_the_resolved_ceiling_reaches_the_request_payload(monkeypatch, tmp_path):
    # A ceiling set on the client but dropped from the payload is no ceiling at all, and the
    # two clients do not agree on the key: langchain-openai 1.6 sends `max_completion_tokens`
    # where langchain-anthropic sends `max_tokens`. Assert the *value* is carried under some
    # key rather than pinning either spelling, so a rename upstream fails loudly here instead
    # of silently unbounding a local reasoning model.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "32000")

    for base_url, model_id in (
        (None, config.DEFAULT_MODEL),
        ("http://127.0.0.1:8080/v1", "mlx-community/Qwen3.8-27B-4bit"),
    ):
        if base_url is None:
            monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("SPEECHWRITER_BASE_URL", base_url)
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        model = _build_model(load_settings())
        # Narrows BaseChatModel for the checker, and pins that each branch built the client
        # it was meant to — the payload assertion below is vacuous on the wrong one.
        assert isinstance(model, ChatAnthropic | ChatOpenAI)
        payload = model._get_request_payload([])
        carrying = {key for key, value in payload.items() if value == 32000}
        assert carrying, f"{model_id}: ceiling absent from payload {sorted(payload)}"


def test_a_blank_openai_key_falls_back_to_the_placeholder(monkeypatch, tmp_path):
    # `export OPENAI_API_KEY=` is how a shell says "unset", and an unstripped blank is
    # truthy -- so it would be sent as the Authorization bearer and a hosted endpoint would
    # 401 far from the typo. Same normalisation as base_url, which had it from the start.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    for blank in ("", "   "):
        monkeypatch.setenv("OPENAI_API_KEY", blank)
        assert load_settings().endpoint_api_key == config.LOCAL_API_KEY_PLACEHOLDER


def test_settings_can_still_be_built_without_the_local_model_fields(tmp_path):
    # `build_agent(settings)` is the documented library entry point, so Settings is part of
    # the public surface: adding an optional capability must not break a caller that predates
    # it. Mirrors the defaulting already argued for on SpeechwriterAgent's own added fields.
    settings = config.Settings(
        model=config.DEFAULT_MODEL,
        anthropic_api_key=None,
        tavily_api_key=None,
        project_root=tmp_path,
        workspace_dir=tmp_path,
        skills_dir=tmp_path,
        store_path=tmp_path / "store.json",
        max_research_results=5,
        max_tokens=None,
    )

    assert settings.base_url is None
    assert settings.uses_local_endpoint is False
    # The field the model picker adds, asserted here rather than in a test of its own: this is
    # the standing guard on the constructor's shape, and a required field would break it.
    assert settings.context_window is None


def test_every_anthropic_model_choice_is_profiled_above_the_floor(monkeypatch, tmp_path):
    # The roster is a promise: everything the picker offers keeps its *own* 64k-128k ceiling
    # rather than dropping to the 32k floor. Nothing structural enforces it — MODEL_CHOICES is a
    # hand-written tuple, and `_build_model` resolves a typo'd id through tier 3 with only a log
    # line, so a reader would discover it by watching a draft come back short.
    #
    # Like `test_default_model_still_resolves_through_tier_two`, this asserts against a
    # third-party table rather than against our own code, and for the same reason: that table is
    # the input the roster rests on and it moves on langchain-anthropic's schedule. A failure
    # here is a prompt to re-pick the roster, not a bug to fix.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    # Explicit: SPEECHWRITER_BASE_URL swaps the client for an OpenAI one, so a developer
    # who exported it to drive the local model would otherwise turn this test red.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

    for choice in config.MODEL_CHOICES:
        assert choice.base_url is None, (
            f"{choice.label} names an endpoint, but the curated roster is Anthropic-only — a "
            f"local entry belongs in `model_choices`, synthesised from the environment."
        )
        monkeypatch.setenv("SPEECHWRITER_MODEL", choice.model)
        model = _build_model(load_settings())
        assert getattr(model, "profile", None) is not None, (
            f"LangChain no longer profiles {choice.model!r}, so offering it drops the ceiling "
            f"to the {config.DEFAULT_MAX_TOKENS}-token floor."
        )
        assert getattr(model, "max_tokens", 0) > config.DEFAULT_MAX_TOKENS


def test_the_configured_pair_is_always_offered(monkeypatch, tmp_path):
    # Streamlit *silently* rewrites a selection that is not among a widget's options to option
    # zero — no exception, no log. So a roster that did not contain the configured model would
    # take a reader running a local endpoint and retarget them onto claude-sonnet-5, which on a
    # machine with no Anthropic key flips the whole app into its "no credentials" state.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    settings = load_settings()
    offered = config.model_choices(settings)

    assert offered[: len(config.MODEL_CHOICES)] == config.MODEL_CHOICES
    assert len(offered) == len(config.MODEL_CHOICES) + 1
    assert offered[-1].model == settings.model
    assert settings.base_url is not None
    assert offered[-1].base_url == settings.base_url
    # Built through the one constructor, so an entry discovered from a live endpoint is
    # indistinguishable from this one and the same model cannot appear twice.
    assert offered[-1] == config.local_choice(settings.model, settings.base_url)

    # ...and a configured model already on the roster is not offered a second time.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5")
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    assert config.model_choices(load_settings()) == config.MODEL_CHOICES


def test_switching_away_from_a_local_model_leaves_a_way_back(monkeypatch, tmp_path):
    # The roster must be widened by *every* configuration that has to stay reachable, not only
    # the live one. Selecting a curated entry sets `base_url` to None, so a roster derived from
    # the post-switch settings alone drops the locally served entry the reader came from — and
    # on a keyless machine that is a dead end, because picking a Claude id there also disables
    # the chat input. The way back would have been removed by the act of leaving.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    configured = load_settings()
    # What the bundle's settings look like after picking a curated Anthropic entry.
    switched = replace(configured, model="claude-opus-5", base_url=None, context_window=None)

    offered = config.model_choices(configured, switched)

    assert any(c.base_url == configured.base_url for c in offered), (
        "the locally served entry vanished once a Claude model was selected — there is no way "
        "back to it without restarting the process"
    )
    # And no duplicate for the curated entry, which is now both current and already on the list.
    assert len(offered) == len(config.MODEL_CHOICES) + 1
    assert [c.model for c in offered].count("claude-opus-5") == 1


def test_the_roster_offers_an_off_list_anthropic_model_without_calling_it_local(
    monkeypatch, tmp_path
):
    # A configured id that is neither curated nor locally served — `claude-opus-4-8`, say — is
    # still a pair that must stay selectable, and labelling it "(local)" would be a lie about
    # where it is served.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-4-8")
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

    offered = config.model_choices(load_settings())

    assert offered[-1] == config.ModelChoice("claude-opus-4-8", "claude-opus-4-8")
    assert offered[-1].base_url is None


def test_a_stalling_endpoint_gives_up_rather_than_hanging():
    # The one failure `DEFAULT_TIMEOUT` exists for, and the only one this module cannot shrug
    # off by itself: a server that *accepts* the connection and then never answers. A refused
    # port fails in under a millisecond, but `urlopen` inherits no default timeout at all, so
    # without one this blocks forever — and the caller is a sidebar button.
    #
    # Not mutation-tested by deleting the timeout, for the obvious reason: that mutation does
    # not fail this test, it hangs the suite. The elapsed-time assertion is what stands in.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        started = time.monotonic()
        found = endpoints.list_models(
            f"http://127.0.0.1:{listener.getsockname()[1]}/v1", timeout=0.25
        )
        elapsed = time.monotonic() - started
    finally:
        listener.close()

    assert found == []
    assert elapsed < 5, f"the request ran for {elapsed:.1f}s — the timeout is not being applied"


def test_a_local_choice_compacts_before_its_context_window(monkeypatch, tmp_path):
    # deepagents sizes its context-compaction trigger from the model's profile. An unprofiled
    # id — which every locally served one is — gets a flat ("tokens", 170000) trigger instead,
    # and no local server has a 170k window, so the plan/draft/critique/revise rhythm outgrows
    # the window and the server errors before compaction ever fires. `_build_model` hands the
    # local client a minimal profile so the trigger scales to the window it actually has.
    #
    # Asserted as "a fraction of *this* model's window" rather than as an exact figure: the
    # fraction is deepagents' own default and may move, but a flat token count sized for
    # somebody else's model is the failure, and that is what this catches.
    import deepagents.graph

    triggers = []
    original = deepagents.graph.create_summarization_middleware

    def spy(*args, **kwargs):
        middleware = original(*args, **kwargs)
        # deepagents keeps the resolved trigger on an inner helper, not on the middleware.
        triggers.append(getattr(getattr(middleware, "_lc_helper", middleware), "trigger", None))
        return middleware

    monkeypatch.setattr(deepagents.graph, "create_summarization_middleware", spy)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    settings = load_settings()
    profile = getattr(_build_model(settings), "profile", None) or {}
    assert profile.get("max_input_tokens") == config.DEFAULT_LOCAL_CONTEXT_WINDOW

    build_agent(settings)

    assert triggers, "no summarization middleware was built — the spy is watching the wrong name"
    assert all(t is not None and t[0] == "fraction" for t in triggers), triggers


def test_a_roster_entry_can_override_the_assumed_local_window(monkeypatch, tmp_path):
    # The floor is conservative on purpose, so a server that really does have a larger window
    # must be able to say so per model — and without a new environment variable, which would be
    # a documentation contract this knob does not deserve.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    settings = replace(load_settings(), context_window=131072)
    profile = getattr(_build_model(settings), "profile", None) or {}

    assert profile.get("max_input_tokens") == 131072


def test_switching_models_carries_learned_memory_across_the_rebuild(monkeypatch, tmp_path):
    # A model switch is a *rebuild*, and `build_agent` rehydrates a brand-new InMemoryStore from
    # the on-disk snapshot — so everything learned since the last save is dropped unless
    # `persist()` runs first. Both front ends switch this way. Reversing the two lines loses
    # voice profiles with no error at all, which is why the order is asserted here rather than
    # only commented at the call sites.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    bundle = build_agent(load_settings())
    bundle.store.put(("speechwriter", "memories"), "mayor.md", {"content": "Plain speaker."})

    bundle.persist()
    switched = build_agent(replace(bundle.settings, model="claude-haiku-4-5"))

    assert switched.store is not bundle.store
    assert [item.key for item in memory.all_items(switched.store)] == ["mayor.md"]
    assert switched.settings.model == "claude-haiku-4-5"
    # The rebuild is what makes the switch real: the ceiling has to track the new model, or the
    # banner keeps quoting the previous one's.
    assert switched.max_tokens != bundle.max_tokens


def test_an_oversized_ceiling_override_is_reported_beside_the_label_not_inside_it(
    monkeypatch, tmp_path
):
    # SPEECHWRITER_MAX_TOKENS is tier 1 and global, so an override sized for one model follows a
    # switch to another: 128k asked of Haiku 4.5, whose real ceiling is 64k, is rejected at the
    # first turn — far from the switch that caused it. Only reachable now that the model can be
    # changed without editing the file the override lives in, which is why it is surfaced at all
    # rather than left to a log line nobody reads.
    #
    # Two facts, two members, and that separation is the point: this began as a suffix on
    # `ceiling_label`, and both front ends interpolate that label into a sentence telling the
    # reader to *raise* the ceiling — producing "raise SPEECHWRITER_MAX_TOKENS (currently
    # 128,000 — above this model's 64,000)", which argues with itself.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "128000")

    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-haiku-4-5")
    over = build_agent(load_settings())
    assert over.ceiling_exceeds_model is True
    assert over.profiled_max_tokens == 64000
    # The label stays a bare figure, so the sentence it lands in still reads correctly.
    assert over.ceiling_label == "128,000"

    # The same override on a model that can honour it raises nothing.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5")
    within = build_agent(load_settings())
    assert within.ceiling_exceeds_model is False
    assert within.ceiling_label == "128,000"

    # And with no override at all there is nothing to exceed, whatever the model.
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-haiku-4-5")
    assert build_agent(load_settings()).ceiling_exceeds_model is False


def test_the_bundle_still_takes_its_fields_in_the_documented_order():
    # `SpeechwriterAgent` is public: `build_agent` returns it and the README documents that
    # path. Fields are appended, never inserted — a defaulted field in the *middle* still shifts
    # every positional argument after it, which is the mistake `config.Settings` spells out and
    # this class made when `profiled_max_tokens` landed ahead of `warner`. A consumer writing
    # `SpeechwriterAgent(agent, store, settings, 32000, my_warner)` had their warner bound to
    # the ceiling field: no truncation signal, and a TypeError from the first comparison.
    fields = [f.name for f in dataclasses.fields(SpeechwriterAgent)]

    assert fields[:5] == ["agent", "store", "settings", "max_tokens", "warner"], fields
    # Anything added later belongs after those, in the order it was added.
    assert fields[5:] == ["profiled_max_tokens"], fields


def test_a_typed_endpoint_is_read_the_way_a_reader_types_it_and_never_raises():
    # One table rather than four tests, in the shape of
    # `test_resolving_a_choice_never_raises_on_a_stray_argument`, and for the same reason: this
    # now runs on whatever a reader types into a text box, so *answering* matters more than any
    # single answer. Two of the entries were measured against a live `mlx_lm.server` serving
    # eight models: without a scheme the Request constructor raises before any socket, and
    # without `/v1` the path is `/models`, which no OpenAI-compatible server exposes. Both came
    # back as `[]` — a healthy server reported exactly like a dead one.
    normalize = endpoints.normalize_endpoint
    expected = {
        "127.0.0.1:8080": "http://127.0.0.1:8080/v1",
        "localhost:8080": "http://localhost:8080/v1",
        "http://127.0.0.1:8080": "http://127.0.0.1:8080/v1",
        "  http://h/v1/  ": "http://h/v1",
        "https://api.example.com/openai/v1": "https://api.example.com/openai/v1",
        # `urlsplit` reads the two spellings of one typo differently — "localhost:8080" parses
        # as scheme "localhost", "127.0.0.1:8080" as no scheme at all — which is why the test
        # for a scheme is `"://" in raw` and not `urlsplit(raw).scheme`.
        "file:///Users/you/private": None,
        "ftp://example.invalid/v1": None,
        "http://": None,
        "": None,
        "   ": None,
        # The one that raises. `configured_endpoint` runs on every Streamlit rerun, so a
        # ValueError here is not a bad caption but a page that throws on every rerun with the
        # offending text still in session state.
        "http://[::1": None,
        "[::1": None,
    }
    for typed, want in expected.items():
        assert normalize(typed) == want, typed

    # Idempotent, which both callers rely on: the sidebar normalises the value seeded from the
    # environment as well as the one typed, so a second pass must be a no-op.
    for produced in filter(None, expected.values()):
        assert normalize(produced) == produced


def test_listing_models_never_raises_on_a_url_it_cannot_parse():
    # This module's docstring promises "every failure is an empty list, never an exception", and
    # the line that decides whether to open a socket at all sat *outside* the try: `urlsplit`
    # raises `ValueError: Invalid IPv6 URL` on an unclosed bracket. Unreachable while an
    # endpoint could only come from a dotenv the operator wrote; one keystroke away once it can
    # be typed. Asserted here rather than left to the caller because the promise is this
    # module's, and both front ends were written against it.
    assert endpoints.list_models("http://[::1") == []
    assert endpoints.list_models("http://[::1]:8080/v1", timeout=0.2) == []


def test_the_key_configured_for_one_endpoint_is_not_sent_to_another(monkeypatch, tmp_path):
    # `list_models` licenses forwarding the reader's key with "no new disclosure: the chat
    # client already sends the very same credential to this very same host". That is exact, and
    # it stops holding the moment the host is *typed*: a reader with a real OPENAI_API_KEY for a
    # hosted gateway who types a colleague's laptop address would hand that key over plaintext
    # HTTP to a machine the operator never named, on the first request — before
    # `_CredentialSafeRedirects`, which guards only the second hop, can see it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    settings = load_settings()

    # The configured server keeps the credential: that is the host it was configured for, and
    # the chat client sends it there every turn anyway.
    assert settings.endpoint_api_key_for("http://127.0.0.1:8080/v1") == "sk-real-key"
    # Same origin, different path — still the same server.
    assert settings.endpoint_api_key_for("http://127.0.0.1:8080/v2") == "sk-real-key"
    # Everything else gets nothing: another port is another server, and so is another host.
    assert settings.endpoint_api_key_for("http://127.0.0.1:1234/v1") is None
    assert settings.endpoint_api_key_for("http://192.168.1.50:8080/v1") is None
    assert settings.endpoint_api_key_for("https://127.0.0.1:8080/v1") is None
    # Total, like `same_origin` itself: an unparseable target is not the configured one.
    assert settings.endpoint_api_key_for("http://[::1") is None

    # And with nothing configured there is no host to trust, so nothing is sent anywhere.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    assert load_settings().endpoint_api_key_for("http://127.0.0.1:8080/v1") is None


def _client_key(model: ChatOpenAI) -> str | None:
    """The plain text behind a built client's key field.

    ``ChatOpenAI.openai_api_key`` is typed ``SecretStr | () -> str | () -> Awaitable[str]``,
    so the attribute alone does not type-check. Unwrapped through ``getattr`` rather than by
    importing ``pydantic.SecretStr``: pydantic reaches this environment only transitively via
    langchain, and importing it directly is the undeclared-dependency trap this repo already
    documents for ``pyyaml``.
    """
    getter = getattr(model.openai_api_key, "get_secret_value", None)
    return getter() if callable(getter) else None


def test_the_key_reaches_the_chat_client_only_for_the_configured_endpoint(monkeypatch, tmp_path):
    # `endpoint_api_key_for` guards the /v1/models probe, and guarding only there was a hole
    # rather than a boundary: the probe withheld the key while the chat client went on sending
    # it to the same typed host on *every turn*, over plaintext HTTP, which is where it actually
    # matters. Measured on the wire before this — one invoke carried `Authorization: Bearer` and
    # the reader's real gateway key — while the sidebar rendered "No credential sent".
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-gateway-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    configured = load_settings()

    # A model detected at an endpoint the reader typed: not the configured origin, so the key
    # is dropped and `endpoint_api_key` falls back to the placeholder a local server ignores.
    typed = config.local_choice("qwen", "http://192.168.1.50:8080/v1")
    elsewhere = _build_model(typed.applied_to(configured))
    assert isinstance(elsewhere, ChatOpenAI)
    assert _client_key(elsewhere) == config.LOCAL_API_KEY_PLACEHOLDER

    # Its own server still gets it, or a configured hosted gateway could never be called.
    same = config.local_choice("gpt-4o", "https://gateway.example.com/v1")
    mine = _build_model(same.applied_to(configured))
    assert isinstance(mine, ChatOpenAI)
    assert _client_key(mine) == "sk-real-gateway-secret"

    # And a curated entry carries it through untouched: the Anthropic client never reads it, so
    # clearing it would strip the key on a detour through Claude and not put it back.
    assert config.MODEL_CHOICES[0].applied_to(configured).openai_api_key == "sk-real-gateway-secret"


def test_one_model_served_by_two_machines_will_not_resolve_by_name(monkeypatch, tmp_path):
    # `local_choice` labels both `<id> (local)` because the label must stay a pure function of
    # the pair, so a roster holding the same id at two endpoints has two rows a name cannot tell
    # apart. Returning the first match pointed the agent at whichever was merged earlier, with
    # nothing said. None is the honest answer: the caller prints the table, whose `base_url`
    # column distinguishes them, and a row number never is ambiguous.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    laptop = config.local_choice("qwen", "http://127.0.0.1:8080/v1")
    workstation = config.local_choice("qwen", "http://192.168.1.50:8080/v1")
    roster = config.model_choices(load_settings(), detected=[laptop, workstation])

    assert config.resolve_choice(roster, "qwen") is None
    assert config.resolve_choice(roster, "qwen (local)") is None
    # The number still resolves, and to the right one of the two.
    assert config.resolve_choice(roster, str(roster.index(workstation) + 1)) == workstation
    # An unambiguous name is unaffected — this must narrow ambiguity, not matching.
    assert config.resolve_choice(roster, "opus 5") == config.MODEL_CHOICES[1]


def test_a_configured_endpoint_and_a_detected_one_are_the_same_row(monkeypatch, tmp_path):
    # `SPEECHWRITER_BASE_URL=http://127.0.0.1:8080/v1/` works perfectly — `list_models` rstrips
    # it — but a detection against that same server is normalised without the trailing slash,
    # and roster dedup is on `(model, base_url)`. Two strings, one server: the reader gets two
    # rows both labelled "qwen (local)", indistinguishable in the picker and, in the REPL,
    # matched by label so `resolve_choice` returns whichever came first. Measured before
    # `load_settings` normalised the configured value.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "  http://127.0.0.1:8080/v1/  ")
    settings = load_settings()

    assert settings.base_url == "http://127.0.0.1:8080/v1"
    detected = [config.local_choice("local/qwen", "http://127.0.0.1:8080/v1")]
    labels = [choice.label for choice in config.model_choices(settings, detected=detected)]
    assert labels.count("local/qwen (local)") == 1, labels

    # A value that does not normalise is kept verbatim rather than dropped, so junk keeps the
    # behaviour it has today — refused at the point of use — instead of silently becoming "no
    # endpoint configured" and building an Anthropic client for a locally served id.
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "file:///Users/you/private")
    assert load_settings().base_url == "file:///Users/you/private"


def test_listing_models_speaks_only_http(tmp_path):
    # `urlopen`'s default opener installs FileHandler, FTPHandler and DataHandler, so a
    # `SPEECHWRITER_BASE_URL` of `file:///Users/you/private` would make the Detect button read
    # `/Users/you/private/models` off local disk and list whatever it found. `build_opener`
    # cannot be relied on to drop those — it re-adds every default whose class was not passed
    # in — so the scheme is checked before a Request is built.
    served = tmp_path / "models"
    served.write_text(json.dumps({"object": "list", "data": [{"id": "read-off-disk"}]}))

    assert endpoints.list_models(f"file://{tmp_path}") == []
    assert endpoints.list_models("ftp://example.invalid/v1") == []
    assert endpoints.list_models(f"data:,{served.read_text()}") == []


def test_listing_models_does_not_carry_the_key_across_a_redirect():
    # `HTTPRedirectHandler.redirect_request` copies every header except content-length and
    # content-type onto the new request — Authorization included. So an endpoint answering
    # /v1/models with a 302 elsewhere hands the reader's real OPENAI_API_KEY to whatever it
    # points at: reachable from a mistyped hosted gateway, an http URL redirecting to https on
    # another host, or a captive portal. Same-origin keeps it; anything else is stripped.
    received: list[tuple[str, str | None]] = []

    def note(handler):
        received.append((handler.path, handler.headers.get("Authorization")))

    with _serving({"object": "list", "data": [{"id": "ok"}]}, observer=note) as elsewhere:
        with _redirecting_to(f"{elsewhere}/models") as configured:
            assert endpoints.list_models(configured, api_key="sk-REAL-SECRET") == ["ok"]

    assert received, "the redirect target was never reached"
    assert all(auth is None for _, auth in received), (
        f"the bearer token was forwarded off-origin: {received}"
    )


def test_listing_models_survives_every_endpoint_that_is_not_one():
    # The Detect button's whole contract, and every case here is one typo away in
    # SPEECHWRITER_BASE_URL. The no-scheme case is the one that bites: `urllib.request.Request`
    # raises at *construction*, before any socket is opened, so a try block wrapped around only
    # the connection would let it escape into the page as a traceback.
    assert endpoints.list_models("http://127.0.0.1:1/v1") == []
    assert endpoints.list_models("definitely-not-a-url") == []
    assert endpoints.list_models("") == []


def test_listing_models_reads_ids_and_skips_rows_that_have_none():
    # Sorted, and rows without a usable id are skipped rather than trusted, so one malformed
    # entry does not cost the rest of an otherwise real answer.
    served = {
        "object": "list",
        "data": [
            {"id": "mlx-community/granite-4.1-8b-4bit"},
            {"id": "mlx-community/Qwen3.8-27B-4bit"},
            {"object": "model"},
            {"id": ""},
        ],
    }
    with _serving(served) as base_url:
        assert endpoints.list_models(base_url) == [
            "mlx-community/Qwen3.8-27B-4bit",
            "mlx-community/granite-4.1-8b-4bit",
        ]


def test_an_endpoint_serving_nothing_is_not_reported_as_unreachable(caplog):
    # A running Ollama with nothing pulled answers `{"object": "list", "data": null}` — that is
    # well-formed, and a real answer meaning "none". Iterating that None raises, and the blanket
    # `except` would then file a healthy server under "could not list", so the two would be
    # indistinguishable in the log.
    #
    # Asserted on the log rather than on the return value, deliberately: both paths return [],
    # so a test that only checked the result would pass with the shape guard deleted — it did,
    # on the first mutation run.
    with _serving({"object": "list", "data": None}) as base_url:
        with caplog.at_level(logging.INFO, logger="speechwriter.endpoints"):
            assert endpoints.list_models(base_url) == []

    assert not caplog.text, f"a well-formed empty answer was logged as a failure: {caplog.text}"


@contextlib.contextmanager
def _serving(payload, observer=None):
    """Run a throwaway OpenAI-shaped ``/models`` endpoint on a free loopback port.

    Loopback only, and on a port the OS picks, so the suite stays free and cannot reach — or be
    reached from — anything outside this process. ``observer`` is handed each request, for
    tests that care about what arrived rather than what came back.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's own spelling
            if observer is not None:
                observer(self)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - the base class's own parameter name
            """Silence the default stderr access log, which pytest would otherwise capture."""

    yield from _running(Handler)


@contextlib.contextmanager
def _redirecting_to(target):
    """An endpoint whose ``/models`` answers 302 pointing at ``target``.

    ``localhost`` rather than ``127.0.0.1`` so the redirect is genuinely cross-origin by
    netloc while still resolving to this machine — a real second host would make the test
    depend on the network.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's own spelling
            self.send_response(302)
            self.send_header("Location", target.replace("127.0.0.1", "localhost"))
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002 - the base class's own parameter name
            """Silence the default stderr access log."""

    yield from _running(Handler)


def _running(handler):
    """Serve ``handler`` on a free loopback port for the life of the block."""
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
