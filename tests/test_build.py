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
from pathlib import Path

import yaml
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
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    # Both halves of the documented default pair, pinned by unsetting them: an exported
    # SPEECHWRITER_BASE_URL or SPEECHWRITER_MODEL would otherwise decide what this asserts.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)

    settings = load_settings()
    assert settings.research_enabled is False
    assert [sa["name"] for sa in build_subagents(settings)] == ["style-critic"]

    bundle = build_agent(settings)
    assert bundle.agent.__class__.__name__ == "CompiledStateGraph"
    # A fresh clone names a coherent pair, not a model with nowhere to send it.
    assert bundle.settings.model == config.DEFAULT_MODEL
    assert bundle.settings.base_url == config.DEFAULT_LOCAL_ENDPOINT


def test_research_subagent_appears_with_tavily(monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    settings = load_settings()
    assert settings.research_enabled is True
    assert [sa["name"] for sa in build_subagents(settings)] == ["researcher", "style-critic"]


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/granite-4.1-8b-4bit")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    assert load_settings().model == "mlx-community/granite-4.1-8b-4bit"


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
    # Pins the documented default endpoint rather than whatever a developer exported.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    for bad in ("0", "-5"):
        monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", bad)
        assert load_settings().max_tokens is None, f"{bad} must not reach the model"
        assert getattr(_build_model(load_settings()), "max_tokens", None) != int(bad)


def test_ceiling_resolution_is_two_tier(monkeypatch, tmp_path):
    # Regression. `ChatOpenAI` defaults `max_tokens` to None -- "let the server decide" -- and
    # on a reasoning model that defaults to a high effort level that is an *unbounded* thinking
    # budget rather than a merely small one: extended thinking bills against the same ceiling,
    # so a subagent can spend the whole response deliberating and emit no text, which deepagents
    # forwards as an empty status="success" task result. A ceiling is therefore always set.
    #
    # There used to be a middle tier that kept the ceiling the client had resolved for itself
    # out of LangChain's profile table. It is gone by construction rather than by choice:
    # `init_chat_model` reads a profile's max_tokens only on the Anthropic path, so with one
    # OpenAI-compatible client it could never fire — see
    # test_a_profiled_id_over_a_local_endpoint_still_gets_a_ceiling, which pins that half.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    # Pins the documented default endpoint rather than whatever a developer exported.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

    served = (config.DEFAULT_MODEL, "mlx-community/not-a-real-model-9")

    # Tier 2: with no override every id lands on our floor, never on the client's own None.
    for model_id in served:
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        model = _build_model(load_settings())
        assert isinstance(model, ChatOpenAI)
        assert model.max_tokens == config.DEFAULT_MAX_TOKENS, model_id

    # Tier 1: an explicit override wins, and still reaches the same client.
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "4242")
    for model_id in served:
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        overridden = _build_model(load_settings())
        assert isinstance(overridden, ChatOpenAI)
        assert overridden.max_tokens == 4242, model_id


def test_the_default_ceiling_fits_inside_the_assumed_context_window():
    # Output and input come out of *one* budget on a local server, which the hosted path never
    # had to think about: DEFAULT_MAX_TOKENS was 32000 while it applied to 128k-window hosted
    # models, and left unchanged it would leave 768 tokens of a 32768-token window for the
    # entire system prompt, the loaded skills and the draft under revision. vLLM rejects that
    # outright at the first turn; others clamp it silently, which is worse. The two constants
    # are declared independently in config.py with nothing structural tying them, and
    # `ceiling_crowds_context` draws its line at half the window — so the shipped default must
    # not itself trip the warning both front ends render.
    assert config.DEFAULT_MAX_TOKENS <= config.DEFAULT_LOCAL_CONTEXT_WINDOW // 2, (
        f"the default ceiling ({config.DEFAULT_MAX_TOKENS}) crowds the assumed window "
        f"({config.DEFAULT_LOCAL_CONTEXT_WINDOW}) — a fresh install would warn about itself."
    )

    # And from below, which the first version of this test missed entirely — a one-sided bound
    # on a constant is satisfied by making it smaller, and smaller is the direction that
    # silently truncates a draft. The ceiling is the *thinking* budget too, so it has to clear
    # the longest speech the committed datasets actually grade plus room to deliberate.
    #
    # Read out of `evals/datasets/` rather than hard-coded, so adding a longer example moves the
    # floor with it instead of leaving this assertion describing a corpus that has changed.
    datasets = Path(__file__).resolve().parents[1] / "evals" / "datasets"
    longest = max(
        (example.get("outputs") or {}).get("target_word_count") or 0
        for path in datasets.glob("*.json")
        for example in json.loads(path.read_text())
    )
    assert longest > 0, "no target_word_count found — this assertion would pass vacuously"
    # ~1.4 tokens per word of English prose, conservative for a model that emits punctuation and
    # stage cues; doubled to leave the reasoning pass the same room again.
    needed = int(longest * 1.4 * 2)
    assert config.DEFAULT_MAX_TOKENS >= needed, (
        f"the default ceiling ({config.DEFAULT_MAX_TOKENS}) cannot hold the longest graded "
        f"draft ({longest} words ≈ {int(longest * 1.4)} tokens) plus a reasoning pass "
        f"({needed}) — the run would be truncated and scored as a short speech."
    )


def test_the_payload_carries_no_sampling_parameters(monkeypatch, tmp_path):
    # `_build_model` sets no temperature, top_p or top_k, and this is what keeps it that way in
    # both directions. `build_agent()` never touches the wire — that is what makes this suite
    # free — so nothing else here would notice a langchain-openai bump that began sending one by
    # default, which it has form for: `ChatOpenAI.temperature` used to default to 0.7 and go out
    # on every request. A sampling parameter appearing unbidden changes every generation from a
    # local reasoning model and raises nothing.
    #
    # This absorbs a sibling that asserted the same thing through `ChatAnthropic`, where the
    # three parameters were rejected outright with a 400. That client is gone, and with it the
    # reason to assert this twice; `_get_request_payload` is still private, like the other
    # LangChain/deepagents internals this file reaches into, so a rename breaks it loudly.
    unset = {"temperature", "top_p", "top_k"}
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)

    for model_id in (config.DEFAULT_MODEL, "gpt-4o"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        # Every ceiling branch, since a stray default could be injected on either call:
        # None exercises the floor, the override exercises tier 1.
        for override in (None, "4242"):
            if override is None:
                monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
            else:
                monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", override)
            model = _build_model(load_settings())
            # Narrows `BaseChatModel` for the type checker, and pins the client type while we
            # are here: `settings.model` is free-form, so that is worth asserting too.
            assert isinstance(model, ChatOpenAI)
            payload = model._get_request_payload([])
            assert unset.isdisjoint(payload), f"{model_id} (override={override}): {payload}"


def test_the_configured_pair_reaches_the_client(monkeypatch, tmp_path):
    # The pair is the whole configuration: `base_url` selects nothing *else* any more, but it is
    # still what the client is pointed at, and the id is free-form, so a build that dropped
    # either would only fail at the first turn. Asserted through `_build_model` rather than on
    # Settings so a branch that forgot to thread the client kwargs through is caught here.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://192.168.1.50:8080/v1")
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")

    settings = load_settings()
    # The point of the gate: no credential of ours anywhere, yet the pair is runnable.
    assert settings.openai_api_key is None
    assert settings.model_endpoint_usable is True

    model = _build_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://192.168.1.50:8080/v1"
    assert model.model_name == "mlx-community/Qwen3.8-27B-4bit"


def test_blank_base_url_falls_back_to_the_documented_default(monkeypatch, tmp_path):
    # `export SPEECHWRITER_BASE_URL=` is how a shell says "unset". Read with a bare
    # `os.environ.get` that empty string is truthy enough to set the field, and every call would
    # be routed at an endpoint no shape check could tell apart from a real one — while the
    # banner still printed a model id. There is no second client to select by leaving this
    # empty, so the fallback is the documented pair, not None.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)

    for blank in ("", "   "):
        monkeypatch.setenv("SPEECHWRITER_BASE_URL", blank)
        settings = load_settings()
        assert settings.base_url == config.DEFAULT_LOCAL_ENDPOINT
        assert settings.model_endpoint_usable is True
        model = _build_model(settings)
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == config.DEFAULT_LOCAL_ENDPOINT


def test_the_endpoint_gate_refuses_only_a_shape_that_cannot_be_called(monkeypatch, tmp_path):
    # The CLI refuses commissions on this and the web UI disables its chat input, so a false
    # negative silently bricks a working setup and a false positive defers the failure to the
    # first turn. It replaced `model_credentials_present` when the hosted client left: a locally
    # served model needs no credential of ours, so "is a key present" stopped being a question
    # with an answer, and a gate that is unconditionally true has quietly stopped running.
    #
    # What is left is *shape*, which is worth checking precisely because `_configured_endpoint`
    # deliberately never rewrites what an operator wrote. Reachability is not part of it — a
    # probe here would break the offline-build invariant, and a server that is merely not
    # running yet is what Detect models and `/endpoint` are for.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    # Nothing configured is not a failure state any more: it is the documented default pair.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    assert load_settings().model_endpoint_usable is True

    for usable in ("http://127.0.0.1:8080/v1", "https://gateway.example.com/openai/v1"):
        monkeypatch.setenv("SPEECHWRITER_BASE_URL", usable)
        assert load_settings().model_endpoint_usable is True, usable

    # `file:///…` is the one that bites: `urlopen`'s default opener would read it off local
    # disk. `http://` has no host, and raises inside urllib at request *construction*.
    for unusable in ("file:///Users/you/private", "ftp://example.invalid/v1", "http://", "[::1"):
        monkeypatch.setenv("SPEECHWRITER_BASE_URL", unusable)
        assert load_settings().model_endpoint_usable is False, unusable


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
    # Pins the documented default endpoint rather than whatever a developer exported.
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
    # Regression, and the reason the middle ceiling tier was *deleted* rather than kept for
    # symmetry. It asked "is there a profile?", which is the same question as "was a ceiling
    # resolved?" for ChatAnthropic and emphatically not for ChatOpenAI: init_chat_model applies
    # a profile's max_tokens only on the Anthropic path. So a *profiled* id served over an
    # OpenAI-compatible endpoint -- `gpt-4o` on LM Studio, LiteLLM or a hosted service, all of
    # which the README names -- skipped the floor and came back with max_tokens=None: no ceiling
    # at all, which is the unbounded thinking budget the floor exists to prevent.
    #
    # With one client that branch could only ever have been dead code reading as live
    # protection, and this is what keeps a well-meaning reinstatement red.
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


def test_the_resolved_ceiling_reaches_the_request_payload(monkeypatch, tmp_path):
    # A ceiling set on the client but dropped from the payload is no ceiling at all, and the key
    # is not the obvious one: langchain-openai 1.6 sends `max_completion_tokens` where
    # langchain-anthropic sent `max_tokens`, and `mlx_lm.server` reads the new spelling while
    # older shims may read only the old. Assert the *value* is carried under some key rather
    # than pinning either, so a rename upstream fails loudly here instead of silently
    # unbounding a local reasoning model.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "12345")

    # The default endpoint and a typed one: the client is the same either way, but the kwargs
    # are threaded through one construction, so a dropped ceiling would show up on both.
    for base_url, model_id in (
        (None, config.DEFAULT_MODEL),
        ("http://192.168.1.50:8080/v1", "mlx-community/granite-4.1-8b-4bit"),
    ):
        if base_url is None:
            monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("SPEECHWRITER_BASE_URL", base_url)
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        model = _build_model(load_settings())
        # Narrows BaseChatModel for the checker, and pins that a client was built at all — the
        # payload assertion below is vacuous on anything else.
        assert isinstance(model, ChatOpenAI)
        payload = model._get_request_payload([])
        carrying = {key for key, value in payload.items() if value == 12345}
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
    #
    # `anthropic_api_key` used to sit second in this list, and removing it shifted every
    # positional argument after it — a deliberate breaking change rather than a vestigial field
    # kept for compatibility, because a key this agent can no longer send would read as a
    # credential in use. Named arguments, so this test says nothing about that ordering; it is
    # `Settings`' own comment that carries the warning.
    settings = config.Settings(
        model=config.DEFAULT_MODEL,
        tavily_api_key=None,
        project_root=tmp_path,
        workspace_dir=tmp_path,
        skills_dir=tmp_path,
        store_path=tmp_path / "store.json",
        max_research_results=5,
        max_tokens=None,
    )

    # A caller who named no machine still gets a coherent *pair*: an id defaulted on its own
    # would be a model with nowhere to send it, and there is no second client to fall back to.
    assert settings.base_url == config.DEFAULT_LOCAL_ENDPOINT
    assert settings.model_endpoint_usable is True
    assert settings.openai_api_key is None
    # The field the model picker adds, asserted here rather than in a test of its own: this is
    # the standing guard on the constructor's shape, and a required field would break it.
    assert settings.context_window is None


def test_the_configured_pair_is_always_offered(monkeypatch, tmp_path):
    # Streamlit *silently* rewrites a selection that is not among a widget's options to option
    # zero — no exception, no log. So a roster that did not contain the configured pair would
    # retarget a reader onto whatever happened to be listed first, on a machine where that entry
    # may not even be served.
    #
    # MODEL_CHOICES is empty, which makes this the *only* roster rather than merely the
    # authoritative one: every row is synthesised from a configuration that exists, because a
    # shipped list of local endpoints would be a guess about which server the reader is running.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://192.168.1.50:8080/v1")

    settings = load_settings()
    offered = config.model_choices(settings)

    assert offered[-1].model == settings.model
    assert offered[-1].base_url == settings.base_url
    # Built through the one constructor, so an entry discovered from a live endpoint is
    # indistinguishable from this one and the same model cannot appear twice. The empty seed is
    # why this is the whole roster — and why it is never *itself* empty, which
    # `streamlit_app.py` depends on when it indexes `choices[0]`.
    assert offered == (config.local_choice(settings.model, settings.base_url),)

    # ...and a configuration already on the roster is not offered a second time, however many
    # times it is handed over — both front ends pass the configured *and* the live settings.
    assert config.model_choices(settings, settings) == offered


def test_switching_away_from_a_server_leaves_a_way_back(monkeypatch, tmp_path):
    # The roster must be widened by *every* configuration that has to stay reachable, not only
    # the live one. A roster derived from the post-switch settings alone drops the pair the
    # reader came from, and with MODEL_CHOICES empty there is no curated list underneath to fall
    # back on: the way back would be removed by the act of leaving, and nothing short of a
    # restart brings it back. Callers therefore pass the configuration the session *started* on
    # as well as the one now in force.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "mlx-community/Qwen3.8-27B-4bit")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1")

    configured = load_settings()
    # What the bundle's settings look like after picking a model detected on another machine.
    elsewhere = config.local_choice("qwen", "http://192.168.1.50:8080/v1")
    switched = elsewhere.applied_to(configured)

    offered = config.model_choices(configured, switched)

    assert any(c.base_url == configured.base_url for c in offered), (
        "the pair the session started on vanished once another server was selected — there is "
        "no way back to it without restarting the process"
    )
    # And no duplicate for the entry that is now both current and already on the list.
    assert len(offered) == 2
    assert [c.model for c in offered].count(elsewhere.model) == 1


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
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    configured = load_settings()

    # A real rebuild is told from a no-op by the *context window*. It used to be told by the
    # resolved ceiling, which separated the two hosted models this switched between (128k and
    # 64k) and separates nothing now that one floor applies to every id. The window still does,
    # because it is the per-entry field a roster row carries — and it is what the switch has to
    # move, since compacting for the previous model's window is the local failure mode.
    first = config.local_choice(config.DEFAULT_MODEL, configured.base_url, 32768)
    second = config.local_choice("mlx-community/granite-4.1-8b-4bit", configured.base_url, 131072)

    bundle = build_agent(first.applied_to(configured))
    bundle.store.put(("speechwriter", "memories"), "mayor.md", {"content": "Plain speaker."})

    bundle.persist()
    switched = build_agent(second.applied_to(configured))

    assert switched.store is not bundle.store
    assert [item.key for item in memory.all_items(switched.store)] == ["mayor.md"]
    assert switched.settings.model == second.model
    assert bundle.context_window == 32768
    assert switched.context_window == 131072


def test_an_oversized_ceiling_override_is_reported_beside_the_label_not_inside_it(
    monkeypatch, tmp_path
):
    # SPEECHWRITER_MAX_TOKENS is tier 1 and global, so an override sized for one model follows a
    # switch to another — and served locally the constraint is harder than any hosted ceiling
    # was: output and input come out of *one* window. A 32,000-token ceiling against a
    # 32,768-token window leaves 768 tokens for the entire system prompt, the loaded skills and
    # the draft under revision. vLLM rejects that at the first turn; others clamp it silently,
    # which is worse, because the reader sees a short speech and no error.
    #
    # Two facts, two members, and that separation is the point: this began as a suffix on
    # `ceiling_label`, and both front ends interpolate that label into a sentence telling the
    # reader to *raise* the ceiling — producing "raise SPEECHWRITER_MAX_TOKENS (currently
    # 32,000 — more than half this model's window)", which argues with itself.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "32000")

    over = build_agent(load_settings())
    assert over.context_window == config.DEFAULT_LOCAL_CONTEXT_WINDOW
    assert over.ceiling_crowds_context is True
    # The label stays a bare figure, so the sentence it lands in still reads correctly.
    assert over.ceiling_label == "32,000"

    # The same override against a window that can carry it raises nothing — which is what
    # `ModelChoice.context_window` is for, and the reason the window travels with the model.
    roomy = config.local_choice(config.DEFAULT_MODEL, config.DEFAULT_LOCAL_ENDPOINT, 131072)
    within = build_agent(roomy.applied_to(load_settings()))
    assert within.ceiling_crowds_context is False
    assert within.ceiling_label == "32,000"

    # And the shipped default is sized to sit under the line, so a fresh install never warns
    # about itself — the ceiling is always resolved now, so "no override" is not "no ceiling".
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS")
    default = build_agent(load_settings())
    assert default.max_tokens == config.DEFAULT_MAX_TOKENS
    assert default.ceiling_crowds_context is False


def test_the_bundle_still_takes_its_fields_in_the_documented_order():
    # `SpeechwriterAgent` is public: `build_agent` returns it and the README documents that
    # path. Fields are appended, never inserted — a defaulted field in the *middle* still shifts
    # every positional argument after it, which is the mistake `config.Settings` spells out and
    # this class made when a second ceiling field landed ahead of `warner`. A consumer writing
    # `SpeechwriterAgent(agent, store, settings, 32000, my_warner)` had their warner bound to
    # that field: no truncation signal, and a TypeError from the first comparison.
    fields = [f.name for f in dataclasses.fields(SpeechwriterAgent)]

    assert fields[:5] == ["agent", "store", "settings", "max_tokens", "warner"], fields
    # Anything added later belongs after those, in the order it was added. `context_window`
    # replaced `profiled_max_tokens` in place, which is the one edit that keeps this order: a
    # field that can only ever be None was swapped for one that answers the same question a
    # locally served model can actually be asked.
    assert fields[5:] == ["context_window"], fields


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
    assert endpoints.list_models("[::1") == []
    # Deliberately no well-formed-but-unreachable address here. An earlier version asserted on
    # `http://[::1]:8080/v1`, which opens a real TCP connection — breaking the suite's offline
    # invariant, and going red for any contributor running the `mlx_lm.server --port 8080` the
    # README recommends, since a listening server answers and the result is no longer `[]`.
    # The unclosed bracket is the whole point: it raises inside `urlsplit`, before any socket.


def test_the_key_configured_for_one_endpoint_is_not_sent_to_another(monkeypatch, tmp_path):
    # `list_models` licenses forwarding the reader's key with "no new disclosure: the chat
    # client already sends the very same credential to this very same host". That is exact, and
    # it stops holding the moment the host is *typed*: a reader with a real OPENAI_API_KEY for a
    # hosted gateway who types a colleague's laptop address would hand that key over plaintext
    # HTTP to a machine the operator never named, on the first request — before
    # `_CredentialSafeRedirects`, which guards only the second hop, can see it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://192.168.1.50:8080/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    settings = load_settings()

    # The configured server keeps the credential: that is the host it was configured for, and
    # the chat client sends it there every turn anyway.
    assert settings.endpoint_api_key_for("http://192.168.1.50:8080/v1") == "sk-real-key"
    # Same origin, different path — still the same server.
    assert settings.endpoint_api_key_for("http://192.168.1.50:8080/v2") == "sk-real-key"
    # Everything else gets nothing: another port is another server, and so is another host —
    # the default endpoint emphatically included, since it is only the default.
    assert settings.endpoint_api_key_for("http://192.168.1.50:1234/v1") is None
    assert settings.endpoint_api_key_for(config.DEFAULT_LOCAL_ENDPOINT) is None
    assert settings.endpoint_api_key_for("https://192.168.1.50:8080/v1") is None
    # Total, like `same_origin` itself: an unparseable target is not the configured one.
    assert settings.endpoint_api_key_for("http://[::1") is None

    # With nothing configured, the real key goes NOWHERE — including to the default endpoint.
    #
    # This is the half that is easy to get backwards, and an earlier version of this test did:
    # it reasoned that since `base_url` now always names a server, the default is "a configured
    # origin like any other" and should keep the key. It is not. `DEFAULT_LOCAL_ENDPOINT` is a
    # guess this repo makes on the reader's behalf, not a host the operator named, and the whole
    # credential boundary is *the operator named this server*.
    #
    # The concrete case: `OPENAI_API_KEY` exported globally for some hosted service — an
    # ordinary thing for a developer to have — plus a fresh clone with no dotenv. While unset
    # meant "no endpoint at all" that key was simply never read. Defaulting `base_url` would
    # have started sending it, on every turn, to whatever process holds 127.0.0.1:8080. Loopback
    # bounds that; it does not make it intended, and the README promises the opposite.
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    defaulted = load_settings()
    assert defaulted.endpoint_configured is False
    assert defaulted.endpoint_api_key == config.LOCAL_API_KEY_PLACEHOLDER
    assert defaulted.endpoint_api_key_for(config.DEFAULT_LOCAL_ENDPOINT) == (
        config.LOCAL_API_KEY_PLACEHOLDER
    ), "the reader's real key was sent to an endpoint they never named"
    assert defaulted.endpoint_api_key_for("http://192.168.1.50:8080/v1") is None

    # Naming that same endpoint explicitly is what opts in — the value is identical, so this
    # pins that the flag and not the string is what carries the permission.
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", config.DEFAULT_LOCAL_ENDPOINT)
    named = load_settings()
    assert named.endpoint_configured is True
    assert named.endpoint_api_key_for(config.DEFAULT_LOCAL_ENDPOINT) == "sk-real-key"


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

    # A choice that merely *defaults* onto the local endpoint is judged on origin like any
    # other — it names a real server, so `same_origin` decides it on the merits.
    defaulted = config.local_choice("qwen")
    assert defaulted.base_url == config.DEFAULT_LOCAL_ENDPOINT
    assert defaulted.applied_to(configured).openai_api_key is None

    # And there is no exemption left, which is the change worth pinning — but pinning it needs a
    # choice the type system now forbids, so read why before deleting this.
    #
    # `applied_to` used to read `if self.base_url is None or same_origin(...)`. That disjunct
    # existed for a curated (Anthropic) entry, which carries no endpoint at all and whose client
    # never reads the key: without it, a detour through Claude and back stripped the key the
    # configured endpoint still needed. Nothing constructs such a choice today, because
    # `base_url` is `str` with a default — which makes the disjunct *unreachable* rather than
    # wrong, and unreachable is exactly why it needs a test. Relax `base_url` back to
    # `str | None` for any reason at all and it silently becomes a live carve-out handing the
    # reader's real `OPENAI_API_KEY` to any choice that arrives without an endpoint.
    #
    # The assertion above cannot see that: a defaulted choice has a non-None `base_url`, so the
    # `is None` half never fires for it. Mutation-tested — restoring the disjunct fails this
    # line and nothing else in the suite.
    exempt = config.ModelChoice("qwen", "qwen", None)  # ty: ignore[invalid-argument-type]
    assert exempt.applied_to(configured).openai_api_key is None


def test_one_model_served_by_two_machines_will_not_resolve_by_name(monkeypatch, tmp_path):
    # `local_choice` labels an entry with the bare model id because the label must stay a pure
    # function of the pair, so a roster holding the same id at two endpoints has two rows a name
    # cannot tell apart. Returning the first match pointed the agent at whichever was merged
    # earlier, with nothing said. None is the honest answer: the caller prints the table, whose
    # `base_url` column distinguishes them, and a row number never is ambiguous.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_BASE_URL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    laptop = config.local_choice("qwen", "http://127.0.0.1:8080/v1")
    workstation = config.local_choice("qwen", "http://192.168.1.50:8080/v1")
    roster = config.model_choices(load_settings(), detected=[laptop, workstation])

    assert config.resolve_choice(roster, "qwen") is None
    # The label is the id now, so the two spellings a reader might try are the same string —
    # which is the point: nothing about the label distinguishes the two servers.
    assert laptop.label == workstation.label == "qwen"
    # The number still resolves, and to the right one of the two.
    assert config.resolve_choice(roster, str(roster.index(workstation) + 1)) == workstation
    # An unambiguous name is unaffected — this must narrow ambiguity, not matching.
    assert config.resolve_choice(roster, config.DEFAULT_MODEL) == config.local_choice(
        config.DEFAULT_MODEL, config.DEFAULT_LOCAL_ENDPOINT
    )


def test_reading_an_endpoint_back_never_rewrites_it(monkeypatch, tmp_path):
    # Two functions on purpose. `normalize_endpoint` edits what a reader *typed*, because a
    # missing scheme or a missing /v1 reports a healthy server as dead. `usable_endpoint` only
    # accepts or rejects, because the value it reads was written deliberately by an operator —
    # and normalising on read is what dropped Azure's required `?api-version=` and appended a
    # 404-producing `/v1` to a proxy serving the OpenAI API at its root.
    deliberate = [
        "https://x.openai.azure.com/openai/deployments/gpt4?api-version=2024-02-01",
        "http://127.0.0.1:4000",
        "http://127.0.0.1:8080/v1/",
    ]
    for written in deliberate:
        assert endpoints.usable_endpoint(f"  {written}  ") == written
    # It still refuses what cannot be called, and still never raises on an unclosed bracket.
    for junk in ("", "   ", "file:///Users/you/private", "localhost:8080", "http://[::1"):
        assert endpoints.usable_endpoint(junk) is None, junk

    # And the typed side lower-cases the host, because "LocalHost" and "localhost" are one
    # server while roster dedup compares the string -- two rows, identical labels, and
    # `same_origin` disagreeing with the dedup key about what "the same server" means.
    assert endpoints.normalize_endpoint("http://LocalHost:8080/v1") == "http://localhost:8080/v1"
    # Userinfo is left alone, where case is significant.
    assert endpoints.normalize_endpoint("http://user:Pa55@h/v1") == "http://user:Pa55@h/v1"
    # A typed URL keeps its query too, for the same reason a configured one does.
    azure = "https://x.openai.azure.com/deployments/g?api-version=2024-02-01"
    assert endpoints.normalize_endpoint(azure) == azure


def test_an_ambiguous_model_name_is_refused_rather_than_passed_through():
    # `resolve_choice` answers None for "no such entry" and for "two entries by that name", and
    # the eval harness's pass-through is only right for the first: it sets SPEECHWRITER_MODEL
    # while leaving SPEECHWRITER_BASE_URL alone, which sends a Claude id to a local server.
    # `matching_choices` is what lets a caller tell the two Nones apart.
    laptop = config.local_choice("qwen", "http://127.0.0.1:8080/v1")
    workstation = config.local_choice("qwen", "http://192.168.1.50:8080/v1")
    roster = (config.local_choice(config.DEFAULT_MODEL), laptop)

    assert config.matching_choices(roster, "nonesuch") == []
    assert config.matching_choices(roster, "qwen") == [laptop]
    assert len(config.matching_choices(roster, config.DEFAULT_MODEL)) == 1

    ambiguous = roster + (workstation,)
    assert len(config.matching_choices(ambiguous, "qwen")) == 2
    assert config.resolve_choice(ambiguous, "qwen") is None


def test_a_configured_endpoint_reaches_the_client_exactly_as_written(monkeypatch, tmp_path):
    # An earlier version normalised SPEECHWRITER_BASE_URL to stop one server appearing twice in
    # the picker, and rewrote endpoints that worked: an Azure deployment URL lost the
    # `?api-version=` query it requires, and a proxy serving the OpenAI API at the root gained a
    # `/v1` that 404s. Both measured. `build_agent` never probes, so each failed at the first
    # turn with no log line. A configured endpoint is an operator's deliberate string.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "gpt-4o")
    for written in (
        "https://x.openai.azure.com/openai/deployments/gpt4?api-version=2024-02-01",
        "http://127.0.0.1:4000",
        "http://127.0.0.1:8080/v1/",
        "http://LocalHost:8080/v1",
    ):
        monkeypatch.setenv("SPEECHWRITER_BASE_URL", f"  {written}  ")
        settings = load_settings()
        assert settings.base_url == written, f"{written} was rewritten to {settings.base_url}"
        model = _build_model(settings)
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == written


def test_a_configured_endpoint_and_a_detected_one_are_the_same_row(monkeypatch, tmp_path):
    # The duplicate normalisation used to prevent, prevented on the read side instead. A dotenv
    # holding a trailing slash works fine — `list_models` rstrips it — but if the endpoint field
    # rewrote the value it was seeded with, detections would carry the tidied spelling while the
    # configured pair carried the raw one. Roster dedup is on `(model, base_url)`, so that is two
    # strings for one server: two rows both labelled "local/qwen", indistinguishable in the
    # picker and, in the REPL, ambiguous by label. `usable_endpoint` accepts a seeded value
    # without touching it, so both sides name the server the same way.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "local/qwen")
    monkeypatch.setenv("SPEECHWRITER_BASE_URL", "http://127.0.0.1:8080/v1/")
    settings = load_settings()

    seeded = endpoints.usable_endpoint(settings.base_url)
    assert seeded == settings.base_url, "reading the configured endpoint back rewrote it"

    detected = [config.local_choice("local/qwen", seeded or "")]
    labels = [choice.label for choice in config.model_choices(settings, detected=detected)]
    assert labels.count("local/qwen") == 1, labels


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
