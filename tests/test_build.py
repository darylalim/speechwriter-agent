"""Offline tests: everything here runs without an API key or network.

Constructing a Deep Agent does not call the model, so we can assert the whole graph
wires up, the research subagent toggles on the Tavily key, memory survives a
save/load round-trip, and every SKILL.md is well-formed — all in CI, for free.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

import speechwriter
from speechwriter import config, memory, prompts
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
    # Pinned by unsetting: an exported SPEECHWRITER_MODEL would otherwise decide what this asserts.
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)

    settings = load_settings()
    assert settings.research_enabled is False
    assert [sa["name"] for sa in build_subagents(settings)] == ["style-critic"]

    bundle = build_agent(settings)
    assert bundle.agent.__class__.__name__ == "CompiledStateGraph"
    assert bundle.settings.model == config.DEFAULT_MODEL


def test_the_agent_builds_offline_without_an_anthropic_key(monkeypatch, tmp_path):
    # The invariant the whole suite rests on, stated for the one credential every turn needs:
    # `ChatAnthropic` accepts a missing key at construction and fails only at the first call. If
    # a langchain-anthropic bump started validating the key eagerly, CI — which sets no key
    # anywhere — would go red here rather than in forty unrelated tests at once.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings()
    assert settings.anthropic_api_key is None
    assert settings.model_credentials_present is False
    assert build_agent(settings).agent.__class__.__name__ == "CompiledStateGraph"


def test_research_subagent_appears_with_tavily(monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dummy")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    settings = load_settings()
    assert settings.research_enabled is True
    assert [sa["name"] for sa in build_subagents(settings)] == ["researcher", "style-critic"]


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    assert load_settings().model == "claude-opus-5-5"


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
    # A zero or negative ceiling is accepted by the client without complaint and only fails at
    # the first API call, with an opaque provider error far from the typo that caused it — so it
    # must be rejected at load time, not forwarded to the client.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    for bad in ("0", "-5"):
        monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", bad)
        assert load_settings().max_tokens is None, f"{bad} must not reach the model"
        assert _build_model(load_settings()).max_tokens != int(bad)


def test_ceiling_resolution_is_three_tier(monkeypatch, tmp_path):
    # Regression, both directions. `ChatAnthropic` takes max_tokens from LangChain's profile
    # table and silently falls back to 4096 for an id it cannot profile — and adaptive thinking
    # bills against that same ceiling, so a subagent can spend the whole budget thinking and
    # emit no text, which deepagents forwards as an empty status="success" task result. But a
    # blunt constant must not *lower* a model LangChain does know: capping the 128k models at 32k
    # would be the same mistake inverted.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    # Tier 2: a profiled model keeps its own, larger ceiling.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5-5")
    assert (_build_model(load_settings()).max_tokens or 0) > config.DEFAULT_MAX_TOKENS

    # Tier 3: an unprofiled id gets our floor, never LangChain's 4096.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")
    assert _build_model(load_settings()).max_tokens == config.DEFAULT_MAX_TOKENS

    # Tier 1: an explicit override beats both.
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "4242")
    for model_id in ("claude-opus-5-5", "claude-not-a-real-model-9"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        assert _build_model(load_settings()).max_tokens == 4242


def test_unprofiled_model_id_warns(monkeypatch, tmp_path, caplog):
    # A model id LangChain cannot profile must not degrade silently. Uses a fabricated id so the
    # test keeps meaning once real ids gain profiles.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")

    with caplog.at_level(logging.WARNING, logger="speechwriter.agent"):
        _build_model(load_settings())

    assert "model profile" in caplog.text


def test_default_model_still_resolves_through_tier_two(monkeypatch, tmp_path):
    # A tripwire on someone else's data, deliberately. `test_ceiling_resolution_is_three_tier`
    # proves tier 2 works through `claude-opus-5-5` — so the day LangChain stops profiling
    # DEFAULT_MODEL, every other assertion in this file still passes while the default
    # configuration quietly drops to the 32k floor. Nothing would surface it: falling back is
    # *correct* behaviour, just four times smaller, and `ceiling_label` is the only place it shows.
    #
    # It has already happened once: langchain-anthropic 1.6.1 did not profile the 5.5 ids, so the
    # default ran on tier 3 until the pin moved to 1.7.5. A failure is not a bug. It is a prompt
    # to re-read the ceiling notes in CLAUDE.md and config.py, and to decide whether to pin
    # SPEECHWRITER_MAX_TOKENS or the dependency.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    model = _build_model(load_settings())

    assert (model.profile or {}).get("max_output_tokens"), (
        f"LangChain no longer profiles {config.DEFAULT_MODEL}, so the default configuration "
        f"now resolves through tier 3 to the {config.DEFAULT_MAX_TOKENS}-token floor."
    )
    # Not implied by the line above: a profile *below* the floor would keep tier 2 and leave
    # the default running under the ceiling an unprofiled id would have been given.
    assert (model.max_tokens or 0) > config.DEFAULT_MAX_TOKENS, (
        f"{config.DEFAULT_MODEL} profiles at {model.max_tokens}, at or below the "
        f"{config.DEFAULT_MAX_TOKENS} floor — re-check the figure CLAUDE.md quotes."
    )


def test_every_model_choice_is_profiled_above_the_floor(monkeypatch, tmp_path):
    # The roster is a promise: everything the picker offers keeps its *own* 128k ceiling rather
    # than dropping to the 32k floor. Nothing structural enforces it — MODEL_CHOICES is a
    # hand-written tuple, and `_build_model` resolves a typo'd id through tier 3 with only a log
    # line, so a reader would discover it by watching a draft come back short. Like the tripwire
    # above, this asserts against a third-party table; a failure is a prompt to re-pick the
    # roster or the pin, not a bug to fix.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    assert config.MODEL_CHOICES, "an empty roster would make this assertion vacuous"
    for choice in config.MODEL_CHOICES:
        monkeypatch.setenv("SPEECHWRITER_MODEL", choice.model)
        model = _build_model(load_settings())
        assert (model.profile or {}).get("max_output_tokens"), (
            f"LangChain no longer profiles {choice.model!r}, so offering it drops the ceiling "
            f"to the {config.DEFAULT_MAX_TOKENS}-token floor."
        )
        assert (model.max_tokens or 0) > config.DEFAULT_MAX_TOKENS, choice.model


def test_the_default_ceiling_clears_the_longest_commission():
    # A one-sided bound on a constant is satisfied by making it smaller, and smaller is the
    # direction that silently truncates a draft. The floor is also the *thinking* budget for an
    # unprofiled id, so it has to clear the longest speech the committed datasets actually grade
    # plus room to deliberate.
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


def test_payload_omits_parameters_current_models_reject(monkeypatch, tmp_path):
    # temperature/top_p/top_k are rejected outright (400) on the 5.5 models. `build_agent()`
    # never touches the wire — that is what makes this suite free — so nothing else here would
    # notice a langchain-anthropic bump that began sending one by default; instead every real
    # turn would fail, far from the upgrade that caused it. `_get_request_payload` builds the
    # dict offline, so the seam is assertable for free. It is private, like the other
    # LangChain/deepagents internals this file reaches into: a rename breaks this test loudly,
    # which is the failure mode we want.
    rejected = {"temperature", "top_p", "top_k"}
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    for model_id in (config.DEFAULT_MODEL, "claude-opus-5-5", "claude-not-a-real-model-9"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        # Every ceiling branch, since a stray default could be injected on any call: None
        # exercises the profile/floor paths, the override exercises tier 1.
        for override in (None, "4242"):
            if override is None:
                monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
            else:
                monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", override)
            payload = _build_model(load_settings())._get_request_payload([])
            assert rejected.isdisjoint(payload), f"{model_id} (override={override}): {payload}"


def test_every_call_carries_the_thinking_settings_the_models_need(monkeypatch, tmp_path):
    # Each of these fails differently, and none of them fails at build time:
    #
    # * effort is set explicitly because the default differs by model (`high` on Sonnet 5.5,
    #   `medium` on Opus 5.5) — a switch would otherwise silently change what every turn costs;
    # * `display: summarized`, because the default `omitted` returns empty thinking blocks and on
    #   Sonnet 5.5 the notes *between* tool calls arrive as thinking blocks — a long turn goes
    #   silent;
    # * `block_binding: drop_block` plus its beta, because deepagents' summarisation edits
    #   history and newer accounts 400 when an edited history replays a thinking block — the
    #   first compaction in a long session would fail the next call outright;
    # * `fallbacks: default` plus its beta, so a classifier decline is retried server-side
    #   rather than handed back as an empty turn.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)

    for model_id in (config.DEFAULT_MODEL, "claude-opus-5-5", "claude-not-a-real-model-9"):
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        payload = _build_model(load_settings())._get_request_payload([])

        assert payload["output_config"]["effort"] == config.DEFAULT_EFFORT, model_id
        thinking = payload["thinking"]
        assert thinking["type"] == "adaptive", model_id
        assert thinking["display"] == "summarized", model_id
        assert thinking["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}, model_id
        assert payload["fallbacks"] == "default", model_id
        assert "thinking-binding-controls-2026-08-01" in payload["betas"], model_id
        assert "server-side-fallback-2026-07-01" in payload["betas"], model_id
        # Forced tool choice is a 400 on the 5.5 models; nothing here may default it on.
        assert "tool_choice" not in payload, model_id


def test_the_model_client_streams(monkeypatch, tmp_path):
    # Profiled ceilings are 128k, and the Anthropic SDK refuses a *non-streaming* request whose
    # max_tokens could outlast its ten-minute timeout — so a client built without streaming
    # would fail every turn at the SDK, before the request is sent.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    assert _build_model(load_settings()).streaming is True


def test_the_credentials_gate_is_a_presence_check(monkeypatch, tmp_path):
    # The CLI refuses commissions on this and the web UI disables its chat input, so a false
    # negative silently bricks a working setup and a false positive defers the failure to a 401
    # at the first turn. Blank is how a copied dotenv template says "unset", and an unstripped
    # blank is *truthy* — it would pass the gate and be sent as the key.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_settings().model_credentials_present is False

    for blank in ("", "   "):
        monkeypatch.setenv("ANTHROPIC_API_KEY", blank)
        settings = load_settings()
        assert settings.anthropic_api_key is None, repr(blank)
        assert settings.model_credentials_present is False, repr(blank)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-dummy  ")
    settings = load_settings()
    assert settings.anthropic_api_key == "sk-ant-dummy"
    assert settings.model_credentials_present is True


def test_the_key_reaches_the_client_exactly_once(monkeypatch, tmp_path):
    # The key is read by `load_settings` and handed to the client explicitly, so the credential
    # the gate checked is the one the client sends — never a second read of the environment
    # that could disagree with it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")

    model = _build_model(load_settings())

    assert model.anthropic_api_key.get_secret_value() == "sk-ant-dummy"


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
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    bundle = build_agent()

    config = bundle.turn_config("thread-1")
    assert config["configurable"]["thread_id"] == "thread-1"

    callbacks = config["callbacks"]
    assert isinstance(callbacks, list)  # narrows the RunnableConfig union
    assert bundle.warner in callbacks


def test_truncation_warner_counts_refusals_and_names_their_category():
    # A refusal the server-side fallback could not rescue is an HTTP 200 with
    # stop_reason="refusal" and little or no text — inside a subagent that reaches the
    # orchestrator as the same empty status="success" result a truncation does, so it is
    # counted the same way. `stop_details` is optional, so an unnamed refusal still counts.
    warner = TruncationWarner()

    def result(**metadata: object) -> LLMResult:
        message = AIMessage(content="", response_metadata=metadata)
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    warner.on_llm_end(
        result(stop_reason="refusal", stop_details={"category": "general_harms"}),
        run_id=uuid.uuid4(),
    )
    warner.on_llm_end(result(stop_reason="refusal"), run_id=uuid.uuid4())
    warner.on_llm_end(result(stop_reason="end_turn"), run_id=uuid.uuid4())

    assert warner.refused == 2
    assert warner.refusal_categories == ["general_harms"]
    # A refusal is not a truncation — the two call for different advice.
    assert warner.truncated == 0

    warner.reset()
    assert (warner.refused, warner.refusal_categories) == (0, [])


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


def test_orchestrator_prompt_names_every_skill(monkeypatch, tmp_path):
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

    # Isolated like every other test that loads settings. It used to call `load_settings()` on
    # the real repo, which read the developer's dotenv into `os.environ` for the rest of the
    # process — invisible until tracing arrived, when a real PHOENIX_COLLECTOR_ENDPOINT leaked
    # that way made every later `build_agent()` in the suite trace to the developer's Phoenix.
    # The skills are read off the real tree above; the prompt does not depend on the home.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
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


def test_an_unprofiled_anthropic_id_does_not_keep_langchains_4096(monkeypatch, tmp_path):
    # The other half of the tier-2 condition, and the reason it cannot be simplified to a bare
    # max_tokens check: ChatAnthropic *always* carries a max_tokens, and for an unprofiled id
    # that value is LangChain's silent 4096 fallback — the original trap. Asserting the resolved
    # ceiling alone would happily accept it.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")

    assert _build_model(load_settings()).max_tokens == config.DEFAULT_MAX_TOKENS


def test_the_resolved_ceiling_reaches_the_request_payload(monkeypatch, tmp_path):
    # A ceiling set on the client but dropped from the payload is no ceiling at all. Assert the
    # *value* is carried under some key rather than pinning the spelling, so a rename upstream
    # fails loudly here instead of silently unbounding the model.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))

    for override, model_id in (
        ("12345", config.DEFAULT_MODEL),
        ("12345", "claude-not-a-real-model-9"),
        (None, "claude-not-a-real-model-9"),
    ):
        if override is None:
            monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
        else:
            monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", override)
        monkeypatch.setenv("SPEECHWRITER_MODEL", model_id)
        expected = int(override) if override else config.DEFAULT_MAX_TOKENS
        payload = _build_model(load_settings())._get_request_payload([])
        carrying = {key for key, value in payload.items() if value == expected}
        assert carrying, f"{model_id}: ceiling {expected} absent from payload {sorted(payload)}"


def test_settings_can_still_be_built_with_only_the_required_fields(tmp_path):
    # `build_agent(settings)` is the documented library entry point, so Settings is part of
    # the public surface: adding an optional capability must not break a caller that predates
    # it. Mirrors the defaulting already argued for on SpeechwriterAgent's own added fields.
    #
    # The local-endpoint fields used to sit after `max_tokens`, and removing them shifted every
    # positional argument after them — a deliberate breaking change, because an endpoint this
    # agent can no longer reach would read as configuration in use. Named arguments, so this
    # test says nothing about that ordering; it is `Settings`' own comment that carries it.
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

    # No key is a coherent, reportable state rather than a construction error: the gate says
    # so, and the build still succeeds offline.
    assert settings.anthropic_api_key is None
    assert settings.model_credentials_present is False


def test_the_configured_model_is_always_offered(monkeypatch, tmp_path):
    # Streamlit *silently* rewrites a selection that is not among a widget's options to option
    # zero — no exception, no log. So a roster that did not contain the configured model would
    # retarget a reader who set SPEECHWRITER_MODEL to an older Claude id onto Sonnet 5.5.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5")

    settings = load_settings()
    offered = config.model_choices(settings)

    assert offered[: len(config.MODEL_CHOICES)] == config.MODEL_CHOICES
    assert offered[-1] == config.ModelChoice("claude-opus-5", "claude-opus-5")

    # ...and a configuration already on the roster is not offered a second time, however many
    # times it is handed over — both front ends pass the configured *and* the live settings.
    assert config.model_choices(settings, settings) == offered
    monkeypatch.setenv("SPEECHWRITER_MODEL", config.DEFAULT_MODEL)
    assert config.model_choices(load_settings()) == config.MODEL_CHOICES


def test_switching_away_from_an_off_roster_model_leaves_a_way_back(monkeypatch, tmp_path):
    # The roster must be widened by *every* configuration that has to stay reachable, not only
    # the live one. A roster derived from the post-switch settings alone drops the off-roster id
    # the reader came from: the way back would be removed by the act of leaving, and nothing
    # short of a restart brings it back. Callers therefore pass the configuration the session
    # *started* on as well as the one now in force.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-opus-5")

    configured = load_settings()
    switched = config.MODEL_CHOICES[0].applied_to(configured)

    offered = config.model_choices(configured, switched)

    assert any(c.model == configured.model for c in offered), (
        "the model the session started on vanished once another was selected — there is no "
        "way back to it without restarting the process"
    )
    # And no duplicate for the entry that is now both current and already on the list.
    assert len(offered) == len(config.MODEL_CHOICES) + 1


def test_switching_models_carries_learned_memory_across_the_rebuild(monkeypatch, tmp_path):
    # A model switch is a *rebuild*, and `build_agent` rehydrates a brand-new InMemoryStore from
    # the on-disk snapshot — so everything learned since the last save is dropped unless
    # `persist()` runs first. Both front ends switch this way. Reversing the two lines loses
    # voice profiles with no error at all, which is why the order is asserted here rather than
    # only commented at the call sites.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.delenv("SPEECHWRITER_MAX_TOKENS", raising=False)
    configured = load_settings()

    first, second = config.MODEL_CHOICES[0], config.MODEL_CHOICES[1]

    bundle = build_agent(first.applied_to(configured))
    bundle.store.put(("speechwriter", "memories"), "mayor.md", {"content": "Plain speaker."})

    bundle.persist()
    switched = build_agent(second.applied_to(configured))

    assert switched.store is not bundle.store
    assert [item.key for item in memory.all_items(switched.store)] == ["mayor.md"]
    assert switched.settings.model == second.model


def test_an_oversized_ceiling_override_is_reported_beside_the_label_not_inside_it(
    monkeypatch, tmp_path
):
    # SPEECHWRITER_MAX_TOKENS is tier 1 and global, so an override sized for one model follows a
    # switch to another — and the API rejects a ceiling above the model's maximum at the first
    # turn rather than clamping it.
    #
    # Two facts, two members, and that separation is the point: a warning folded into
    # `ceiling_label` lands in the sentence both front ends use to tell the reader to *raise*
    # the ceiling, which argues with itself.
    monkeypatch.setenv("SPEECHWRITER_HOME", str(tmp_path))
    monkeypatch.delenv("SPEECHWRITER_MODEL", raising=False)
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "200000")

    over = build_agent(load_settings())
    assert over.profiled_max_tokens is not None
    assert over.ceiling_exceeds_model is True
    # The label stays a bare figure, so the sentence it lands in still reads correctly.
    assert over.ceiling_label == "200,000"

    # An override within the model's maximum raises nothing.
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "32000")
    assert build_agent(load_settings()).ceiling_exceeds_model is False

    # Nor does an unprofiled id: with no maximum to compare against there is nothing to claim.
    monkeypatch.setenv("SPEECHWRITER_MODEL", "claude-not-a-real-model-9")
    monkeypatch.setenv("SPEECHWRITER_MAX_TOKENS", "200000")
    unknown = build_agent(load_settings())
    assert unknown.profiled_max_tokens is None
    assert unknown.ceiling_exceeds_model is False


def test_the_bundle_still_takes_its_fields_in_the_documented_order():
    # `SpeechwriterAgent` is public: `build_agent` returns it and the README documents that
    # path. Fields are appended, never inserted — a defaulted field in the *middle* still shifts
    # every positional argument after it, which is the mistake `config.Settings` spells out and
    # this class once made when a second ceiling field landed ahead of `warner`. A consumer
    # writing `SpeechwriterAgent(agent, store, settings, 32000, my_warner)` had their warner
    # bound to that field: no truncation signal, and a TypeError from the first comparison.
    fields = [f.name for f in dataclasses.fields(SpeechwriterAgent)]

    assert fields[:5] == ["agent", "store", "settings", "max_tokens", "warner"], fields
    # Anything added later belongs after those, in the order it was added. The sixth slot has
    # held `profiled_max_tokens`, then `context_window` while the agent ran on local models, and
    # `profiled_max_tokens` again — each swapped in place, which is the one edit that keeps this
    # order. `tracing` came after it, appended.
    assert fields[5:] == ["profiled_max_tokens", "tracing"], fields

    # Deliberately no well-formed-but-unreachable address here. An earlier version asserted on
    # `http://[::1]:8080/v1`, which opens a real TCP connection — breaking the suite's offline
    # invariant, and going red for any contributor running the `mlx_lm.server --port 8080` the
    # README recommends, since a listening server answers and the result is no longer `[]`.
    # The unclosed bracket is the whole point: it raises inside `urlsplit`, before any socket.


def test_an_ambiguous_model_name_is_refused_rather_than_passed_through():
    # `resolve_choice` answers None for "no such entry" and for "two entries by that name", and
    # the eval harness's pass-through is only right for the first: it would set
    # SPEECHWRITER_MODEL to the literal text the reader typed. `matching_choices` is what lets a
    # caller tell the two Nones apart.
    sonnet, opus = config.MODEL_CHOICES[0], config.MODEL_CHOICES[1]
    roster = (sonnet, opus)

    assert config.matching_choices(roster, "nonesuch") == []
    assert config.matching_choices(roster, "opus 5.5") == [opus]
    assert config.matching_choices(roster, opus.model) == [opus]

    # Two entries that share a label — the only way a curated roster could collide.
    twin = config.ModelChoice(opus.label, "claude-opus-5")
    ambiguous = roster + (twin,)
    assert len(config.matching_choices(ambiguous, opus.label)) == 2
    assert config.resolve_choice(ambiguous, opus.label) is None
    # A row number is never ambiguous.
    assert config.resolve_choice(ambiguous, "3") == twin
