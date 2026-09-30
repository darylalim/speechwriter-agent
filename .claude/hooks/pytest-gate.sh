#!/usr/bin/env bash
# Stop-hook test gate for speechwriter-agent.
#
# CLAUDE.md requires `uv run pytest` to stay clean. CI (.github/workflows/ci.yml) runs it
# too, matrixed over 3.11-3.13, but only after a push. The suite is fully offline
# (build_agent() never calls the model or the network) and runs in ~1.2s, so gating the end
# of a turn on it is cheap and keeps the failure from becoming a commit in the first place.
#
# Scope note: this fires when the WORKING TREE is dirty under the watched paths, not when
# "this turn" touched them -- a Stop hook has no reliable per-turn diff. Once there are
# uncommitted changes under src/ or tests/, the gate runs on every turn until they are
# committed. At ~1.2s that is the intended trade.
#
# Fail OPEN on missing tooling (jq, git, uv); fail CLOSED only on a real test failure.
set -uo pipefail

payload=$(cat)

# Never re-block a turn this hook already blocked.
if command -v jq >/dev/null 2>&1; then
  active=$(printf '%s' "$payload" | jq -r '.stop_hook_active // false' 2>/dev/null)
else
  # jq is not guaranteed to be installed; without a fallback the loop guard is silently
  # lost and the gate can block repeatedly.
  active=false
  case "$payload" in
    *'"stop_hook_active":true'* | *'"stop_hook_active": true'*) active=true ;;
  esac
fi
[ "$active" = "true" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 0
command -v git >/dev/null 2>&1 || exit 0

# Paths the suite covers. `git status --porcelain` is required: `git diff` sees only
# TRACKED files, so a new tests/test_*.py or a new skills/<slug>/SKILL.md -- exactly what
# trips the hard-coded `len(skill_dirs) == 4` assertion -- would slip past untested.
#
# .streamlit is watched because test_streamlit_config_parses_and_offers_both_theme_modes
# reads config.toml off disk: a TOML typo or a dropped [theme.dark] table is caught by the
# suite but by nothing else in the inner loop, since that file is not Python and so never
# reaches ruff-ty-gate.sh. Ignored files stay invisible here -- --untracked-files=all lists
# untracked but not ignored paths -- so a local, gitignored secrets file never pins the gate on.
#
# evals is watched for the same reason as .streamlit: test_eval_datasets_match_the_live_contract
# checks the committed dataset JSON against config.py, and JSON never reaches ruff-ty-gate.sh
# either. Without it, editing a dataset -- the most likely edit in that folder by far -- would
# leave the inner loop silent and the drift would surface only in CI.
watched=(src tests skills .streamlit evals pyproject.toml uv.lock)
dirty=$(git status --porcelain --untracked-files=all -- "${watched[@]}" 2>/dev/null)
[ -n "$dirty" ] || exit 0

# Exiting 2 because uv is missing would block the turn with a "command not found"
# masquerading as a test failure. Exit 1 rather than 0: both are non-blocking, but only
# exit 1 surfaces the note -- a stderr line on exit 0 is swallowed, so a gate that has
# quietly stopped running looks exactly like a gate that is passing. Same discipline as
# ruff-ty-gate.sh, which exits 1 for missing jq/python3/uvx.
if ! command -v uv >/dev/null 2>&1; then
  printf 'pytest gate skipped: uv is not on PATH, so the suite did not run.\n' >&2
  exit 1
fi

out=$(uv run pytest -q 2>&1)
status=$?
[ "$status" -eq 0 ] && exit 0

# A dependency change makes two specific, documented fragilities the prime suspects.
dep_hint=""
case "$dirty" in
  *pyproject.toml* | *uv.lock*)
    dep_hint=$'\n\nA dependency file is among the changes, so check the three private third-party symbols tests/test_build.py reaches into first: `deepagents.middleware.filesystem._check_fs_permission`, `agent.nodes["tools"].bound.tools_by_name`, and `ChatAnthropic._get_request_payload`. deepagents is pinned >=0.6.12,<0.8 because implicit Store namespaces were removed in 0.7. test_build.py imports speechwriter.agent at module scope, so a deepagents API break surfaces as a collection error across all tests, not a single failure.'
    ;;
esac

# Cap the feedback: a collection error dumps every traceback into context.
printf '`uv run pytest` is failing. CLAUDE.md requires this gate to stay clean, and CI will fail on it the moment this is pushed. Fix the failures now, then stop.%s\n\n%s\n' \
  "$dep_hint" "$(printf '%s\n' "$out" | tail -c 4000)" >&2
exit 2
