#!/usr/bin/env bash
# Stop-hook test gate for speechwriter-agent.
#
# There is no CI and no pre-commit here, and CLAUDE.md requires `uv run pytest` to stay
# clean. The suite is fully offline (build_agent() never calls the model or the network)
# and runs in ~1.2s, so gating the end of a turn on it is cheap.
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
watched=(src tests skills pyproject.toml uv.lock)
dirty=$(git status --porcelain --untracked-files=all -- "${watched[@]}" 2>/dev/null)
[ -n "$dirty" ] || exit 0

# Exiting 2 because uv is missing would block the turn with a "command not found"
# masquerading as a test failure.
if ! command -v uv >/dev/null 2>&1; then
  printf 'pytest gate skipped: uv is not on PATH.\n' >&2
  exit 0
fi

out=$(uv run pytest -q 2>&1)
status=$?
[ "$status" -eq 0 ] && exit 0

# A dependency change makes two specific, documented fragilities the prime suspects.
dep_hint=""
case "$dirty" in
  *pyproject.toml* | *uv.lock*)
    dep_hint=$'\n\nA dependency file is among the changes, so check the two known-fragile imports in tests/test_build.py first: `yaml` (pyyaml is not a declared dependency -- it arrives transitively via langchain) and `deepagents.middleware.filesystem._check_fs_permission` (a private API). deepagents is pinned >=0.6.12,<0.8 because implicit Store namespaces were removed in 0.7. test_build.py imports speechwriter.agent at module scope, so a deepagents API break surfaces as a collection error across all tests, not a single failure.'
    ;;
esac

# Cap the feedback: a collection error dumps every traceback into context.
printf '`uv run pytest` is failing. CLAUDE.md requires this gate to stay clean, and nothing else (no CI, no pre-commit) checks it. Fix the failures, then stop.%s\n\n%s\n' \
  "$dep_hint" "$(printf '%s\n' "$out" | tail -c 4000)" >&2
exit 2
