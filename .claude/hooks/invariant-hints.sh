#!/usr/bin/env bash
# PostToolUse(Edit|Write): surfaces the cross-file invariants CLAUDE.md documents but
# nothing in the code enforces. Advisory only -- this hook emits context and never blocks.
#
# Path filtering is done here rather than via the settings.json "if" field so the matching
# is explicit and directly testable (see the self-test at the bottom of this file).
set -uo pipefail

payload=$(cat)

command -v python3 >/dev/null 2>&1 || exit 0   # hints are advisory; never fail a turn over tooling

if command -v jq >/dev/null 2>&1; then
  file=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
else
  file=$(printf '%s' "$payload" | python3 -c \
    'import json,sys;d=json.load(sys.stdin);print((d.get("tool_input") or {}).get("file_path") or "")' \
    2>/dev/null)
fi
[ -n "$file" ] || exit 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 0
root="${CLAUDE_PROJECT_DIR:-${script_dir%/.claude/hooks}}"
root=$(cd -- "$root" 2>/dev/null && pwd -P) || exit 0

case "$file" in /*) ;; *) file="$PWD/$file" ;; esac
case "$file" in "$root"/*) ;; *) exit 0 ;; esac
rel="${file#"$root"/}"

emit() {
  python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":sys.argv[1]}}))' "$1"
}

case "$rel" in
  skills/*)
    # The skill-count assertion is hard-coded and carries no assert message, so adding a
    # skill fails the suite on a line that looks unrelated to the change.
    [ -d "$root/skills" ] && [ -f "$root/tests/test_build.py" ] || exit 0
    n=$(find "$root/skills" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    e=$(grep -oE 'len\(skill_dirs\) == [0-9]+' "$root/tests/test_build.py" | grep -oE '[0-9]+' | head -1)
    [ -n "$e" ] || exit 0
    [ "$n" = "$e" ] && exit 0
    emit "skills/ now holds $n skill directories, but tests/test_build.py hard-codes 'assert len(skill_dirs) == $e'. CLAUDE.md: adding or removing a skill requires updating that assertion, in the same change. A new SKILL.md also needs YAML frontmatter whose name matches its directory slug exactly, a non-empty description, and all four sections: '## Overview', '## When to Use', '## Instructions', '## Pitfalls'. Consider also adding the skill to the parenthetical list in orchestrator_prompt() in src/speechwriter/prompts.py, which names the library inline."
    ;;
  src/speechwriter/config.py)
    emit "config.py is the single source for memories_vpath / skills_vpath / workspace_vpath. CLAUDE.md: three consumers must agree and nothing enforces the agreement -- (1) agent.py backend() routes, (2) agent.py _write_sandbox() allow rules, (3) prompts.py prompt interpolation. README.md's routing table is an undeclared fourth consumer. Note the deliberate asymmetry: skills_vpath and memories_vpath carry a trailing slash, workspace_vpath does not, because prompts.py renders '{workspace_vpath}/speeches/<slug>.md'. Normalizing the three for consistency silently produces '/workspace//speeches/'. If a path changed, propagate it to all four and re-read prompts.py."
    ;;
esac
exit 0
