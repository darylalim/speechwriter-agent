#!/usr/bin/env bash
# PostToolUse(Edit|Write) gate: keeps `uvx ruff check .` and `uvx ty check` green on every
# Python edit. Exit 2 shows stderr to Claude so it can fix the diagnostics inline.
#
# CLAUDE.md declares all three gates clean and says "Keep them that way", but there is no
# CI, no pre-commit, and no git hooks here. Measured cost: ruff ~0.18s + ty ~0.30s.
set -uo pipefail

payload=$(cat)

# --- Parse the edited path. Never no-op silently if the parser is missing. ---
if command -v jq >/dev/null 2>&1; then
  file=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
elif command -v python3 >/dev/null 2>&1; then
  file=$(printf '%s' "$payload" | python3 -c \
    'import json,sys;d=json.load(sys.stdin);print((d.get("tool_input") or {}).get("file_path") or "")' \
    2>/dev/null)
else
  echo "ruff-ty-gate: neither jq nor python3 is on PATH; the ruff/ty gate cannot run." >&2
  exit 1
fi
[ -n "$file" ] || exit 0

# --- Project root: prefer CLAUDE_PROJECT_DIR, else derive from this script's own location.
# Never fall back to "." -- an absolute file_path can never match a "./*" prefix test,
# which would make the whole gate a silent no-op.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 1
root="${CLAUDE_PROJECT_DIR:-${script_dir%/.claude/hooks}}"
root=$(cd -- "$root" 2>/dev/null && pwd -P) || {
  echo "ruff-ty-gate: cannot resolve project root '$root'." >&2
  exit 1
}

# --- Normalise the path (absolute + symlink-resolved) before any prefix comparison. ---
case "$file" in /*) ;; *) file="$PWD/$file" ;; esac
file_dir=$(cd -- "$(dirname -- "$file")" 2>/dev/null && pwd -P) || exit 0
file="$file_dir/$(basename -- "$file")"

case "$file" in *.py | *.pyi) ;; *) exit 0 ;; esac
case "$file" in "$root"/*) ;; *) exit 0 ;; esac
case "${file#"$root"/}" in
  .venv/* | .git/* | build/* | dist/* | .ruff_cache/* | .pytest_cache/*) exit 0 ;;
esac
[ -f "$file" ] || exit 0   # renamed or deleted: ruff would report a bogus E902

cd -- "$root" || exit 1
rel="${file#"$root"/}"

command -v uvx >/dev/null 2>&1 || {
  echo "ruff-ty-gate: 'uvx' is not on PATH, so the ruff/ty gate did not run for $rel." >&2
  exit 1
}

# --- Autofix what is mechanically fixable (line-length 100, isort via lint select "I"). ---
before=$(cksum < "$file")
uvx ruff format --quiet -- "$file" >/dev/null 2>&1
uvx ruff check --fix --quiet -- "$file" >/dev/null 2>&1
after=$(cksum < "$file")

lint_out=$(uvx ruff check -- "$file" 2>&1)
lint_status=$?
type_out=$(uvx ty check 2>&1)
type_status=$?

if [ "$lint_status" -ne 0 ] || [ "$type_status" -ne 0 ]; then
  {
    echo "The quality gates CLAUDE.md requires to stay clean are failing after your edit to $rel:"
    if [ "$lint_status" -ne 0 ]; then
      printf '\n[uvx ruff check %s]\n%s\n' "$rel" "$lint_out"
    fi
    if [ "$type_status" -ne 0 ]; then
      printf '\n[uvx ty check] (whole project; diagnostics may be in files you did not edit)\n%s\n' "$type_out"
    fi
    echo
    echo "Fix these before continuing. Prefer typing something precisely over widening to Any;"
    echo "if a suppression is truly unavoidable use a rule-specific '# ty: ignore[rule-name]'."
    if [ "$before" != "$after" ]; then
      echo "Note: ruff also reformatted $rel on disk, so re-read it before editing it again."
    fi
  } >&2
  exit 2
fi

# Gates are green. If ruff rewrote the file, tell Claude so it does not edit from a stale copy.
# (JSON on stdout is only read on exit 0 -- never combine it with exit 2.)
if [ "$before" != "$after" ]; then
  safe_rel=$(printf '%s' "$rel" | tr -d '"\\')
  printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"ruff format/--fix rewrote %s after your edit, so the file on disk differs from what you wrote. Re-read it before making further edits to it."}}\n' "$safe_rel"
fi
exit 0
