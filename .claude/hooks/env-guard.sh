#!/usr/bin/env bash
# PreToolUse(Bash): block shell access to .env files.
#
# WHY THIS EXISTS. `permissions.deny` in .claude/settings.json is TOOL-SCOPED:
# "Read(./.env)" stops the Read tool and nothing else. `cat .env`, `grep KEY .env`,
# `sed -n 1p .env`, `source .env` and a heredoc that overwrites it all go through Bash,
# which no deny rule covered -- so the file holding ANTHROPIC_API_KEY and TAVILY_API_KEY
# was one tool away from readable. Claude Code's auto mode makes that the *default* path,
# since it instructs the model to read files with cat/sed/grep rather than the Read tool.
#
# FAIL CLOSED, deliberately -- the inverse of every other hook here. The other three guard
# code quality, where a missing `uv` must never masquerade as a broken build, so they fail
# open. This one guards credentials: a parser it cannot find must not silently switch it
# off, because then you believe you have a guard that is not running.
#
# Scope note: this matches `.env` and `.env.<suffix>`, INCLUDING the committed, secret-free
# `.env.example`, to mirror the existing `Read(./.env.*)` deny rule exactly. Narrowing it is
# one line (add an .env.example exemption below) if the friction outweighs the consistency.
set -uo pipefail

payload=$(cat)

deny() {
  printf '%s\n' "$1" >&2
  exit 2
}

command -v python3 >/dev/null 2>&1 || deny \
  "env-guard: python3 is not on PATH, so the .env guard cannot inspect this command.
Refusing it rather than silently disabling a credential guard. Install python3, or remove
this hook from .claude/settings.json if you accept the risk."

# The pattern deliberately uses lookarounds instead of a delimiter character class, so it
# needs no quote characters and survives shell quoting intact. It keeps `.envrc` (direnv)
# and `settings.environment` out, while catching `.env`, `./.env`, `$HOME/.env`,
# `.env.local` and `.env.example` wherever they appear as a path token. `.streamlit/
# secrets.toml` is covered too: CLAUDE.md names it as the web UI's secrets location.
hit=$(CLAUDE_ENV_GUARD_PAYLOAD="$payload" python3 -c '
import json, os, re
try:
    data = json.loads(os.environ.get("CLAUDE_ENV_GUARD_PAYLOAD") or "")
except Exception:
    data = None
if not isinstance(data, dict):
    print("!parse")
    raise SystemExit(0)
cmd = (data.get("tool_input") or {}).get("command") or ""
pat = re.compile(r"(?<![A-Za-z0-9_-])(?:\.env(?:\.[A-Za-z0-9_-]+)?|secrets\.toml)(?![A-Za-z0-9_-])")
m = pat.search(cmd)
print(m.group(0) if m else "")
') || deny "env-guard: the .env guard failed to run, so this command was refused."

[ -n "$hit" ] || exit 0

if [ "$hit" = "!parse" ]; then
  deny "env-guard: could not parse the tool payload, so the .env guard could not inspect
this command. Refusing rather than silently disabling a credential guard."
fi

deny "Blocked: this Bash command references '$hit', which is a secrets file and off-limits.

\`permissions.deny\` only covers the Read and Edit *tools*; Bash routes around them, which
is the gap this hook closes. .env holds ANTHROPIC_API_KEY and TAVILY_API_KEY.

If you need a value from it: ask the user to check it and report back, or verify it
indirectly (assert the variable is non-empty, run the offline test suite) without printing
it. To change a setting, ask the user to edit .env themselves.

If you only meant to write '.env' in prose -- a commit message, a doc, a comment -- rephrase
to avoid the literal token, or make the edit with the Write/Edit tool instead of a heredoc."
