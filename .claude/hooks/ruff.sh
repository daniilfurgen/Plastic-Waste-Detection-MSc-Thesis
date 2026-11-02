#!/usr/bin/env bash
# PostToolUse hook: format and lint a Python file after Claude edits it.
# Reads the hook JSON from stdin. Exit 2 feeds remaining (unfixable) lint
# errors back to Claude; any other outcome is silent.

f=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))')

case "$f" in
    *.py) ;;
    *) exit 0 ;;
esac
[ -f "$f" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

uv run ruff format --quiet "$f" >&2 || exit 0
if ! out=$(uv run ruff check --fix --quiet "$f" 2>&1); then
    printf 'ruff check found issues that could not be auto-fixed:\n%s\n' "$out" >&2
    exit 2
fi
