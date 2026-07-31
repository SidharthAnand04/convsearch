#!/usr/bin/env bash
# Start convsearch locally, creating and optionally populating a workspace first.
# Usage: bash scripts/convsearch-up.sh [WORKSPACE] [CHATGPT_EXPORT_ZIP]
# The export may also be supplied as CONVSEARCH_EXPORT_ZIP (or CHATGPT_EXPORT_ZIP).

set -euo pipefail

WORKSPACE="${1:-./workspace}"
EXPORT_ZIP="${2:-${CONVSEARCH_EXPORT_ZIP:-${CHATGPT_EXPORT_ZIP:-}}}"
DB_PATH="$WORKSPACE/database/convsearch.sqlite3"

# In a checkout with a uv project, use its locked environment. --no-sync avoids
# waiting on the uv lock when a previously started server is still running.
if { [ -f pyproject.toml ] || [ -d .venv ]; } && command -v uv >/dev/null 2>&1; then
  CONVSEARCH=(uv run --no-sync convsearch)
elif command -v convsearch >/dev/null 2>&1; then
  CONVSEARCH=(convsearch)
elif command -v uv >/dev/null 2>&1; then
  CONVSEARCH=(uv run --no-sync convsearch)
else
  echo "error: neither 'uv' nor 'convsearch' is available on PATH." >&2
  echo "Install the engine first: uv sync --extra ml --extra llm" >&2
  exit 1
fi

run_convsearch() { "${CONVSEARCH[@]}" "$@"; }

if [ ! -f "$DB_PATH" ]; then
  if [ -n "$EXPORT_ZIP" ] && [ ! -f "$EXPORT_ZIP" ]; then
    echo "error: ChatGPT export ZIP not found: $EXPORT_ZIP" >&2
    exit 1
  fi
  echo "No workspace database found; initializing: $WORKSPACE"
  run_convsearch init "$WORKSPACE"
  if [ -n "$EXPORT_ZIP" ]; then
    echo "Importing ChatGPT export: $EXPORT_ZIP"
    run_convsearch import "$EXPORT_ZIP" -w "$WORKSPACE"
    echo "Building the local search index..."
    run_convsearch index -w "$WORKSPACE"
  else
    echo "Workspace is empty. To add history later, run:"
    echo "  ${CONVSEARCH[*]} import <your-ChatGPT-export.zip> -w \"$WORKSPACE\""
    echo "  ${CONVSEARCH[*]} index -w \"$WORKSPACE\""
  fi
else
  echo "Using existing workspace: $WORKSPACE"
fi

echo "Starting local convsearch server: http://127.0.0.1:8756"
echo "Next: load the unpacked extension/ folder in chrome://extensions, then open its side panel."
echo "Press Ctrl+C to stop."
exec "${CONVSEARCH[@]}" serve -w "$WORKSPACE"
