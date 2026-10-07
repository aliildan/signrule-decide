#!/bin/sh
# Rehearse the CI steps (lint, format, tests, publish check, tables) on the committed tree only: no docs/, data/, CLAUDE.md, runs/ or other
# git-ignored or untracked files, exactly what GitHub Actions sees. Catches code and tests that
# silently depend on private local files. Used by the pre-push hook; also runnable by hand.
set -eu
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ref="${1:-HEAD}"
git -C "$repo" archive "$ref" | tar -x -C "$tmp"
cd "$tmp"
# The same lint/format step as CI (ruff also formats Python blocks in Markdown files).
"$repo/.venv/bin/ruff" check . && "$repo/.venv/bin/ruff" format --check .
# The committed src/ must win over the editable install of the working tree.
PYTHONPATH="$tmp/src" SIGNRULE_DATA_DIR="$tmp/data" \
  "$repo/.venv/bin/python" -m pytest -q -p no:cacheprovider -p no:warnings -x
(cd "$repo" && python3 scripts/publish_check.py --publish)  # needs git ls-files
python3 scripts/make_tables.py --check
