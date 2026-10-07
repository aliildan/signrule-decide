#!/usr/bin/env bash
# Full Norwegian Fullmakt fetch as a detached, resumable job (plan-02 T5).
# Tier 1 (AS, ASA) first, seeded shuffle within tiers; cached lookups are skipped on restart.
# Progress (aggregate only) is appended to runs/ingest/no-fullmakt.log.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG="$REPO/runs/ingest/no-fullmakt.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"
CMD=("$REPO/.venv/bin/python" -m signrule.ingest.ingest_no fullmakt --workers 4 --progress 5000 "$@")
if command -v systemd-run >/dev/null && systemd-run --user --quiet true 2>/dev/null; then
  systemd-run --user --unit=signrule-ingest-no --collect --same-dir \
    bash -c "${CMD[*]} >> '$LOG' 2>&1"
  echo "started systemd user unit signrule-ingest-no; log: $LOG"
else
  setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
  echo "started pid $!; log: $LOG"
fi
