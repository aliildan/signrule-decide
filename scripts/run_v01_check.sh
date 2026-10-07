#!/usr/bin/env bash
# v0.1 (RI-with-text fix, same entities as v0): 4B train -> bench -> score -> demo.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY="$REPO/.venv/bin/python"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "train no-4b-v01"
"$PY" -m signrule.train.kev_wrapper train --config configs/train/4b-v01.yaml > runs/no-4b-v01.log 2>&1 || { log "train failed"; exit 1; }
for part in val test; do
  "$PY" -m signrule.train.kev_wrapper bench --run runs/no-4b-v01 --split random --part $part --raw > /dev/null
done
"$PY" eval/run_all.py --split random --part val --kev "no-4b-v01=runs/no-4b-v01-bench-no-random-val,runs/no-4b-v01-bench-no-random-test"
"$PY" eval/run_all.py --split random --part test --allow-test --kev "no-4b-v01=runs/no-4b-v01-bench-no-random-val,runs/no-4b-v01-bench-no-random-test"
log "demo"
"$PY" server/app.py --run runs/no-4b-v01 --policy results/no-random/no-4b-v01.test.json --port 8300 > runs/serve-4b-v01.log 2>&1 &
SERVER=$!
until curl -s localhost:8300/healthz > /dev/null 2>&1; do sleep 3; done
"$PY" scripts/demo.py
kill $SERVER
log "done"
