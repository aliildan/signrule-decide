#!/usr/bin/env bash
# Ablation: 4B without roles in the state (same records/ids/labels), validation only (preregistration).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY="$REPO/.venv/bin/python"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "train no-4b-noroles"
"$PY" -m signrule.train.kev_wrapper train --config configs/train/4b-noroles.yaml > runs/no-4b-noroles.log 2>&1 || { log "train failed"; exit 1; }
log "bench val (raw logits)"
"$PY" -m signrule.train.kev_wrapper bench --run runs/no-4b-noroles --split random_noroles --part val --raw > /dev/null
"$PY" eval/run_all.py --split random_noroles --part val \
  --kev "no-4b-noroles=runs/no-4b-noroles-bench-no-random_noroles-val,runs/no-4b-noroles-bench-no-random_noroles-val"
log "paired comparison vs 4B with roles (val)"
"$PY" eval/compare.py --split random --b-split random_noroles --part val --a no-4b-base --b no-4b-noroles
log "done"
