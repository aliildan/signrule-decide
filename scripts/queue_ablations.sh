#!/usr/bin/env bash
# After the main benchmark: 4B random + 4B temporal (H4), benchmarked and scored.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY="$REPO/.venv/bin/python"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
for cfg in 4b 4b-temporal; do
  name=$(grep '^name:' "configs/train/$cfg.yaml" | cut -d' ' -f2)
  split=$(grep '^split:' "configs/train/$cfg.yaml" | cut -d' ' -f2)
  log "train $name ($split)"
  "$PY" -m signrule.train.kev_wrapper train --config "configs/train/$cfg.yaml" > "runs/$name.log" 2>&1 || { log "train $name failed"; continue; }
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$name" --split "$split" --part val --raw > /dev/null
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$name" --split "$split" --part test --raw > /dev/null
  "$PY" eval/run_all.py --split "$split" --part val --kev "$name=runs/$name-bench-no-$split-val,runs/$name-bench-no-$split-test"
  "$PY" eval/run_all.py --split "$split" --part test --allow-test --kev "$name=runs/$name-bench-no-$split-val,runs/$name-bench-no-$split-test"
done
"$PY" eval/make_tables.py --split random --part test
"$PY" eval/make_tables.py --split temporal --part test
log "done"
