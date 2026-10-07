#!/usr/bin/env bash
# Overnight v1 chain (plan-11 Task 3, part 1). Runs unattended as a systemd user unit:
#   1. wait until the Norway Fullmakt fetch has passed tier 1 (AS/ASA, 866,848 lookups)
#   2. free the GPU from the local QA model (Ollama keep_alive 0)
#   3. keep the v0.1 data, rebuild NO data (v1), data-check (stop on failure)
#   4. run A: train no-4b-v1, bench NO val/test + AT pilot
#   5. train no-9b-v1 (size comparison on the same data), same benches
# Runs B′/C′ and all scoring follow in scripts/at_runs.sh (unit signrule-at-runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/overnight-v1.log"
THRESH="${THRESH:-870000}"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

done_lookups() {
  grep -oE '^ +[0-9]+/1274984 done' runs/ingest/no-fullmakt.log | tail -1 | grep -oE '[0-9]+' | head -1
}

log "waiting for $THRESH Norway lookups"
until [ "$(done_lookups || echo 0)" -ge "$THRESH" ]; do sleep 300; done
log "tier 1 reached: $(done_lookups) lookups"

curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3.5:35b-a3b-q4_K_M","keep_alive":0}' >/dev/null || true

if [ ! -d data/processed/no-v01 ]; then
  cp -a data/processed/no data/processed/no-v01
  for s in random temporal; do
    cp -a "data/splits/no_${s}.json" "data/splits/no_${s}.v01.json"
  done
  log "v0.1 data kept as processed/no-v01"
fi

log "pipeline_no run"
"$PY" -m signrule.normalize.pipeline_no run >>"$LOG" 2>&1
log "pipeline_no check"
if ! "$PY" -m signrule.normalize.pipeline_no check --near-dups >>"$LOG" 2>&1; then
  log "DATA-CHECK FAILED - stopping before training"
  exit 1
fi

bench() {  # run, jurisdiction, split, part
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$1" --jurisdiction "$2" --split "$3" \
    --part "$4" --raw >>"runs/$1.bench.log" 2>&1 || log "bench $1 $2/$3/$4 FAILED"
}

for cfg in 4b-v1 9b-v1; do
  name="$(grep '^name:' "configs/train/$cfg.yaml" | awk '{print $2}')"
  log "train $name"
  if [ "$cfg" = "9b-v1" ]; then export KEV_DTYPE=bf16; fi
  if ! "$PY" -m signrule.train.kev_wrapper train --config "configs/train/$cfg.yaml" \
      >>"runs/$name.log" 2>&1; then
    log "train $name FAILED (see runs/$name.log)"
    continue
  fi
  log "bench $name"
  bench "$name" no random val
  bench "$name" no random test
  bench "$name" at pilot test
done
log "overnight chain done"
