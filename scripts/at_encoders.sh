#!/usr/bin/env bash
# plan-10: encoder baselines for the Austrian story — trained on AT random train, calibrated on AT
# val, scored on at:gold (400) and the AT pattern-disjoint test. Unit: signrule-atenc.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/at-encoders.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
for enc in ${ENCODERS:-"jhu-clsp/mmBERT-base@1024" "FacebookAI/xlm-roberta-large@512/8"}; do
  log "encoder $enc -> at:gold"
  "$PY" eval/run_all.py --jurisdiction at --split random --target at:gold --allow-test \
    --baselines "enc:$enc" >>"$LOG" 2>&1 || log "  $enc at:gold FAILED"
  log "encoder $enc -> at random test"
  "$PY" eval/run_all.py --jurisdiction at --split random --part test --allow-test \
    --baselines "enc:$enc" >>"$LOG" 2>&1 || log "  $enc at test FAILED"
done
log "at encoders done"
