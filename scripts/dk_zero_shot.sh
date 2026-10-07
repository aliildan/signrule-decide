#!/usr/bin/env bash
# plan-14 T3: Denmark zero-shot (systemd user unit signrule-dk). Waits until the C″ chain
# (signrule-c2c) has finished and freed the GPU, then benches C″ and C′ on dk:gold, scores them
# with the strictest merge of the AT and NO val policies (no Danish calibration exists) and writes
# results/plan14/dk-report.json (pre-registered H17/H18).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/dk-zero-shot.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "waiting for the C″ chain (signrule-c2c)"
while systemctl --user is-active --quiet signrule-c2c; do sleep 300; done
log "C″ chain finished"

for run in noat-4b-v2 noat-4b-v1; do
  out="runs/$run-bench-dk-gold-test"
  [ -d "$out" ] && rm -rf "$out"
  if [ ! -d "runs/$run" ] || [ ! -d "runs/$run-bench-at-random-val" ] || [ ! -d "runs/$run-bench-no-random-val" ]; then
    log "$run: model or val benchmarks missing - skipped"; continue
  fi
  log "bench $run on dk/gold"
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$run" --jurisdiction dk --split gold \
    --part test --raw >>"runs/$run.bench.log" 2>&1 || { log "bench $run dk FAILED"; continue; }
  "$PY" eval/run_all.py --jurisdiction at --split random --target dk:gold --allow-test \
    --strictest-with no \
    --kev "$run=runs/$run-bench-at-random-val,$out,runs/$run-bench-no-random-val" \
    >>"$LOG" 2>&1 || log "score $run dk FAILED"
done
models=$(for r in noat-4b-v2 noat-4b-v1; do
  [ -f "results/at+no-random-to-dk-gold/$r.test.json" ] && echo "$r"; done | paste -sd, -)
if [ -n "$models" ]; then
  "$PY" eval/dk_report.py --models "$models" >>"$LOG" 2>&1 || log "report FAILED"
fi
log "dk zero-shot done"
