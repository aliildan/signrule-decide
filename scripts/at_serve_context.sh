#!/usr/bin/env bash
# 2026-10-07: re-benchmark the release model C″ (noat-4b-v2) on every Austrian part under the
# serving context (kev.benchmark skipped states over 384 tokens: ~10 % of every AT part), then
# re-score AT (own val policy), DK (strictest AT+NO) and the plan-16 report. The 384-token
# benchmarks are kept as <dir>-ctx384. Unit: signrule-atctx.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/at-serve-context.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
run=noat-4b-v2
for t in "random val" "random test" "pilot test" "gold test"; do
  set -- $t
  out="runs/$run-bench-at-$1-$2"
  [ -d "$out" ] && [ ! -d "$out-ctx384" ] && mv "$out" "$out-ctx384"
  [ -d "$out" ] && rm -rf "$out"
  log "bench $run at/$1/$2 (serving context)"
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$run" --jurisdiction at --split "$1" \
    --part "$2" --raw >>"runs/$run.bench.log" 2>&1 || log "bench at/$1/$2 FAILED"
done
val="runs/$run-bench-at-random-val"
"$PY" eval/run_all.py --jurisdiction at --split random --part test --allow-test \
  --baselines phrase_at,rules_at --kev "$run=$val,runs/$run-bench-at-random-test" >>"$LOG" 2>&1 || log "score at FAILED"
for t in at:gold at:pilot; do
  ev="runs/$run-bench-${t%%:*}-${t##*:}-test"
  "$PY" eval/run_all.py --jurisdiction at --split random --target "$t" --allow-test \
    --baselines phrase_at,rules_at --kev "$run=$val,$ev" >>"$LOG" 2>&1 || log "score $t FAILED"
done
"$PY" eval/run_all.py --jurisdiction at --split random --target dk:gold --allow-test --strictest-with no \
  --kev "$run=$val,runs/$run-bench-dk-gold-test,runs/$run-bench-no-random-val" >>"$LOG" 2>&1 || log "score dk FAILED"
"$PY" eval/dk_report.py --models "$run" >>"$LOG" 2>&1 || log "dk report FAILED"
"$PY" eval/plan16_report.py --model "$run" >>"$LOG" 2>&1 || log "plan16 report FAILED"
log "at serve-context done"
