#!/usr/bin/env bash
# plan-13 C″, second attempt (systemd user unit signrule-c2b): the first attempt (c2_run.sh) built
# and checked the v2 data and gold splits, then ran out of GPU memory at training step 465. This
# script continues from the training step with the shared-prefix forward (configs/train/
# 4b-noat-v2.yaml) and expandable allocator segments (attempt 2 was stopped at step 500 for the
# latter: 31.1 of 32.6 GB in use, fragmentation was a co-cause of the first crash); everything
# after training is the same as in c2_run.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/c2-run.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3.5:35b-a3b-q4_K_M","keep_alive":0}' >/dev/null || true

bench() {  # run, jurisdiction, split, part
  local out="runs/$1-bench-$2-$3-$4"
  [ -d "$out" ] && rm -rf "$out"
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$1" --jurisdiction "$2" --split "$3" \
    --part "$4" --raw >>"runs/$1.bench.log" 2>&1 || log "bench $1 $2/$3/$4 FAILED"
}
score() {  # name, calibration jurisdiction, target (empty = in-jurisdiction test)
  local name="$1" cal="$2" target="${3:-}" val="runs/$1-bench-$2-random-val" ev
  if [ -n "$target" ]; then
    ev="runs/$name-bench-${target%%:*}-${target##*:}-test"
    "$PY" eval/run_all.py --jurisdiction "$cal" --split random --target "$target" --allow-test \
      --kev "$name=$val,$ev" >>"$LOG" 2>&1 || log "score $name $cal->$target FAILED"
  else
    ev="runs/$name-bench-$cal-random-test"
    "$PY" eval/run_all.py --jurisdiction "$cal" --split random --part test --allow-test \
      --kev "$name=$val,$ev" >>"$LOG" 2>&1 || log "score $name $cal FAILED"
  fi
}

log "train noat-4b-v2 (attempt 3: shared prefix, expandable allocator segments)"
"$PY" -m signrule.train.kev_wrapper train --config configs/train/4b-noat-v2.yaml \
  >>runs/noat-4b-v2.log 2>&1 || { log "train noat-4b-v2 FAILED"; exit 1; }

for run in noat-4b-v2 noat-4b-v1; do
  log "bench $run"
  for t in "at random val" "at random test" "at pilot test" "at gold test" \
           "no random val" "no random test" "no gold test"; do
    # shellcheck disable=SC2086
    bench "$run" $t
  done
  score "$run" at
  for t in at:pilot at:gold; do score "$run" at "$t"; done
  score "$run" no
  score "$run" no no:gold
done
"$PY" eval/plan12_report.py --c noat-4b-v2 --a noat-4b-v1 --a-tag at-random-to-at-gold \
  --out report-c2.json >>"$LOG" 2>&1 || log "report FAILED"
"$PY" eval/seen_patterns.py --models noat-4b-v2,noat-4b-v1 >>"$LOG" 2>&1 || true
log "c2 run done"
