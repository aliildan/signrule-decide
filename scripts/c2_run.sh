#!/usr/bin/env bash
# plan-13 run C″ (noat-4b-v2), unattended (systemd user unit signrule-c2):
#   1. wait until the Norway fetch has finished; free the GPU from the local QA model
#   2. keep the v1 NO data (A′/B′/C′ were trained on it), rebuild NO (v2, complete fetch) and AT
#      data with the plan-13 labels; data-checks must pass
#   3. rebuild both gold evaluation splits (new coalition labels)
#   4. train C″; bench C″ on all sets and C′ on the rebuilt NO/gold sets; score with the
#      pre-registered protocol; write results/plan12/report-c2.json (C″ vs C′)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/c2-run.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "waiting for the Norway fetch to finish"
while systemctl --user is-active --quiet signrule-ingest-no; do sleep 300; done
log "Norway fetch finished"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3.5:35b-a3b-q4_K_M","keep_alive":0}' >/dev/null || true

if [ ! -d data/processed/no-v1 ]; then
  cp -a data/processed/no data/processed/no-v1
  for s in random temporal; do cp -a "data/splits/no_${s}.json" "data/splits/no_${s}.v1.json"; done
  log "v1 NO data kept as processed/no-v1"
fi

"$PY" -m signrule.normalize.pipeline_no run >>"$LOG" 2>&1
if ! "$PY" -m signrule.normalize.pipeline_no check --near-dups >>"$LOG" 2>&1; then
  log "NO DATA-CHECK FAILED - stopping"; exit 1
fi
"$PY" -m signrule.normalize.normalize_at build --split pilot >>"$LOG" 2>&1
"$PY" -m signrule.normalize.pipeline_at run >>"$LOG" 2>&1
if ! "$PY" -m signrule.normalize.pipeline_at check >>"$LOG" 2>&1; then
  log "AT DATA-CHECK FAILED - stopping"; exit 1
fi
"$PY" -m signrule.evaluation.gold_eval build --set at-text --a ali --b annotator2 >>"$LOG" 2>&1
"$PY" -m signrule.evaluation.gold_eval build --set no-rt --a annotator1 --b annotator2 \
  --jurisdiction no --adjudicate 2,10,16,22,28,34,40,61,148,226,238 --adjudicate-by annotator2 \
  --reason "guideline §2 changed 2026-10-05: clear text with too few registered members is a rule" \
  >>"$LOG" 2>&1

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

log "train noat-4b-v2"
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
