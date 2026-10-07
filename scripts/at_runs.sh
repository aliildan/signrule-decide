#!/usr/bin/env bash
# plan-11/12 runs B′ and C′, unattended (systemd user unit). Waits until the overnight chain
# (A′ no-4b-v1, 9B′ no-9b-v1) and the Austrian fetch are finished, then:
#   1. AT data from the full cache (pilot rebuilt from its frozen ids), data-check (stop on failure)
#   2. AT gold evaluation split rebuilt (agreed annotations)
#   3. train B′ (at-4b-v1) and C′ (noat-4b-v1)
#   4. benches for A′, 9B′, B′, C′ on the same sets, and scoring with the pre-registered
#      protocol: A′/9B′ zero-shot on Austria with NO-val calibration; B′/C′ with AT-val calibration
#      on Austria and NO-val calibration on Norway. Gold sets are never used for fitting.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG="$PWD/runs/at-runs.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "waiting for the overnight chain (A′); the Austrian fetch may still be running (snapshot of the cache)"
while systemctl --user is-active --quiet signrule-overnight-v1; do
  sleep 300
done
log "overnight chain finished"
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3.5:35b-a3b-q4_K_M","keep_alive":0}' >/dev/null || true

"$PY" -m signrule.normalize.normalize_at build --split pilot >>"$LOG" 2>&1
"$PY" -m signrule.normalize.pipeline_at run >>"$LOG" 2>&1
if ! "$PY" -m signrule.normalize.pipeline_at check --near-dups >>"$LOG" 2>&1; then
  log "AT DATA-CHECK FAILED - stopping before training"
  exit 1
fi
"$PY" -m signrule.evaluation.gold_eval build --set at-text --a ali --b annotator2 >>"$LOG" 2>&1

bench() {  # run, jurisdiction, split, part
  local out="runs/$1-bench-$2-$3-$4"
  [ -d "$out" ] && rm -rf "$out"
  "$PY" -m signrule.train.kev_wrapper bench --run "runs/$1" --jurisdiction "$2" --split "$3" \
    --part "$4" --raw >>"runs/$1.bench.log" 2>&1 || log "bench $1 $2/$3/$4 FAILED"
}

train() {  # config file stem
  local name
  name="$(grep '^name:' "configs/train/$1.yaml" | awk '{print $2}')"
  log "train $name"
  "$PY" -m signrule.train.kev_wrapper train --config "configs/train/$1.yaml" \
    >>"runs/$name.log" 2>&1 || log "train $name FAILED (see runs/$name.log)"
}

bench_all() {  # run
  local run="$1"
  [ -d "runs/$run" ] || { log "skip $run (not trained)"; return; }
  log "bench $run"
  for t in "at random val" "at random test" "at pilot test" "at gold test" \
           "no random val" "no random test" "no gold test"; do
    # shellcheck disable=SC2086
    bench "$run" $t
  done
}

score() {  # name, calibration jurisdiction, target (empty = in-jurisdiction test)
  local name="$1" cal="$2" target="${3:-}"
  local val="runs/$name-bench-$cal-random-val"
  local ev
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

score_zero_shot() {  # NO-trained run: NO calibration everywhere (zero-shot protocol for Austria)
  score "$1" no
  score "$1" no no:gold
  for t in at:random at:pilot at:gold; do score "$1" no "$t"; done
}

score_with_at() {  # AT-trained run: AT-val calibration on Austria, NO-val on Norway
  score "$1" at
  for t in at:pilot at:gold; do score "$1" at "$t"; done
  score "$1" no
  score "$1" no no:gold
}

# C′ first (the main result), scored together with A′ as soon as it is trained; then B′.
train 4b-noat-v1
bench_all no-4b-v1
bench_all noat-4b-v1
score_zero_shot no-4b-v1
[ -d runs/noat-4b-v1 ] && score_with_at noat-4b-v1
log "C′ and A′ scored"
train 4b-at-v1
bench_all at-4b-v1
[ -d runs/at-4b-v1 ] && score_with_at at-4b-v1

# baselines next to the models (phrase_at = the training phrase table used as a predictor)
for t in at:pilot at:gold; do
  "$PY" eval/run_all.py --jurisdiction at --split random --target "$t" --allow-test \
    --baselines majority,rules_at,phrase_at >>"$LOG" 2>&1 || log "baselines $t FAILED"
done
"$PY" eval/run_all.py --jurisdiction at --split random --part test --allow-test \
  --baselines majority,rules_at,phrase_at >>"$LOG" 2>&1 || log "baselines at test FAILED"
log "at runs done"
