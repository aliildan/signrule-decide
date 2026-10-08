#!/usr/bin/env bash
# plan-17 T6: evaluate the Strands Decider-format model S after training, export it to Ollama and
# check parity. Waits for the training unit. Unit: signrule-strands-eval. Log: runs/strands-eval.log.
#   1. raw benchmarks (strands-decider's engine, serving window 16k) on every part C″ was scored on
#   2. run_all with the same calibration protocol as C″ (val-fitted temperatures + LTT per register;
#      DK with the strictest AT+NO merge)
#   3. Ollama: raw export (bf16, q8) -> parity vs. PyTorch on the demo states and at:gold (H23)
#   4. final export with the AT-val temperatures -> `signrule-decide` in the local Ollama
#   5. results/plan17/strands-report.json (H23-H26)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=strands-4b-v1
CKPT="runs/$RUN/ckpt"
PY="uv run --group strands python"
LOG="$PWD/runs/strands-eval.log"
OLLAMA=/data/signrule/ollama/0.40.0/bin/ollama
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/data/signrule/ollama/models PYTHONUNBUFFERED=1
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

while systemctl --user is-active --quiet signrule-strands; do sleep 60; done
[ -f "$CKPT/slot_head.pt" ] || { log "no checkpoint at $CKPT; training failed?"; exit 1; }
log "training finished; benchmarks"

for t in "no random val" "no random test" "no gold test" "at random val" "at random test" \
         "at pilot test" "at gold test" "dk gold test"; do
  set -- $t
  log "bench $1/$2/$3"
  $PY -m signrule.train.strands_bench --ckpt "$CKPT" --jurisdiction "$1" --split "$2" --part "$3" \
    >>"runs/$RUN.bench.log" 2>&1 || log "bench $1/$2/$3 FAILED"
done

b="runs/$RUN-bench"
log "scoring"
$PY eval/run_all.py --jurisdiction no --split random --part test --allow-test \
  --kev "$RUN=$b-no-random-val,$b-no-random-test" >>"$LOG" 2>&1 || log "score no FAILED"
$PY eval/run_all.py --jurisdiction no --split random --target no:gold --allow-test \
  --kev "$RUN=$b-no-random-val,$b-no-gold-test" >>"$LOG" 2>&1 || log "score no:gold FAILED"
$PY eval/run_all.py --jurisdiction at --split random --part test --allow-test \
  --kev "$RUN=$b-at-random-val,$b-at-random-test" >>"$LOG" 2>&1 || log "score at FAILED"
for t in at:gold at:pilot; do
  $PY eval/run_all.py --jurisdiction at --split random --target "$t" --allow-test \
    --kev "$RUN=$b-at-random-val,$b-${t%%:*}-${t##*:}-test" >>"$LOG" 2>&1 || log "score $t FAILED"
done
$PY eval/run_all.py --jurisdiction at --split random --target dk:gold --allow-test --strictest-with no \
  --kev "$RUN=$b-at-random-val,$b-dk-gold-test,$b-no-random-val" >>"$LOG" 2>&1 || log "score dk FAILED"
$PY eval/plan17_report.py --model "$RUN" >>"$LOG" 2>&1 || log "report (pre-parity) FAILED"

log "ollama: raw export + parity"
$PY scripts/export_ollama.py --ckpt "$CKPT" --out "runs/$RUN/ollama-raw" >>"$LOG" 2>&1 || log "raw export FAILED"
$OLLAMA create "$RUN-raw" -f "runs/$RUN/ollama-raw/Modelfile" >>"$LOG" 2>&1 || log "create raw FAILED"
$OLLAMA create "$RUN-raw-q8" -q q8 -f "runs/$RUN/ollama-raw/Modelfile" >>"$LOG" 2>&1 || log "create raw q8 FAILED"
for tag in bf16 q8; do
  model="$RUN-raw"; [ "$tag" = q8 ] && model="$RUN-raw-q8"
  for states in demo gold; do
    name="demo-$tag"; [ "$states" = gold ] && name="at-gold-$tag"
    $OLLAMA stop "$RUN-raw" >/dev/null 2>&1; $OLLAMA stop "$RUN-raw-q8" >/dev/null 2>&1
    $PY scripts/ollama_parity.py --ckpt "$CKPT" --model "$model" --states "$states" --jurisdiction at \
      --out "results/plan17/parity-$name.json" >>"$LOG" 2>&1 || log "parity $name FAILED"
  done
done
$OLLAMA stop "$RUN-raw" >/dev/null 2>&1; $OLLAMA stop "$RUN-raw-q8" >/dev/null 2>&1

log "final export with the AT-val temperatures"
$PY - <<'EOF' >>"$LOG" 2>&1 || log "temperatures FAILED"
import json
pol = json.load(open("results/at-random-to-at-gold/strands-4b-v1.test.json"))["policy"]
json.dump(pol["temperatures"], open("runs/strands-4b-v1/temperatures.json", "w"))
print("temperatures", pol["temperatures"])
EOF
$PY scripts/export_ollama.py --ckpt "$CKPT" --out "runs/$RUN/ollama" \
  --temperatures "runs/$RUN/temperatures.json" >>"$LOG" 2>&1 || log "final export FAILED"
$OLLAMA create signrule-decide -f "runs/$RUN/ollama/Modelfile" >>"$LOG" 2>&1 || log "create final FAILED"
$PY eval/plan17_report.py --model "$RUN" >>"$LOG" 2>&1 || log "report FAILED"
log "strands eval done"
