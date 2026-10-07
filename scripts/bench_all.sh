#!/usr/bin/env bash
# Post-training benchmark for SignRule-Decide v0 (plan-06 T5/T6, plan-08 T4). Sequential GPU jobs.
#   scripts/bench_all.sh <run-name> [split]
# Writes results/<jur>-<split>/*.json, private item sidecars in runs/eval/, tables in results/tables/.
set -uo pipefail
RUN=${1:?run name, e.g. no-9b-base}
SPLIT=${2:-random}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY="$REPO/.venv/bin/python"
P="data/processed/no/$SPLIT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "bench $RUN (raw logits) on val/test"
"$PY" -m signrule.train.kev_wrapper bench --run "runs/$RUN" --split "$SPLIT" --part val --raw > /dev/null
"$PY" -m signrule.train.kev_wrapper bench --run "runs/$RUN" --split "$SPLIT" --part test --raw > /dev/null

KEV_ARGS=(--kev "$RUN=runs/$RUN-bench-no-$SPLIT-val,runs/$RUN-bench-no-$SPLIT-test")
for s in 0.8b 4b 9b; do
  out="runs/zs-kev-$s-bench-no-$SPLIT"
  log "zero-shot jaredpalmer/kev-$s"
  KEV_TEMPERATURE=1.0 "$PY" -m kev.benchmark --run "jaredpalmer/kev-$s" --data "$P/val.jsonl" --out "$out-val" > /dev/null 2>&1 \
    || log "  kev-$s val failed"
  KEV_TEMPERATURE=1.0 "$PY" -m kev.benchmark --run "jaredpalmer/kev-$s" --data "$P/test.jsonl" --out "$out-test" --allow-test > /dev/null 2>&1 \
    || log "  kev-$s test failed"
  [ -f "$out-test/predictions.jsonl" ] && KEV_ARGS+=(--kev "kev-$s-zeroshot=$out-val,$out-test")
done

log "Clef-Flash zero-shot (test; val used for its temperatures)"
"$PY" eval/run_all.py --split "$SPLIT" --part test --allow-test --baselines clef || log "  clef failed"

log "score decision models (val, test)"
"$PY" eval/run_all.py --split "$SPLIT" --part val "${KEV_ARGS[@]}"
"$PY" eval/run_all.py --split "$SPLIT" --part test --allow-test "${KEV_ARGS[@]}"

log "CPU baselines (test, with sidecars)"
"$PY" eval/run_all.py --split "$SPLIT" --part test --allow-test --baselines majority,rules_no,tfidf_lr

for enc in jhu-clsp/mmBERT-base microsoft/mdeberta-v3-base FacebookAI/xlm-roberta-large; do
  log "encoder baseline $enc"
  "$PY" eval/run_all.py --split "$SPLIT" --part test --allow-test --baselines "enc:$enc" || log "  $enc failed"
done

log "paired comparisons (H1)"
for b in enc_mmBERT-base enc_mdeberta-v3-base enc_xlm-roberta-large tfidf_lr kev-9b-zeroshot clef-flash-zeroshot; do
  "$PY" eval/compare.py --split "$SPLIT" --part test --a "$RUN" --b "$b" || true
done

log "analyses (confusion, length, legal form, top errors)"
for m in "$RUN" enc_mmBERT-base tfidf_lr kev-9b-zeroshot; do
  "$PY" eval/analysis.py --split "$SPLIT" --part test --model "$m" || true
done

log "tables"
"$PY" eval/make_tables.py --split "$SPLIT" --part test
"$PY" eval/make_tables.py --split "$SPLIT" --part val
log "done"
