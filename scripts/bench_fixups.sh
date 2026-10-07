#!/usr/bin/env bash
# Re-run the two baselines that failed in bench_all: Kev-9B zero-shot (needs bf16: fp32 9B > 32 GB)
# and mDeBERTa-v3 (NaN under bf16 autocast; now trained in fp32). Then comparisons and tables.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY="$REPO/.venv/bin/python"
P="data/processed/no/random"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
out="runs/zs-kev-9b-bench-no-random"
log "zero-shot jaredpalmer/kev-9b (bf16)"
KEV_DTYPE=bf16 KEV_TEMPERATURE=1.0 "$PY" -m kev.benchmark --run jaredpalmer/kev-9b --data "$P/val.jsonl" --out "$out-val" > runs/zs-kev-9b.log 2>&1 || log "  val failed (runs/zs-kev-9b.log)"
KEV_DTYPE=bf16 KEV_TEMPERATURE=1.0 "$PY" -m kev.benchmark --run jaredpalmer/kev-9b --data "$P/test.jsonl" --out "$out-test" --allow-test >> runs/zs-kev-9b.log 2>&1 || log "  test failed"
if [ -f "$out-test/predictions.jsonl" ]; then
  "$PY" eval/run_all.py --split random --part val --kev "kev-9b-zeroshot=$out-val,$out-test"
  "$PY" eval/run_all.py --split random --part test --allow-test --kev "kev-9b-zeroshot=$out-val,$out-test"
  "$PY" eval/compare.py --split random --part test --a no-9b-base --b kev-9b-zeroshot
  "$PY" eval/analysis.py --split random --part test --model kev-9b-zeroshot || true
fi
log "mDeBERTa-v3-base (fp32)"
"$PY" eval/run_all.py --split random --part test --allow-test --baselines enc:microsoft/mdeberta-v3-base
"$PY" eval/compare.py --split random --part test --a no-9b-base --b enc_mdeberta-v3-base
"$PY" eval/make_tables.py --split random --part test
"$PY" eval/make_tables.py --split random --part val
log "done"
