"""Benchmark a Strands Decider-format checkpoint in the `kev.benchmark` output format (plan-17 T6).

    uv run --group strands python -m signrule.train.strands_bench --ckpt runs/<name>/ckpt \
        --jurisdiction at --split gold --part test [--window 16384]

Writes `runs/<name>-bench-<jur>-<split>-<part>/predictions.jsonl` (one line per request:
`{"id", "prediction": {"probabilities": {qid: {option: p}}}}`, raw probabilities at temperature 1)
and `meta.json`, so `eval/run_all.py --kev NAME=VALDIR,EVALDIR` fits temperatures and thresholds on
val exactly as for Kev runs. strands-decider's own engine computes the answers (the prompt Ollama
serves). Every request is scored within the serving window; a request that does not fit is
recorded and left without predictions (run_all then counts its questions as unanswered).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from signrule.evaluation.harness import load_requests
from signrule.train.kev_wrapper import RUNS_DIR

KEEP = ("type", "instructions", "criteria")


def probabilities(answer: dict[str, Any]) -> dict[str, float]:
    """A strands-decider answer as {option: probability} (a noul as false/true)."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def question_payload(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        q: {k: v for k, v in spec.items() if k in KEEP} for q, spec in request["questions"].items()
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--jurisdiction", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--part", default="test")
    ap.add_argument("--window", type=int, default=16384, help="serving window (as the export)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N requests only (checks)")
    ap.add_argument("--out", type=Path, help="output directory (default runs/<run>-bench-...)")
    a = ap.parse_args(argv)

    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderModel
    from strands_decider.schema import SystemOneRequest

    run_name = a.ckpt.parent.name
    out = a.out or RUNS_DIR / f"{run_name}-bench-{a.jurisdiction}-{a.split}-{a.part}"
    out.mkdir(parents=True, exist_ok=True)
    model = StrandsDeciderModel.load(str(a.ckpt))
    cfg = model.config
    if cfg.temperature != 1.0 or cfg.temperature_by_kind:
        raise SystemExit("checkpoint carries temperatures; benchmark the raw training checkpoint")
    cfg.max_length = a.window
    engine = SystemOneEngine(model, EngineConfig(device=a.device, strict_window=True))

    reqs = load_requests(a.jurisdiction, a.split, a.part)
    if a.limit:
        reqs = reqs[: a.limit]
    over: list[str] = []
    t0 = time.time()
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in reqs:
            rid = r["_meta"]["id"]
            try:
                resp = engine.evaluate(
                    SystemOneRequest(state=r["state"], questions=question_payload(r))
                )
            except ValueError as e:  # strict_window: does not fit the serving window
                if "context window" not in str(e):
                    raise
                over.append(rid)
                continue
            answers = resp.model_dump()["answers"]
            pred = {"probabilities": {q: probabilities(ans) for q, ans in answers.items()}}
            f.write(json.dumps({"id": rid, "prediction": pred}) + "\n")
    meta = {
        "ckpt": str(a.ckpt),
        "data": f"{a.jurisdiction}/{a.split}/{a.part}",
        "requests": len(reqs),
        "over_window": len(over),
        "window": a.window,
        "seconds": round(time.time() - t0, 1),
        "temperature": "raw (1.0)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
