"""Latency and throughput of a running /v1/systemone server (plan-09 T3).

    uv run python scripts/bench_latency.py --model no-9b-base --n 200 --concurrency 1,4,8

Sends masked test-split requests (labels stripped) and writes aggregate timings to
results/serving/<model>.latency.json. Start the server first (`make serve RUN=runs/<name>`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests


def api_request(r: dict) -> dict:
    return {
        "model": "signrule-decide",
        "state": r["state"],
        "questions": {
            qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            for qid, q in r["questions"].items()
        },
    }


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def run(url: str, reqs: list[dict], concurrency: int) -> dict:
    lat: list[float] = []

    def one(body: dict) -> float:
        t = time.perf_counter()
        resp = client.post(f"{url}/v1/systemone", json=body)
        resp.raise_for_status()
        return time.perf_counter() - t

    with httpx.Client(timeout=120) as client:
        one(reqs[0])  # warm-up (CUDA graphs, caches)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            lat = list(pool.map(one, reqs))
        wall = time.perf_counter() - t0
    n_q = sum(len(r["questions"]) for r in reqs)
    return {
        "concurrency": concurrency,
        "n_requests": len(reqs),
        "questions_per_request": n_q / len(reqs),
        "p50_ms": 1000 * pct(lat, 50),
        "p95_ms": 1000 * pct(lat, 95),
        "mean_ms": 1000 * statistics.mean(lat),
        "throughput_rps": len(reqs) / wall,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:8300")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", default="1,4,8")
    a = ap.parse_args(argv)
    reqs = [api_request(r) for r in load_requests("no", "random", "test")[: a.n]]
    rows = [run(a.url, reqs, int(c)) for c in a.concurrency.split(",")]
    out = REPO_ROOT / "results" / "serving"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}.latency.json").write_text(
        json.dumps({"model": a.model, "runs": rows}, indent=1)
    )
    for r in rows:
        print(
            f"{a.model} c={r['concurrency']}: p50 {r['p50_ms']:.0f} ms, p95 {r['p95_ms']:.0f} ms, "
            f"{r['throughput_rps']:.1f} req/s ({r['questions_per_request']:.1f} questions/request)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
