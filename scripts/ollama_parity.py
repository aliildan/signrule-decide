"""Ollama vs. PyTorch on the same requests: is the exported model the trained one? (plan-17, H23)

    uv run --group strands python scripts/ollama_parity.py --ckpt runs/<name>/ckpt \
        --model signrule-decide --states demo [--host 127.0.0.1:11435] [--out results/...json]
    ... --states gold --jurisdiction at          # the frozen gold items, local Ollama only

The PyTorch side is strands-decider's own engine on the checkpoint (LoRA unmerged, bf16); the
Ollama side is the exported, merged model behind a local Ollama. Both read the checkpoint's raw
probabilities when the export carries no temperatures. Requests go only to a local server (the
host must be 127.0.0.1/localhost); only aggregates are printed and written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from signrule.format.request import load_questions

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def ask_payload(questions: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    keep = ("type", "instructions", "criteria")
    return {q: {k: v for k, v in questions[q].items() if k in keep} for q in qids}


def probs_of(answer: dict[str, Any]) -> dict[str, float]:
    """Answer -> {option: probability}; a noul as {false, true}."""
    if answer["type"] == "noul":
        return {"false": 1.0 - float(answer["noul"]), "true": float(answer["noul"])}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def compare(a: dict[str, float], b: dict[str, float]) -> tuple[bool, float]:
    """(same argmax, max |Δp|) over the union of options."""
    keys = set(a) | set(b)
    same = max(a, key=a.__getitem__) == max(b, key=b.__getitem__)
    return same, max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def summarise(pairs: list[tuple[bool, float]]) -> dict[str, Any]:
    n = len(pairs)
    deltas = [d for _, d in pairs]
    return {
        "answers": n,
        "argmax_agreement": sum(s for s, _ in pairs) / n if n else None,
        "max_abs_dp": max(deltas) if deltas else None,
        "mean_abs_dp": sum(deltas) / n if n else None,
    }


def _demo_states() -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("demo", Path(__file__).with_name("demo.py"))
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return [state for _, state in demo.EXAMPLES_DE]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--host", default="127.0.0.1:11435")
    ap.add_argument("--states", choices=["demo", "gold"], default="demo")
    ap.add_argument("--jurisdiction", default="at")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--torch-only", type=Path, help=argparse.SUPPRESS)  # child: write answers here
    a = ap.parse_args(argv)
    base = a.host if "://" in a.host else f"http://{a.host}"
    if urlparse(base).hostname not in LOCAL_HOSTS:
        raise SystemExit("parity runs against a local Ollama only")

    qc = load_questions().questions
    if a.states == "demo":
        items = [(s, list(qc)) for s in _demo_states()]
    else:
        from signrule.evaluation.harness import load_requests

        items = [
            (r["state"], list(r["questions"]))
            for r in load_requests(a.jurisdiction, "gold", "test")
        ]

    if a.torch_only:
        from strands_decider.infer import load_engine
        from strands_decider.schema import SystemOneRequest

        engine = load_engine(a.ckpt, device="cuda")
        answers = [
            engine.evaluate(
                SystemOneRequest(state=state, questions=ask_payload(qc, qids))
            ).model_dump()["answers"]
            for state, qids in items
        ]
        a.torch_only.write_text(json.dumps(answers))
        return 0

    # One model on the GPU at a time: unload it from Ollama (its MLX pool holds memory while
    # loaded), compute the PyTorch answers in a child process (all its GPU memory is returned when
    # it exits), then query Ollama.
    for m in httpx.get(f"{base}/api/ps", timeout=30).json().get("models", []):
        httpx.post(f"{base}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=120)
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "torch.json"
        child = [sys.executable, __file__, "--ckpt", a.ckpt, "--model", a.model]
        child += ["--host", a.host, "--states", a.states, "--jurisdiction", a.jurisdiction]
        child += ["--torch-only", str(cache)]
        subprocess.run(child, check=True)
        torch_answers = json.loads(cache.read_text())

    pairs: list[tuple[bool, float]] = []
    by_kind: dict[str, list[tuple[bool, float]]] = {}
    worst: list[tuple[float, str, float, float]] = []
    restarts = failed = 0
    with httpx.Client(timeout=600) as client:
        for (state, qids), torch_ans in zip(items, torch_answers, strict=True):
            body = {"model": a.model, "state": state, "questions": ask_payload(qc, qids)}
            r = client.post(f"{base}/v1/systemone", json=body)
            if r.status_code == 500:
                # Linux MLX/CUDA runner out of memory: unload, reload, retry once (counted)
                restarts += 1
                client.post(f"{base}/api/generate", json={"model": a.model, "keep_alive": 0})
                r = client.post(f"{base}/v1/systemone", json=body)
            if r.status_code != 200:
                failed += 1
                pairs += [(False, 1.0)] * len(qids)  # unserved answers count against parity
                continue
            ollama_ans = r.json()["answers"]
            for q in qids:
                tp, op = probs_of(torch_ans[q]), probs_of(ollama_ans[q])
                pair = compare(tp, op)
                pairs.append(pair)
                by_kind.setdefault(qc[q]["type"], []).append(pair)
                worst.append((pair[1], q, round(max(tp.values()), 4), round(max(op.values()), 4)))
    report = {
        "model": a.model,
        "ckpt": a.ckpt,
        "states": a.states if a.states == "demo" else f"{a.jurisdiction}:gold",
        "requests": len(items),
        **summarise(pairs),
        "by_kind": {k: summarise(v) for k, v in sorted(by_kind.items())},
        # the largest differences: question, PyTorch and Ollama top-option probability
        "largest": [
            {"abs_dp": round(d, 4), "qid": q, "torch_top": t, "ollama_top": o}
            for d, q, t, o in sorted(worst, reverse=True)[:8]
        ],
        "abs_dp_over_0.01": sum(d > 0.01 for d, *_ in worst),
        "runner_restarts": restarts,
        "failed_requests": failed,
    }
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
