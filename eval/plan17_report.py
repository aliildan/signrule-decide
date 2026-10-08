"""Plan-17 outcome: the Strands Decider-format model S against its pre-registered criteria H23–H26.

    uv run python eval/plan17_report.py --model strands-4b-v1 [--reference noat-4b-v2]

Reads only results/ and the item sidecars of the scorings `eval/run_all.py` wrote (AT gold with the
AT val policy, NO gold with the NO val policy, DK gold with the strictest merge) and the parity
reports of scripts/ollama_parity.py (results/plan17/parity-*.json). Writes
results/plan17/strands-report.json. Thresholds are those of docs/preregistration.md (2026-10-07).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from signrule.common.paths import REPO_ROOT
from signrule.ontology.coalitions import ALL_COALITIONS

RESULTS = REPO_ROOT / "results"
SIDECARS = REPO_ROOT / "runs" / "eval"
OUT = RESULTS / "plan17"

_spec = importlib.util.spec_from_file_location(
    "plan16_report", Path(__file__).with_name("plan16_report.py")
)
assert _spec is not None and _spec.loader is not None
plan16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan16)


def coalition_mean(per_question: dict[str, Any]) -> tuple[float | None, int]:
    hits = n = 0.0
    for q in ALL_COALITIONS:
        c = per_question.get(q, {}).get("calibrated")
        if c and c.get("n"):
            hits += c["accuracy"] * c["n"]
            n += c["n"]
    return (hits / n if n else None), int(n)


def accuracy(per_question: dict[str, Any], q: str) -> float | None:
    c = per_question.get(q, {}).get("calibrated")
    return c["accuracy"] if c else None


def at_gold(model: str) -> dict[str, Any]:
    tag = "at-random-to-at-gold"
    res = json.loads((RESULTS / tag / f"{model}.test.json").read_text())
    rows = [
        json.loads(x)
        for x in (SIDECARS / tag / f"{model}.test.items.jsonl").read_text().splitlines()
    ]
    return plan16.summarise(rows, res["policy"]["thresholds"].get("0.02", {}))


def summary(tag: str, model: str) -> dict[str, Any] | None:
    p = RESULTS / tag / f"{model}.test.json"
    if not p.exists():
        return None
    pq = json.loads(p.read_text())["per_question"]
    cm, n = coalition_mean(pq)
    return {
        "coalition_mean": cm,
        "coalition_items": n,
        "min_signers": accuracy(pq, "min_signers"),
        "rule_type": accuracy(pq, "rule_type"),
        "ceo_alone": accuracy(pq, "ceo_alone"),
    }


def parity(name: str) -> dict[str, Any] | None:
    p = OUT / f"parity-{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def h23(reports: list[dict[str, Any] | None]) -> bool | None:
    present = [r for r in reports if r is not None]
    if len(present) != len(reports):
        return None
    return all(
        r["argmax_agreement"] >= 0.995 and r["max_abs_dp"] <= 0.02 and r["mean_abs_dp"] <= 0.005
        for r in present
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="strands-4b-v1")
    ap.add_argument("--reference", default="noat-4b-v2")
    a = ap.parse_args(argv)

    at = at_gold(a.model)
    no = summary("no-random-to-no-gold", a.model)
    dk = summary("at+no-random-to-dk-gold", a.model)
    par = {k: parity(k) for k in ("demo-bf16", "at-gold-bf16", "demo-q8", "at-gold-q8")}
    out: dict[str, Any] = {
        "model": a.model,
        "reference": a.reference,
        "at:gold": at,
        "no:gold": no,
        "dk:gold": dk,
        "reference_at:gold": at_gold(a.reference),
        "reference_no:gold": summary("no-random-to-no-gold", a.reference),
        "parity": par,
        "H23": h23([par["demo-bf16"], par["at-gold-bf16"]]),
        "H23_q8": h23([par["demo-q8"], par["at-gold-q8"]]),
        "H24": (at["coalition_mean"] or 0) >= 0.983
        and (at["min_signers"] or 0) >= 0.957
        and (at["ceo_alone"] or 0) >= 0.99
        and (at["dangerous_rate"] if at["dangerous_rate"] is not None else 1) <= 0.01,
        "H25": None
        if no is None
        else (no["rule_type"] or 0) >= 0.90 and (no["coalition_mean"] or 0) >= 0.97,
        "H26": at["alpha_2pct"]["risk"] is not None and at["alpha_2pct"]["risk"] <= 0.02,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strands-report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("H23", "H23_q8", "H24", "H25", "H26")}))
    print(
        f"at:gold coalitions {at['coalition_mean']:.4f} min_signers {at['min_signers']:.4f} "
        f"ceo_alone {at['ceo_alone']} dangerous {at['dangerous']}/{at['yes_no_items']} "
        f"risk@2% {at['alpha_2pct']['risk']} coverage {at['alpha_2pct']['coverage']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
