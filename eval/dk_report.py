"""Pre-registered Denmark zero-shot hypotheses H17/H18 (plan-14 T3; aggregate output only).

    uv run python eval/dk_report.py --models noat-4b-v2,noat-4b-v1

Reads results/<tag>/<model>.test.json and the item sidecars of the strictest-merge scoring
(tag at+no-random-to-dk-gold: calibrated on the AT and NO val parts, never on Danish data).
H17: coalition mean >= 0.90 and min_signers >= 0.85 (all items answered). H18: risk on the
answered questions at alpha 5 % <= 0.05. Secondary: by wording stratum and company-weighted
(each item weighted by the companies of its sampling cell / the items drawn from that cell).
Writes results/plan14/dk-report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.gold import _dk_bucket
from signrule.evaluation.harness import load_requests
from signrule.ontology.coalitions import ALL_COALITIONS

RESULTS = REPO_ROOT / "results"
SIDECARS = REPO_ROOT / "runs" / "eval"
TAG = "at+no-random-to-dk-gold"


def cell(wording: str, state: dict[str, Any]) -> tuple[str, str, str]:
    return (wording, *_dk_bucket(state))


def cell_weights(pool: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, float]:
    """Item id -> companies of its (wording, board, directors) cell / gold items in that cell."""
    companies = Counter(cell(r["_meta"]["stratum"], r["state"]) for r in pool)
    gold_cells = {r["_meta"]["id"]: cell(r["_meta"]["strata"]["wording"], r["state"]) for r in gold}
    drawn = Counter(gold_cells.values())
    return {rid: companies[c] / drawn[c] for rid, c in gold_cells.items()}


def accuracy(rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> float | None:
    w = [(weights or {}).get(r["rid"], 1.0) if weights else 1.0 for r in rows]
    total = sum(w)
    return (
        sum(wi for wi, r in zip(w, rows, strict=True) if r["pred"] == r["y"]) / total
        if total
        else None
    )


def selective(rows: list[dict[str, Any]], thresholds: dict[str, float | None]) -> dict[str, Any]:
    """Answered = confidence at or above the question type's threshold (None = never answered)."""
    answered = [
        r
        for r in rows
        if thresholds.get(r["qtype"]) is not None and r["conf"] >= thresholds[r["qtype"]]
    ]
    risk = sum(r["pred"] != r["y"] for r in answered) / len(answered) if answered else None
    return {
        "coverage": len(answered) / len(rows) if rows else None,
        "risk": risk,
        "answered": len(answered),
    }


def report(model: str, weights: dict[str, float], wording: dict[str, str]) -> dict[str, Any]:
    res = json.loads((RESULTS / TAG / f"{model}.test.json").read_text())
    rows = [
        json.loads(x)
        for x in (SIDECARS / TAG / f"{model}.test.items.jsonl").read_text().splitlines()
    ]
    coal = [r for r in rows if r["qid"] in ALL_COALITIONS]
    mins = [r for r in rows if r["qid"] == "min_signers"]
    sel = selective(rows, res["policy"]["thresholds"].get("0.05", {}))
    out: dict[str, Any] = {
        "calibration": res.get("calibration"),
        "policy_merge": res.get("policy_merge"),
        "coalition_mean": accuracy(coal),
        "coalition_items": len(coal),
        "min_signers": accuracy(mins),
        "min_signers_items": len(mins),
        "rule_type": accuracy([r for r in rows if r["qid"] == "rule_type"]),
        "alpha_5pct": sel,
        "company_weighted": {
            "coalition_mean": accuracy(coal, weights),
            "min_signers": accuracy(mins, weights),
        },
        "by_wording": {
            w: {
                "coalition_mean": accuracy([r for r in coal if wording.get(r["rid"]) == w]),
                "min_signers": accuracy([r for r in mins if wording.get(r["rid"]) == w]),
            }
            for w in ("dk-standard", "dk-tail")
        },
    }
    out["H17"] = (out["coalition_mean"] or 0) >= 0.90 and (out["min_signers"] or 0) >= 0.85
    out["H18"] = sel["risk"] is not None and sel["risk"] <= 0.05
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True)
    a = ap.parse_args(argv)
    gold = load_requests("dk", "gold", "test")
    weights = cell_weights(load_requests("dk", "pool", "test"), gold)
    wording = {r["_meta"]["id"]: r["_meta"]["strata"]["wording"] for r in gold}
    out = {m: report(m, weights, wording) for m in a.models.split(",")}
    d = RESULTS / "plan14"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dk-report.json").write_text(json.dumps(out, indent=1))
    for m, r in out.items():
        s = r["alpha_5pct"]
        risk = "n/a" if s["risk"] is None else f"{s['risk']:.4f}"
        print(
            f"{m:>14}: coalitions {r['coalition_mean']:.3f} (n={r['coalition_items']}), "
            f"min_signers {r['min_signers']:.3f}, rule_type {r['rule_type']:.3f} | alpha 5 %: "
            f"coverage {s['coverage']:.3f}, risk {risk} "
            f"| H17 {'met' if r['H17'] else 'NOT met'}, H18 {'met' if r['H18'] else 'NOT met'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
