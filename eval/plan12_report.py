"""Pre-registered plan-12 hypotheses H11–H14 from results + item sidecars (aggregate output only).

    uv run python eval/plan12_report.py --c noat-4b-v1 --a no-4b-v1 [--b at-4b-v1]

Reads results/<tag>/<model>.test.json and runs/eval/<tag>/<model>.test.items.jsonl for the AT gold
set (C′/B′: tag at-random-to-at-gold, A′: no-random-to-at-gold) and NO random/test, plus the
phrase_at baseline. Derived outputs: min_signers and rule_type from the predicted coalition answers
(design-04). Writes results/plan12/report.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.format.request import load_questions
from signrule.ontology.cir import Group, SigningRule, derive_answers
from signrule.ontology.coalitions import (
    ALL_COALITIONS,
    MEMBER,
    min_signers_from,
    minimal_coalitions,
)

RESULTS = REPO_ROOT / "results"
SIDECARS = REPO_ROOT / "runs" / "eval"
PERSON_ROLE = {
    "CEO": "CEO",
    "CHAIR": "CHAIR",
    "DEPUTY": "DEPUTY_CHAIR",
    "PROKURIST": "PROKURIST",
    MEMBER: "BOARD_MEMBER",
}


def per_question(tag: str, model: str) -> dict[str, dict[str, Any]]:
    return json.loads((RESULTS / tag / f"{model}.test.json").read_text())["per_question"]


def policy(tag: str, model: str) -> dict[str, Any]:
    return json.loads((RESULTS / tag / f"{model}.test.json").read_text())["policy"]


def items(tag: str, model: str) -> list[dict[str, Any]]:
    path = SIDECARS / tag / f"{model}.test.items.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def coalition_mean(pq: dict[str, dict[str, Any]]) -> tuple[float, int]:
    accs = [
        (v["calibrated"]["accuracy"], v["calibrated"]["n"])
        for q, v in pq.items()
        if q in ALL_COALITIONS
    ]
    n = sum(k for _, k in accs)
    return (sum(a * k for a, k in accs) / n if n else float("nan")), n


def derived(tag: str, model: str) -> dict[str, float]:
    """Accuracy of min_signers and rule_type derived from the predicted coalition answers."""
    qc = load_questions()
    gold = {r["_meta"]["id"]: r["questions"] for r in load_requests("at", "gold", "test")}
    pred: dict[str, dict[str, bool]] = defaultdict(dict)
    for it in items(tag, model):
        if it["qid"] in ALL_COALITIONS:
            pred[it["rid"]][it["qid"]] = qc.keys(it["qid"])[it["pred"]] == "true"
    ok: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for rid, answers in pred.items():
        g = gold.get(rid, {})
        if "min_signers" in g:
            n["min_signers"] += 1
            ok["min_signers"] += min_signers_from(answers) == g["min_signers"]["label"]
        if "rule_type" in g:
            groups = [
                Group.of(*Counter(PERSON_ROLE[p] for p in ALL_COALITIONS[q]).items())
                for q in minimal_coalitions(answers)
            ]
            rt = (
                derive_answers(SigningRule("rule", frozenset(groups), False, None), None).get(
                    "rule_type"
                )
                if groups
                else None
            )
            n["rule_type"] += 1
            ok["rule_type"] += rt == g["rule_type"]["label"]
    return {k: ok[k] / n[k] for k in n}


def coverage_risk(tag: str, model: str, alpha: str = "0.02") -> tuple[float, float | None]:
    thr = policy(tag, model)["thresholds"].get(alpha, {}).get("noul")
    co = [it for it in items(tag, model) if it["qid"] in ALL_COALITIONS]
    if thr is None or not co:
        return float("nan"), None
    answered = [it for it in co if it["conf"] >= thr]
    risk = sum(it["pred"] != it["y"] for it in answered) / len(answered) if answered else None
    return len(answered) / len(co), risk


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--a-tag", default="no-random-to-at-gold", help="AT gold tag of --a")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args(argv)
    tags = {
        a.c: "at-random-to-at-gold",
        a.a: a.a_tag,
        "phrase_at": "at-random-to-at-gold",
    }
    if a.b:
        tags[a.b] = "at-random-to-at-gold"
    rows: dict[str, dict[str, Any]] = {}
    for model, tag in tags.items():
        pq = per_question(tag, model)
        cm, cn = coalition_mean(pq)
        row: dict[str, Any] = {
            "coalition_mean": cm,
            "coalition_items": cn,
            "min_signers": pq.get("min_signers", {}).get("calibrated", {}).get("accuracy"),
            "rule_type": pq.get("rule_type", {}).get("calibrated", {}).get("accuracy"),
        }
        if model != "phrase_at":
            row.update({f"derived_{k}": v for k, v in derived(tag, model).items()})
            row["coverage_2pct"], row["risk_2pct"] = coverage_risk(tag, model)
        rows[model] = row
    c, a_ = rows[a.c], rows[a.a]
    no_c, no_a = per_question("no-random", a.c), per_question("no-random", a.a)
    deltas = {
        q: no_c[q]["calibrated"]["accuracy"] - no_a[q]["calibrated"]["accuracy"]
        for q in no_a
        if q in no_c
    }
    # Pre-registered (2026-10-05/06): H11 and H14 use the DERIVED min_signers / rule_type
    # (from the predicted coalitions). The direct heads are reported as secondary, never mixed in.
    d_ms, d_rt = c.get("derived_min_signers") or 0, c.get("derived_rule_type") or 0
    hyp = {
        "H11": c["coalition_mean"] >= 0.95 and d_ms >= 0.90 and d_rt >= 0.85,
        "H12": (c.get("coverage_2pct") or 0) >= 0.70 and (c.get("risk_2pct") or 1) <= 0.05,
        "H13": c["coalition_mean"] - a_["coalition_mean"] >= 0.10
        and all(abs(d) <= 0.005 for d in deltas.values()),
        "H14": (c["coalition_mean"] + d_ms) / 2
        > (rows["phrase_at"]["coalition_mean"] + (rows["phrase_at"]["min_signers"] or 0)) / 2,
    }
    secondary = {
        "H11_direct_heads": c["coalition_mean"] >= 0.95
        and (c.get("min_signers") or 0) >= 0.90
        and (c.get("rule_type") or 0) >= 0.85,
        "H14_direct_heads": (c["coalition_mean"] + (c.get("min_signers") or 0)) / 2
        > (rows["phrase_at"]["coalition_mean"] + (rows["phrase_at"]["min_signers"] or 0)) / 2,
    }
    report = {
        "at_gold": rows,
        "no_test_delta_c_minus_a": deltas,
        "hypotheses": hyp,
        "secondary_not_preregistered": secondary,
    }
    out = RESULTS / "plan12"
    out.mkdir(parents=True, exist_ok=True)
    (out / a.out).write_text(json.dumps(report, indent=1))
    keys = [
        "coalition_mean",
        "min_signers",
        "derived_min_signers",
        "rule_type",
        "derived_rule_type",
        "coverage_2pct",
        "risk_2pct",
    ]
    print("AT gold".ljust(22) + "".join(m[:14].rjust(15) for m in rows))
    for k in keys:
        cells = []
        for r in rows.values():
            v = r.get(k)
            cells.append(("-" if v is None or v != v else f"{100 * v:.1f}").rjust(15))
        print(k.ljust(22) + "".join(cells))
    worst = min(deltas.items(), key=lambda kv: kv[1])
    print(
        f"NO test, C′ − A′: worst {worst[0]} {100 * worst[1]:+.2f} pp over {len(deltas)} questions"
    )
    for h, v in hyp.items():
        print(f"{h} (pre-registered): {'met' if v else 'NOT met'}")
    for h, v in secondary.items():
        print(f"{h} (secondary, not pre-registered): {'met' if v else 'NOT met'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
