"""Pre-registered H21/H22 on the Austrian focused batch `at-text-2` (plan-16; aggregates only).

    uv run python eval/plan16_report.py --model noat-4b-v2

Reads the item sidecar of the AT-calibrated scoring of `at:gold` (tag at-random-to-at-gold), keeps
the rows of `at-text-2` (`_meta.stratum`), and reports: coalition mean, min_signers, dangerous
yes/no errors (model "yes", gold "no") and the risk at alpha 2 % on the answered structural
questions (ambiguity excluded: experimental). Also per structure (Vorstand / partners / GmbH with
>= 2 GF) and for the whole `at:gold`. Writes results/plan16/at-report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.format.request import load_questions
from signrule.ontology.coalitions import ALL_COALITIONS

RESULTS = REPO_ROOT / "results"
SIDECARS = REPO_ROOT / "runs" / "eval"
TAG = "at-random-to-at-gold"
OFFICE = {"ceo_alone", "chair_alone", "two_board_members_jointly", "ceo_with_one_board_member"}
YES_NO = set(ALL_COALITIONS) | OFFICE


def summarise(rows: list[dict[str, Any]], thresholds: dict[str, float | None]) -> dict[str, Any]:
    qc = load_questions()
    acc = lambda xs: sum(r["pred"] == r["y"] for r in xs) / len(xs) if xs else None  # noqa: E731
    coal = [r for r in rows if r["qid"] in ALL_COALITIONS]
    mins = [r for r in rows if r["qid"] == "min_signers"]
    yn = [r for r in rows if r["qid"] in YES_NO]
    dangerous = [
        r
        for r in yn
        if qc.keys(r["qid"])[r["pred"]] == "true" and qc.keys(r["qid"])[r["y"]] == "false"
    ]
    structural = [r for r in rows if r["qid"] != "ambiguity"]
    answered = [
        r
        for r in structural
        if thresholds.get(r["qtype"]) is not None and r["conf"] >= thresholds[r["qtype"]]
    ]
    risk = sum(r["pred"] != r["y"] for r in answered) / len(answered) if answered else None
    return {
        "coalition_mean": acc(coal),
        "coalition_items": len(coal),
        "min_signers": acc(mins),
        "min_signers_items": len(mins),
        "rule_type": acc([r for r in rows if r["qid"] == "rule_type"]),
        "ceo_alone": acc([r for r in rows if r["qid"] == "ceo_alone"]),
        "ceo_alone_items": sum(r["qid"] == "ceo_alone" for r in rows),
        "chair_alone": acc([r for r in rows if r["qid"] == "chair_alone"]),
        "dangerous_rate": len(dangerous) / len(yn) if yn else None,
        "dangerous": len(dangerous),
        # how sure the model was when it said "can sign" wrongly (abstention cannot catch these)
        "dangerous_conf_ge_0_9": sum(r["conf"] >= 0.9 for r in dangerous),
        "yes_no_items": len(yn),
        "alpha_2pct": {
            "coverage": len(answered) / len(structural) if structural else None,
            "risk": risk,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="noat-4b-v2")
    a = ap.parse_args(argv)
    gold = load_requests("at", "gold", "test")
    batch = {r["_meta"]["id"]: r["_meta"].get("stratum") for r in gold}
    focus = {r["_meta"]["id"]: (r["_meta"].get("strata") or {}).get("focus") for r in gold}
    res = json.loads((RESULTS / TAG / f"{a.model}.test.json").read_text())
    thr = res["policy"]["thresholds"].get("0.02", {})
    rows = [
        json.loads(x)
        for x in (SIDECARS / TAG / f"{a.model}.test.items.jsonl").read_text().splitlines()
    ]
    new = [r for r in rows if batch.get(r["rid"]) == "at-text-2"]
    out = {
        "model": a.model,
        "at-text-2": summarise(new, thr),
        "by_structure": {
            k: summarise([r for r in new if focus.get(r["rid"]) == k], thr)
            for k in ("vorstand", "partner", "gmbh_multi")
        },
        "at:gold (both batches)": summarise(rows, thr),
        "at-text (first batch)": summarise(
            [r for r in rows if batch.get(r["rid"]) == "at-text"], thr
        ),
    }
    s = out["at-text-2"]
    out["H21"] = (
        (s["coalition_mean"] or 0) >= 0.95
        and (s["min_signers"] or 0) >= 0.90
        and (s["dangerous_rate"] if s["dangerous_rate"] is not None else 1) <= 0.01
    )
    out["H22"] = s["alpha_2pct"]["risk"] is not None and s["alpha_2pct"]["risk"] <= 0.02
    d = RESULTS / "plan16"
    d.mkdir(parents=True, exist_ok=True)
    (d / "at-report.json").write_text(json.dumps(out, indent=1))
    for name in ("at-text-2", "at-text (first batch)", "at:gold (both batches)"):
        v = out[name]
        print(
            f"{name:>24}: coalitions {v['coalition_mean']:.3f} (n={v['coalition_items']}), "
            f"min_signers {v['min_signers']:.3f}, rule_type {v['rule_type']:.3f}, "
            f"ceo_alone {v['ceo_alone']}, dangerous {v['dangerous']}/{v['yes_no_items']}, "
            f"alpha 2 %: coverage {v['alpha_2pct']['coverage']:.3f} risk {v['alpha_2pct']['risk']}"
        )
    for k, v in out["by_structure"].items():
        print(
            f"{k:>24}: coalitions {v['coalition_mean']:.3f}, min_signers {v['min_signers']:.3f}, "
            f"dangerous {v['dangerous']}/{v['yes_no_items']}"
        )
    print(f"H21 {'met' if out['H21'] else 'NOT met'}, H22 {'met' if out['H22'] else 'NOT met'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
