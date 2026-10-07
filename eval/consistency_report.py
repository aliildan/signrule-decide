"""Effect of the abstention gates on a scored model (parseable gate + consistency gate).

    uv run python eval/consistency_report.py --split random --part test --model no-4b-v01

Reads the private item sidecar and the model's results JSON (for its LTT thresholds), rebuilds the
per-request predictions, and reports how many requests the consistency gate flags, accuracy of the
structural answers on flagged vs unflagged requests, and end-to-end coverage/risk at α = 2 % with
LTT only, + parseable gate, + consistency gate. Aggregate output only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from signrule.calib.consistency import STRUCTURAL, has_named_signatory, is_consistent
from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.format.request import load_questions


def _value(qid: str, idx: int, keys: list[str]) -> object:
    k = keys[idx]
    return (
        {"true": True, "false": False}.get(k, k) if qid not in ("rule_type", "min_signers") else k
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jurisdiction", default="no")
    ap.add_argument("--split", default="random")
    ap.add_argument("--part", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--alpha", default="0.02")
    a = ap.parse_args(argv)
    tag = f"{a.jurisdiction}-{a.split}"
    qc = load_questions()
    items = [
        json.loads(x)
        for x in (REPO_ROOT / "runs" / "eval" / tag / f"{a.model}.{a.part}.items.jsonl")
        .read_text()
        .splitlines()
    ]
    policy = json.loads((REPO_ROOT / "results" / tag / f"{a.model}.{a.part}.json").read_text())[
        "policy"
    ]
    lam = policy["thresholds"][a.alpha]
    states = {r["_meta"]["id"]: r["state"] for r in load_requests(a.jurisdiction, a.split, a.part)}

    by_req: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_req[it["rid"]].append(it)

    flagged = 0
    struct_ok = {"flagged": [0, 0], "clean": [0, 0]}
    modes = {"ltt": [0, 0, 0], "ltt+parseable": [0, 0, 0], "ltt+parseable+consistency": [0, 0, 0]}
    for rid, its in by_req.items():
        pred = {
            i["qid"]: _value(i["qid"], i["pred"], qc.keys(i["qid"]))
            for i in its
            if i["qid"] in STRUCTURAL
        }
        consistent = is_consistent(pred, has_named_signatory(states.get(rid)))
        flagged += not consistent
        bucket = "clean" if consistent else "flagged"
        parse = next((i for i in its if i["qid"] == "parseable"), None)
        gate_parse = parse is not None and parse["pred"] == 0  # predicted "false"
        for i in its:
            ok = i["pred"] == i["y"]
            if i["qid"] in STRUCTURAL:
                struct_ok[bucket][0] += ok
                struct_ok[bucket][1] += 1
            thr = lam.get(i["qtype"])
            answer = thr is not None and i["conf"] >= thr
            for mode, (gp, gc) in {
                "ltt": (False, False),
                "ltt+parseable": (True, False),
                "ltt+parseable+consistency": (True, True),
            }.items():
                ans = answer
                if gp and gate_parse and i["qid"] != "parseable":
                    ans = False
                if gc and not consistent and i["qid"] in STRUCTURAL:
                    ans = False
                m = modes[mode]
                m[0] += 1
                m[1] += ans
                m[2] += ans and not ok

    out = {
        "model": a.model,
        "split": a.split,
        "part": a.part,
        "alpha": float(a.alpha),
        "n_requests": len(by_req),
        "consistency_flagged": flagged,
        "flagged_share": flagged / max(len(by_req), 1),
        "structural_accuracy": {k: (v[0] / v[1] if v[1] else None) for k, v in struct_ok.items()},
        "structural_items": {k: v[1] for k, v in struct_ok.items()},
        "selective": {
            k: {"coverage": v[1] / v[0], "risk": (v[2] / v[1]) if v[1] else None}
            for k, v in modes.items()
        },
    }
    d = REPO_ROOT / "results" / tag / "consistency"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.model}.{a.part}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{a.model} [{a.part}]: {flagged}/{len(by_req)} requests flagged inconsistent "
        f"({100 * out['flagged_share']:.1f} %); structural accuracy clean "
        f"{out['structural_accuracy']['clean']} vs flagged {out['structural_accuracy']['flagged']}"
    )
    for k, v in out["selective"].items():
        risk = "–" if v["risk"] is None else f"{100 * v['risk']:.2f} %"
        print(f"   {k:28s} coverage {100 * v['coverage']:.1f} %  risk {risk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
