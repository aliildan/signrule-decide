"""Error rate per text pattern on a cross-jurisdiction target (aggregate, k-anonymous).

    uv run python eval/pattern_errors.py --tag no-random-to-at-pilot --model no-4b-v01 \
        --target at:pilot --qid ceo_alone

Joins the private item sidecar with the masked target requests, normalises dates, and prints the
error rate of each state pattern shared by at least --min companies (patterns rarer than that
are summed into one line and never printed). Output goes to stdout only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.ingest.ingest_at import template_key


def pattern(state: dict) -> str:
    parts = [
        f"{k}: {template_key(str(state[k]))}"
        for k in ("legal_form", "signature_rule", "procuration_rule")
        if state.get(k)
    ]
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, help="JUR:SPLIT")
    ap.add_argument("--qid", required=True)
    ap.add_argument("--min", type=int, default=20)
    a = ap.parse_args(argv)
    jur, split = a.target.split(":", 1)
    reqs = {r["_meta"]["id"]: r for r in load_requests(jur, split, "test")}
    side = REPO_ROOT / "runs" / "eval" / a.tag / f"{a.model}.test.items.jsonl"
    groups: dict[str, list[tuple[bool, int, float]]] = defaultdict(list)
    for line in side.read_text().splitlines():
        it = json.loads(line)
        if it["qid"] != a.qid:
            continue
        groups[pattern(reqs[it["rid"]]["state"])].append(
            (it["pred"] == it["y"], it["y"], it["conf"])
        )
    rare = [x for g in groups.values() if len(g) < a.min for x in g]
    rows = sorted(
        (
            (len(g), sum(not ok for ok, _, _ in g), k, g)
            for k, g in groups.items()
            if len(g) >= a.min
        ),
        key=lambda r: -r[1],
    )
    total = sum(len(g) for g in groups.values())
    errors = sum(not ok for g in groups.values() for ok, _, _ in g)
    print(f"{a.qid}: {errors}/{total} errors; {len(groups)} patterns, {len(rows)} with >= {a.min}")
    for n, e, k, g in rows:
        ys = sum(y for _, y, _ in g)
        conf = sum(c for ok, _, c in g if not ok) / e if e else 0.0
        print(f"  n={n:5d} err={e:4d} ({100 * e / n:5.1f} %) y=1:{ys:5d} conf(err)={conf:.2f}  {k}")
    if rare:
        e = sum(not ok for ok, _, _ in rare)
        print(f"  rare patterns (< {a.min} companies each): n={len(rare)} err={e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
