"""Gold accuracy split by whether the item's date-normalised pattern occurs in the AT training data.

    uv run python eval/seen_patterns.py --tag at-random-to-at-gold --models noat-4b-v1,phrase_at

Pattern = (legal form, date-normalised signature text, date-normalised procuration text, roles) —
the same key the AT training data is deduplicated by. Aggregate output only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from signrule.common.paths import REPO_ROOT
from signrule.evaluation.harness import load_requests
from signrule.normalize.dedup import roles_signature
from signrule.normalize.pipeline_at import pattern_key
from signrule.ontology.coalitions import ALL_COALITIONS


def key(state: dict) -> tuple[str, str, str, str]:
    return (
        str(state.get("legal_form")),
        pattern_key(state.get("signature_rule")),
        pattern_key(state.get("procuration_rule")),
        roles_signature(state),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="at-random-to-at-gold")
    ap.add_argument("--models", default="noat-4b-v1,phrase_at")
    a = ap.parse_args(argv)
    train = {key(r["state"]) for r in load_requests("at", "random", "train")}
    gold = {r["_meta"]["id"]: r for r in load_requests("at", "gold", "test")}
    seen = {rid: key(r["state"]) in train for rid, r in gold.items()}
    print(f"gold items: {len(gold)}; pattern seen in AT training data: {sum(seen.values())}")
    focus = {*ALL_COALITIONS, "min_signers", "rule_type"}
    for model in a.models.split(","):
        side = REPO_ROOT / "runs" / "eval" / a.tag / f"{model}.test.items.jsonl"
        c: Counter[tuple[str, bool]] = Counter()
        n: Counter[tuple[str, bool]] = Counter()
        for line in side.read_text().splitlines():
            it = json.loads(line)
            if it["qid"] not in focus:
                continue
            grp = "coalitions" if it["qid"] in ALL_COALITIONS else it["qid"]
            k = (grp, seen.get(it["rid"], False))
            n[k] += 1
            c[k] += it["pred"] == it["y"]
        print(f"\n{model}")
        for grp in ("coalitions", "min_signers", "rule_type"):
            cells = []
            for s in (True, False):
                k = (grp, s)
                cells.append(
                    f"{'seen' if s else 'unseen'} {100 * c[k] / n[k]:.1f} % (n={n[k]})"
                    if n[k]
                    else f"{'seen' if s else 'unseen'} -"
                )
            print(f"  {grp:12s} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
