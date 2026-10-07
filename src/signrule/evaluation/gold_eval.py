"""Gold annotations -> an evaluation split (processed/<jur>/gold/test.jsonl).

    python -m signrule.evaluation.gold_eval build --set at-text --a ali --b annotator2

A question is labelled only where both annotators' derived answers agree (disagreements are left
out, not adjudicated by Claude). For Austria, questions about an ordinary board
(two_board_members_jointly, ceo_with_one_board_member) are dropped (a GmbH/KG/OG has no
representing board), ceo_alone needs a Geschäftsführer and chair_alone a chair in the roles.
The file carries `_meta.label_source` and `_meta.provenance`; it is never a training input
(training reads only the configured split's train part). Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from signrule.common.paths import GOLD_DIR, PROCESSED_DIR, require_vault
from signrule.evaluation.gold import annotation_answers, load_annotations
from signrule.format.request import load_questions, to_request
from signrule.ingest.pii import redact_text
from signrule.normalize.normalize_at import applicable_at
from signrule.normalize.normalize_dk import applicable_dk


def consensus(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ra, rb = annotation_answers(a), annotation_answers(b)
    return {q: v for q, v in ra.items() if v is not None and rb.get(q) == v}


def applicable(jur: str, state: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """Austria: the same office filter as the training labels (normalize_at.applicable_at);
    Denmark: the same filter on Danish role labels (normalize_dk.applicable_dk)."""
    if jur == "at":
        return applicable_at(state, answers)
    if jur == "dk":
        return applicable_dk(state, answers)
    return answers


def _set_rows(
    set_name: str,
    ann_a: str,
    ann_b: str,
    jur: str,
    provenance: str,
    adjudicate: dict[int, str],
    reason: str,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    qc = load_questions()
    batch = [
        json.loads(x)
        for x in (GOLD_DIR / set_name / "batch.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    a, b = load_annotations(set_name, ann_a), load_annotations(set_name, ann_b)
    rows = []
    for n, it in enumerate(batch, 1):
        if it["item_id"] not in a or it["item_id"] not in b:
            stats["not annotated by both"] += 1
            continue
        if n in adjudicate:
            chosen = {ann_a: a, ann_b: b}[adjudicate[n]][it["item_id"]]
            agreed = {q: v for q, v in annotation_answers(chosen).items() if v is not None}
            stats["adjudicated"] += 1
        else:
            agreed = consensus(a[it["item_id"]], b[it["item_id"]])
        ans = applicable(jur, it["state"], agreed)
        if not ans:
            stats["no agreed question"] += 1
            continue
        meta = {
            "id": it["item_id"],
            "group_id": it["item_id"],
            "has_text": True,
            "label_source": (
                f"gold:{set_name}:adjudicated:{adjudicate[n]} ({reason})"
                if n in adjudicate
                else f"gold:{set_name}:{ann_a}+{ann_b}:agreed"
            ),
            "provenance": provenance,
            "stratum": set_name,
            "strata": it.get("strata"),
        }
        state = {
            k: (redact_text(v)[0] if k in ("signature_rule", "procuration_rule") and v else v)
            for k, v in it["state"].items()
        }
        rows.append(to_request(state, ans, qc, source=jur, meta=meta))
        stats["items"] += 1
        stats.update(f"q:{q}" for q in ans)
    return rows


def build(
    set_name: str,
    ann_a: str,
    ann_b: str,
    jur: str,
    provenance: str,
    adjudicate: dict[int, str] | None = None,
    reason: str = "",
    also: list[tuple[str, str, str]] | None = None,
) -> Counter[str]:
    """`adjudicate` maps batch item numbers (1-based) of `set_name` to the annotator whose answers
    stand for that item (a documented adjudication; labels always come from a human). `also`:
    further (set, annotator a, annotator b) of the same jurisdiction, written into the same split;
    every row carries its set as `_meta.stratum`."""
    stats: Counter[str] = Counter()
    rows = _set_rows(set_name, ann_a, ann_b, jur, provenance, adjudicate or {}, reason, stats)
    for other, oa, ob in also or []:
        rows += _set_rows(other, oa, ob, jur, provenance, {}, "", stats)
    out = PROCESSED_DIR / jur / "gold" / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gold_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--jurisdiction", default="at")
    ap.add_argument("--provenance", default="unverified")
    ap.add_argument("--adjudicate", help="item numbers whose answers come from --adjudicate-by")
    ap.add_argument("--adjudicate-by", help="annotator chosen for those items")
    ap.add_argument("--reason", default="")
    ap.add_argument(
        "--also", action="append", default=[], help="SET:A:B, another set into the same split"
    )
    args = ap.parse_args(argv)
    require_vault(GOLD_DIR)
    adj: dict[int, str] = {}
    if args.adjudicate:
        if args.adjudicate_by not in (args.a, args.b) or not args.reason:
            raise SystemExit("--adjudicate needs --adjudicate-by (one of --a/--b) and --reason")
        adj = {int(x): args.adjudicate_by for x in args.adjudicate.split(",")}
    also = [tuple(x.split(":")) for x in args.also]
    if any(len(x) != 3 for x in also):
        raise SystemExit("--also takes SET:A:B")
    stats = build(
        args.set, args.a, args.b, args.jurisdiction, args.provenance, adj, args.reason, also
    )
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
