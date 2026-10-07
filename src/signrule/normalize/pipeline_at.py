"""Austria: cache -> masked, court-coded, deduplicated, split training data (plan-11 Task 1).

    python -m signrule.normalize.pipeline_at run     # processed/at/random/{train,val,test}.jsonl
    python -m signrule.normalize.pipeline_at check   # leakage + pilot exclusion + PII -> check.json

Same scheme as Norway (dedup.py), with one difference: Austrian texts carry the function's start
date ("vertritt seit 21.12.2018 …"), which would make every company unique. Dedup and split keys
therefore use the date-normalised text (`template_key`); the group's representative keeps its real
wording. Pilot companies (data/splits/at_pilot.json) are excluded from every split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from signrule.common.ids import fnr_valid, isikukood_valid
from signrule.common.paths import PROCESSED_DIR, SPLITS_DIR, require_vault, stamped_files
from signrule.format.request import load_questions
from signrule.ingest.ingest_at import template_key
from signrule.normalize.dedup import DedupGroup, Member, dedup, leakage_report, random_split
from signrule.normalize.mask import residual_name_risk
from signrule.normalize.normalize_at import PILOT_MANIFEST, iter_records, to_at_request
from signrule.normalize.text import text_key
from signrule.ontology.mapping import load_roles

OUT = PROCESSED_DIR / "at"
SPLIT = "random"
_ELEVEN = re.compile(r"(?<![0-9A-Za-z])\d{11}(?![0-9A-Za-z])")


def pattern_key(text: str | None) -> str:
    return text_key(template_key(text)) if text else ""


def n_managing_directors(state: dict[str, Any]) -> int:
    return sum(r["count"] for r in state.get("roles") or [] if r["role"] == "Geschäftsführer")


def collect() -> tuple[list[Member], Counter[str]]:
    roles, questions = load_roles(), load_questions()
    pilot = set(json.loads(PILOT_MANIFEST.read_text())["ids"]) if PILOT_MANIFEST.exists() else set()
    stats: Counter[str] = Counter()
    members: list[Member] = []
    for rec in iter_records():
        if rec.fnr in pilot:
            stats["pilot (excluded)"] += 1
            continue
        r = to_at_request(rec, roles, questions)
        if r is None:
            stats["sole trader / no current function"] += 1
            continue
        if r.residual_risk:
            stats["residual name risk (dropped)"] += 1
            continue
        if not r.request["questions"]:
            stats["no coded label"] += 1
            continue
        st = r.request["state"]
        sig, prok = st.get("signature_rule"), st.get("procuration_rule")
        req = {
            "state": st,
            "questions": r.request["questions"],
            "_stratum": r.stratum,
            "_label_source": r.request["_meta"].get("label_source", "codes"),
        }
        members.append(Member(rec.fnr, req, pattern_key(sig), pattern_key(prok), None, True))
        stats["kept"] += 1
    return members, stats


def _row(g: DedupGroup, split_name: str) -> dict[str, Any]:
    first = g.members[0].request
    return {
        "state": g.request["state"],
        "questions": g.request["questions"],
        "_meta": {
            "id": f"at/{g.cluster}/{hashlib.sha256(repr(g.key).encode()).hexdigest()[:10]}",
            "group_id": f"at/{g.cluster}",
            "n_duplicates": g.n_duplicates,
            "has_text": True,
            "split": split_name,
            "stratum": first["_stratum"],
            "label_source": first["_label_source"],
            "n_gf": n_managing_directors(g.request["state"]),
        },
    }


def _write(path: Any, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cmd_run(args: argparse.Namespace) -> int:
    require_vault()
    members, stats = collect()
    groups = dedup(members)
    assign = random_split(groups)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        sp = assign[g.cluster]
        rows[sp].append(_row(g, sp))
        manifest[sp].extend(g.entity_ids)
    for sp in ("train", "val", "test"):
        _write(OUT / SPLIT / f"{sp}.jsonl", rows[sp])
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / f"at_{SPLIT}.json").write_text(
        json.dumps({sp: sorted(ids) for sp, ids in manifest.items()})
    )
    sources = Counter(g.members[0].request["_label_source"] for g in groups)
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for g in groups:
        for qid, q in g.request["questions"].items():
            labels[qid][str(q["label"])] += 1
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "records": dict(stats),
        "n_groups": len(groups),
        "n_entities": sum(g.n_duplicates for g in groups),
        "label_source_groups": dict(sources),
        "splits": {
            sp: {
                "groups": len(rows[sp]),
                "entities": sum(r["_meta"]["n_duplicates"] for r in rows[sp]),
            }
            for sp in ("train", "val", "test")
        },
        "labels": {q: dict(c.most_common()) for q, c in sorted(labels.items())},
    }
    (OUT / f"summary_{SPLIT}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def scan_row(line: str) -> Counter[str]:
    issues: Counter[str] = Counter()
    if re.search(r"(?i)geburtsdatum|synniaeg|birth_?date", line):
        issues["birthdate_key"] += 1
    if any(fnr_valid(m) or isikukood_valid(m) for m in _ELEVEN.findall(line)):
        issues["national_id"] += 1
    st = json.loads(line)["state"]
    for k in ("signature_rule", "procuration_rule"):
        if residual_name_risk(st.get(k) or "", "AT"):
            issues["residual_name"] += 1
    return issues


def file_hashes() -> dict[str, str]:
    return {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in stamped_files(OUT)
    }


def cmd_check(args: argparse.Namespace) -> int:
    ok = True
    manifest = json.loads((SPLITS_DIR / f"at_{SPLIT}.json").read_text())
    ent: dict[str, set[str]] = defaultdict(set)
    for sp, ids in manifest.items():
        for e in ids:
            ent[e].add(sp)
    overlap = sum(1 for s in ent.values() if len(s) > 1)
    pilot = set(json.loads(PILOT_MANIFEST.read_text())["ids"]) if PILOT_MANIFEST.exists() else set()
    in_pilot = len(pilot & set(ent))
    patterns: dict[tuple[str, str], set[str]] = defaultdict(set)
    issues: Counter[str] = Counter()
    for sp in ("train", "val", "test"):
        with (OUT / SPLIT / f"{sp}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                st = json.loads(line)["state"]
                key = (
                    pattern_key(st.get("signature_rule")),
                    pattern_key(st.get("procuration_rule")),
                )
                patterns[key].add(sp)
                issues.update(scan_row(line))
    for p in sorted(OUT.glob("pilot/*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            issues.update(scan_row(line))
    tmpl_overlap = sum(1 for s in patterns.values() if "train" in s and "test" in s)
    print(
        f"entity overlap={overlap} pilot ids in splits={in_pilot} template overlap={tmpl_overlap}"
    )
    print(f"PII scan: {dict(issues) or 'clean'}")
    ok = overlap == 0 and in_pilot == 0 and tmpl_overlap == 0 and not issues
    if args.near_dups:
        members, _ = collect()
        groups = dedup(members)
        rep = leakage_report(groups, random_split(groups))
        print(f"near-duplicate rate test vs train: {rep.near_dup_rate:.4f} ({rep.n_test_clusters})")
    print("data-check:", "OK" if ok else "FAILED")
    if ok:
        stamp = {
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": file_hashes(),
        }
        (OUT / "check.json").write_text(json.dumps(stamp, indent=1))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pipeline_at",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(func=cmd_run)
    p = sub.add_parser("check")
    p.add_argument("--near-dups", action="store_true")
    p.set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
