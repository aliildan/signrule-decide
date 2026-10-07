"""Norway data pipeline: cache -> masked, labelled, deduplicated, split requests (plan-03 T8).

    python -m signrule.normalize.pipeline_no run     # writes data/processed/no/ + data/splits/
    python -m signrule.normalize.pipeline_no check   # leakage + PII scan + labels; exit 1 if bad

Prints aggregates only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from signrule.common.ids import fnr_valid
from signrule.common.paths import PROCESSED_DIR, SPLITS_DIR, raw_dir, require_vault, stamped_files
from signrule.format.request import load_questions
from signrule.normalize.dedup import (
    DedupGroup,
    Member,
    dedup,
    leakage_report,
    random_split,
    temporal_split,
)
from signrule.normalize.mask import residual_name_risk
from signrule.normalize.normalize_no import build_record, to_masked_request
from signrule.ontology.mapping import load_legal_vocab, load_no_rules, load_roles

RAW = raw_dir("no")
OUT = PROCESSED_DIR / "no"
CUTOFF = date(2025, 1, 1)
_FNR = re.compile(r"(?<![0-9A-Za-z])\d{11}(?![0-9A-Za-z])")
_BIRTH = re.compile(r"(?i)f[oø]e?dsels(dato|nummer)|birth_?date")


def _entities() -> dict[str, dict[str, Any]]:
    snaps = sorted((RAW / "enheter").glob("enheter_*.jsonl.gz"))
    if not snaps:
        raise SystemExit("no entities snapshot; run ingest_no entities")
    out = {}
    with gzip.open(snaps[-1], "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            out[e["orgnr"]] = e
    return out


def _load(path: Path) -> dict[str, Any] | None:
    env = json.loads(path.read_text(encoding="utf-8"))
    return env["body"] if env.get("status_code") == 200 and env.get("body") else None


def _name_leak(req: dict[str, Any], names: list[str]) -> bool:
    """True if a name token (>= 4 chars, not legal vocabulary) survives in the masked state."""
    blob = json.dumps(req["state"], ensure_ascii=False).casefold()
    vocab = load_legal_vocab("NO")
    for n in names:
        for tok in re.split(r"[\s\-]+", n):
            t = tok.casefold()
            if len(t) >= 4 and t not in vocab and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", blob):
                return True
    return False


def collect(
    limit: int | None = None, only: set[str] | None = None
) -> tuple[list[Member], Counter[str]]:
    entities = _entities()
    rules, roles, questions = load_no_rules(), load_roles(), load_questions()
    stats: Counter[str] = Counter()
    members: list[Member] = []
    sig_files = sorted((RAW / "fullmakt" / "signatur").glob("*/*.json"))
    for i, path in enumerate(sig_files):
        if limit and i >= limit:
            break
        if only is not None and path.stem not in only:
            continue
        stats["signatur_cached"] += 1
        sig = _load(path)
        if sig is None:
            stats["signatur_not_200"] += 1
            continue
        orgnr = path.stem
        prok_path = RAW / "fullmakt" / "prokura" / orgnr[:3] / f"{orgnr}.json"
        prok = _load(prok_path) if prok_path.exists() else None
        if not prok_path.exists():
            stats["prokura_missing"] += 1
        rec = build_record(sig, prok, entities.get(orgnr), rules)
        if rec is None:
            stats["rutine_na"] += 1
            continue
        stats[f"regel_{rec.signing.status}"] += 1
        if rec.unmapped_codes:
            stats["unmapped_records"] += 1
            for c in rec.unmapped_codes:
                stats[f"unmapped_code:{c}"] += 1
        m = to_masked_request(rec, roles=roles, questions=questions, rules=rules)
        if m.residual_risk:
            stats["dropped_residual_name_risk"] += 1
            continue
        if _name_leak(m.request, rec.person_names):
            stats["dropped_name_leak"] += 1
            continue
        stats["kept"] += 1
        stats["kept_with_text" if m.has_text else "kept_roles_only"] += 1
        members.append(
            Member(orgnr, m.request, m.signature_key, m.procuration_key, rec.registered, m.has_text)
        )
    return members, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _kev_row(g: DedupGroup, split_name: str) -> dict[str, Any]:
    return {
        **g.request,
        "_meta": {
            "id": f"no/{g.cluster}/{hashlib.sha256(repr(g.key).encode()).hexdigest()[:10]}",
            "group_id": f"no/{g.cluster}",
            "n_duplicates": g.n_duplicates,
            "has_text": g.members[0].has_text,
            "label_source": g.members[0].request.get("_meta", {}).get("label_source", "register"),
            "split": split_name,
        },
    }


def label_distribution(groups: list[DedupGroup]) -> dict[str, dict[str, int]]:
    dist: dict[str, Counter[str]] = defaultdict(Counter)
    for g in groups:
        for qid, q in g.request["questions"].items():
            dist[qid][str(q["label"])] += 1
    return {q: dict(c.most_common()) for q, c in sorted(dist.items())}


def cmd_run(args: argparse.Namespace) -> int:
    require_vault()
    only = None
    if args.only_entities:  # a split manifest: reproduce an earlier entity set exactly
        manifest = json.loads(Path(args.only_entities).read_text())
        only = {e for ids in manifest.values() for e in ids}
        print(f"restricting to {len(only)} entities from {args.only_entities}")
    members, stats = collect(args.limit, only)
    groups = dedup(members)
    rnd = random_split(groups)
    tmp_assign, seen_new = temporal_split(groups, CUTOFF)
    manifests: dict[str, dict[str, list[str]]] = {
        "random": defaultdict(list),
        "temporal": defaultdict(list),
    }
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "random": defaultdict(list),
        "temporal": defaultdict(list),
    }
    for g in groups:
        for name, assign in (("random", rnd), ("temporal", tmp_assign)):
            sp = assign[g.cluster]
            rows[name][sp].append(_kev_row(g, sp))
            manifests[name][sp].extend(g.entity_ids)
    for name in rows:
        for sp in ("train", "val", "test"):
            _write_jsonl(OUT / name / f"{sp}.jsonl", rows[name][sp])
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        (SPLITS_DIR / f"no_{name}.json").write_text(
            json.dumps({sp: sorted(ids) for sp, ids in manifests[name].items()})
        )
    conflicts: Counter[str] = Counter()
    for g in groups:
        conflicts.update(g.conflicts)
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "records": dict(stats),
        "n_groups": len(groups),
        "n_entities": sum(g.n_duplicates for g in groups),
        "label_conflicts": dict(conflicts),
        "temporal_seen_text_new_entities": seen_new,
        "splits": {
            name: {sp: len(rows[name][sp]) for sp in ("train", "val", "test")} for name in rows
        },
        "labels": label_distribution(groups),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "labels"}, indent=1))
    return 0


def _scan_rows(path: Path) -> Counter[str]:
    issues: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if _BIRTH.search(line):
                issues["birthdate_key"] += 1
            if any(fnr_valid(m.group(0)) for m in _FNR.finditer(line)):
                issues["fnr"] += 1
            st = json.loads(line)["state"]
            for k in ("signature_rule", "procuration_rule"):
                if residual_name_risk(st.get(k) or "", "NO"):
                    issues["residual_name"] += 1
    return issues


def cmd_check(args: argparse.Namespace) -> int:
    summary = json.loads((OUT / "summary.json").read_text())
    ok = True
    # leakage, recomputed from the written files
    for name in ("random", "temporal"):
        texts: dict[str, set[str]] = defaultdict(set)
        ents: dict[str, set[str]] = defaultdict(set)
        manifest = json.loads((SPLITS_DIR / f"no_{name}.json").read_text())
        for sp, ids in manifest.items():
            for e in ids:
                ents[e].add(sp)
        for sp in ("train", "val", "test"):
            with (OUT / name / f"{sp}.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    st = json.loads(line)["state"]
                    if st.get("signature_rule") or st.get("procuration_rule"):
                        texts[
                            f"{st.get('signature_rule')}|{st.get('procuration_rule')}".casefold()
                        ].add(sp)
        ent_overlap = sum(1 for s in ents.values() if len(s) > 1)
        text_overlap = sum(1 for s in texts.values() if "train" in s and "test" in s)
        print(f"[{name}] entity overlap={ent_overlap} exact text overlap train/test={text_overlap}")
        ok &= ent_overlap == 0 and text_overlap == 0
    # PII scan
    issues: Counter[str] = Counter()
    for p in sorted(OUT.glob("*/*.jsonl")):
        issues.update(_scan_rows(p))
    print(f"PII scan: {dict(issues) or 'clean'}")
    ok &= not issues
    # unmapped share
    rec = summary["records"]
    unmapped = rec.get("unmapped_records", 0) / max(
        rec.get("kept", 0) + rec.get("dropped_residual_name_risk", 0), 1
    )
    print(f"unmapped share={unmapped:.4f} (limit 0.02)")
    ok &= unmapped < 0.02
    # near-duplicates (informational), recomputed with MinHash on the in-memory groups
    if args.near_dups:
        members, _ = collect()
        groups = dedup(members)
        rep = leakage_report(groups, random_split(groups))
        print(
            f"[random] near-duplicate rate test vs train (Jaccard>=0.95): {rep.near_dup_rate:.4f} "
            f"over {rep.n_test_clusters} test texts"
        )
    print("data-check:", "OK" if ok else "FAILED")
    if ok:
        (OUT / "check.json").write_text(
            json.dumps(
                {
                    "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "files": file_hashes(),
                },
                indent=1,
            )
        )
    return 0 if ok else 1


def file_hashes() -> dict[str, str]:
    """sha256 of every processed split file; training refuses data whose hashes differ."""
    return {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in stamped_files(OUT)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pipeline_no",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--limit", type=int)
    p.add_argument("--only-entities", help="split manifest JSON whose entity ids to keep")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("check")
    p.add_argument("--near-dups", action="store_true")
    p.set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
