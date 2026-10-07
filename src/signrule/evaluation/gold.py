"""Human gold sets (plan-05 T5-T7): sampling, annotation records -> CIR -> answers, agreement.

    python -m signrule.evaluation.gold sample-no-rt --n 250
    python -m signrule.evaluation.gold sample-dk --n 300
    python -m signrule.evaluation.gold agreement --set no-rt --a ann1 --b ann2

Batches and annotations live in data/gold/<set>/ (encrypted vault, never published). Only
aggregate agreement statistics are written to results/gold/<set>/. Items come from `val`/`test`
only, so no gold text was ever trained on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from signrule.common.paths import GOLD_DIR, REPO_ROOT, require_vault
from signrule.evaluation.harness import RulesNo, load_requests
from signrule.ontology.cir import (
    ROLES,
    Group,
    ProcurationRule,
    SigningRule,
    derive_answers,
)

COLLECTIVE_CHOICES = ("", "ALL_BOARD", "ALL_PARTNERS", "ALL_EXECUTIVES")
ANSWER_KEYS = (
    "rule_type",
    "min_signers",
    "ceo_alone",
    "chair_alone",
    "two_board_members_jointly",
    "ceo_with_one_board_member",
    "prokura_present",
    "prokura_joint",
    "parseable",
    "ambiguity",
)


# ---- sampling -------------------------------------------------------------------------------


def _len_bucket(text: str) -> str:
    n = len(text)
    return "<40" if n < 40 else "<80" if n < 80 else "<160" if n < 160 else ">=160"


def sample_no_rt(n: int, seed: int = 13, parts: tuple[str, ...] = ("val", "test")) -> list[dict]:
    """RT items (register: rule present but not interpretable), stratified by whether the text
    matches one of the register's rule descriptions (role-mismatch RT) and by text length."""
    rules = RulesNo()
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for part in parts:
        for r in load_requests("no", "random", part):
            q = r["questions"].get("parseable")
            if q is None or q["label"] is not False:
                continue
            text = r["state"].get("signature_rule") or ""
            match = rules._match(text)[0]
            stratum = ("matches_rule" if match == "rule" else "unmatched", _len_bucket(text))
            pool[stratum].append(
                {
                    "item_id": r["_meta"]["id"],
                    "set": "no-rt",
                    "part": part,
                    "state": r["state"],
                    "strata": {"match": stratum[0], "length": stratum[1]},
                }
            )
    rng = random.Random(seed)
    strata = sorted(pool)
    for s in strata:
        rng.shuffle(pool[s])
    out: list[dict] = []
    while len(out) < n and any(pool[s] for s in strata):  # round-robin across strata
        for s in strata:
            if pool[s] and len(out) < n:
                out.append(pool[s].pop())
    return out


def _gf_bucket(state: dict[str, Any]) -> str:
    n = sum(r["count"] for r in state.get("roles") or [] if r["role"] == "Geschäftsführer")
    return "gf0" if n == 0 else "gf1" if n == 1 else "gf2" if n == 2 else "gf3+"


def sample_at(n: int, seed: int = 13) -> list[dict]:
    """Austrian free-text items (stratum at-text) that no model has trained on: the frozen pilot
    and the pattern-disjoint test part of `at/random`. One item per text pattern (group_id),
    stratified by number of managing directors, procuration present, and legal form."""
    pool: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for split, part in (("pilot", "test"), ("random", "test")):
        try:
            reqs = load_requests("at", split, part)
        except FileNotFoundError:
            continue
        for r in reqs:
            m = r["_meta"]
            if m.get("stratum") != "at-text" or m["group_id"] in seen:
                continue
            seen.add(m["group_id"])
            st = r["state"]
            stratum = (
                _gf_bucket(st),
                "prok" if st.get("procuration_rule") else "no-prok",
                "GmbH" if st.get("legal_form") == "GmbH" else "other",
            )
            pool[stratum].append(
                {
                    "item_id": m["id"],
                    "set": "at-text",
                    "part": f"{split}/{part}",
                    "state": st,
                    "strata": {"gf": stratum[0], "prok": stratum[1], "form": stratum[2]},
                }
            )
    rng = random.Random(seed)
    strata = sorted(pool)
    for s in strata:
        rng.shuffle(pool[s])
    out: list[dict] = []
    while len(out) < n and any(pool[s] for s in strata):
        for s in strata:
            if pool[s] and len(out) < n:
                out.append(pool[s].pop())
    return out


def _round_robin(pool: dict[Any, list[dict]], n: int, rng: random.Random) -> list[dict]:
    strata = sorted(pool)
    for s in strata:
        rng.shuffle(pool[s])
    out: list[dict] = []
    while len(out) < n and any(pool[s] for s in strata):
        for s in strata:
            if pool[s] and len(out) < n:
                out.append(pool[s].pop())
    return out


def _dk_bucket(state: dict[str, Any]) -> tuple[str, str]:
    roles = {r["role"]: r["count"] for r in state.get("roles") or []}
    board = (
        "board" if roles.keys() & {"Bestyrelsesmedlem", "Formand", "Næstformand"} else "no-board"
    )
    n_dir = roles.get("Direktør", 0) + roles.get("Administrerende direktør", 0)
    return board, "dir0" if n_dir == 0 else "dir1" if n_dir == 1 else "dir2+"


def sample_dk(n: int, seed: int = 13, tail_share: float = 2 / 3) -> list[dict]:
    """Danish zero-shot gold items (no model ever trained on Danish text): one per text group
    (masked text + roles); about two thirds from the long tail (texts used by < 20 companies),
    the rest from standard wording; round-robin over board present x number of directors."""
    pools: dict[str, dict[tuple[str, str], list[dict]]] = {
        "dk-standard": defaultdict(list),
        "dk-tail": defaultdict(list),
    }
    seen: set[str] = set()
    for r in load_requests("dk", "pool", "test"):
        m = r["_meta"]
        if m["group_id"] in seen:
            continue
        seen.add(m["group_id"])
        board, dirs = _dk_bucket(r["state"])
        pools[m["stratum"]][(board, dirs)].append(
            {
                "item_id": m["id"],
                "set": "dk-text",
                "part": "pool/test",
                "state": r["state"],
                "strata": {
                    "wording": m["stratum"],
                    "board": board,
                    "directors": dirs,
                    "text_companies": m.get("text_companies"),
                },
            }
        )
    rng = random.Random(seed)
    n_tail = round(n * tail_share)
    tail = _round_robin(pools["dk-tail"], n_tail, rng)
    return tail + _round_robin(pools["dk-standard"], n - len(tail), rng)


VORSTAND_ROLES = frozenset(
    {
        "Vorstandsmitglied",
        "Vorsitzender des Vorstands",
        "Stellvertreter des Vorsitzenden",
        "stellvertretendes Vorstandsmitglied",
        "Obmann",
        "Obmann-Stellvertreter",
    }
)
FOCUS_QUOTAS = {"vorstand": 110, "partner": 50, "gmbh_multi": 40}


def _at_focus(state: dict[str, Any]) -> str:
    roles = {r["role"]: r["count"] for r in state.get("roles") or []}
    if roles.keys() & VORSTAND_ROLES:
        return "vorstand"
    if "unbeschränkt haftender Gesellschafter" in roles:
        return "partner"
    if roles.get("Geschäftsführer", 0) >= 2:
        return "gmbh_multi"
    return "other"


def sample_at_focus(
    n: int,
    seed: int = 13,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
    quotas: dict[str, int] | None = None,
    set_name: str = "at-text-2",
) -> list[dict]:
    """A second Austrian free-text batch focused on thin structures (plan-16): one item per text
    pattern from the parts no model trained on, never a pattern of an excluded item; quotas per
    structure (Vorstand, partners, GmbH with >= 2 GF); a shortfall goes round-robin to the rest."""
    quotas = quotas or FOCUS_QUOTAS
    reqs = []
    for split, part in (("pilot", "test"), ("random", "test")):
        try:
            reqs += load_requests("at", split, part)
        except FileNotFoundError:
            continue
    seen = {r["_meta"]["group_id"] for r in reqs if r["_meta"]["id"] in exclude_ids}
    pools: dict[str, list[dict]] = {k: [] for k in quotas}
    for r in reqs:
        m = r["_meta"]
        if m.get("stratum") != "at-text" or m["group_id"] in seen:
            continue
        seen.add(m["group_id"])
        focus = _at_focus(r["state"])
        if focus not in pools:
            continue
        st = r["state"]
        pools[focus].append(
            {
                "item_id": m["id"],
                "set": set_name,
                "part": "pool",
                "state": st,
                "strata": {
                    "focus": focus,
                    "form": st.get("legal_form"),
                    "prok": "prok" if st.get("procuration_rule") else "no-prok",
                },
            }
        )
    rng = random.Random(seed)
    for k in sorted(pools):
        rng.shuffle(pools[k])
    out: list[dict] = []
    for k in sorted(quotas):
        out += pools[k][: quotas[k]]
        pools[k] = pools[k][quotas[k] :]
    while len(out) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out[:n]


# ---- annotations -> CIR -> answers ----------------------------------------------------------


def annotation_to_rules(a: dict[str, Any]) -> tuple[SigningRule, ProcurationRule]:
    groups = []
    for g in a.get("alternatives") or []:
        # Vocabulary harmonisation (2026-10-06): the guideline said EXECUTIVE_MEMBER for a
        # Vorstandsmitglied, the training data (roles.yaml AT, phrase table) says BOARD_MEMBER.
        roles = [
            ("BOARD_MEMBER" if r == "EXECUTIVE_MEMBER" else str(r), int(c))
            for r, c in g.get("roles") or []
            if r in ROLES and int(c) >= 1
        ]
        coll = g.get("collective") or None
        if coll:
            groups.append(Group(collective=coll))
        elif roles:
            groups.append(Group(frozenset(roles)))
    person = bool(a.get("person_specific")) or any(
        r == "SIGNATORY" for g in groups for r, _ in g.roles
    )
    s = SigningRule(
        a["status"],
        frozenset(groups) if a["status"] == "rule" else frozenset(),
        person,
        Group(collective="ALL_BOARD"),
    )
    proc = a.get("procuration") or {}
    present = {"yes": True, "no": False}.get(proc.get("present", ""))
    mode = proc.get("mode") if proc.get("mode") in ("sole", "joint", "mixed") else None
    return s, ProcurationRule(present, mode if present else None)


def annotation_answers(a: dict[str, Any]) -> dict[str, Any]:
    s, p = annotation_to_rules(a)
    ans: dict[str, Any] = dict(derive_answers(s, p))
    if a.get("person_specific") and a.get("status") == "rule":
        # §6: the flag records one person's sole power (Austria: "vertritt selbständig"), so one
        # person can bind the company, whatever the role-based groups require.
        ans["min_signers"] = "1"
    if a.get("ambiguity") is not None:
        ans["ambiguity"] = int(a["ambiguity"])
    return ans


def load_annotations(set_name: str, annotator: str, root: Path | None = None) -> dict[str, dict]:
    """Latest non-skipped annotation per item."""
    path = (root or GOLD_DIR) / set_name / f"{annotator}.jsonl"
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["item_id"]] = rec
    return {k: v for k, v in latest.items() if not v.get("skipped")}


def agreement(a: dict[str, dict], b: dict[str, dict]) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    common = sorted(set(a) & set(b))
    out: dict[str, Any] = {"n_items": len(common), "per_question": {}}
    for q in ANSWER_KEYS:
        pairs = []
        for i in common:
            x, y = annotation_answers(a[i]).get(q), annotation_answers(b[i]).get(q)
            if x is not None and y is not None:
                pairs.append((str(x), str(y)))
        if not pairs:
            continue
        xs, ys = zip(*pairs, strict=True)
        agree = sum(x == y for x, y in pairs) / len(pairs)
        if len(set(xs) | set(ys)) < 2:
            kappa = 1.0 if agree == 1.0 else 0.0
        else:
            kappa = float(
                cohen_kappa_score(xs, ys, weights="quadratic" if q == "ambiguity" else None)
            )
        out["per_question"][q] = {"n": len(pairs), "agreement": agree, "kappa": kappa}
    return out


# ---- CLI ------------------------------------------------------------------------------------


def cmd_sample_no_rt(args: argparse.Namespace) -> int:
    require_vault(GOLD_DIR)
    items = sample_no_rt(args.n, args.seed)
    d = GOLD_DIR / "no-rt"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "batch.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    strata = Counter(f"{i['strata']['match']}/{i['strata']['length']}" for i in items)
    print(f"no-rt batch: {len(items)} items -> data/gold/no-rt/batch.jsonl; strata {dict(strata)}")
    return 0


def _batch_ids(set_name: str) -> set[str]:
    path = GOLD_DIR / set_name / "batch.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return {json.loads(x)["item_id"] for x in lines if x.strip()}


def cmd_sample_at(args: argparse.Namespace) -> int:
    require_vault(GOLD_DIR)
    d = GOLD_DIR / args.set
    d.mkdir(parents=True, exist_ok=True)
    batch = d / "batch.jsonl"
    if batch.exists() and not args.force:
        raise SystemExit(
            f"{args.set} batch exists (annotations refer to it); use --force to replace"
        )
    if args.focus:
        exclude = set().union(*(_batch_ids(x) for x in filter(None, args.exclude.split(","))))
        items = sample_at_focus(args.n, args.seed, exclude, set_name=args.set)
    else:
        items = sample_at(args.n, args.seed)
    with batch.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    strata = Counter("/".join(i["strata"].values()) for i in items)
    parts = Counter(i["part"] for i in items)
    print(f"{args.set} batch: {len(items)} items; parts {dict(parts)}; strata {dict(strata)}")
    return 0


def cmd_sample_dk(args: argparse.Namespace) -> int:
    require_vault(GOLD_DIR)
    d = GOLD_DIR / "dk-text"
    d.mkdir(parents=True, exist_ok=True)
    batch = d / "batch.jsonl"
    if batch.exists() and not args.force:
        raise SystemExit("dk-text batch exists (annotations refer to it); use --force to replace")
    items = sample_dk(args.n, args.seed)
    with batch.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    strata = Counter(
        f"{i['strata']['wording']}/{i['strata']['board']}/{i['strata']['directors']}" for i in items
    )
    print(f"dk-text batch: {len(items)} items; strata {dict(sorted(strata.items()))}")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    res = agreement(load_annotations(args.set, args.a), load_annotations(args.set, args.b))
    out = REPO_ROOT / "results" / "gold" / args.set
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(res, indent=1))
    print(f"{args.set}: {res['n_items']} items annotated by both")
    for q, v in res["per_question"].items():
        print(f"  {q:28s} n={v['n']:4d} agreement={v['agreement']:.3f} kappa={v['kappa']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gold", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample-no-rt")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_sample_no_rt)
    p = sub.add_parser("sample-at")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--set", default="at-text")
    p.add_argument("--focus", action="store_true", help="structure quotas (plan-16)")
    p.add_argument("--exclude", default="", help="sets whose patterns are never sampled again")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sample_at)
    p = sub.add_parser("sample-dk")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sample_dk)
    p = sub.add_parser("agreement")
    p.add_argument("--set", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(func=cmd_agreement)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
