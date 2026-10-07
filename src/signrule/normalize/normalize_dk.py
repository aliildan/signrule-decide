"""Denmark normaliser (plan-14): CVR ES projection -> masked evaluation requests (no labels).

    python -m signrule.normalize.normalize_dk vocab   # capitalised words in >= 20 companies' texts
    python -m signrule.normalize.normalize_dk build   # processed/dk/pool/test.jsonl + stats.json

Denmark has no official labels for us, so the pool carries no questions; the human gold set
(`gold sample-dk`, `gold_eval build --jurisdiction dk`) adds them. Nothing here is ever trained on.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signrule.common.paths import INTERIM_DIR, PROCESSED_DIR, require_vault
from signrule.format.request import QuestionsConfig, build_state, load_questions, to_request
from signrule.ingest.ingest_dk import latest_projection
from signrule.ingest.pii import redact_text
from signrule.normalize.mask import mask_text, residual_name_risk
from signrule.ontology.mapping import RolesConfig, load_legal_vocab, load_roles

# Forms whose short name we are sure of; others keep the register's code (evaluation input only).
LEGAL_SHORT = {
    "APS": "ApS",
    "A/S": "A/S",
    "K/S": "K/S",
    "I/S": "I/S",
    "IVS": "IVS",
    "KAS": "P/S",
    "ABA": "A.M.B.A.",
    "FMA": "F.M.B.A.",
    "SMA": "S.M.B.A.",
}
VOCAB_MIN = 20
_CAP_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

CEO_ROLES = frozenset({"Direktør", "Administrerende direktør"})
CHAIR_ROLES = frozenset({"Formand"})
DEPUTY_ROLES = frozenset({"Næstformand"})
BOARD_ROLES = frozenset({"Bestyrelsesmedlem", "Formand", "Næstformand"})
MEMBER_ROLES = BOARD_ROLES | {"Interessent", "Komplementar"}
NEEDS_MEMBER = ("member_alone", "chair_with_member", "three_members")
NEEDS_BOARD = ("two_board_members_jointly", "ceo_with_one_board_member", "ceo_with_two_members")
NEEDS_DEPUTY = ("chair_with_deputy", "deputy_with_member")
NEEDS_CEO = (
    "ceo_alone",
    "two_ceos",
    "ceo_with_chair",
    "ceo_with_prokurist",
    "ceo_with_one_board_member",
    "ceo_with_two_members",
)
NEEDS_CHAIR = ("chair_alone", "chair_with_member", "ceo_with_chair")


@dataclass(frozen=True)
class DkRequest:
    request: dict[str, Any]
    residual_risk: bool


def dk_code(fn: dict[str, Any], roles: RolesConfig) -> str:
    """roles.yaml "DK" key of a function; other Direktion titles count as Direktør."""
    code = f"{fn.get('body')}:{fn.get('function')}"
    other_title = fn.get("body") == "Direktion" and str(fn.get("function")).upper() != "SUPPLEANT"
    if code not in roles.codes.get("DK", {}) and other_title:
        return "Direktion:DIREKTØR"
    return code


def _holder(fn: dict[str, Any], row: dict[str, Any]) -> tuple[str, str | None]:
    """(holder key, name). The first projection filed ANDEN_DELTAGER names under firm_names, so
    the name list is chosen by the index key, the token type by enhedstype."""
    if "person" in fn:
        return f"p{fn['person']}", row["mask_names"][fn["person"]]
    if "firm" in fn:
        return f"f{fn['firm']}", row["firm_names"][fn["firm"]]
    return f"?{id(fn)}", None


def current_text(row: dict[str, Any]) -> str | None:
    cur = [t for t in row.get("rule_texts") or [] if not t.get("to") and t.get("text")]
    if not cur:
        return None
    return max(cur, key=lambda t: t.get("from") or "")["text"]


def render_dk(
    row: dict[str, Any], roles: RolesConfig
) -> tuple[str | None, list[tuple[str, int]], bool]:
    """(masked signature rule, role counts per distinct holder, residual name risk)."""
    persons: list[str] = []
    firms: list[str] = []
    holders: dict[str, set[str]] = {}
    for fn in row.get("functions") or []:
        key, name = _holder(fn, row)
        if name:
            target = firms if fn.get("enhedstype") == "VIRKSOMHED" else persons
            if name not in target:
                target.append(name)
        label = roles.info("DK", dk_code(fn, roles), str(fn.get("function"))).label
        holders.setdefault(label, set()).add(key)
    text = current_text(row)
    if text is not None:
        text, _ = redact_text(" ".join(text.split()))
        for i, firm in enumerate(sorted(firms, key=len, reverse=True)):
            text = re.sub(rf"(?<!\w){re.escape(firm)}(?!\w)", f"\x01{i}\x01", text, flags=re.I)
        text = mask_text(text, persons, "DK")
        numbering: dict[str, int] = {}
        text = re.sub(
            "\x01(\\d+)\x01",
            lambda m: f"[FIRMA_{numbering.setdefault(m.group(1), len(numbering) + 1)}]",
            text,
        )
    order = {roles.info("DK", c).label: i for i, c in enumerate(roles.order.get("DK", []))}
    role_list = sorted(
        ((label, len(keys)) for label, keys in holders.items()),
        key=lambda kv: (order.get(kv[0], 99), kv[0]),
    )
    risk = residual_name_risk(text or "", "DK")
    return text, role_list, risk


def applicable_dk(state: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """Drop questions about offices the company does not have (as normalize_at.applicable_at)."""
    present = {r["role"] for r in state.get("roles") or []}
    drop: set[str] = set()
    if not present & MEMBER_ROLES:
        drop |= set(NEEDS_MEMBER)
    if not present & BOARD_ROLES:
        drop |= set(NEEDS_BOARD)
    if not present & CEO_ROLES:
        drop |= set(NEEDS_CEO)
    if not present & CHAIR_ROLES:
        drop |= set(NEEDS_CHAIR)
    if not present & DEPUTY_ROLES:
        drop |= set(NEEDS_DEPUTY)
    return {q: v for q, v in answers.items() if q not in drop}


def text_key(signature: str, roles: list[tuple[str, int]]) -> str:
    blob = json.dumps([signature, roles], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def to_dk_request(
    row: dict[str, Any], roles: RolesConfig, questions: QuestionsConfig
) -> DkRequest | None:
    sig, role_list, risk = render_dk(row, roles)
    if not sig:
        return None
    legal = LEGAL_SHORT.get(row.get("legal_form") or "", row.get("legal_form"))
    state = build_state("DK", legal, sig, None, role_list)
    meta = {
        "id": row["cvr"],
        "group_id": text_key(sig, role_list),
        "legal_form_code": row.get("legal_form"),
        "has_text": True,
        "label_source": "none (evaluation pool)",
    }
    return DkRequest(to_request(state, {}, questions, source="dk", meta=meta), risk)


def iter_rows() -> Any:
    with gzip.open(latest_projection(), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def cmd_vocab(args: argparse.Namespace) -> int:
    """Capitalised non-initial words of the masked texts and how many companies use each (only
    words used by >= --min companies are printed: standard wording, reviewed by hand)."""
    require_vault()
    roles = load_roles()
    vocab = load_legal_vocab("DK")
    counts: Counter[str] = Counter()
    for row in iter_rows():
        sig, _, _ = render_dk(row, roles)
        words: set[str] = set()
        for sentence in re.split(r"[.!?]\s+", sig or ""):
            toks = _CAP_WORD.findall(sentence)
            words |= {w for w in toks[1:] if w[0].isupper() and not w.isupper()}
        counts.update(w for w in words if w.lower() not in vocab)
    for w, k in counts.most_common():
        if k >= args.min:
            print(f"{k:7d}  {w}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    require_vault()
    roles, questions = load_roles(), load_questions()
    out_dir = PROCESSED_DIR / "dk" / "pool"
    risk_dir = INTERIM_DIR / "dk"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter[str]] = {
        k: Counter() for k in ("records", "stratum", "legal_form", "roles")
    }
    kept: list[dict[str, Any]] = []
    texts: Counter[str] = Counter()
    with open(risk_dir / "residual_risk.jsonl.part", "w", encoding="utf-8") as risk_f:
        for row in iter_rows():
            r = to_dk_request(row, roles, questions)
            if r is None:
                stats["records"]["skipped (no current tegningsregel)"] += 1
                continue
            if r.residual_risk:
                stats["records"]["residual name risk -> interim"] += 1
                risk_f.write(json.dumps(r.request, ensure_ascii=False) + "\n")
                continue
            kept.append(r.request)
            texts[r.request["state"]["signature_rule"]] += 1
    Path(risk_f.name).replace(risk_dir / "residual_risk.jsonl")
    with open(out_dir / "test.jsonl.part", "w", encoding="utf-8") as f:
        for req in kept:
            n_same = texts[req["state"]["signature_rule"]]
            req["_meta"]["text_companies"] = n_same
            # standard wording (shared by >= VOCAB_MIN companies) vs. the long tail
            req["_meta"]["stratum"] = "dk-standard" if n_same >= VOCAB_MIN else "dk-tail"
            stats["stratum"][req["_meta"]["stratum"]] += 1
            stats["records"]["written"] += 1
            stats["legal_form"][req["state"].get("legal_form") or ""] += 1
            for role in req["state"].get("roles") or []:
                stats["roles"][role["role"]] += 1
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    Path(f.name).replace(out_dir / "test.jsonl")
    summary: dict[str, Any] = {k: dict(v.most_common()) for k, v in stats.items()}
    summary["unique_texts"] = len(texts)
    summary["groups"] = len({r["_meta"]["group_id"] for r in kept})
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="normalize_dk",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocab")
    p.add_argument("--min", type=int, default=VOCAB_MIN)
    p.set_defaults(func=cmd_vocab)
    sub.add_parser("build").set_defaults(func=cmd_build)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
