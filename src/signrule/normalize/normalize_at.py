"""Austria: cached HVD extracts -> masked evaluation requests, register-coded labels (plan-05 T3).

    python -m signrule.normalize.normalize_at vocab   # capitalised words shared by >= 20 companies
    python -m signrule.normalize.normalize_at build --split pilot   # fixed pilot set (+ manifest)
    python -m signrule.normalize.normalize_at build --split xjur    # everything except the pilot

State (same keys as Norway, original language, no translation): one line per current function
holder, "<role> [PERSON_n]: <register TEXT> <TXTVERTR…>", representing organs in `signature_rule`,
procurators in `procuration_rule`. Natural persons become [PERSON_n], companies acting as partners
[FIRMA_n], numbered by first appearance across both fields, so references inside free texts keep
pointing at the right line.

Labels come only from the court's per-function codes (`VART` E = Einzelvertretung, G = gemeinsame
Vertretung; `VSBEIDE` V = chair of the Vorstand) and from which functions exist. A question is
labelled only when the codes determine it; everything else is masked (left to the gold set):

- ceo_alone      all current GF are E without TXTVERTR -> true; all are G -> false
- chair_alone    the Vorstand chair (VM/V) is E without TXTVERTR -> true; G -> false
- min_signers    some representing person is E without TXTVERTR -> "1"
- prokura_present  any current PR -> true; none -> false
- prokura_joint  all PR are G -> true; all are E -> false
Companies with a liquidator (AB) or insolvency administrator (MW) get no structural labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signrule.common.paths import INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR, require_vault
from signrule.format.request import QuestionsConfig, build_state, load_questions, to_request
from signrule.ingest.ingest_at import RAW, AtFunction, AtRecord, extract_at, template_key
from signrule.ingest.pii import redact_text
from signrule.normalize.mask import residual_name_risk
from signrule.ontology.at_phrases import phrase_cir
from signrule.ontology.cir import Group, SigningRule, derive_answers
from signrule.ontology.mapping import RolesConfig, load_roles

REPRESENTING = frozenset({"GF", "VM", "KP", "GL"})
PROCURATION = frozenset({"PR"})
ABNORMAL = frozenset(
    {"AB", "MW", "SV"}
)  # liquidator, insolvency administrator, branch representative
LEGAL_SHORT = {
    "GES": "GmbH",
    "AG": "AG",
    "KG": "KG",
    "OG": "OG",
    "FKG": "FlexKapG",
    "GEN": "Genossenschaft",
    "PST": "Privatstiftung",
    "SE": "SE",
}
VOCAB_MIN = 20
_CAP_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")


@dataclass(frozen=True)
class AtRequest:
    request: dict[str, Any]
    residual_risk: bool
    stratum: str  # "at-coded" (all representing functions E/G without free text) or "at-text"


def _role_code(f: AtFunction) -> str:
    if f.fken == "VM" and f.vsbeide_code in ("V", "O", "B", "R", "S"):
        return f"VM:{f.vsbeide_code}"
    return f.fken or "?"


_KEEP_HYPHEN_BEFORE = ("und ", "oder ", "bzw")


def join_fragments(frags: list[str]) -> str:
    """Join line-wrapped register fragments: a space between fragments, except after a line-break
    hyphen ("Geschäfts-" + "führer" -> "Geschäftsführer"; "zeichnungs- und" stays as written)."""
    out = ""
    for frag in (x.strip() for x in frags):
        if not frag:
            continue
        if out.endswith("-") and frag[:1].islower() and not frag.startswith(_KEEP_HYPHEN_BEFORE):
            out = out[:-1] + frag
        else:
            out = f"{out} {frag}" if out else frag
    return out


def _line_text(f: AtFunction) -> str:
    return redact_text(_line_text_raw(f))[0]


def _line_text_raw(f: AtFunction) -> str:
    """The register wording of one function entry. TEXT already carries the TXTVERTR fragments
    (as wrapped lines after "vertritt seit … gemeinsam mit"), sometimes with better spacing than
    TXTVERTR ("einer/einemProkuristin"); TXTVERTR is appended only when its wording, ignoring
    whitespace, is missing from TEXT."""
    text = join_fragments(f.text)
    squeezed = re.sub(r"[\s-]+", "", "".join(f.text))
    extra = [t for t in f.txtvertr if t.strip() and re.sub(r"[\s-]+", "", t) not in squeezed]
    return join_fragments([text, *extra]) if extra else (text or f.vart_text or "")


def render(
    rec: AtRecord, roles: RolesConfig
) -> tuple[str | None, str | None, list[tuple[str, int]], bool]:
    """(signature_rule, procuration_rule, role counts, residual name risk), names masked."""
    order = sorted(
        rec.functions,
        key=lambda f: (f.fken in PROCURATION, roles.order_key("AT", _role_code(f))),
    )
    tokens: dict[str, str] = {}
    n_person = n_firma = 0
    for f in order:
        p = rec.persons.get(f.pnr or "")
        if f.pnr in tokens:
            continue
        if p is not None and p.legal:
            n_firma += 1
            tokens[f.pnr or ""] = f"[FIRMA_{n_firma}]"
        else:
            n_person += 1
            tokens[f.pnr or ""] = f"[PERSON_{n_person}]"
    # Every spelling of every name on the extract, longest first, mapped to its holder's token
    # (persons without a current function get their own token, so they are masked as well).
    for pnr, p in rec.persons.items():
        if pnr not in tokens:
            if p.legal:
                n_firma += 1
                tokens[pnr] = f"[FIRMA_{n_firma}]"
            else:
                n_person += 1
                tokens[pnr] = f"[PERSON_{n_person}]"
    spellings = sorted(
        ((name, tokens[pnr]) for pnr, p in rec.persons.items() for name in p.names),
        key=lambda x: -len(x[0]),
    )

    def mask(text: str) -> str:
        for name, tok in spellings:
            text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", tok, text, flags=re.I)
        # the register refers to people by their letter ("gemeinsam mit Person A")
        return re.sub(
            r"\bPerson ([A-Z]{1,2})\b",
            lambda m: tokens.get(m.group(1), m.group(0)),
            text,
        )

    sig, prok = [], []
    counts: Counter[str] = Counter()
    for f in order:
        info = roles.info("AT", _role_code(f), f.fkentext)
        counts[info.label] += 1
        line = f"{info.label} {tokens.get(f.pnr or '', '[PERSON_?]')}: {mask(_line_text(f))}"
        (prok if f.fken in PROCURATION else sig).append(line)
    signature = "; ".join(sig) or None
    procuration = "; ".join(prok) or None
    risk = residual_name_risk(signature or "", "AT") or residual_name_risk(procuration or "", "AT")
    label_order = {roles.info("AT", c).label: i for i, c in enumerate(roles.order.get("AT", []))}
    role_list = sorted(counts.items(), key=lambda kv: (label_order.get(kv[0], 99), kv[0]))
    return signature, procuration, role_list, risk


def _sole(fs: list[AtFunction]) -> bool | None:
    """True if all are E without free text, False if all are G, None otherwise."""
    if not fs:
        return None
    if all(f.vart_code == "E" and not f.txtvertr for f in fs):
        return True
    if all(f.vart_code == "G" for f in fs):
        return False
    return None


def coded_answers(rec: AtRecord) -> dict[str, Any]:
    fken = {f.fken for f in rec.functions}
    rep = [f for f in rec.functions if f.fken in REPRESENTING]
    if not rep:
        return {}
    pr = [f for f in rec.functions if f.fken in PROCURATION]
    out: dict[str, Any] = {"prokura_present": bool(pr)}
    if pr and all(f.vart_code == "G" for f in pr):
        out["prokura_joint"] = True
    elif pr and all(f.vart_code == "E" for f in pr):
        out["prokura_joint"] = False
    if fken & ABNORMAL:
        return out
    gf = [f for f in rep if f.fken == "GF"]
    if (v := _sole(gf)) is not None:
        out["ceo_alone"] = v
    chair = [f for f in rep if f.fken == "VM" and f.vsbeide_code in ("V", "O")]
    if len(chair) == 1 and (v := _sole(chair)) is not None:
        out["chair_alone"] = v
    if any(f.vart_code == "E" and not f.txtvertr for f in rep):
        out["min_signers"] = "1"
    return out


def coded_cir(rec: AtRecord, roles: RolesConfig) -> SigningRule | None:
    """Signing rule from the court codes alone (plan-12 T2), for records whose representing
    functions carry no free text. An office whose holders are all E ("selbständig") is a
    single-person group; an office with any G holder (joint, partner not coded) or with mixed codes
    makes the rule person-specific, so nothing the codes cannot decide is answered "no".
    Holders coded X ("nicht vertretungsbefugt") do not represent. None if not decidable."""
    rep = [f for f in rec.functions if f.fken in REPRESENTING]
    if not rep or {f.fken for f in rec.functions} & ABNORMAL or any(f.txtvertr for f in rep):
        return None
    holders: dict[str, list[str | None]] = {}
    for f in rep:
        if f.vart_code == "X":
            continue
        if f.vart_code not in ("E", "G"):
            return None
        cir_role = roles.info("AT", _role_code(f)).cir
        if cir_role == "OTHER":
            return None
        holders.setdefault(cir_role, []).append(f.vart_code)
    if not holders:
        return None
    groups = [Group.of((r, 1)) for r, codes in holders.items() if set(codes) == {"E"}]
    person_specific = any(set(codes) != {"E"} for codes in holders.values())
    return SigningRule("rule", frozenset(groups), person_specific, None)


MEMBER_ROLES = frozenset(
    {
        "Vorstandsmitglied",
        "Vorsitzender des Vorstands",
        "Obmann",
        "Obmann-Stellvertreter",
        "Stellvertreter des Vorsitzenden",
        "stellvertretendes Vorstandsmitglied",
        "unbeschränkt haftender Gesellschafter",
    }
)
BOARD_ROLES = MEMBER_ROLES - {"unbeschränkt haftender Gesellschafter"}  # a Vorstand, not partners
CHAIR_ROLES = frozenset({"Vorsitzender des Vorstands", "Obmann"})
DEPUTY_ROLES = frozenset({"Stellvertreter des Vorsitzenden", "Obmann-Stellvertreter"})
CEO_ROLES = frozenset({"Geschäftsführer", "Geschäftsleiter"})
NEEDS_MEMBER = ("member_alone", "chair_with_member", "three_members")
NEEDS_BOARD = (
    "two_board_members_jointly",
    "ceo_with_one_board_member",
    "ceo_with_two_members",
)  # worded for a board
NEEDS_DEPUTY = ("chair_with_deputy", "deputy_with_member")
NEEDS_CEO = ("ceo_alone", "two_ceos", "ceo_with_chair", "ceo_with_prokurist")
NEEDS_CHAIR = ("chair_alone", "chair_with_member", "ceo_with_chair")


def applicable_at(state: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """Drop office questions about offices the company does not have (a GmbH has no
    representing board; a KG has no Geschäftsführer). Used for training and evaluation alike."""
    present = {r["role"] for r in state.get("roles") or []}
    drop: set[str] = set()
    if not present & MEMBER_ROLES:
        drop |= set(NEEDS_MEMBER)
    if not present & BOARD_ROLES:
        drop |= set(NEEDS_BOARD)
    if not present & CEO_ROLES:
        drop |= {*NEEDS_CEO, "ceo_with_one_board_member"}
    if not present & CHAIR_ROLES:
        drop |= set(NEEDS_CHAIR)
    if not present & DEPUTY_ROLES:
        drop |= set(NEEDS_DEPUTY)
    if not present & CEO_ROLES:
        drop.add("ceo_with_two_members")
    return {q: v for q, v in answers.items() if q not in drop}


def text_key(signature: str | None, procuration: str | None, roles: list[tuple[str, int]]) -> str:
    blob = json.dumps(
        [template_key(signature or ""), template_key(procuration or ""), roles], ensure_ascii=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def to_at_request(
    rec: AtRecord, roles: RolesConfig, questions: QuestionsConfig
) -> AtRequest | None:
    if rec.rechtsform_code == "EU" or not rec.functions:
        return None
    sig, prok, role_list, risk = render(rec, roles)
    legal = LEGAL_SHORT.get(rec.rechtsform_code or "", rec.rechtsform_text)
    state = build_state("AT", legal, sig, prok, role_list)
    answers = coded_answers(rec)
    cir, source = None, "codes"
    rep_functions = [f for f in rec.functions if f.fken in REPRESENTING]
    if rep_functions and not {f.fken for f in rec.functions} & ABNORMAL:
        lines = [
            (roles.info("AT", _role_code(f)).cir, _line_text(f), f.vart_code) for f in rep_functions
        ]
        cir = phrase_cir(lines)
        if cir is not None and any(f.vart_code == "G" for f in rep_functions):
            source = "codes+phrase_table"
    from_codes_only = cir is None
    if cir is None:
        cir = coded_cir(rec, roles)
    if cir is not None:
        derived = {
            q: v
            for q, v in derive_answers(cir, None).items()
            if q not in ("prokura_present", "prokura_joint", "ambiguity")
        }
        if from_codes_only and cir.person_specific:
            derived.pop("parseable", None)  # a joint partner the codes do not name
        for q, v in derived.items():
            answers.setdefault(q, v)  # direct code answers (e.g. all-G -> not alone) win
    answers = applicable_at(state, answers)
    rep = [f for f in rec.functions if f.fken in REPRESENTING | PROCURATION]
    coded = bool(rep) and all(f.vart_code in ("E", "G") and not f.txtvertr for f in rep)
    stratum = "at-coded" if coded else "at-text"
    meta = {
        "id": rec.fnr,
        "stratum": stratum,
        "legal_form_code": rec.rechtsform_code,
        "text_key": text_key(sig, prok, role_list),
        "group_id": text_key(sig, prok, role_list),
        "has_text": True,
        "has_txtvertr": any(f.txtvertr for f in rec.functions),
        "label_source": source if cir is not None else "codes",
    }
    request = to_request(state, answers, questions, source="at", meta=meta)
    return AtRequest(request=request, residual_risk=risk, stratum=stratum)


def iter_records() -> Any:
    for p in sorted((RAW / "auszug").glob("*.json")):
        env = json.loads(p.read_text(encoding="utf-8"))
        if env.get("body"):
            yield extract_at(env["body"])


def cmd_vocab(args: argparse.Namespace) -> int:
    """Capitalised words in register texts, by number of companies, never a person's name token.
    Output is for hand review into configs/ontology/legal_vocab.yaml ("AT")."""
    require_vault()
    per_word: Counter[str] = Counter()
    for rec in iter_records():
        name_tokens = {
            t.casefold()
            for p in rec.persons.values()
            for n in p.names
            for t in re.split(r"[\s\-,.]+", n)
            if t
        }
        words = set()
        for f in rec.functions:
            for w in _CAP_WORD.findall(_line_text(f)):
                if w[0].isupper() and not w.isupper() and w.casefold() not in name_tokens:
                    words.add(w.casefold())
        per_word.update(words)
    shared = [(w, n) for w, n in per_word.most_common() if n >= args.min]
    print(f"{len(per_word)} distinct capitalised words; {len(shared)} in >= {args.min} companies:")
    print(", ".join(f"{w} ({n})" for w, n in shared))
    return 0


PILOT_MANIFEST = SPLITS_DIR / "at_pilot.json"


def cmd_build(args: argparse.Namespace) -> int:
    """Write processed/at/<split>/test.jsonl (requests with at least one coded label).

    `pilot`: the first build freezes the ids of the current cache as the pilot (split manifest);
    later builds rebuild exactly those ids. `xjur`: everything except the pilot ids, so numbers
    looked at during development never enter the reported Austrian test set."""
    require_vault()
    roles, questions = load_roles(), load_questions()
    frozen = set(json.loads(PILOT_MANIFEST.read_text())["ids"]) if PILOT_MANIFEST.exists() else None
    if args.split == "xjur" and frozen is None:
        raise SystemExit("no pilot manifest; build --split pilot first")
    out_dir = PROCESSED_DIR / "at" / args.split
    risk_dir = INTERIM_DIR / "at"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter[str]] = {k: Counter() for k in ("records", "stratum", "labels")}
    with (
        open(out_dir / "test.jsonl.part", "w", encoding="utf-8") as ok_f,
        open(risk_dir / "residual_risk.jsonl.part", "w", encoding="utf-8") as risk_f,
    ):
        written_ids: list[str] = []
        for rec in iter_records():
            if args.split == "pilot" and frozen is not None and rec.fnr not in frozen:
                continue
            if args.split == "xjur" and frozen is not None and rec.fnr in frozen:
                stats["records"]["pilot (excluded)"] += 1
                continue
            r = to_at_request(rec, roles, questions)
            if r is None:
                stats["records"]["skipped (sole trader / no current function)"] += 1
                continue
            if r.residual_risk:
                stats["records"]["residual name risk -> interim"] += 1
                risk_f.write(json.dumps(r.request, ensure_ascii=False) + "\n")
                continue
            written_ids.append(rec.fnr)
            if not r.request["questions"]:
                stats["records"]["no coded label (gold pool only)"] += 1
                continue
            stats["records"]["written"] += 1
            stats["stratum"][r.stratum] += 1
            for qid, q in r.request["questions"].items():
                stats["labels"][f"{qid}={q['label']}"] += 1
            ok_f.write(json.dumps(r.request, ensure_ascii=False) + "\n")
    Path(ok_f.name).replace(out_dir / "test.jsonl")
    Path(risk_f.name).replace(risk_dir / "residual_risk.jsonl")
    summary = {k: dict(sorted(v.items())) for k, v in stats.items()}
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.split == "pilot" and frozen is None:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        PILOT_MANIFEST.write_text(
            json.dumps({"jurisdiction": "at", "split": "pilot", "ids": sorted(written_ids)})
        )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="normalize_at",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocab")
    p.add_argument("--min", type=int, default=VOCAB_MIN)
    p.set_defaults(func=cmd_vocab)
    p = sub.add_parser("build")
    p.add_argument("--split", choices=["pilot", "xjur"], required=True)
    p.set_defaults(func=cmd_build)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
