"""Norway: cached Fullmakt responses -> records with CIR -> masked, labelled requests (plan-03).

Records keep person names only in memory (for masking); nothing with names is written outside
the vault. Output requests contain masked texts, role labels and counts, never names.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from signrule.format.request import QuestionsConfig, build_state, load_questions, to_request
from signrule.ingest.pii import redact_text
from signrule.normalize.mask import mask_text, residual_name_risk
from signrule.normalize.text import normalize_text, text_key
from signrule.ontology.cir import Group, ProcurationRule, SigningRule, derive_answers
from signrule.ontology.mapping import NoRuleSpec, RolesConfig, load_no_rules, load_roles
from signrule.ontology.no_compose import compose

STATUS = {"RF": "rule", "RT": "uninterpretable", "RI": "none"}


@dataclass
class NoRecord:
    entity_id: str
    legal_form: str | None
    registered: date | None
    signature_text: str | None
    procuration_text: str | None
    role_counts: dict[str, int]  # source role code -> count of distinct persons
    role_labels: dict[str, str]  # source role code -> register label (fallback for unknown codes)
    person_names: list[str]  # masking only; never written
    signing: SigningRule
    procuration: ProcurationRule
    rule_codes: tuple[str, ...]  # R-codes behind `signing` (for ambiguity level 0)
    source_status: dict[str, str | None]
    unmapped_codes: list[str] = field(default_factory=list)


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _text(body: dict[str, Any]) -> str | None:
    t = _get(body, "signeringsGrunnlag", "signaturProkuraRoller", "signaturProkuraFritekst")
    if isinstance(t, str) and t.strip():
        return redact_text(normalize_text(t))[0]  # birth dates/years, IDs, e-mails in the text
    return None


def _person_lists(body: dict[str, Any]) -> list[list[dict[str, Any]]]:
    g = body.get("signeringsGrunnlag") or {}
    lists = [
        _get(g, "muligeSigneringsRoller", "personRolleGrunnlag") or [],
        _get(g, "signaturProkuraRoller", "personRolleGrunnlag") or [],
    ]
    for k in _get(body, "signeringsKombinasjon", "kombinasjon") or []:
        lists.append(k.get("personRolleKombinasjon") or [])
    return lists


def _parse_date(s: Any) -> date | None:
    try:
        return date.fromisoformat(str(s)) if s else None
    except ValueError:
        return None


def _signing(
    body: dict[str, Any], rules: dict[str, NoRuleSpec]
) -> tuple[SigningRule, tuple[str, ...], list[str]]:
    regel = _get(body, "status", "regelStatus", "kode")
    status = STATUS.get(regel or "", "none")
    if status == "none" and _text(body):
        # RI with a free text: the register's interpreter did not recognise the text (typically
        # a combination of rules). That is "not interpretable by the register", not "no rule".
        status = "uninterpretable"
    combos = _get(body, "signeringsKombinasjon", "kombinasjon") or []
    ident = _get(body, "status", "regelStatus", "regelIdent")
    alternatives: set[Group] = set()
    statutory: Group | None = None
    person_specific = False
    rule_codes: list[str] = []
    unmapped: list[str] = []
    for c in combos + [{"kode": ident}] if status == "rule" else combos:
        code = c.get("kode")
        if not code:
            continue
        spec = rules.get(code)
        if spec is None:
            unmapped.append(code)
            continue
        if spec.kind == "rule":
            if status == "rule":
                alternatives.update(spec.alternatives)
                if code not in rule_codes:
                    rule_codes.append(code)
        elif spec.kind == "statutory":
            statutory = statutory or spec.alternatives[0]
        elif spec.kind == "person":
            person_specific = True
            n = len(c.get("personRolleKombinasjon") or []) if spec.mode == "joint" else 1
            alternatives.add(Group.of(("SIGNATORY", max(n, 2) if spec.mode == "joint" else 1)))
    return (
        SigningRule(status, frozenset(alternatives), person_specific, statutory),  # type: ignore[arg-type]
        tuple(rule_codes),
        unmapped,
    )


def _procuration(
    body: dict[str, Any] | None, rules: dict[str, NoRuleSpec]
) -> tuple[ProcurationRule, list[str]]:
    if not body or _get(body, "status", "rutineStatus", "kode") != "OK":
        return ProcurationRule(None, None), []
    regel = _get(body, "status", "regelStatus", "kode")
    combos = _get(body, "signeringsKombinasjon", "kombinasjon") or []
    ident = _get(body, "status", "regelStatus", "regelIdent")
    modes: set[str] = set()
    unmapped: list[str] = []
    rule_groups: list[Group] = []
    seen_rule = False
    entries = [c.get("kode") for c in combos] + ([ident] if regel == "RF" and ident else [])
    for code in entries:
        if not code:
            continue
        spec = rules.get(code)
        if spec is None:
            unmapped.append(code)
        elif spec.kind == "procuration":
            modes.add(spec.mode or "sole")
        elif spec.kind == "rule" and regel == "RF":
            seen_rule = True
            rule_groups.extend(spec.alternatives)
    if seen_rule and rule_groups:
        modes.add("sole" if any(g.size == 1 for g in rule_groups) else "joint")
    present = regel in ("RF", "RT") or bool(modes) or _text(body) is not None
    mode = (modes.pop() if len(modes) == 1 else "mixed") if modes else None
    return ProcurationRule(present, mode if present else None), unmapped  # type: ignore[arg-type]


def build_record(
    sig: dict[str, Any],
    prok: dict[str, Any] | None,
    entity: dict[str, Any] | None,
    rules: dict[str, NoRuleSpec] | None = None,
) -> NoRecord | None:
    """None when the register could not run its routine (`rutineStatus != OK`)."""
    if _get(sig, "status", "rutineStatus", "kode") != "OK":
        return None
    rules = rules or load_no_rules()
    signing, rule_codes, unmapped = _signing(sig, rules)
    procuration, unmapped_p = _procuration(prok, rules)

    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    names: list[str] = []
    for body in (sig, prok or {}):
        for people in _person_lists(body):
            for p in people:
                name = unicodedata.normalize("NFC", " ".join(str(p.get("navn") or "").split()))
                code = _get(p, "rolle", "kode") or "?"
                labels.setdefault(code, _get(p, "rolle", "tekstforklaring") or code)
                if name and name not in names:
                    names.append(name)
                if (name, code) not in seen:
                    seen.add((name, code))
                    counts[code] += 1
    return NoRecord(
        entity_id=str(_get(sig, "enhet", "organisasjonsnummer")),
        legal_form=_get(sig, "enhet", "organisasjonsform", "kode"),
        registered=_parse_date((entity or {}).get("registered")),
        signature_text=_text(sig),
        procuration_text=_text(prok) if prok else None,
        role_counts=dict(counts),
        role_labels=labels,
        person_names=names,
        signing=signing,
        procuration=procuration,
        rule_codes=rule_codes,
        source_status={
            "regel": _get(sig, "status", "regelStatus", "kode"),
            "regel_ident": _get(sig, "status", "regelStatus", "regelIdent"),
            "regel_text": _get(sig, "status", "regelStatus", "tekstforklaring"),
            "prokura_regel": _get(prok, "status", "regelStatus", "kode") if prok else None,
        },
        unmapped_codes=unmapped + unmapped_p,
    )


@dataclass
class MaskedRequest:
    request: dict[str, Any]
    has_text: bool
    residual_risk: bool
    signature_key: str
    procuration_key: str


def ambiguity_level(rec: NoRecord, rules: dict[str, NoRuleSpec]) -> int | None:
    """Decision O3 (a): 0 = text equals the register's description of its single matched rule,
    1 = interpreted rule with other wording, None otherwise (masked)."""
    if rec.signing.status != "rule" or not rec.signature_text or not rec.rule_codes:
        return None
    if len(rec.rule_codes) == 1 and text_key(rec.signature_text) == text_key(
        rules[rec.rule_codes[0]].description
    ):
        return 0
    return 1


def to_masked_request(
    rec: NoRecord,
    *,
    roles: RolesConfig | None = None,
    questions: QuestionsConfig | None = None,
    rules: dict[str, NoRuleSpec] | None = None,
) -> MaskedRequest:
    roles = roles or load_roles()
    questions = questions or load_questions()
    rules = rules or load_no_rules()
    sig = mask_text(rec.signature_text, rec.person_names) if rec.signature_text else None
    prok = mask_text(rec.procuration_text, rec.person_names) if rec.procuration_text else None
    risk = residual_name_risk(sig or "", "NO") or residual_name_risk(prok or "", "NO")
    role_items = sorted(rec.role_counts.items(), key=lambda kv: roles.order_key("NO", kv[0]))
    role_list = [(roles.info("NO", c, rec.role_labels.get(c)).label, n) for c, n in role_items]
    state = build_state("NO", rec.legal_form, sig, prok, role_list)
    signing, label_source = rec.signing, "register"
    if rec.signing.status == "uninterpretable":
        composed = compose(rec.signature_text, rules)
        if composed is not None:  # plan-13: the text is the register's own rule descriptions
            signing, label_source = composed, "register+composition"
    answers: dict[str, Any] = derive_answers(
        signing, rec.procuration, roles.not_applicable("NO", rec.legal_form)
    )
    if answers.get("parseable") is False:
        # plan-12 T3: "the register's interpreter did not cover this wording" is not a property of
        # the text (humans read 96 % of these texts as definite rules); never train it as one.
        del answers["parseable"]
    amb = ambiguity_level(rec, rules)
    if amb is not None:
        answers["ambiguity"] = amb
    request = to_request(
        state, answers, questions, source="no", meta={"label_source": label_source}
    )
    return MaskedRequest(
        request=request,
        has_text=bool(sig or prok),
        residual_risk=risk,
        signature_key=text_key(sig),
        procuration_key=text_key(prok),
    )
