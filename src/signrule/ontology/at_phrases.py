"""Austrian representation lines -> CIR via the hand-reviewed phrase table (plan-12).

A line is one function holder: "vertritt seit <date> selbständig" (sole) or "… gemeinsam mit
<partner> [oder <partner> …]" (joint). With the court code, E must read "selbständig" and G must
read "gemeinsam"; a mismatch, an unknown partner phrase or a named partner ("Person A",
[PERSON_n]) makes the whole company undetermined (None) — nothing is guessed.

Office-level groups (design-04 "by virtue of the office"): a group counts for an office only if
every holder of that office can use it (a sole holder can use any group that contains them); if
the holders of an office differ, the rule is person-specific.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from pathlib import Path

import yaml

from signrule.common.paths import CONFIG_DIR
from signrule.ontology.cir import Group, SigningRule

PHRASES_PATH = CONFIG_DIR / "ontology" / "at_phrases.yaml"
SIGNING_OFFICES = frozenset(
    {"CEO", "CHAIR", "DEPUTY_CHAIR", "BOARD_MEMBER", "EXECUTIVE_MEMBER", "PARTNER"}
)
_SPLIT = re.compile(r"\s+oder\s+")
_NAMED = re.compile(r"\[(?:PERSON|FIRMA)_\d+\]|\bperson [a-z]{1,2}\b")


@cache
def load_phrases(path: Path = PHRASES_PATH) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for role, phrases in raw.items():
        for p in phrases:
            key = normalise(p)
            if key in out and out[key] != role:
                raise ValueError(f"at_phrases.yaml: {p!r} listed for two roles")
            out[key] = role
    return out


def normalise(phrase: str) -> str:
    p = re.sub(r"\s+", " ", phrase.lower()).strip().rstrip(".,;:")
    p = re.sub(r"^(?:gemeinsam\s+)?mit\s+", "", p)
    return p.strip()


def parse_line(text: str) -> str | list[str] | None:
    """'sole', 'none' (no power), a list of partner roles, or None if not in the table."""
    t = re.sub(r"\s+", " ", text.lower())
    if "nicht vertretungsbefugt" in t:
        return "none"
    if "gemeinsam" not in t:
        return "sole" if "selbständig" in t or "selbstständig" in t else None
    if "selbständig" in t or "selbstständig" in t:
        return None  # both words: a conditional rule, not a table case
    rest = t.split("gemeinsam", 1)[1]
    rest = re.sub(r"^\s*mit\s+", "", rest).strip().rstrip(".,;:")
    if not rest or _NAMED.search(rest):
        return None
    table = load_phrases()
    partners = []
    for part in _SPLIT.split(rest):
        role = table.get(normalise(part))
        if role is None:
            return None
        partners.append(role)
    return partners


def _contains(big: Counter[str], small: Counter[str]) -> bool:
    return all(big[r] >= n for r, n in small.items())


def phrase_cir(lines: list[tuple[str, str, str | None]]) -> SigningRule | None:
    """`lines` = (CIR office of the holder, line text, court code E/G/X or None)."""
    per_office: dict[str, list[list[Counter[str]]]] = {}
    for office, text, code in lines:
        if office not in SIGNING_OFFICES:
            return None
        parsed = parse_line(text)
        if parsed is None or code == "X" or parsed == "none":
            if parsed is None:
                return None
            continue
        if code is not None and (parsed == "sole") != (code == "E"):
            return None  # the wording contradicts the court code
        if parsed == "sole":
            groups = [Counter({office: 1})]
        else:
            groups = [Counter({office: 1}) + Counter({p: 1}) for p in parsed]
        per_office.setdefault(office, []).append(groups)
    if not per_office:
        return None
    alternatives: set[Group] = set()
    person_specific = False
    for holders in per_office.values():
        candidates = {tuple(sorted(g.items())) for h in holders for g in h}
        for cand in candidates:
            c = Counter(dict(cand))
            if all(any(_contains(c, g) for g in h) for h in holders):
                alternatives.add(Group(frozenset(c.items())))
        if len({tuple(sorted(tuple(sorted(g.items())) for g in h)) for h in holders}) > 1:
            person_specific = True
    # keep minimal groups only
    minimal = {
        g
        for g in alternatives
        if not any(
            o != g and o.collective is None and _contains(Counter(g.counts()), Counter(o.counts()))
            for o in alternatives
        )
    }
    if not minimal:
        return None
    return SigningRule("rule", frozenset(minimal), person_specific, None)


def state_lines(state: dict, label_to_office: dict[str, str]) -> list[tuple[str, str, None]] | None:
    """Rendered signature lines "<role> [PERSON_n]: <text>" -> (office, text, None)."""
    out = []
    for part in (state.get("signature_rule") or "").split("; "):
        head, sep, body = part.partition(": ")
        if not sep:
            return None
        label = re.sub(r"\s*\[(?:PERSON|FIRMA)_\d+\]$", "", head).strip()
        office = label_to_office.get(label)
        if office is None:
            return None
        out.append((office, body, None))
    return out or None
