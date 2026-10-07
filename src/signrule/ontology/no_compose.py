"""Norwegian signing texts composed of the register's own rule descriptions (plan-13).

The register's interpreter returns RT/RI for texts such as
"Daglig leder og styrets leder i fellesskap eller to styremedlemmer i fellesskap." although every
part is, word for word, one of its own 18 rule descriptions. Splitting at sentence ends and at
" eller " and matching every part exactly (normalised, `text_key`) against the descriptions gives
the rule as the union of those R-codes' alternatives. Anything that does not match completely is
not composed (owner decision 2026-10-06, CLAUDE.md §2.1; validated: 100 % agreement with the
human NO gold labels on the 117 composable items).
"""

from __future__ import annotations

import re

from signrule.normalize.text import text_key
from signrule.ontology.cir import Group, SigningRule
from signrule.ontology.mapping import NoRuleSpec

_SPLIT = re.compile(r"(?:[.;]\s+|,?\s+eller\s+)", re.I)


def description_index(rules: dict[str, NoRuleSpec]) -> dict[str, str]:
    return {text_key(s.description): code for code, s in rules.items() if s.kind == "rule"}


def compose(text: str | None, rules: dict[str, NoRuleSpec]) -> SigningRule | None:
    """The rule a text states if every part is an official rule description, else None."""
    if not text:
        return None
    index = description_index(rules)
    parts = [p.strip(" .;,") for p in _SPLIT.split(text) if p and p.strip(" .;,")]
    groups: list[Group] = []
    for part in parts:
        code = index.get(text_key(part))
        if code is None:
            return None
        groups.extend(rules[code].alternatives)
    if not groups:
        return None
    return SigningRule("rule", frozenset(groups), False, Group(collective="ALL_BOARD"))
