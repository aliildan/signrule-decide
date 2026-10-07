"""How far do Norwegian texts decompose into the register's own rule descriptions? (aggregate)

    uv run python eval/no_composition_check.py

Splits a signature text into parts at sentence ends and at " eller " and matches every part exactly
(normalised) against the 18 official R-code descriptions (no_rules.yaml). If all parts match, the
composed CIR is the union of their alternatives. Reports, on the NO gold set: coverage and agreement
of the composed answers with the human gold labels. Prints counts only.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from signrule.evaluation.harness import RulesNo, load_requests
from signrule.ontology.cir import derive_answers
from signrule.ontology.no_compose import compose


def main() -> int:
    rules = RulesNo()
    gold = load_requests("no", "gold", "test")
    c: Counter[str] = Counter()
    for r in gold:
        s = compose(r["state"].get("signature_rule"), rules.rules)
        multi = bool(re.search(r"\beller\b", r["state"].get("signature_rule") or "", re.I))
        c["items"] += 1
        c["multi (contains 'eller')"] += multi
        if s is None:
            continue
        c["composed"] += 1
        c["composed multi"] += multi
        ans = derive_answers(s, None)
        for q, v in r["questions"].items():
            if q in ans and q not in ("parseable", "prokura_present", "prokura_joint", "ambiguity"):
                c[f"{q}: agree" if ans[q] == v["label"] else f"{q}: DISAGREE"] += 1
    for k, v in sorted(c.items()):
        print(f"{v:5d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
