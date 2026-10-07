"""Logical-consistency gate: predicted answers must fit at least one canonical signing rule.

Every structural answer is a pure function of the canonical signing representation (design-02),
so a set of predictions that no rule can produce (e.g. `rule_type = sole_ceo` with
`ceo_alone = no`) is self-contradictory. That happens mostly on unfamiliar input and is a cheap,
label-free signal to abstain. The universe of rules is built from the register's groups plus a few
extra executive/partner/procuration groups, as alternatives of up to three groups, with and without
a named signatory, plus the statutory defaults.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations

from signrule.ontology.cir import Group, ProcurationRule, SigningRule, derive_answers
from signrule.ontology.mapping import load_no_rules

STRUCTURAL = (
    "rule_type",
    "min_signers",
    "ceo_alone",
    "chair_alone",
    "two_board_members_jointly",
    "ceo_with_one_board_member",
)
G = Group.of
EXTRA_GROUPS = (
    G(("CEO", 2)),
    G(("CEO", 1), ("PROKURIST", 1)),
    G(("EXECUTIVE_MEMBER", 1)),
    G(("EXECUTIVE_MEMBER", 2)),
    G(("EXECUTIVE_MEMBER", 1), ("PROKURIST", 1)),
    G(("PARTNER", 2)),
    G(("PROKURIST", 2)),
    Group(collective="ALL_EXECUTIVES"),
)
STATUTORY = (Group(collective="ALL_BOARD"), G(("PARTNER", 1)))
MAX_ALTERNATIVES = 3


def _vector(s: SigningRule) -> tuple[object, ...]:
    ans = derive_answers(s, ProcurationRule(None, None))
    return tuple(ans.get(q) for q in STRUCTURAL)


@cache
def _universe(with_signatory: bool = True) -> frozenset[tuple[object, ...]]:
    atoms = sorted(
        {g for spec in load_no_rules().values() if spec.kind == "rule" for g in spec.alternatives}
        | set(EXTRA_GROUPS),
        key=repr,
    )
    signatory = G(("SIGNATORY", 1))
    vectors: set[tuple[object, ...]] = set()
    for k in range(1, MAX_ALTERNATIVES + 1):
        for combo in combinations(atoms, k):
            for person in (False, True) if with_signatory else (False,):
                alts = frozenset(combo) | ({signatory} if person else set())
                vectors.add(_vector(SigningRule("rule", alts, person, STATUTORY[0])))
    for default in STATUTORY:
        vectors.add(_vector(SigningRule("none", frozenset(), False, default)))
        if with_signatory:
            vectors.add(_vector(SigningRule("none", frozenset({signatory}), True, default)))
    return frozenset(vectors)


def universe_size() -> int:
    return len(_universe())


def _matches(v: tuple[object, ...], asked: list[tuple[int, object]]) -> bool:
    for i, p in asked:
        if v[i] is None and STRUCTURAL[i] != "rule_type":
            continue  # undetermined yes/no answer (e.g. a named signatory may be the CEO)
        if v[i] != p:  # an undetermined rule_type only fits when no rule_type is predicted
            return False
    return True


def has_named_signatory(state: object) -> bool:
    """Named (person-specific) signatories are visible as signature roles in the state."""
    if not isinstance(state, dict):
        return True  # unknown structure: don't rule them out
    return any("signatur" in str(r.get("role", "")).lower() for r in state.get("roles") or [])


def _implications_hold(pred: dict[str, object]) -> bool:
    """Implications true for every rule, named signatories included (signing is monotone)."""
    if pred.get("ceo_alone") is True and pred.get("ceo_with_one_board_member") is False:
        return False  # whoever may sign alone may also sign together with someone else
    alone = pred.get("ceo_alone") is True or pred.get("chair_alone") is True
    return not (alone and pred.get("min_signers") not in (None, "1"))


def is_consistent(pred: dict[str, object], allow_named_signatory: bool = True) -> bool:
    """True if some canonical rule yields all predicted structural answers.

    Rules with a named signatory leave CEO/chair answers open (the person may hold the office), so
    they are only considered when the input shows such a role (`has_named_signatory`).
    """
    asked = [(i, pred[q]) for i, q in enumerate(STRUCTURAL) if q in pred]
    if len(asked) < 2:
        return True
    if not _implications_hold(pred):
        return False
    return any(_matches(v, asked) for v in _universe(allow_named_signatory))
