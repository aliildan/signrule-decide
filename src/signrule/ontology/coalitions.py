"""Coalition questions (design-04): "can these office holders, together, bind the company?"

A CIR is a monotone set of sufficient AND-groups. Each coalition in `NEW_COALITIONS` is a concrete
set of office holders; it is sufficient if some alternative group can be filled from it by distinct
people, with roles qualifying upwards (a chair or deputy chair is also a board member; the generic
"member" is an ordinary member of whatever body the company has: board, partners, executive board).
Answers come from the same CIR as every other label, so no new labelling is needed; a
person-specific rule masks what its role groups cannot decide (never "no").

The four older coalition-like questions (ceo_alone, chair_alone, two_board_members_jointly,
ceo_with_one_board_member) keep their derivation in `cir.derive_answers`, so their labels do not
change.
"""

from __future__ import annotations

from collections import Counter

from signrule.ontology.cir import Group, SigningRule

MEMBER = "MEMBER"  # generic body member (only used for coalition persons)
# Slots (roles in a CIR group) each coalition person can fill.
FILLS: dict[str, frozenset[str]] = {
    "CEO": frozenset({"CEO"}),
    "CHAIR": frozenset({"CHAIR", "BOARD_MEMBER", "EXECUTIVE_MEMBER"}),
    "DEPUTY": frozenset({"DEPUTY_CHAIR", "BOARD_MEMBER", "EXECUTIVE_MEMBER"}),
    "PROKURIST": frozenset({"PROKURIST"}),
    MEMBER: frozenset({"BOARD_MEMBER", "PARTNER", "EXECUTIVE_MEMBER"}),
}
NEW_COALITIONS: dict[str, tuple[str, ...]] = {
    "member_alone": (MEMBER,),
    "two_ceos": ("CEO", "CEO"),
    "ceo_with_chair": ("CEO", "CHAIR"),
    "ceo_with_prokurist": ("CEO", "PROKURIST"),
    "chair_with_member": ("CHAIR", MEMBER),
    "three_members": (MEMBER, MEMBER, MEMBER),
    "prokurist_alone": ("PROKURIST",),
    "two_prokurists": ("PROKURIST", "PROKURIST"),
    # plan-13: pairs with the deputy chair and a three-person group, so that the coalition answers
    # can reconstruct the common Norwegian and Austrian board rules
    "chair_with_deputy": ("CHAIR", "DEPUTY"),
    "deputy_with_member": ("DEPUTY", MEMBER),
    "ceo_with_two_members": ("CEO", MEMBER, MEMBER),
}
# The older questions, expressed as coalitions (for derived min_signers and consistency only).
OLD_COALITIONS: dict[str, tuple[str, ...]] = {
    "ceo_alone": ("CEO",),
    "chair_alone": ("CHAIR",),
    "two_board_members_jointly": (MEMBER, MEMBER),
    "ceo_with_one_board_member": ("CEO", MEMBER),
}
ALL_COALITIONS = {**OLD_COALITIONS, **NEW_COALITIONS}


def _assign(slots: list[str], people: list[str]) -> bool:
    if not slots:
        return True
    slot, rest = slots[0], slots[1:]
    for i, person in enumerate(people):
        if slot in FILLS[person] and _assign(rest, people[:i] + people[i + 1 :]):
            return True
    return False


def fills(group: Group, coalition: tuple[str, ...]) -> bool:
    """Can the distinct persons of `coalition` fill every slot of `group`?"""
    if group.collective is not None:
        return False
    slots = [role for role, n in sorted(group.roles) for _ in range(n)]
    return len(slots) <= len(coalition) and _assign(slots, list(coalition))


def coalition_answers(s: SigningRule) -> dict[str, bool]:
    """Answers for the new coalition questions the CIR determines (absent = masked)."""
    if s.status != "rule":
        return {}
    role_groups = [g for g in s.alternatives if "SIGNATORY" not in g.counts()]
    out: dict[str, bool] = {}
    for qid, coalition in NEW_COALITIONS.items():
        if any(fills(g, coalition) for g in role_groups):
            out[qid] = True
        elif not s.person_specific:
            out[qid] = False
    return out


def minimal_coalitions(answers: dict[str, bool]) -> list[str]:
    """Coalitions answered yes that contain no smaller yes-coalition (multiset inclusion)."""
    yes = [q for q, v in answers.items() if v is True and q in ALL_COALITIONS]

    def contains(big: str, small: str) -> bool:
        b, sm = Counter(ALL_COALITIONS[big]), Counter(ALL_COALITIONS[small])
        return (
            big != small
            and sum(b.values()) > sum(sm.values())
            and all(b[r] >= n for r, n in sm.items())
        )

    return [q for q in yes if not any(contains(q, other) for other in yes)]


def min_signers_from(answers: dict[str, bool]) -> str | None:
    """Size of the smallest coalition answered yes; None if no coalition is a yes."""
    sizes = [len(ALL_COALITIONS[q]) for q in minimal_coalitions(answers)]
    if not sizes:
        return None
    m = min(sizes)
    return "1" if m == 1 else "2" if m == 2 else "3+"
