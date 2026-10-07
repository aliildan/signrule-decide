"""Canonical signing representation (CIR) and the typed answers derived from it (design-02).

Every jurisdiction maps its source label into `SigningRule` / `ProcurationRule`; the questions
the model answers are pure functions of these. A question whose answer the source cannot
determine is *absent* from the result, which is how Kev masks a label (omit the question).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal[
    "CEO",
    "CHAIR",
    "DEPUTY_CHAIR",
    "BOARD_MEMBER",
    "PARTNER",
    "EXECUTIVE_MEMBER",
    "PROKURIST",
    "SIGNATORY",
    "OTHER",
]
Collective = Literal["ALL_BOARD", "ALL_PARTNERS", "ALL_EXECUTIVES"]
Status = Literal["rule", "uninterpretable", "none"]

ROLES: frozenset[str] = frozenset(Role.__args__)  # type: ignore[attr-defined]
COLLECTIVES: frozenset[str] = frozenset(Collective.__args__)  # type: ignore[attr-defined]
ANY_MEMBER_ROLES = frozenset({"BOARD_MEMBER", "PARTNER", "EXECUTIVE_MEMBER"})

RULE_TYPES = (
    "sole_ceo",
    "sole_chair",
    "sole_any_board_member",
    "sole_specific_roles",
    "joint_two_any",
    "joint_two_specific",
    "joint_role_combo",
    "joint_three_plus",
    "board_jointly",
    "mixed_alternatives",
    "joint_alternatives",
    "prokura_only_rule",
    "no_rule_registered",
    "other_unmapped",
)
MIN_SIGNERS = ("1", "2", "3+", "all_board")


@dataclass(frozen=True)
class Group:
    """AND-group: all listed role holders together (or the whole collective body)."""

    roles: frozenset[tuple[str, int]] = frozenset()
    collective: str | None = None

    def __post_init__(self) -> None:
        for role, n in self.roles:
            if role not in ROLES or n < 1:
                raise ValueError(f"bad group member {(role, n)}")
        if self.collective is not None and self.collective not in COLLECTIVES:
            raise ValueError(f"bad collective {self.collective}")
        if self.collective is None and not self.roles:
            raise ValueError("empty group")

    @staticmethod
    def of(*pairs: tuple[str, int], collective: str | None = None) -> Group:
        return Group(frozenset(pairs), collective)

    @property
    def size(self) -> int | Literal["all"]:
        return "all" if self.collective else sum(n for _, n in self.roles)

    def counts(self) -> dict[str, int]:
        return dict(self.roles)

    def is_exactly(self, **counts: int) -> bool:
        return self.collective is None and self.counts() == counts


@dataclass(frozen=True)
class SigningRule:
    status: Status
    alternatives: frozenset[Group]
    person_specific: bool
    statutory_default: Group | None


@dataclass(frozen=True)
class ProcurationRule:
    present: bool | None
    mode: Literal["sole", "joint", "mixed"] | None


def _role_based(s: SigningRule) -> list[Group]:
    return [g for g in s.alternatives if "SIGNATORY" not in g.counts()]


def _rule_type(s: SigningRule) -> str | None:
    if s.status == "uninterpretable":
        return None
    alts = _role_based(s)
    if not alts:
        return None if s.person_specific else "no_rule_registered"
    if len(alts) >= 2:
        sole = [g.size == 1 for g in alts]
        if all(sole):
            return "sole_specific_roles"
        return "mixed_alternatives" if any(sole) else "joint_alternatives"
    g = alts[0]
    if g.collective:
        return "board_jointly"
    c = g.counts()
    if set(c) == {"PROKURIST"}:
        return "prokura_only_rule"
    if g.size == 1:
        if c == {"CEO": 1}:
            return "sole_ceo"
        if c == {"CHAIR": 1}:
            return "sole_chair"
        if set(c) <= ANY_MEMBER_ROLES:
            return "sole_any_board_member"
        return "sole_specific_roles"
    specific = set(c) - ANY_MEMBER_ROLES
    if g.size == 2:
        if not specific:
            return "joint_two_any"
        if set(c) == specific:
            return "joint_two_specific"
        return "joint_role_combo"
    return "joint_role_combo" if specific else "joint_three_plus"


def _min_signers(s: SigningRule) -> str | None:
    if s.status == "uninterpretable":
        return None
    groups = list(s.alternatives)
    if s.status == "none" and s.statutory_default is not None:
        groups.append(s.statutory_default)
    if not groups:
        return None
    sizes = [g.size for g in groups if g.size != "all"]
    if not sizes:
        return "all_board"
    m = min(sizes)  # type: ignore[type-var]
    return "1" if m == 1 else "2" if m == 2 else "3+"


def _has(s: SigningRule, *wanted: dict[str, int]) -> bool:
    return any(g.collective is None and g.counts() in wanted for g in s.alternatives)


def _role_answer(s: SigningRule, *wanted: dict[str, int]) -> bool | None:
    """True if an alternative grants it; False only when nothing person-specific could."""
    if s.status == "uninterpretable":
        return None
    if _has(s, *wanted):
        return True
    return None if s.person_specific else False


def derive_answers(
    s: SigningRule,
    p: ProcurationRule | None,
    not_applicable: frozenset[str] = frozenset(),
) -> dict[str, str | bool | int]:
    """Typed answers for every question the source determines (absent = masked)."""
    out: dict[str, str | bool | int | None] = {
        "rule_type": _rule_type(s),
        "min_signers": _min_signers(s),
        "ceo_alone": _role_answer(s, {"CEO": 1}),
        "chair_alone": _role_answer(s, {"CHAIR": 1}, {"BOARD_MEMBER": 1}),
        "two_board_members_jointly": _role_answer(s, {"BOARD_MEMBER": 1}, {"BOARD_MEMBER": 2}),
        "ceo_with_one_board_member": _role_answer(
            s, {"CEO": 1}, {"BOARD_MEMBER": 1}, {"CEO": 1, "BOARD_MEMBER": 1}
        ),
        "parseable": s.status != "uninterpretable",
    }
    from signrule.ontology.coalitions import coalition_answers  # coalitions imports this module

    out.update(coalition_answers(s))
    if p is not None and p.present is not None:
        out["prokura_present"] = p.present
        if p.present and p.mode in ("sole", "joint"):
            out["prokura_joint"] = p.mode == "joint"
    return {k: v for k, v in out.items() if v is not None and k not in not_applicable}


def procuration_mode(groups: list[Group]) -> Literal["sole", "joint"] | None:
    """Prokura held alone if any alternative is a single person, jointly if all need two or more."""
    if not groups:
        return None
    return "sole" if any(g.size == 1 for g in groups) else "joint"
