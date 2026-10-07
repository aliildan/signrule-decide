"""Coalition questions (design-04) derived from hand-written CIRs (no register data)."""

from __future__ import annotations

from signrule.ontology.cir import Group, SigningRule, derive_answers
from signrule.ontology.coalitions import coalition_answers, min_signers_from, minimal_coalitions


def rule(*groups: Group, person_specific: bool = False) -> SigningRule:
    return SigningRule("rule", frozenset(groups), person_specific, Group(collective="ALL_BOARD"))


def test_chair_and_one_member():  # "Styrets leder og ett styremedlem i fellesskap"
    a = coalition_answers(rule(Group.of(("CHAIR", 1), ("BOARD_MEMBER", 1))))
    assert a["chair_with_member"] is True
    assert a["member_alone"] is False and a["three_members"] is False  # needs the chair
    assert a["two_ceos"] is False and a["ceo_with_chair"] is False


def test_two_board_members():  # "To styremedlemmer i fellesskap"
    a = coalition_answers(rule(Group.of(("BOARD_MEMBER", 2))))
    assert a["chair_with_member"] is True  # the chair is a board member
    assert a["three_members"] is True and a["member_alone"] is False


def test_ceo_alone_or_two_members_is_mixed():  # "Daglig leder alene eller to styremedlemmer …"
    s = rule(Group.of(("CEO", 1)), Group.of(("BOARD_MEMBER", 2)))
    ans = derive_answers(s, None)
    assert ans["rule_type"] == "mixed_alternatives"
    assert (
        ans["two_ceos"] is True and ans["ceo_with_chair"] is True
    )  # supersets of a sufficient group


def test_austrian_two_gf_or_gf_with_prokurist_is_joint_alternatives():
    s = rule(Group.of(("CEO", 2)), Group.of(("CEO", 1), ("PROKURIST", 1)))
    ans = derive_answers(s, None)
    assert ans["rule_type"] == "joint_alternatives"
    assert ans["two_ceos"] is True and ans["ceo_with_prokurist"] is True
    assert ans["ceo_alone"] is False and ans["member_alone"] is False
    assert ans["prokurist_alone"] is False and ans["two_prokurists"] is False


def test_person_specific_masks_what_role_groups_cannot_decide():
    s = rule(Group.of(("CEO", 2)), person_specific=True)
    a = coalition_answers(s)
    assert a["two_ceos"] is True
    assert "ceo_with_prokurist" not in a and "member_alone" not in a


def test_partnership_member():  # "Deltakerne hver for seg"
    a = coalition_answers(rule(Group.of(("PARTNER", 1))))
    assert a["member_alone"] is True and a["three_members"] is True


def test_named_signatory_groups_are_not_offices():
    s = rule(Group.of(("SIGNATORY", 1)), Group.of(("CHAIR", 1)), person_specific=True)
    a = coalition_answers(s)
    assert a["chair_with_member"] is True and "two_ceos" not in a


def test_uninterpretable_or_none_gives_no_coalitions():
    assert coalition_answers(SigningRule("uninterpretable", frozenset(), False, None)) == {}
    none = SigningRule("none", frozenset(), False, Group(collective="ALL_BOARD"))
    assert coalition_answers(none) == {}


def test_min_signers_from_coalitions():
    assert (
        min_signers_from({"ceo_alone": False, "two_ceos": True, "ceo_with_prokurist": True}) == "2"
    )
    assert min_signers_from({"ceo_alone": True, "two_ceos": True}) == "1"
    assert min_signers_from({"three_members": True, "two_board_members_jointly": False}) == "3+"
    assert min_signers_from({"ceo_alone": False}) is None
    assert minimal_coalitions({"ceo_alone": True, "two_ceos": True, "ceo_with_chair": True}) == [
        "ceo_alone"
    ]


def test_min_signers_agrees_with_cir_on_typical_rules():
    for s in (
        rule(Group.of(("CHAIR", 1), ("BOARD_MEMBER", 1))),
        rule(Group.of(("CEO", 1)), Group.of(("BOARD_MEMBER", 2))),
        rule(Group.of(("CEO", 2)), Group.of(("CEO", 1), ("PROKURIST", 1))),
    ):
        ans = derive_answers(s, None)
        coal = {k: v for k, v in ans.items() if isinstance(v, bool)}
        assert min_signers_from(coal) == ans["min_signers"]


def test_deputy_and_three_person_coalitions():
    deputy_member = coalition_answers(rule(Group.of(("DEPUTY_CHAIR", 1), ("BOARD_MEMBER", 1))))
    assert deputy_member["deputy_with_member"] is True
    assert deputy_member["chair_with_deputy"] is True  # the chair is also a board member
    assert deputy_member["ceo_with_chair"] is False and deputy_member["member_alone"] is False
    two_members = coalition_answers(rule(Group.of(("BOARD_MEMBER", 2))))
    assert two_members["chair_with_deputy"] is True  # chair and deputy are two board members
    ceo_two = derive_answers(rule(Group.of(("CEO", 1), ("BOARD_MEMBER", 2))), None)
    assert ceo_two["ceo_with_two_members"] is True and ceo_two["ceo_with_one_board_member"] is False
