"""Tests for the canonical signing representation (plan-03 Task 1, design-02 §2-§4)."""

from __future__ import annotations

import pytest

from signrule.ontology.cir import Group, ProcurationRule, SigningRule, derive_answers

G = Group.of
ALL_BOARD = Group(collective="ALL_BOARD")
ALL_PARTNERS = Group(collective="ALL_PARTNERS")

R = {
    "R0001": [G(("CHAIR", 1))],
    "R0002": [G(("CEO", 1))],
    "R0003": [ALL_BOARD],
    "R0004": [G(("CHAIR", 1), ("BOARD_MEMBER", 1))],
    "R0005": [G(("BOARD_MEMBER", 2))],
    "R0006": [G(("BOARD_MEMBER", 1))],
    "R0007": [G(("CEO", 1)), G(("CHAIR", 1))],
    "R0008": [G(("CEO", 1), ("CHAIR", 1))],
    "R0009": [ALL_PARTNERS],
    "R0010": [G(("PARTNER", 1))],
    "R0011": [G(("CHAIR", 1)), G(("DEPUTY_CHAIR", 1))],
    "R0012": [G(("CHAIR", 1)), G(("BOARD_MEMBER", 2))],
    "R0013": [G(("CEO", 1)), G(("BOARD_MEMBER", 1))],
    "R0014": [G(("CEO", 1), ("BOARD_MEMBER", 1))],
    "R0015": [G(("CHAIR", 1), ("DEPUTY_CHAIR", 1))],
    "R0016": [G(("BOARD_MEMBER", 3))],
    "R0017": [G(("CHAIR", 1), ("BOARD_MEMBER", 2))],
    "R0018": [G(("DEPUTY_CHAIR", 1), ("BOARD_MEMBER", 1))],
}

# code: rule_type, min_signers, ceo_alone, chair_alone, two_bm, ceo_with_bm
TABLE = {
    "R0001": ("sole_chair", "1", False, True, False, False),
    "R0002": ("sole_ceo", "1", True, False, False, True),
    "R0003": ("board_jointly", "all_board", False, False, False, False),
    "R0004": ("joint_role_combo", "2", False, False, False, False),
    "R0005": ("joint_two_any", "2", False, False, True, False),
    "R0006": ("sole_any_board_member", "1", False, True, True, True),
    "R0007": ("sole_specific_roles", "1", True, True, False, True),
    "R0008": ("joint_two_specific", "2", False, False, False, False),
    "R0009": ("board_jointly", "all_board", False, False, False, False),
    "R0010": ("sole_any_board_member", "1", False, False, False, False),
    "R0011": ("sole_specific_roles", "1", False, True, False, False),
    "R0012": ("mixed_alternatives", "1", False, True, True, False),
    "R0013": ("sole_specific_roles", "1", True, True, True, True),
    "R0014": ("joint_role_combo", "2", False, False, False, True),
    "R0015": ("joint_two_specific", "2", False, False, False, False),
    "R0016": ("joint_three_plus", "3+", False, False, False, False),
    "R0017": ("joint_role_combo", "3+", False, False, False, False),
    "R0018": ("joint_role_combo", "2", False, False, False, False),
}


def rule(code: str, **kw) -> SigningRule:
    return SigningRule(
        status="rule",
        alternatives=frozenset(R[code]),
        person_specific=kw.get("person_specific", False),
        statutory_default=ALL_BOARD,
    )


@pytest.mark.parametrize("code", sorted(TABLE))
def test_r_code_answers(code):
    a = derive_answers(rule(code), None)
    rt, ms, ceo, chair, two, ceo_bm = TABLE[code]
    assert a["rule_type"] == rt
    assert a["min_signers"] == ms
    assert a["ceo_alone"] is ceo
    assert a["chair_alone"] is chair
    assert a["two_board_members_jointly"] is two
    assert a["ceo_with_one_board_member"] is ceo_bm
    assert a["parseable"] is True
    assert "prokura_present" not in a and "prokura_joint" not in a


def test_uninterpretable_masks_structure():
    s = SigningRule("uninterpretable", frozenset(), False, ALL_BOARD)
    assert derive_answers(s, None) == {"parseable": False}


def test_none_uses_statutory_default():
    s = SigningRule("none", frozenset(), False, ALL_BOARD)
    a = derive_answers(s, None)
    assert a["rule_type"] == "no_rule_registered"
    assert a["min_signers"] == "all_board"
    assert a["ceo_alone"] is False and a["chair_alone"] is False
    assert a["parseable"] is True


def test_person_specific_masks_negative_role_answers():
    sig = Group.of(("SIGNATORY", 1))
    s = SigningRule("none", frozenset({sig}), True, ALL_BOARD)
    a = derive_answers(s, None)
    assert "ceo_alone" not in a and "chair_alone" not in a and "rule_type" not in a
    assert a["min_signers"] == "1"
    s2 = SigningRule("rule", frozenset({sig, G(("CEO", 1))}), True, ALL_BOARD)
    a2 = derive_answers(s2, None)
    assert a2["ceo_alone"] is True and a2["rule_type"] == "sole_ceo"
    assert "chair_alone" not in a2


def test_not_applicable_questions_absent():
    a = derive_answers(rule("R0010"), None, not_applicable=frozenset({"chair_alone"}))
    assert "chair_alone" not in a and "ceo_alone" in a


def test_procuration_answers():
    s = rule("R0002")
    a = derive_answers(s, ProcurationRule(present=True, mode="joint"))
    assert a["prokura_present"] is True and a["prokura_joint"] is True
    b = derive_answers(s, ProcurationRule(present=False, mode=None))
    assert b["prokura_present"] is False and "prokura_joint" not in b
    c = derive_answers(s, ProcurationRule(present=True, mode="mixed"))
    assert c["prokura_present"] is True and "prokura_joint" not in c
    d = derive_answers(s, ProcurationRule(present=None, mode=None))
    assert "prokura_present" not in d


def test_prokura_only_rule():
    s = SigningRule("rule", frozenset({G(("PROKURIST", 2))}), False, ALL_BOARD)
    assert derive_answers(s, None)["rule_type"] == "prokura_only_rule"


def test_group_size():
    assert G(("CHAIR", 1), ("BOARD_MEMBER", 2)).size == 3
    assert ALL_BOARD.size == "all"
