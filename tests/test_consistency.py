"""Tests for the logical-consistency gate (answers must fit at least one canonical rule)."""

from __future__ import annotations

from signrule.calib.consistency import (
    STRUCTURAL,
    has_named_signatory,
    is_consistent,
    universe_size,
)


def test_universe_is_reasonable():
    assert 20 < universe_size() < 10_000  # distinct answer vectors, not rules
    assert {"rule_type", "ceo_alone", "chair_alone", "min_signers"} <= set(STRUCTURAL)


def test_consistent_standard_answers():
    assert is_consistent(
        {
            "rule_type": "sole_ceo",
            "ceo_alone": True,
            "chair_alone": False,
            "min_signers": "1",
            "ceo_with_one_board_member": True,
        }
    )
    assert is_consistent(
        {
            "rule_type": "mixed_alternatives",
            "min_signers": "1",
            "chair_alone": True,
            "two_board_members_jointly": True,
        }
    )
    assert is_consistent(
        {"rule_type": "no_rule_registered", "min_signers": "all_board", "ceo_alone": False}
    )


def test_contradictions_detected():
    # the two confident errors from the v0.1 demo
    assert not is_consistent({"rule_type": "sole_ceo", "ceo_alone": False})
    demo = {"rule_type": "sole_chair", "ceo_alone": True, "chair_alone": True}
    assert not is_consistent(demo, allow_named_signatory=False)
    assert is_consistent(demo)  # possible if a named signatory exists who is the CEO
    assert not is_consistent({"rule_type": "sole_chair", "chair_alone": False})
    j2 = {"rule_type": "joint_two_any", "min_signers": "1"}
    assert not is_consistent(j2, allow_named_signatory=False)
    assert is_consistent(j2)  # two members jointly OR a named person alone
    assert not is_consistent({"ceo_alone": True, "ceo_with_one_board_member": False})


def test_single_or_unknown_questions_never_flagged():
    assert is_consistent({"ceo_alone": False})
    assert is_consistent({"prokura_present": True, "ceo_alone": True})  # prokura isn't structural
    assert is_consistent({})


def test_named_signatory_masks_role_answers():
    # a person-specific rule leaves CEO/chair questions undetermined: any value is consistent
    assert is_consistent({"min_signers": "1", "ceo_alone": False, "chair_alone": False})


def test_has_named_signatory_from_roles():
    assert has_named_signatory({"roles": [{"role": "Signatur hver for seg", "count": 2}]})
    assert not has_named_signatory({"roles": [{"role": "Daglig leder", "count": 1}]})
    assert has_named_signatory("free text state")


def test_monotone_implications_hold_even_with_named_signatories():
    assert not is_consistent({"ceo_alone": True, "min_signers": "2"})
    assert not is_consistent({"chair_alone": True, "min_signers": "all_board"})
    assert is_consistent({"ceo_alone": True, "min_signers": "1", "ceo_with_one_board_member": True})
