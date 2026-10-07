"""Phrase-table CIR for Austrian representation lines (hand-written fixture lines)."""

from __future__ import annotations

from signrule.ontology.at_phrases import parse_line, phrase_cir
from signrule.ontology.cir import Group, derive_answers

SOLE = "vertritt seit 01.01.2020 selbständig"
GF_OR_PROK = (
    "vertritt seit 01.01.2020 gemeinsam mit einem weiteren Geschäftsführer oder einem Prokuristen"
)


def test_parse_line_variants():
    assert parse_line(SOLE) == "sole"
    assert parse_line(GF_OR_PROK) == ["CEO", "PROKURIST"]
    gendered = (
        "vertritt seit 01.01.2020 gemeinsam mit einem/einer weiteren Geschäftsführer/in "
        "oder einer/einem Prokuristin/Prokuristen"
    )
    assert parse_line(gendered) == ["CEO", "PROKURIST"]
    assert parse_line("vertritt seit 01.01.2020 gemeinsam mit gemeinsam mit einem Prokuristen") == [
        "PROKURIST"
    ]
    assert parse_line("nicht vertretungsbefugt seit 01.01.2020") == "none"


def test_unknown_named_or_conditional_lines_are_undetermined():
    assert parse_line("vertritt seit 01.01.2020 gemeinsam mit Person A") is None
    assert parse_line("vertritt gemeinsam mit dem Geschäftsführer [PERSON_2]") is None
    assert parse_line("vertritt gemeinsam mit einem Mitglied des Aufsichtsrats") is None
    assert parse_line("vertritt selbständig, bei Geschäften über EUR 1 Mio gemeinsam mit …") is None


def test_two_joint_gf_give_joint_alternatives():
    s = phrase_cir([("CEO", GF_OR_PROK, "G"), ("CEO", GF_OR_PROK, "G")])
    assert s is not None and not s.person_specific
    assert s.alternatives == {Group.of(("CEO", 2)), Group.of(("CEO", 1), ("PROKURIST", 1))}
    ans = derive_answers(s, None)
    assert ans["rule_type"] == "joint_alternatives" and ans["min_signers"] == "2"
    assert ans["ceo_alone"] is False and ans["two_ceos"] is True
    assert ans["ceo_with_prokurist"] is True and ans["prokurist_alone"] is False


def test_one_sole_one_joint_gf_is_person_specific():
    s = phrase_cir([("CEO", SOLE, "E"), ("CEO", GF_OR_PROK, "G")])
    assert s is not None and s.person_specific
    assert s.alternatives == {Group.of(("CEO", 2)), Group.of(("CEO", 1), ("PROKURIST", 1))}
    ans = derive_answers(s, None)
    assert "ceo_alone" not in ans  # depends on the person
    assert ans["two_ceos"] is True and ans["ceo_with_prokurist"] is True


def test_cooperative_board_any_two_of_chair_deputy_member():
    chair = (
        "vertritt seit 01.01.2020 gemeinsam mit dem Obmannstellvertreter "
        "oder einem weiteren Vorstandsmitglied"
    )
    deputy = (
        "vertritt seit 01.01.2020 gemeinsam mit dem Obmann oder einem weiteren Vorstandsmitglied"
    )
    member = "vertritt seit 01.01.2020 gemeinsam mit dem Obmann oder dem Obmannstellvertreter"
    s = phrase_cir(
        [("CHAIR", chair, "G"), ("DEPUTY_CHAIR", deputy, "G"), ("BOARD_MEMBER", member, "G")]
    )
    assert s is not None and not s.person_specific
    assert s.alternatives == {
        Group.of(("CHAIR", 1), ("DEPUTY_CHAIR", 1)),
        Group.of(("CHAIR", 1), ("BOARD_MEMBER", 1)),
        Group.of(("DEPUTY_CHAIR", 1), ("BOARD_MEMBER", 1)),
    }
    assert derive_answers(s, None)["two_board_members_jointly"] is False  # two ordinary members


def test_code_must_match_wording():
    assert phrase_cir([("CEO", SOLE, "G")]) is None
    assert phrase_cir([("CEO", GF_OR_PROK, "E")]) is None
    assert phrase_cir([("OTHER", SOLE, "E")]) is None


def test_no_power_holders_are_ignored():
    s = phrase_cir([("CEO", SOLE, "E"), ("CEO", "nicht vertretungsbefugt seit 2020", "X")])
    assert s is not None and s.alternatives == {Group.of(("CEO", 1))} and not s.person_specific
