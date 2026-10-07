"""Tests for gold-set helpers: annotation -> CIR -> answers, and agreement (plan-05 T5)."""

from __future__ import annotations

import pytest

from signrule.evaluation import gold
from signrule.evaluation.gold import agreement, annotation_answers


def ann(item: str, status: str = "rule", alts=None, prok=None, amb=0) -> dict:
    return {
        "item_id": item,
        "status": status,
        "alternatives": alts or [],
        "procuration": prok or {"present": "no", "mode": ""},
        "ambiguity": amb,
    }


def test_chair_alone_or_two_members():
    a = ann("x", alts=[{"roles": [["CHAIR", 1]]}, {"roles": [["BOARD_MEMBER", 2]]}])
    ans = annotation_answers(a)
    assert ans["rule_type"] == "mixed_alternatives"
    assert ans["chair_alone"] is True and ans["ceo_alone"] is False
    assert ans["min_signers"] == "1" and ans["prokura_present"] is False
    assert ans["ambiguity"] == 0


def test_uninterpretable_and_named_signatory():
    assert annotation_answers(ann("x", status="uninterpretable", amb=3)) == {
        "parseable": False,
        "prokura_present": False,
        "ambiguity": 3,
    }
    named = annotation_answers(ann("y", alts=[{"roles": [["SIGNATORY", 1]]}]))
    assert "ceo_alone" not in named and named["min_signers"] == "1"


def test_collective_and_procuration_joint():
    a = ann("x", alts=[{"collective": "ALL_BOARD"}], prok={"present": "yes", "mode": "joint"})
    ans = annotation_answers(a)
    assert ans["rule_type"] == "board_jointly" and ans["prokura_joint"] is True


def test_agreement_perfect_and_partial():
    a = {i: ann(i, alts=[{"roles": [["CEO", 1]]}]) for i in ("1", "2", "3", "4")}
    b = dict(a)
    b["4"] = ann("4", alts=[{"roles": [["CHAIR", 1]]}])
    res = agreement(a, b)
    assert res["n_items"] == 4
    assert res["per_question"]["rule_type"]["agreement"] == pytest.approx(0.75)
    assert res["per_question"]["prokura_present"]["kappa"] == 1.0


def test_austrian_mixed_powers_are_person_specific():
    # GF A "selbständig" (only that person), GF B "gemeinsam mit einem weiteren Geschäftsführer
    # oder einem Prokuristen": role-based groups are the joint ones; the office question stays open.
    a = ann(
        "x",
        alts=[{"roles": [["CEO", 2]]}, {"roles": [["CEO", 1], ["PROKURIST", 1]]}],
        prok={"present": "yes", "mode": "joint"},
        amb=2,
    )
    a["person_specific"] = True
    ans = annotation_answers(a)
    assert ans.get("ceo_alone") is None
    assert ans["min_signers"] == "1"  # the person-specific power is a sole power
    a["person_specific"] = False
    assert annotation_answers(a)["min_signers"] == "2"
    assert annotation_answers(a)["ceo_alone"] is False


def test_executive_member_is_harmonised_to_board_member():
    a = ann("x", alts=[{"roles": [["EXECUTIVE_MEMBER", 2]]}])
    b = ann("x", alts=[{"roles": [["BOARD_MEMBER", 2]]}])
    assert annotation_answers(a) == annotation_answers(b)
    assert annotation_answers(a)["two_board_members_jointly"] is True


def _dk_req(i: int, text: str, n_same: int, roles: list[tuple[str, int]], group: str | None = None):
    return {
        "state": {
            "jurisdiction": "DK",
            "legal_form": "ApS",
            "signature_rule": text,
            "roles": [{"role": r, "count": c} for r, c in roles],
        },
        "questions": {},
        "_meta": {
            "id": f"9999{i:04d}",
            "group_id": group or f"g{i}",
            "stratum": "dk-standard" if n_same >= 20 else "dk-tail",
            "text_companies": n_same,
        },
    }


def test_dk_sample_is_one_per_group_and_two_thirds_long_tail(monkeypatch):
    pool = [
        _dk_req(i, "Selskabet tegnes af en direktør.", 500, [("Direktør", 1)], "std")
        for i in range(50)
    ]
    pool += [_dk_req(100 + i, f"Standard {i}", 30, [("Direktør", 2)]) for i in range(40)]
    pool += [
        _dk_req(
            200 + i, f"Tail {i}", 1, [("Direktør", 1), ("Formand", 1), ("Bestyrelsesmedlem", 2)]
        )
        for i in range(100)
    ]
    monkeypatch.setattr(gold, "load_requests", lambda jur, split, part: pool)
    items = gold.sample_dk(30, seed=13)
    assert len(items) == 30
    assert len({it["item_id"] for it in items}) == 30
    assert sum(it["strata"]["wording"] == "dk-tail" for it in items) == 20
    assert (
        sum(
            it["item_id"].startswith("9999")
            and it["state"]["signature_rule"] == "Selskabet tegnes af en direktør."
            for it in items
        )
        == 1
    )
    assert all(it["set"] == "dk-text" and it["part"] == "pool/test" for it in items)


def _at_req(i: int, roles: list[tuple[str, int]], group: str | None = None, form: str = "GmbH"):
    return {
        "state": {
            "jurisdiction": "AT",
            "legal_form": form,
            "signature_rule": f"Fixture line {i}",
            "roles": [{"role": r, "count": c} for r, c in roles],
        },
        "questions": {},
        "_meta": {"id": f"FN{i}", "group_id": group or f"g{i}", "stratum": "at-text"},
    }


def test_at_focus_sample_fills_quotas_excludes_and_redistributes(monkeypatch):
    vorstand = [
        _at_req(i, [("Vorsitzender des Vorstands", 1), ("Vorstandsmitglied", 2)], form="AG")
        for i in range(30)
    ]
    obmann = [
        _at_req(100 + i, [("Obmann", 1), ("Obmann-Stellvertreter", 1)], form="Genossenschaft")
        for i in range(5)
    ]
    partners = [
        _at_req(200 + i, [("unbeschränkt haftender Gesellschafter", 2)], form="OG")
        for i in range(3)
    ]
    gmbh = [_at_req(300 + i, [("Geschäftsführer", 2), ("Prokurist", 1)]) for i in range(40)]
    single = [_at_req(400 + i, [("Geschäftsführer", 1)]) for i in range(40)]
    excluded = [
        _at_req(500, [("Vorstandsmitglied", 2)], group="taken", form="AG"),
        _at_req(501, [("Vorstandsmitglied", 2)], group="taken", form="AG"),  # same pattern
    ]
    pool = vorstand + obmann + partners + gmbh + single + excluded
    monkeypatch.setattr(
        gold, "load_requests", lambda jur, split, part: pool if split == "pilot" else []
    )
    items = gold.sample_at_focus(
        20, seed=13, exclude_ids={"FN500"}, quotas={"vorstand": 10, "partner": 6, "gmbh_multi": 4}
    )
    kinds = [it["strata"]["focus"] for it in items]
    assert len(items) == 20 and len({it["item_id"] for it in items}) == 20
    assert kinds.count("partner") == 3  # only three exist; the rest is redistributed
    assert kinds.count("vorstand") + kinds.count("gmbh_multi") == 17
    assert not {it["item_id"] for it in items} & {"FN500", "FN501"}  # the whole pattern is excluded
    assert all(it["set"] == "at-text-2" for it in items)
    assert gold._at_focus(obmann[0]["state"]) == "vorstand"
    assert gold._at_focus(single[0]["state"]) == "other"
