"""Austrian normaliser on the fictitious fixture (tests/fixtures/at): masking, rendering, labels."""

from __future__ import annotations

import copy
from pathlib import Path

from signrule.format.request import load_questions
from signrule.ingest import ingest_at as at
from signrule.normalize import normalize_at as nat
from signrule.ontology.mapping import load_roles

FIX = Path(__file__).parent / "fixtures" / "at"


def _rec() -> at.AtRecord:
    obj = at.sanitize(at.parse_soap_bytes((FIX / "auszug_ges.xml").read_bytes()))
    return at.extract_at(obj)


def test_render_masks_names_and_keeps_references():
    sig, prok, roles, risk = nat.render(_rec(), load_roles())
    assert sig == "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 selbständig"
    assert prok == (
        "Prokurist [PERSON_2]: vertritt seit 01.06.2021 gemeinsam mit "
        "einem Geschäftsführer oder [PERSON_1]"
    )
    assert roles == [("Geschäftsführer", 1), ("Prokurist", 1)]
    assert not risk
    assert "Fixture" not in sig + prok


def test_join_fragments_line_breaks_and_conjunctions():
    assert nat.join_fragments(["gemeinsam mit", "einem Geschäfts-", "führer"]) == (
        "gemeinsam mit einem Geschäftsführer"
    )
    assert nat.join_fragments(["kollektiv zeichnungs-", "und vertretungsbefugt"]) == (
        "kollektiv zeichnungs- und vertretungsbefugt"
    )
    assert nat.join_fragments(["Obfrau-", "Obmannstellvertreter"]) == "Obfrau- Obmannstellvertreter"


def test_free_text_not_duplicated_when_text_carries_it():
    f = _rec().functions[1]
    assert nat._line_text(f).count("Fixture Person A") == 1
    f.text = ["vertritt seit 01.06.2021 gemeinsam mit"]
    assert nat._line_text(f).endswith("gemeinsam mit einem Geschäftsführer oder Fixture Person A")


def test_glued_txtvertr_copy_is_not_appended():
    f = _rec().functions[1]
    f.text = ["vertritt seit 01.06.2021 gemeinsam mit", "einer/einem Prokuristin/Prokuristen"]
    f.txtvertr = ["einer/einemProkuristin/Prokuristen"]
    assert nat._line_text(f) == (
        "vertritt seit 01.06.2021 gemeinsam mit einer/einem Prokuristin/Prokuristen"
    )


def test_future_end_date_is_current():
    entries = [{"DATBIS": "2099-12-31"}, {"DATBIS": "2000-01-01"}, {"@AUFRECHT": "false"}, {}]
    assert at._current(entries, today="2026-10-05") == [entries[0], entries[3]]


def test_coded_answers_from_vart():
    assert nat.coded_answers(_rec()) == {
        "prokura_present": True,
        "prokura_joint": True,
        "ceo_alone": True,
        "min_signers": "1",
    }


def test_free_text_on_managing_director_masks_structural_labels():
    rec = _rec()
    rec.functions[0].txtvertr = ["jedoch bei Geschäften über EUR 100.000 gemeinsam"]
    ans = nat.coded_answers(rec)
    assert "ceo_alone" not in ans and "min_signers" not in ans
    assert ans["prokura_present"] is True


def test_joint_managing_directors_and_no_procuration():
    rec = _rec()
    rec.functions = [copy.deepcopy(rec.functions[0]) for _ in range(2)]
    for f in rec.functions:
        f.vart_code = "G"
    assert nat.coded_answers(rec) == {"prokura_present": False, "ceo_alone": False}


def test_liquidation_keeps_only_procuration_labels():
    rec = _rec()
    liq = copy.deepcopy(rec.functions[0])
    liq.fken, liq.pnr = "AB", "Z"
    rec.functions.append(liq)
    assert set(nat.coded_answers(rec)) == {"prokura_present", "prokura_joint"}


def test_company_partner_gets_firma_token():
    rec = _rec()
    rec.persons["A"] = at.AtPerson(legal=True, names=["Fixture Komplementär GmbH"])
    rec.functions[0].fken = "KP"
    sig, prok, _, _ = nat.render(rec, load_roles())
    assert sig.startswith("unbeschränkt haftender Gesellschafter [FIRMA_1]:")
    assert prok.startswith("Prokurist [PERSON_1]:")


def test_request_has_meta_stratum_and_only_coded_questions():
    r = nat.to_at_request(_rec(), load_roles(), load_questions())
    assert r is not None and r.stratum == "at-text"  # the procurator has a TXTVERTR
    req = r.request
    assert req["state"]["jurisdiction"] == "AT" and req["state"]["legal_form"] == "GmbH"
    # direct code labels + the signing structure from the codes (the GF line has no free text;
    # the Prokurist's text is about procuration): plan-12 T2
    assert set(req["questions"]) == {
        "prokura_present",
        "prokura_joint",
        "ceo_alone",
        "min_signers",
        "rule_type",
        "parseable",
        "two_ceos",
        "ceo_with_prokurist",
        "prokurist_alone",
        "two_prokurists",
    }
    assert req["questions"]["rule_type"]["label"] == "sole_ceo"
    assert req["_meta"]["id"] == "999999x"


def test_sole_trader_stub_yields_no_request():
    stub = {"AUSZUG_V2_RESPONSE": {"@FNR": "1x", "@SKIPPED": "sole_trader"}}
    assert nat.to_at_request(at.extract_at(stub), load_roles(), load_questions()) is None


def test_coded_cir_two_sole_managing_directors():
    rec = _rec()
    gf2 = copy.deepcopy(rec.functions[0])
    gf2.pnr = "Z"
    rec.functions = [rec.functions[0], gf2]
    cir = nat.coded_cir(rec, load_roles())
    assert cir is not None and not cir.person_specific
    r = nat.to_at_request(rec, load_roles(), load_questions())
    labels = {q: v["label"] for q, v in r.request["questions"].items()}
    assert labels["ceo_alone"] is True and labels["two_ceos"] is True
    assert labels["rule_type"] == "sole_ceo" and labels["parseable"] is True
    assert "member_alone" not in labels  # a GmbH has no representing board


def test_coded_cir_mixed_codes_are_person_specific():
    rec = _rec()
    gf2 = copy.deepcopy(rec.functions[0])
    gf2.pnr, gf2.vart_code = "Z", "G"
    rec.functions = [rec.functions[0], gf2]
    cir = nat.coded_cir(rec, load_roles())
    assert cir is not None and cir.person_specific and not cir.alternatives
    labels = {
        q: v["label"]
        for q, v in nat.to_at_request(rec, load_roles(), load_questions())
        .request["questions"]
        .items()
    }
    assert "ceo_alone" not in labels and "rule_type" not in labels and "parseable" not in labels
    assert labels["min_signers"] == "1"


def test_coded_cir_skips_free_text_on_representing_function():
    rec = _rec()
    rec.functions[0].txtvertr = ["jedoch nur gemeinsam für Grundstücksgeschäfte"]
    assert nat.coded_cir(rec, load_roles()) is None
