"""Tests for Norwegian record building and masked requests (plan-03 T3, T6)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from signrule.format.request import load_questions
from signrule.normalize.normalize_no import build_record, to_masked_request
from signrule.ontology.cir import Group

FIX = Path(__file__).parent / "fixtures" / "no"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def test_rf_record_has_cir_and_names():
    rec = build_record(load("fullmakt_signatur_rf.json"), load("fullmakt_prokura_ri.json"), None)
    assert rec is not None
    assert rec.signing.status == "rule"
    assert rec.signing.alternatives == frozenset({Group.of(("CEO", 1))})
    assert rec.rule_codes == ("R0002",)
    assert rec.procuration.present is False
    assert set(rec.person_names) == {"Fixture Person A", "Fixture Person B"}
    assert rec.role_counts == {"DAGL": 1, "LEDE": 1}


def test_rt_and_na():
    rt = build_record(load("fullmakt_signatur_rt.json"), None, None)
    assert rt is not None and rt.signing.status == "uninterpretable"
    assert rt.procuration.present is None
    assert build_record(load("fullmakt_signatur_na.json"), None, None) is None


def test_masked_request_shape_and_labels():
    rec = build_record(load("fullmakt_signatur_rf.json"), load("fullmakt_prokura_ri.json"), None)
    assert rec is not None
    m = to_masked_request(rec)
    req = m.request
    assert list(req["state"]) == ["jurisdiction", "legal_form", "signature_rule", "roles"]
    assert req["state"]["roles"] == [
        {"role": "Daglig leder", "count": 1},
        {"role": "Styrets leder", "count": 1},
    ]
    q = req["questions"]
    assert q["rule_type"]["label"] == "sole_ceo"
    assert q["ceo_alone"]["label"] is True
    assert q["prokura_present"]["label"] is False
    assert "prokura_joint" not in q
    assert q["ambiguity"]["label"] == 0  # "Daglig leder alene." equals the R0002 description
    assert q["rule_type"]["criteria"] == load_questions().questions["rule_type"]["criteria"]
    assert "Fixture Person" not in json.dumps(req)
    assert m.has_text and not m.residual_risk


def test_names_in_text_are_masked():
    sig = load("fullmakt_signatur_rf.json")
    sig = copy.deepcopy(sig)
    sig["signeringsGrunnlag"]["signaturProkuraRoller"]["signaturProkuraFritekst"] = (
        "Fixture Person A alene."
    )
    rec = build_record(sig, None, None)
    assert rec is not None
    m = to_masked_request(rec)
    assert m.request["state"]["signature_rule"] == "[PERSON_1] alene."
    assert m.request["questions"]["ambiguity"]["label"] == 1


def test_uninterpretable_request_has_no_training_label():
    # plan-12 T3: "the register could not interpret it" is not trained as parseable = false.
    rec = build_record(load("fullmakt_signatur_rt.json"), None, None)
    assert rec is not None
    q = to_masked_request(rec).request["questions"]
    assert q == {}


def test_ri_with_text_is_uninterpretable_not_statutory():
    """Register RI + free text = the interpreter didn't recognise the text (e.g. a combination
    of two rules). It must not become 'no rule registered' with statutory answers."""
    sig = copy.deepcopy(load("fullmakt_signatur_rf.json"))
    sig["status"]["regelStatus"] = {"kode": "RI", "tekstforklaring": "Regel ikke registrert"}
    sig["signeringsKombinasjon"] = {"kombinasjon": [{"kode": "STYR", "personRolleKombinasjon": []}]}
    sig["signeringsGrunnlag"]["signaturProkuraRoller"]["signaturProkuraFritekst"] = (
        "Daglig leder alene eller styrets leder og ett styremedlem i fellesskap."
    )
    rec = build_record(sig, None, None)
    assert rec is not None and rec.signing.status == "uninterpretable"
    # plan-13: both parts are official rule descriptions -> composed labels, not statutory ones
    m = to_masked_request(rec)
    q = m.request["questions"]
    assert m.request["_meta"]["label_source"] == "register+composition"
    assert q["rule_type"]["label"] == "mixed_alternatives"
    assert q["ceo_alone"]["label"] is True and q["min_signers"]["label"] == "1"
    assert q["parseable"]["label"] is True


def test_ri_with_unknown_wording_stays_unlabelled():
    sig = copy.deepcopy(load("fullmakt_signatur_rf.json"))
    sig["status"]["regelStatus"] = {"kode": "RI", "tekstforklaring": "Regel ikke registrert"}
    sig["signeringsKombinasjon"] = {"kombinasjon": [{"kode": "STYR", "personRolleKombinasjon": []}]}
    sig["signeringsGrunnlag"]["signaturProkuraRoller"]["signaturProkuraFritekst"] = (
        "Daglig leder alene eller etter nærmere avtale med styret."
    )
    rec = build_record(sig, None, None)
    q = to_masked_request(rec).request["questions"]
    assert "parseable" not in q and "rule_type" not in q and "two_ceos" not in q


def test_ri_without_text_stays_statutory():
    sig = copy.deepcopy(load("fullmakt_signatur_rf.json"))
    sig["status"]["regelStatus"] = {"kode": "RI"}
    sig["signeringsKombinasjon"] = {"kombinasjon": [{"kode": "STYR", "personRolleKombinasjon": []}]}
    sig["signeringsGrunnlag"]["signaturProkuraRoller"].pop("signaturProkuraFritekst")
    rec = build_record(sig, None, None)
    q = to_masked_request(rec).request["questions"]
    assert q["rule_type"]["label"] == "no_rule_registered" and q["parseable"]["label"] is True


def test_prokura_text_with_ri_counts_as_present():
    prok = copy.deepcopy(load("fullmakt_prokura_ri.json"))
    prok["signeringsGrunnlag"] = {
        "kode": "SF",
        "signaturProkuraRoller": {"signaturProkuraFritekst": "Fiktiv prokuratekst."},
    }
    rec = build_record(load("fullmakt_signatur_rf.json"), prok, None)
    assert rec is not None and rec.procuration.present is True and rec.procuration.mode is None
