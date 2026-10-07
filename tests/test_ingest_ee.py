"""Estonia ingest: projection, redaction and aggregate stats on fictitious records (fixtures)."""

from __future__ import annotations

from signrule.common.ids import isikukood_valid, make_isikukood
from signrule.ingest.ingest_ee import project_company, stats
from signrule.ingest.pii import redact_text, scan_for_personal_ids


def _code() -> str:
    return make_isikukood("3" + "800101" + "123")


def _record(**over):
    rec = {
        "ariregistri_kood": "10000001",
        "nimi": "Fixture OÜ",
        "esindusoiguse_eritingimused": [
            {
                "esinduse_sisu": f"Juhatuse liikmed esindavad ühiselt. Fixture Isik ({_code()})",
                "esinduse_tyyp": "YHESJ",
                "algus_kpv": "01.02.2020",
                "lopp_kpv": None,
                "kande_nr": "3",
            }
        ],
        "esindusoiguse_normaalregulatsioonid": [{"roll": "JUHL", "sisu": "Iga juhatuse liige"}],
        "hooneyhistu_liikmed": [{"nimi": "Fixture Member"}],
        "kaardile_kantud_isikud": [
            {
                "isiku_roll": "JUHL",
                "isiku_tyyp": "F",
                "eesnimi": "Fixture",
                "nimi_arinimi": "Isik",
                "synniaeg": "01.01.1980",
                "isikukood_hash": "0f0f0f0f",
                "email": "fixture@example.test",
                "aadress_ehak": "0784",
                "valis_kood": "40000000000",
                "algus_kpv": "01.02.2020",
            }
        ],
    }
    rec.update(over)
    return rec


def test_isikukood_check_digit():
    code = _code()
    assert isikukood_valid(code)
    assert not isikukood_valid(code[:-1] + str((int(code[-1]) + 1) % 10))
    assert not isikukood_valid("38013010000")  # month 13


def test_projection_keeps_only_whitelisted_fields():
    proj, redacted = project_company(_record())
    assert proj is not None and redacted == 1
    person = proj["persons"][0]
    assert person == {"isiku_roll": "JUHL", "isiku_tyyp": "F", "algus_kpv": "01.02.2020"}
    assert proj["mask_names"] == ["Fixture Isik"]
    assert "hooneyhistu_liikmed" not in proj
    special = proj["representation_special"][0]
    assert special["esinduse_tyyp"] == "YHESJ" and "kande_nr" not in special
    assert _code() not in special["text"] and "[ID]" in special["text"]
    assert scan_for_personal_ids(proj) == []
    flat = str(proj)
    for leaked in ("01.01.1980", "0f0f0f0f", "fixture@example.test", "40000000000"):
        assert leaked not in flat


def test_sole_traders_are_dropped():
    rec = _record()
    rec["kaardile_kantud_isikud"][0]["isiku_roll"] = "FIE"
    assert project_company(rec) == (None, 0)


def test_redact_text_birth_year_after_marker():
    text, n = redact_text("gemeinsam mit dem Prokuristen [PERSON_4], geb. 1971, oder")
    assert n == 1 and "1971" not in text and "geb. [DATE]" in text
    kept = "vertritt seit 01.01.2020 selbständig; Kapital EUR 35000"
    assert redact_text(kept) == (kept, 0)


def test_redact_text_birth_dates_and_email():
    text, n = redact_text("Prokurist Fixture (s. 01.01.1980), fixture@example.test")
    assert n == 2 and "1980" not in text and "@" not in text
    unchanged = "Tehingud üle 10000 euro 12345678901"  # not a valid ID code
    assert redact_text(unchanged) == (unchanged, 0)


def test_api_personal_fields_are_stripped_by_key():
    from signrule.ingest.pii import strip_personal_ids

    person = {
        "fyysilise_isiku_eesnimi": "Fixture",
        "fyysilise_isiku_kood": _code(),
        "fyysilise_isiku_synniaeg": "1980-01-01",
        "isikukood_hash": "0f0f",
        "ainuesindusoigus_olemas": "JAH",
    }
    clean, removed = strip_personal_ids({"isikud": [person]})
    assert removed == 3
    assert clean["isikud"][0] == {
        "fyysilise_isiku_eesnimi": "Fixture",
        "ainuesindusoigus_olemas": "JAH",
    }


def test_stats_are_aggregate_only():
    proj, _ = project_company(_record())
    basic = [{"ariregistri_kood": "10000001", "ettevotja_oiguslik_vorm": "Osaühing"}]
    s = stats([proj], basic)
    assert s["companies_with_active_special_rule"] == 1
    assert s["by_legal_form"]["Osaühing"] == {"companies": 1, "with_special_rule": 1}
    assert s["special_rule_types"]["YHESJ"]["n"] == 1
    assert s["special_texts_naming_a_card_person"] == 1
    flat = str(s)
    assert "Fixture" not in flat and "10000001" not in flat
