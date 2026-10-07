"""Tests for the Denmark ingest helpers (credential loading never prints values; the CVR ES
projection on fictitious documents)."""

from __future__ import annotations

import pytest

from signrule.common import secrets as secrets_mod
from signrule.ingest import ingest_dk


@pytest.fixture
def secrets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(secrets_mod, "SECRETS", tmp_path / "secrets")
    monkeypatch.setattr(secrets_mod, "REPO_ROOT", tmp_path)
    (tmp_path / "secrets").mkdir()
    return tmp_path / "secrets"


def test_file_layout(secrets_dir, capsys):
    (secrets_dir / "svc.env").write_text("SVC_API_KEY=fixture-value-1\n")
    assert secrets_mod.load_secret("svc", "SVC_API_KEY") == "fixture-value-1"
    assert "fixture-value-1" not in capsys.readouterr().out


def test_folder_layout_and_alias(secrets_dir, capsys):
    (secrets_dir / "svc").mkdir()
    content = "# comment\nexport API" + "_KEY=" + "'fixture-value-2'\n"  # split: not a real secret
    (secrets_dir / "svc" / ".env").write_text(content)
    assert secrets_mod.load_secret("svc", "SVC_API_KEY") == "fixture-value-2"
    out = capsys.readouterr().out
    assert "API_KEY" in out and "fixture-value-2" not in out


def test_bare_value(secrets_dir):
    (secrets_dir / "svc.env").write_text("fixture-value-3\n")
    assert secrets_mod.load_secret("svc", "SVC_API_KEY") == "fixture-value-3"


def test_ambiguous_entries_refused_without_values(secrets_dir):
    (secrets_dir / "svc.env").write_text("A=fixture-a\nB=fixture-b\n")
    with pytest.raises(SystemExit) as exc:
        secrets_mod.load_secret("svc", "SVC_API_KEY")
    assert "fixture-a" not in str(exc.value) and "['A', 'B']" in str(exc.value)


def test_parse_sdl_types():
    sdl = "type CVR_Virksomhed {\n  CVRNummer: Int\n  status: String\n}\n"
    assert ingest_dk.parse_sdl_types(sdl) == {"CVR_Virksomhed": ["CVRNummer", "status"]}


def _period(start: str, end: str | None = None) -> dict:
    return {"gyldigFra": start, "gyldigTil": end}


def _member(navn: str, enhedstype: str, hovedtype: str, org: str, funktion: str, end=None):
    return {
        "deltager": {
            "enhedstype": enhedstype,
            "navne": [{"navn": navn, "periode": _period("2019-01-01")}],
        },
        "organisationer": [
            {
                "hovedtype": hovedtype,
                "organisationsNavn": [{"navn": org, "periode": _period("2019-01-01")}],
                "medlemsData": [
                    {
                        "attributter": [
                            {
                                "type": "FUNKTION",
                                "vaerdier": [
                                    {"vaerdi": funktion, "periode": _period("2019-01-01", end)}
                                ],
                            },
                            {
                                "type": "EJERANDEL_PROCENT",
                                "vaerdier": [{"vaerdi": "0.5", "periode": _period("2019-01-01")}],
                            },
                        ]
                    }
                ],
            }
        ],
    }


def _doc(**over) -> dict:
    v = {
        "cvrNummer": "99999901",
        "reklamebeskyttet": False,
        "virksomhedMetadata": {
            "sammensatStatus": "Normal",
            "nyesteVirksomhedsform": {"kortBeskrivelse": "APS", "virksomhedsformkode": 80},
        },
        "attributter": [
            {
                "type": "TEGNINGSREGEL",
                "vaerdier": [
                    {
                        "vaerdi": "Selskabet tegnes af en direktør alene.",
                        "periode": _period("2015-01-01", "2019-12-31"),
                    },
                    {
                        "vaerdi": "Selskabet tegnes af direktør Fixture Person alene. "
                        "Kontakt fixture@example.test",
                        "periode": _period("2020-01-01"),
                    },
                ],
            },
            {
                "type": "FORMÅL",
                "vaerdier": [{"vaerdi": "Fixture purpose", "periode": _period("2015-01-01")}],
            },
        ],
        "deltagerRelation": [
            _member("Fixture Person", "PERSON", "LEDELSESORGAN", "Direktion", "DIREKTØR"),
            _member(
                "Former Fixture",
                "PERSON",
                "LEDELSESORGAN",
                "Bestyrelse",
                "FORMAND",
                end="2021-06-30",
            ),
            _member(
                "Fixture Holding ApS",
                "VIRKSOMHED",
                "LEDELSESORGAN",
                "Bestyrelse",
                "BESTYRELSESMEDLEM",
            ),
            _member("Owner Fixture", "PERSON", "REGISTER", "EJERREGISTER", "EJERREGISTER"),
            _member("Founder Fixture", "PERSON", "STIFTERE", "Stiftere", "STIFTERE"),
            _member("Auditor Fixture", "VIRKSOMHED", "REVISION", "Revision", "REVISION"),
        ],
        **over,
    }
    return {"Vrvirksomhed": v}


def test_projection_keeps_rule_texts_with_periods_and_current_management_only():
    out, n_redacted = ingest_dk.project_dk_company(_doc())
    assert out["cvr"] == "99999901" and out["legal_form"] == "APS" and out["formkode"] == 80
    assert out["status"] == "Normal"
    assert [t["to"] for t in out["rule_texts"]] == ["2019-12-31", None]
    assert out["rule_texts"][-1]["text"].endswith("Kontakt [EMAIL]") and n_redacted == 1
    assert [(f["body"], f["function"], f["enhedstype"]) for f in out["functions"]] == [
        ("Direktion", "DIREKTØR", "PERSON"),
        ("Bestyrelse", "BESTYRELSESMEDLEM", "VIRKSOMHED"),
    ]
    assert out["mask_names"] == ["Fixture Person"]
    assert out["firm_names"] == ["Fixture Holding ApS"]
    assert out["functions"][0]["person"] == 0 and out["functions"][1]["firm"] == 0


def test_projection_drops_owners_founders_auditors_and_other_attributes():
    out, _ = ingest_dk.project_dk_company(_doc())
    flat = repr(out)
    for absent in (
        "Owner Fixture",
        "Founder Fixture",
        "Auditor Fixture",
        "Former Fixture",
        "EJERANDEL",
        "Fixture purpose",
    ):
        assert absent not in flat


def test_advertising_protected_companies_are_dropped():
    out, _ = ingest_dk.project_dk_company(_doc(reklamebeskyttet=True))
    assert out is None


def test_partnership_partners_count_as_management():
    doc = _doc(
        deltagerRelation=[
            _member(
                "Partner Fixture",
                "PERSON",
                "FULDT_ANSVARLIG_DELTAGERE",
                "Interessenter",
                "INTERESSENTER",
            )
        ]
    )
    out, _ = ingest_dk.project_dk_company(doc)
    assert [(f["hovedtype"], f["function"]) for f in out["functions"]] == [
        ("FULDT_ANSVARLIG_DELTAGERE", "INTERESSENTER")
    ]


def test_other_participants_are_masked_as_persons():
    doc = _doc(
        deltagerRelation=[
            _member("Foreign Fixture", "ANDEN_DELTAGER", "LEDELSESORGAN", "Bestyrelse", "FORMAND")
        ]
    )
    out, _ = ingest_dk.project_dk_company(doc)
    assert out["mask_names"] == ["Foreign Fixture"] and out["firm_names"] == []
    assert out["functions"][0]["person"] == 0
