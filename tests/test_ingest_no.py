"""Tests for the Norway ingester's pure helpers (src/ingest/ingest_no.py)."""

from __future__ import annotations

import json
from pathlib import Path

from signrule.ingest.ingest_no import (
    extract_fullmakt,
    key_paths,
    load_scope,
    project_entity,
    summarize,
)

FIXTURES = Path(__file__).parent / "fixtures" / "no"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_scope_tiers_and_exclusions():
    scope = load_scope()
    assert scope["AS"] == 1 and scope["ASA"] == 1
    assert "ENK" not in scope and "NUF" not in scope


def test_project_entity_keeps_only_whitelisted_fields():
    e = {
        "organisasjonsnummer": "999000001",
        "navn": "FIXTURE EKSEMPEL AS",
        "organisasjonsform": {"kode": "AS", "beskrivelse": "Aksjeselskap"},
        "registreringsdatoEnhetsregisteret": "2020-01-02",
        "stiftelsesdato": "2019-12-01",
        "konkurs": False,
        "underAvvikling": False,
        "underTvangsavviklingEllerTvangsopplosning": False,
        "naeringskode1": {"kode": "62.010", "beskrivelse": "x"},
        "forretningsadresse": {"adresse": ["Fiktiv gate 1"]},
    }
    p = project_entity(e)
    assert p == {
        "orgnr": "999000001",
        "orgform": "AS",
        "registered": "2020-01-02",
        "founded": "2019-12-01",
        "konkurs": False,
        "under_avvikling": False,
        "under_tvangsavvikling": False,
        "deleted": None,
        "nace": "62.010",
    }


def test_extract_rf():
    x = extract_fullmakt(load("fullmakt_signatur_rf.json"))
    assert x.orgnr == "999000001" and x.orgform == "AS"
    assert (x.rutine, x.regel, x.regel_ident) == ("OK", "RF", "R0002")
    assert x.kombinasjon_codes == ("R0002",)
    assert x.fritekst == "Daglig leder alene."
    assert x.mulige_roller == {"DAGL": 1, "LEDE": 1}


def test_extract_rt_ri_na():
    rt = extract_fullmakt(load("fullmakt_signatur_rt.json"))
    assert rt.regel == "RT" and rt.kombinasjon_status == "IK" and rt.fritekst
    ri = extract_fullmakt(load("fullmakt_prokura_ri.json"))
    assert ri.regel == "RI" and ri.grunnlag == "SI" and ri.fritekst is None
    na = extract_fullmakt(load("fullmakt_signatur_na.json"))
    assert na.rutine == "NA" and na.regel is None


def test_summarize_counts_codes_not_text():
    rows = [
        extract_fullmakt(load(n))
        for n in [
            "fullmakt_signatur_rf.json",
            "fullmakt_signatur_rt.json",
            "fullmakt_signatur_na.json",
        ]
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["regel"] == {"RF": 1, "RT": 1, None: 1}
    assert s["has_fritekst"] == {True: 2, False: 1}
    assert "Daglig leder alene." not in json.dumps(s, default=str)


def test_key_paths_reports_structure_without_values():
    obj = [{"a": {"kode": "X", "navn": "Fixture Person"}, "b": [1, 2]}]
    paths = key_paths(obj)
    assert paths == {"[]": 1, "[].a": 1, "[].a.kode": 1, "[].a.navn": 1, "[].b": 1, "[].b[]": 2}


def test_commands_require_vault(monkeypatch):
    import pytest

    from signrule.common.paths import VaultNotMounted
    from signrule.ingest import ingest_no

    def boom(*a, **k):
        raise VaultNotMounted("not mounted")

    def no_client():
        raise AssertionError("client must not be created before the vault check")

    monkeypatch.setattr(ingest_no, "require_vault", boom)
    monkeypatch.setattr(ingest_no, "make_client", no_client)
    for cmd in (["entities"], ["fullmakt", "--limit", "1"]):
        with pytest.raises(VaultNotMounted):
            ingest_no.main(cmd)


def test_select_orgnrs_shuffles_within_tiers(tmp_path, monkeypatch):
    from signrule.ingest import ingest_no

    ents = [
        {"orgnr": f"9{i:08d}", "orgform": "AS" if i % 3 else "ANS", "deleted": None}
        for i in range(60)
    ]
    monkeypatch.setattr(ingest_no, "latest_entities_snapshot", lambda: tmp_path)
    monkeypatch.setattr(ingest_no, "iter_entities", lambda p: iter(ents))
    scope = {"AS": 1, "ANS": 2}
    a = ingest_no.select_orgnrs(scope, None, None, seed=1)
    b = ingest_no.select_orgnrs(scope, None, None, seed=1)
    tier1 = [o for o in a if int(o) % 3]
    assert a == b  # deterministic
    assert a[: len(tier1)] == tier1  # tier 1 first
    assert tier1 != sorted(tier1)  # shuffled, not orgnr order
    t2 = ingest_no.select_orgnrs(scope, None, None, seed=1, tiers={2})
    assert set(t2) == set(a[len(tier1) :])
