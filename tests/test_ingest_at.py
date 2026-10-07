"""Tests for the Austrian HVD ingester on fictitious fixtures (plan-05 T2)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from signrule.ingest import ingest_at as at
from signrule.ingest.http import RegisterClient

FIX = Path(__file__).parent / "fixtures" / "at"


def test_soap_envelope_fields_and_escaping():
    body = at.auszug_request("999999 x", "2026-01-01").decode()
    assert "<aus:FNR>999999 x</aus:FNR>" in body
    assert "<aus:UMFANG>Kurzinformation</aus:UMFANG>" in body
    assert 'xmlns:aus="ns://firmenbuch.justiz.gv.at/Abfrage/v2/AuszugRequest"' in body
    v = at.veraenderungen_request("2026-01-02", "2026-01-02", rechtsform="A&B").decode()
    assert "<ver:RECHTSFORM>A&amp;B</ver:RECHTSFORM>" in v


def test_parse_strips_birthdates_and_addresses():
    obj = at.parse_soap_bytes((FIX / "auszug_ges.xml").read_bytes())
    clean = at.drop_person_addresses(obj)
    blob = json.dumps(clean, ensure_ascii=False)
    assert "PE_DKZ03" not in blob and "Fiktivgasse" not in blob
    assert "AUSZUG_V2_RESPONSE" in clean


def test_extract_current_functions_and_names():
    obj = at.drop_person_addresses(at.parse_soap_bytes((FIX / "auszug_ges.xml").read_bytes()))
    rec = at.extract_at(obj)
    assert rec.fnr == "999999x" and rec.rechtsform_code == "GES"
    assert [(f.pnr, f.fken, f.vart_code) for f in rec.functions] == [
        ("A", "GF", "E"),
        ("B", "PR", "G"),
    ]
    assert rec.functions[1].txtvertr == ["einem Geschäfts-", "führer oder Fixture Person A"]
    assert rec.person_names == {"A": "Fixture Person A", "B": "Fixture Person B"}
    assert rec.functions[0].text == ["vertritt seit 01.01.2020 selbständig"]


def test_frame_fnrs_deduplicated():
    obj = at.parse_soap_bytes((FIX / "veraenderungen.xml").read_bytes())
    assert at.frame_fnrs(obj) == ["888888y", "999999x"]


def test_fetch_sends_key_header_and_caches_without_birthdates(tmp_path):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, content=(FIX / "auszug_ges.xml").read_bytes())

    client = RegisterClient(
        user_agent="t",
        max_rps=5,
        transport=httpx.MockTransport(handler),
        clock=lambda: 0.0,
        sleep=lambda s: None,
    )
    cache = tmp_path / "auszug" / "999999x.json"
    r = at.fetch_auszug(client, "fixture-key", "999999 x", "2026-01-01", cache)
    assert r.status_code == 200
    assert seen[0].headers["X-API-KEY"] == "fixture-key"
    assert seen[0].headers["Content-Type"].startswith("application/soap+xml")
    stored = cache.read_text(encoding="utf-8")
    assert "FIXTURE-DATE" not in stored and "Fiktivgasse" not in stored
    assert "fixture-key" not in stored
    assert json.loads(stored)["params"] == {"operation": "AUSZUG_V2", "fnr": "999999x"}


def _sole_trader_obj():
    obj = at.drop_person_addresses(at.parse_soap_bytes((FIX / "auszug_ges.xml").read_bytes()))
    resp = obj["AUSZUG_V2_RESPONSE"]
    for e in resp["FIRMA"][0]["FI_DKZ07"]:
        e["RECHTSFORM"] = [{"CODE": "EU", "TEXT": "Einzelunternehmer"}]
    return obj


def test_sanitize_replaces_sole_trader_by_stub():
    clean = at.sanitize(_sole_trader_obj())
    assert clean == {"AUSZUG_V2_RESPONSE": {"@FNR": "999999x", "@SKIPPED": "sole_trader"}}
    rec = at.extract_at(clean)
    assert rec.rechtsform_code == "EU" and rec.functions == [] and rec.person_names == {}
    ges = at.drop_person_addresses(at.parse_soap_bytes((FIX / "auszug_ges.xml").read_bytes()))
    assert at.sanitize(ges) == ges


def test_scrub_rewrites_cached_sole_trader(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"body": _sole_trader_obj(), "status_code": 200}))
    assert at.scrub_cached(path) is True
    env = json.loads(path.read_text())
    assert "Fixture Person" not in json.dumps(env) and "scrubbed_at" in env
    assert at.scrub_cached(path) is False
