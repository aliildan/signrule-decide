"""Tests for the shared register client (src/ingest/http.py) and PII stripping (pii.py)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from signrule.ingest.http import IngestError, RegisterClient
from signrule.ingest.pii import strip_personal_ids

FIXTURES = Path(__file__).parent / "fixtures" / "no"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def make_client(handler, tmp_path: Path, clock: FakeClock | None = None, **kw) -> RegisterClient:
    clock = clock or FakeClock()
    return RegisterClient(
        user_agent="signrule-decide-test/0",
        max_rps=kw.pop("max_rps", 5.0),
        transport=httpx.MockTransport(handler),
        clock=clock.time,
        sleep=clock.sleep,
        **kw,
    )


def fixture_with_birthdates() -> dict:
    body = json.loads((FIXTURES / "fullmakt_signatur_rf.json").read_text())
    for p in body["signeringsGrunnlag"]["muligeSigneringsRoller"]["personRolleGrunnlag"]:
        p["fodselsdato"] = "1970-01-01"
        p["fodselsnummer"] = "0" * 11
    return body


# ---- PII stripping --------------------------------------------------------------------------


def test_strip_removes_birthdate_and_id_keys_recursively():
    body = fixture_with_birthdates()
    clean, removed = strip_personal_ids(body)
    text = json.dumps(clean)
    assert "fodselsdato" not in text and "fodselsnummer" not in text
    assert removed == 2 * 2 + 1  # 2 persons x 2 keys in grunnlag, 1 in kombinasjon
    assert clean["enhet"]["organisasjonsnummer"] == "999000001"  # orgnr is kept
    assert "fodselsdato" in json.dumps(body)  # input not mutated


def test_strip_matches_variants_across_jurisdictions():
    obj = {
        "Fødselsdato": "x",
        "foedselsdato": "x",
        "birthDate": "x",
        "geburtsdatum": "x",
        "isikukood": "x",
        "cprNummer": "x",
        "navn": "kept",
    }
    clean, removed = strip_personal_ids(obj)
    assert clean == {"navn": "kept"} and removed == 6


# ---- client: caching ------------------------------------------------------------------------


def test_get_json_fetches_sanitizes_and_caches(tmp_path):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json=fixture_with_birthdates())

    client = make_client(handler, tmp_path)
    cache = tmp_path / "raw/no/fullmakt/signatur/999/999000001.json"
    r = client.get_json("https://example.test/fullmakt/enheter/999000001/signatur", cache)

    assert r.status_code == 200 and not r.from_cache
    assert calls[0].headers["user-agent"] == "signrule-decide-test/0"
    on_disk = cache.read_text()
    assert "fodselsdato" not in on_disk and "fodselsnummer" not in on_disk
    env = json.loads(on_disk)
    assert env["status_code"] == 200 and env["removed_fields"] == 5
    assert len(env["sha256_response"]) == 64 and env["sanitizer"].startswith("strip_personal_ids")

    r2 = client.get_json("https://example.test/fullmakt/enheter/999000001/signatur", cache)
    assert r2.from_cache and r2.body == r.body
    assert len(calls) == 1  # never re-download what is cached
    assert client.stats.requests == 1 and client.stats.cache_hits == 1


def test_404_is_cached_without_body(tmp_path):
    n = 0

    def handler(req):
        nonlocal n
        n += 1
        return httpx.Response(404, json={"feilmelding": "x"})

    client = make_client(handler, tmp_path)
    cache = tmp_path / "c.json"
    assert client.get_json("https://example.test/a", cache).status_code == 404
    r = client.get_json("https://example.test/a", cache)
    assert r.from_cache and r.body is None and n == 1


def test_other_4xx_raises_without_caching_or_leaking_body(tmp_path):
    def handler(req):
        return httpx.Response(400, text="SECRET-BODY-CONTENT")

    client = make_client(handler, tmp_path)
    cache = tmp_path / "c.json"
    with pytest.raises(IngestError) as exc:
        client.get_json("https://example.test/a", cache)
    assert "SECRET-BODY-CONTENT" not in str(exc.value)
    assert not cache.exists()


# ---- client: rate limit and backoff ---------------------------------------------------------


def test_rate_limit_spaces_requests(tmp_path):
    clock = FakeClock()
    client = make_client(lambda req: httpx.Response(200, json={}), tmp_path, clock, max_rps=5.0)
    for i in range(3):
        client.get_json(f"https://example.test/{i}", tmp_path / f"{i}.json")
    assert clock.now >= 0.4 - 1e-9  # 3 requests at 5 req/s need >= 2 intervals of 0.2 s


def test_retries_on_429_honouring_retry_after(tmp_path):
    clock = FakeClock()
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"ok": 1})]
    )
    client = make_client(lambda req: next(responses), tmp_path, clock)
    r = client.get_json("https://example.test/a", tmp_path / "a.json")
    assert r.body == {"ok": 1}
    assert any(s >= 7 for s in clock.sleeps)
    assert client.stats.retries == 1


def test_retries_on_5xx_then_gives_up(tmp_path):
    clock = FakeClock()
    client = make_client(lambda req: httpx.Response(503), tmp_path, clock, max_retries=3)
    with pytest.raises(IngestError):
        client.get_json("https://example.test/a", tmp_path / "a.json")
    assert client.stats.retries == 3


def test_non_json_200_raises(tmp_path):
    client = make_client(lambda req: httpx.Response(200, text="<html>"), tmp_path)
    with pytest.raises(IngestError):
        client.get_json("https://example.test/a", tmp_path / "a.json")


# ---- value-based scanner (plan-02 T1) -------------------------------------------------------


def test_scanner_flags_birthdate_under_unknown_key():
    from signrule.ingest.pii import scan_for_personal_ids

    hits = scan_for_personal_ids({"x": [{"navn": "Fixture Person", "fdato": "1970-01-01"}]})
    assert hits == ["x[0].fdato"]


def test_scanner_ignores_timestamps_outside_person_objects():
    from signrule.ingest.pii import scan_for_personal_ids

    assert scan_for_personal_ids({"oppslagsTidspunkt": "2026-01-01T00:00:00"}) == []
    assert scan_for_personal_ids({"registrert": "2020-01-02", "orgnr": "999000001"}) == []


def test_scanner_flags_valid_fnr_anywhere():
    from signrule.common.ids import make_fnr
    from signrule.ingest.pii import scan_for_personal_ids

    fnr = next(f for i in range(100, 999) if (f := make_fnr(f"010190{i}")))
    assert scan_for_personal_ids({"a": {"b": fnr}}) == ["a.b"]


def test_get_json_refuses_to_write_when_scanner_hits(tmp_path):
    body = {"personer": [{"navn": "Fixture Person", "fdato": "1970-01-01"}]}
    client = make_client(lambda req: httpx.Response(200, json=body), tmp_path)
    cache = tmp_path / "c.json"
    with pytest.raises(IngestError) as exc:
        client.get_json("https://example.test/a", cache)
    assert "1970-01-01" not in str(exc.value)
    assert not cache.exists()


def test_reserve_slot_is_thread_safe_and_spaced(tmp_path):
    import threading

    client = make_client(lambda req: httpx.Response(200, json={}), tmp_path, max_rps=5.0)
    slots: list[float] = []
    lock = threading.Lock()

    def worker():
        for _ in range(5):
            s = client._reserve_slot(100.0)
            with lock:
                slots.append(s)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    slots.sort()
    gaps = [b - a for a, b in zip(slots, slots[1:], strict=False)]
    assert len(slots) == 40 and min(gaps) >= 0.2 - 1e-9


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    import signrule.ingest.http as h

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(h.os, "replace", interrupted)
    client = make_client(lambda req: httpx.Response(200, json={"a": 1}), tmp_path)
    with pytest.raises(KeyboardInterrupt):
        client.get_json("https://example.test/a", tmp_path / "x" / "c.json")
    assert list((tmp_path / "x").iterdir()) == []


def test_scanner_ignores_dates_on_organisation_objects():
    from signrule.ingest.pii import scan_for_personal_ids

    enhet = {"organisasjonsnummer": "999000001", "navn": "FIXTURE AS", "slettedato": "2024-05-01"}
    assert scan_for_personal_ids({"enhet": enhet}) == []


def test_cache_write_redacts_birth_year_in_free_text(tmp_path):
    body = {"fritekst": "eller styrets leder, f. 1971, hver for seg", "orgnr": "999000001"}
    client = make_client(lambda req: httpx.Response(200, json=body), tmp_path)
    r = client.get_json("https://example.test/a", tmp_path / "c.json")
    cached = json.loads((tmp_path / "c.json").read_text())
    assert "1971" not in json.dumps(cached) and "f. [DATE]" in cached["body"]["fritekst"]
    assert r.body["orgnr"] == "999000001"
