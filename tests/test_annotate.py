"""Tests for the local annotation tool on a fictitious batch (plan-05 T5)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("annotate_app", ROOT / "eval/annotate/app.py")
assert _spec and _spec.loader
app_mod = importlib.util.module_from_spec(_spec)
sys.modules["annotate_app"] = app_mod
_spec.loader.exec_module(app_mod)


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    d = tmp_path / "no-rt"
    d.mkdir()
    items = [
        {
            "item_id": "fx/1",
            "set": "no-rt",
            "state": {
                "jurisdiction": "NO",
                "legal_form": "AS",
                "signature_rule": "To styremedlemmer i fellesskap <b>x</b>.",
                "roles": [{"role": "Styrets leder", "count": 1}],
            },
        },
        {
            "item_id": "fx/2",
            "set": "no-rt",
            "state": {"jurisdiction": "NO", "legal_form": "AS", "signature_rule": "Fiktiv tekst."},
        },
    ]
    (d / "batch.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    return TestClient(app_mod.create_app("no-rt", "fixture-ann", tmp_path)), d


def test_page_shows_escaped_state(client):
    c, _ = client
    page = c.get("/item/0").text
    assert "To styremedlemmer i fellesskap &lt;b&gt;x&lt;/b&gt;." in page
    assert "Styrets leder" in page and "item 1/2" in page


def test_save_and_skip(client):
    c, d = client
    form = {
        "status": "uninterpretable",
        "prok_present": "no",
        "prok_mode": "unknown",
        "ambiguity": "3",
        "note": "only one board member",
        "action": "save",
    }
    r = c.post("/item/0", data=form, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/item/1"
    c.post("/item/1", data={"action": "skip"})
    lines = [json.loads(x) for x in (d / "fixture-ann.jsonl").read_text().splitlines()]
    assert lines[0]["status"] == "uninterpretable" and lines[0]["ambiguity"] == 3
    assert lines[1] == {**lines[1], "item_id": "fx/2", "skipped": True}


def test_rule_needs_groups_and_groups_parse(client):
    c, d = client
    bad = c.post("/item/0", data={"status": "rule", "ambiguity": "0", "action": "save"})
    assert bad.status_code == 422
    ok = {
        "status": "rule",
        "ambiguity": "0",
        "action": "save",
        "prok_present": "no",
        "g0_r0_role": "CHAIR",
        "g0_r0_count": "1",
        "g1_r0_role": "BOARD_MEMBER",
        "g1_r0_count": "2",
        "g2_collective": "ALL_BOARD",
    }
    c.post("/item/0", data=ok)
    rec = json.loads((d / "fixture-ann.jsonl").read_text().splitlines()[-1])
    assert rec["alternatives"] == [
        {"roles": [["CHAIR", 1]], "collective": None},
        {"roles": [["BOARD_MEMBER", 2]], "collective": None},
        {"roles": [], "collective": "ALL_BOARD"},
    ]


def test_austrian_lines_and_person_specific_flag(tmp_path):
    from fastapi.testclient import TestClient

    d = tmp_path / "at-text"
    d.mkdir()
    item = {
        "item_id": "fx/at",
        "set": "at-text",
        "state": {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 01.01.2021 gemeinsam mit einem "
            "weiteren Geschäftsführer",
        },
    }
    (d / "batch.jsonl").write_text(json.dumps(item) + "\n")
    c = TestClient(app_mod.create_app("at-text", "fixture-ann", tmp_path))
    page = c.get("/item/0").text
    assert page.count("<li>Geschäftsführer [PERSON_") == 2 and 'name="person_specific"' in page
    form = {
        "action": "save",
        "status": "rule",
        "g0_r0_role": "CEO",
        "g0_r0_count": "2",
        "person_specific": "on",
        "prok_present": "no",
        "ambiguity": "1",
    }
    c.post("/item/0", data=form, follow_redirects=False)
    saved = [json.loads(x) for x in (d / "fixture-ann.jsonl").read_text().splitlines()]
    assert saved[-1]["person_specific"] is True
