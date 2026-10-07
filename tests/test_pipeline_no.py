"""End-to-end pipeline test on fictitious fixtures in a temporary data dir (plan-03 T8)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "no"


def _envelope(body: dict) -> str:
    return json.dumps({"status_code": 200, "body": body})


def test_pipeline_end_to_end_on_fixtures(tmp_path, monkeypatch):
    from signrule.normalize import pipeline_no as p

    raw = tmp_path / "raw" / "no"
    (raw / "enheter").mkdir(parents=True)
    with gzip.open(raw / "enheter" / "enheter_2026-01-01.jsonl.gz", "wt") as f:
        for orgnr, reg in (("999000001", "2020-01-01"), ("999000002", "2025-06-01")):
            f.write(json.dumps({"orgnr": orgnr, "orgform": "AS", "registered": reg}) + "\n")
    for kind, orgnr, fixture in (
        ("signatur", "999000001", "fullmakt_signatur_rf.json"),
        ("prokura", "999000001", "fullmakt_prokura_ri.json"),
        ("signatur", "999000002", "fullmakt_signatur_rt.json"),
    ):
        d = raw / "fullmakt" / kind / orgnr[:3]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{orgnr}.json").write_text(_envelope(json.loads((FIX / fixture).read_text())))

    monkeypatch.setattr(p, "RAW", raw)
    monkeypatch.setattr(p, "OUT", tmp_path / "processed" / "no")
    monkeypatch.setattr(p, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(p, "require_vault", lambda *a, **k: None)

    assert p.main(["run"]) == 0
    rows = []
    for split_file in (tmp_path / "processed" / "no" / "random").glob("*.jsonl"):
        rows += [json.loads(line) for line in split_file.read_text().splitlines()]
    assert len(rows) == 1  # the RT fixture has no trainable label since plan-12 T3
    blob = json.dumps(rows)
    assert "Fixture Person" not in blob and "fodsel" not in blob
    assert all(r["questions"] and "_meta" in r for r in rows)
    assert p.main(["check"]) == 0
    shutil.rmtree(tmp_path / "processed")
