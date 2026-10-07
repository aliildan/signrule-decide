"""Gold -> evaluation split: agreed answers, documented adjudication (fictitious items)."""

from __future__ import annotations

import json

import signrule.evaluation.gold_eval as ge


def _ann(item: str, status: str, groups: list[list], amb: int) -> dict:
    return {
        "item_id": item,
        "status": status,
        "alternatives": [{"roles": groups, "collective": None}] if groups else [],
        "procuration": {"present": "no", "mode": "unknown"},
        "ambiguity": amb,
    }


def test_adjudication_takes_the_chosen_annotator(tmp_path, monkeypatch):
    gold, processed = tmp_path / "gold", tmp_path / "processed"
    d = gold / "fx"
    d.mkdir(parents=True)
    state = {"jurisdiction": "NO", "legal_form": "AS", "signature_rule": "To styremedlemmer."}
    (d / "batch.jsonl").write_text(
        "".join(json.dumps({"item_id": f"fx/{i}", "state": state}) + "\n" for i in (1, 2))
    )
    a = [_ann("fx/1", "uninterpretable", [], 3), _ann("fx/2", "rule", [["BOARD_MEMBER", 2]], 0)]
    b = [
        _ann("fx/1", "rule", [["BOARD_MEMBER", 2]], 1),
        _ann("fx/2", "rule", [["BOARD_MEMBER", 2]], 0),
    ]
    (d / "a.jsonl").write_text("".join(json.dumps(x) + "\n" for x in a))
    (d / "b.jsonl").write_text("".join(json.dumps(x) + "\n" for x in b))
    monkeypatch.setattr(ge, "GOLD_DIR", gold)
    monkeypatch.setattr("signrule.evaluation.gold.GOLD_DIR", gold)
    monkeypatch.setattr(ge, "PROCESSED_DIR", processed)

    stats = ge.build("fx", "a", "b", "no", "fixture", {1: "b"}, "guideline change")
    rows = [
        json.loads(x) for x in (processed / "no" / "gold" / "test.jsonl").read_text().splitlines()
    ]
    assert stats["adjudicated"] == 1 and len(rows) == 2
    first = rows[0]
    assert first["questions"]["min_signers"]["label"] == "2"
    assert first["_meta"]["label_source"].startswith("gold:fx:adjudicated:b")
    assert rows[1]["_meta"]["label_source"] == "gold:fx:a+b:agreed"

    ge.build(
        "fx", "a", "b", "no", "fixture"
    )  # without adjudication: item 1 has no agreed structure
    rows = [
        json.loads(x) for x in (processed / "no" / "gold" / "test.jsonl").read_text().splitlines()
    ]
    assert "min_signers" not in rows[0]["questions"]


def test_danish_gold_uses_the_danish_office_filter():
    state = {"roles": [{"role": "Direktør", "count": 1}]}
    answers = {"ceo_alone": True, "chair_alone": False, "two_board_members_jointly": False}
    assert ge.applicable("dk", state, answers) == {"ceo_alone": True}


def test_several_sets_build_one_split_marked_by_set(tmp_path, monkeypatch):
    gold, processed = tmp_path / "gold", tmp_path / "processed"
    state = {"jurisdiction": "NO", "legal_form": "AS", "signature_rule": "To styremedlemmer."}
    for name in ("fx", "fx2"):
        d = gold / name
        d.mkdir(parents=True)
        (d / "batch.jsonl").write_text(json.dumps({"item_id": f"{name}/1", "state": state}) + "\n")
        ann = [_ann(f"{name}/1", "rule", [["BOARD_MEMBER", 2]], 0)]
        for who in ("a", "b"):
            (d / f"{who}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ann))
    monkeypatch.setattr(ge, "GOLD_DIR", gold)
    monkeypatch.setattr("signrule.evaluation.gold.GOLD_DIR", gold)
    monkeypatch.setattr(ge, "PROCESSED_DIR", processed)
    stats = ge.build("fx", "a", "b", "no", "fixture", also=[("fx2", "a", "b")])
    rows = [
        json.loads(x) for x in (processed / "no" / "gold" / "test.jsonl").read_text().splitlines()
    ]
    assert stats["items"] == 2
    assert [r["_meta"]["stratum"] for r in rows] == ["fx", "fx2"]
    assert rows[1]["_meta"]["label_source"] == "gold:fx2:a+b:agreed"
