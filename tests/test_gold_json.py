"""Offline JSON annotation round trip on a fictitious batch."""

from __future__ import annotations

import json

import pytest

from signrule.evaluation.gold import annotation_answers
from signrule.evaluation.gold_json import export_file, import_file, merge_file, parse_group


@pytest.fixture
def batch_root(tmp_path, monkeypatch):
    d = tmp_path / "at-text"
    d.mkdir()
    item = {
        "item_id": "fx/1",
        "set": "at-text",
        "state": {
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 01.01.2021 gemeinsam mit einem weiteren "
            "Geschäftsführer",
            "roles": [{"role": "Geschäftsführer", "count": 2}],
        },
    }
    (d / "batch.jsonl").write_text(
        json.dumps(item) + "\n" + json.dumps({**item, "item_id": "fx/2"})
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "annotation_guidelines.md").write_text("# fixture guide\n")
    return tmp_path


def export(root):
    return export_file("at-text", "fixture-ann", root, docs_dir=root / "docs")


def test_parse_group():
    assert parse_group("CEO:1 + PROKURIST:1") == {
        "roles": [["CEO", 1], ["PROKURIST", 1]],
        "collective": None,
    }
    assert parse_group("ALL_BOARD") == {"roles": [], "collective": "ALL_BOARD"}
    with pytest.raises(ValueError):
        parse_group("GESCHAEFTSFUEHRER:1")


def test_export_fill_import_round_trip(batch_root):
    path = export(batch_root)
    doc = json.loads(path.read_text())
    assert [i["show"]["signing_lines"][0][:15] for i in doc["items"]] == ["Geschäftsführer"] * 2
    assert all(i["answer"]["status"] == "" for i in doc["items"])
    doc["items"][0]["answer"].update(
        status="rule",
        alternatives=["CEO:2"],
        person_specific=True,
        procuration_present="no",
        ambiguity=1,
    )
    path.write_text(json.dumps(doc, ensure_ascii=False))
    res = import_file("at-text", "fixture-ann", batch_root)
    assert res == {"valid": 1, "blank": 1, "errors": [], "written": True}
    rec = json.loads((batch_root / "at-text" / "fixture-ann.jsonl").read_text().splitlines()[0])
    assert annotation_answers(rec).get("ceo_alone") is None


def test_import_reports_errors_without_writing(batch_root):
    path = export(batch_root)
    doc = json.loads(path.read_text())
    doc["items"][0]["answer"].update(status="rule", alternatives=["CEO"], ambiguity="1")
    path.write_text(json.dumps(doc, ensure_ascii=False))
    res = import_file("at-text", "fixture-ann", batch_root)
    assert res["written"] is False
    assert res["errors"] == [
        "item 1: alternatives, alternatives (status rule needs at least one group), ambiguity"
    ]
    assert not (batch_root / "at-text" / "fixture-ann.jsonl").exists()


def test_export_refuses_to_overwrite(batch_root):
    export(batch_root)
    with pytest.raises(FileExistsError):
        export(batch_root)


def test_export_needs_the_guides(batch_root):
    with pytest.raises(FileNotFoundError):
        export_file("no-rt", "fixture-ann", batch_root, docs_dir=batch_root / "docs")


@pytest.fixture
def dk_root(tmp_path):
    d = tmp_path / "dk-text"
    d.mkdir()
    item = {
        "item_id": "99999901",
        "set": "dk-text",
        "state": {
            "legal_form": "ApS",
            "signature_rule": "Selskabet tegnes af direktionen.",
            "roles": [{"role": "Direktør", "count": 1}],
        },
    }
    (d / "batch.jsonl").write_text(json.dumps(item) + "\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "annotation_guidelines.md").write_text("# fixture guide\n")
    (docs / "annotation_guide_dk.md").write_text("# fixture dk guide\n")
    return tmp_path


def test_danish_set_ships_its_supplement_and_refuses_executive_member(dk_root):
    path = export_file("dk-text", "fixture-ann", dk_root, docs_dir=dk_root / "docs")
    doc = json.loads(path.read_text())
    assert "annotation_guide_dk.md" in doc["how_to"][0]
    assert (dk_root / "dk-text" / "annotation_guide_dk.md").exists()
    doc["items"][0]["answer"].update(
        status="rule", alternatives=["EXECUTIVE_MEMBER:1"], ambiguity=0
    )
    path.write_text(json.dumps(doc, ensure_ascii=False))
    res = import_file("dk-text", "fixture-ann", dk_root)
    assert res["errors"] == ["item 1: alternatives (use CEO for the Danish direktion)"]
    doc["items"][0]["answer"].update(alternatives=["CEO:1"])
    path.write_text(json.dumps(doc, ensure_ascii=False))
    assert import_file("dk-text", "fixture-ann", dk_root)["written"] is True


def test_merge_copies_answers_by_item_number_only_where_ids_match(batch_root):
    path = export(batch_root)
    retyped = json.loads(path.read_text())
    retyped["items"][0]["answer"].update(status="rule", alternatives=["CEO:2"], ambiguity=1)
    retyped["items"][1]["id"] = "fx/2x"  # a typo made while retyping
    retyped["items"][1]["answer"].update(status="rule", alternatives=["CEO:1"], ambiguity=0)
    filled = batch_root / "at-text" / "filled_fixture-ann.json"
    filled.write_text(json.dumps(retyped, ensure_ascii=False))
    res = merge_file("at-text", "fixture-ann", filled, batch_root)
    assert res == {"copied": 1, "id_mismatch": [2], "missing": []}
    merged = json.loads(path.read_text())
    assert merged["items"][0]["answer"]["alternatives"] == ["CEO:2"]
    assert merged["items"][1]["answer"]["status"] == ""  # not copied
    assert merged["items"][1]["id"] == "fx/2"


def test_merge_accepts_an_answers_only_file(batch_root):
    path = export(batch_root)
    answers = {
        "set": "at-text",
        "answers": [
            {
                "n": 1,
                "id": "fx/1",
                "answer": {"status": "rule", "alternatives": ["CEO:2"], "ambiguity": 1},
            },
            {
                "n": 2,
                "id": "fx/2",
                "answer": {"status": "rule", "alternatives": ["CEO:1"], "ambiguity": 0},
            },
        ],
    }
    filled = batch_root / "at-text" / "filled_fixture-ann.json"
    filled.write_text(json.dumps(answers))
    assert merge_file("at-text", "fixture-ann", filled, batch_root) == {
        "copied": 2,
        "id_mismatch": [],
        "missing": [],
    }
    assert json.loads(path.read_text())["items"][1]["answer"]["alternatives"] == ["CEO:1"]


def test_merge_accepts_a_bare_list_of_answers(batch_root):
    path = export(batch_root)
    filled = batch_root / "at-text" / "filled_fixture-ann.json"
    filled.write_text(
        json.dumps(
            [{"n": 2, "id": "fx/2", "answer": {"status": "rule", "alternatives": ["CEO:1"]}}]
        )
    )
    res = merge_file("at-text", "fixture-ann", filled, batch_root)
    assert res == {"copied": 1, "id_mismatch": [], "missing": [1]}
    assert json.loads(path.read_text())["items"][1]["answer"]["alternatives"] == ["CEO:1"]
