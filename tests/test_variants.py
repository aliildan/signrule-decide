"""The ablation variant deletes fields only: same records, ids, labels and order."""

from __future__ import annotations

import json

from signrule.normalize.variants import drop_state_keys


def test_drop_roles_keeps_everything_else(tmp_path):
    src, dst = tmp_path / "random", tmp_path / "random_noroles"
    src.mkdir()
    rows = [
        {
            "state": {
                "jurisdiction": "NO",
                "signature_rule": "Styrets leder alene.",
                "roles": [{"role": "Styrets leder", "count": 1}],
            },
            "questions": {"chair_alone": {"type": "noul", "label": True}},
            "_meta": {"id": "no/a/1", "group_id": "no/a"},
        },
    ]
    for part in ("train", "val", "test"):
        (src / f"{part}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = drop_state_keys(src, dst, ["roles"])
    assert counts == {"train": 1, "val": 1, "test": 1}
    out = json.loads((dst / "val.jsonl").read_text())
    assert "roles" not in out["state"]
    assert out["state"]["signature_rule"] == "Styrets leder alene."
    assert out["questions"] == rows[0]["questions"]
    assert out["_meta"]["id"] == "no/a/1" and out["_meta"]["variant"] == "drop:roles"
