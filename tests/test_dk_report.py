"""Denmark report helpers on fictitious rows (sampling weights, selective risk)."""

from __future__ import annotations

import importlib.util

spec = importlib.util.spec_from_file_location("dk_report", "eval/dk_report.py")
assert spec is not None and spec.loader is not None
dk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dk)


def _req(rid, wording, roles, gold=False):
    meta = {"id": rid, "stratum": wording}
    if gold:
        meta["strata"] = {"wording": wording}
    return {"_meta": meta, "state": {"roles": [{"role": r, "count": c} for r, c in roles]}}


def test_cell_weights_are_companies_per_drawn_item():
    pool = [_req(f"p{i}", "dk-standard", [("Direktør", 1)]) for i in range(90)]
    pool += [_req(f"t{i}", "dk-tail", [("Direktør", 1)]) for i in range(10)]
    gold = [
        _req("g1", "dk-standard", [("Direktør", 1)], True),
        _req("g2", "dk-tail", [("Direktør", 1)], True),
    ]
    gold.append(_req("g3", "dk-tail", [("Direktør", 1)], True))
    w = dk.cell_weights(pool, gold)
    assert w == {"g1": 90.0, "g2": 5.0, "g3": 5.0}
    rows = [
        {"rid": "g1", "pred": 1, "y": 1},
        {"rid": "g2", "pred": 0, "y": 1},
        {"rid": "g3", "pred": 0, "y": 1},
    ]
    assert dk.accuracy(rows) == 1 / 3
    assert dk.accuracy(rows, w) == 90 / 100


def test_selective_risk_uses_per_type_thresholds_and_never_answers_uncertified_types():
    rows = [
        {"rid": "a", "qtype": "noul", "conf": 0.99, "pred": 1, "y": 1},
        {"rid": "b", "qtype": "noul", "conf": 0.60, "pred": 0, "y": 1},
        {"rid": "c", "qtype": "choice", "conf": 0.99, "pred": 0, "y": 1},
    ]
    s = dk.selective(rows, {"noul": 0.9, "choice": None})
    assert s == {"coverage": 1 / 3, "risk": 0.0, "answered": 1}
