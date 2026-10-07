"""plan16_report.summarise on fictitious sidecar rows."""

from __future__ import annotations

import importlib.util

spec = importlib.util.spec_from_file_location("plan16_report", "eval/plan16_report.py")
assert spec is not None and spec.loader is not None
p16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p16)


def test_dangerous_errors_and_risk_exclude_ambiguity():
    # noul keys are ["false", "true"]: index 1 = "true"
    rows = [
        {"qid": "ceo_alone", "qtype": "noul", "pred": 1, "y": 0, "conf": 0.99},  # dangerous
        {"qid": "two_ceos", "qtype": "noul", "pred": 0, "y": 1, "conf": 0.99},  # cautious
        {"qid": "member_alone", "qtype": "noul", "pred": 1, "y": 1, "conf": 0.40},  # not answered
        {"qid": "ambiguity", "qtype": "score", "pred": 0, "y": 2, "conf": 0.99},  # excluded
    ]
    s = p16.summarise(rows, {"noul": 0.9, "score": 0.5})
    assert s["dangerous"] == 1 and s["yes_no_items"] == 3
    assert s["alpha_2pct"] == {"coverage": 2 / 3, "risk": 1.0}
