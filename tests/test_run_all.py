"""run_all: policy for a jurisdiction no model was calibrated on (strictest merge)."""

from __future__ import annotations

import importlib.util

import numpy as np

from signrule.calib.calibration import strictest
from signrule.evaluation.harness import Item, fit_policy

spec = importlib.util.spec_from_file_location("run_all", "eval/run_all.py")
assert spec is not None and spec.loader is not None
run_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all)


class Stub:
    name = "stub"

    def __init__(self, logits):
        self.logits = logits

    def predict(self, reqs):
        return {k: v for k, v in self.logits.items() if k[0] in {r["_meta"]["id"] for r in reqs}}


def _items(prefix: str, n: int, sharp: float):
    items, logits, reqs = [], {}, []
    rng = np.random.default_rng(0)
    for i in range(n):
        rid = f"{prefix}{i}"
        y = int(rng.integers(0, 2))
        wrong = i % 10 == 0  # 10 % errors, all confident
        z = np.array([sharp, 0.0]) if (y == 0) != wrong else np.array([0.0, sharp])
        items.append(Item(rid, rid, "ceo_alone", "noul", y, True))
        logits[(rid, "ceo_alone")] = z
        reqs.append({"_meta": {"id": rid}})
    return reqs, items, logits


def test_calibration_policy_is_the_strictest_merge_of_every_calibration_set():
    no_reqs, no_items, no_logits = _items("no", 400, 6.0)
    at_reqs, at_items, at_logits = _items("at", 400, 2.0)
    pred = Stub({**no_logits, **at_logits})
    merged = run_all.calibration_policy(pred, [(no_reqs, no_items), (at_reqs, at_items)])
    expected = strictest(
        [fit_policy(no_items, pred.predict(no_reqs)), fit_policy(at_items, pred.predict(at_reqs))]
    )
    assert merged.temperatures == expected.temperatures
    assert merged.thresholds == expected.thresholds
    single = run_all.calibration_policy(pred, [(no_reqs, no_items)])
    assert single.thresholds == fit_policy(no_items, pred.predict(no_reqs)).thresholds


def test_kev_spec_accepts_extra_calibration_benchmarks(monkeypatch):
    seen = {}

    def fake(name, files, qc):
        seen[name] = [str(f) for f in files]
        return name

    monkeypatch.setattr(run_all, "KevPredictions", fake)
    args = type(
        "A", (), {"baselines": "", "kev": ["m=runs/m-at-val,runs/m-dk-test,runs/m-no-val"]}
    )()
    assert run_all.build_predictors(args, [], None) == ["m"]
    assert seen["m"] == [
        "runs/m-at-val/predictions.jsonl",
        "runs/m-dk-test/predictions.jsonl",
        "runs/m-no-val/predictions.jsonl",
    ]


def test_encoder_spec_takes_an_optional_max_length(monkeypatch):
    import signrule.evaluation.encoders as enc

    made = []

    class Fake:
        def __init__(self, name, train, qc, **kw):
            made.append((name, kw))
            self.name = name

    monkeypatch.setattr(enc, "EncoderPredictor", Fake)
    specs = "enc:jhu-clsp/mmBERT-base@1024,enc:x/y,enc:a/b@512/8"
    args = type("A", (), {"baselines": specs, "kev": []})()
    run_all.build_predictors(args, [], None)
    assert made == [
        ("jhu-clsp/mmBERT-base", {"max_len": 1024}),
        ("x/y", {}),
        ("a/b", {"max_len": 512, "batch": 8}),
    ]
