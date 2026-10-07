"""Tests for metrics (plan-06 T1) and calibration (plan-08 T1-T2) on hand-computed arrays."""

from __future__ import annotations

import numpy as np
import pytest

from signrule.calib.calibration import AbstainPolicy, fit_temperature, ltt_threshold, softmax_t
from signrule.evaluation import metrics as m


def test_accuracy_brier_nll():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    assert m.accuracy(p, y) == pytest.approx(2 / 3)
    assert m.brier(p, y) == pytest.approx(((0.1**2 * 2) + (0.2**2 * 2) + (0.6**2 * 2)) / 3)
    assert m.nll(p, y) == pytest.approx(-(np.log(0.9) + np.log(0.8) + np.log(0.4)) / 3)


def test_macro_f1_two_classes():
    p = np.eye(2)[[0, 0, 1, 1]]
    y = np.array([0, 1, 1, 1])
    # class0: tp1 fp1 fn0 -> 2/3; class1: tp2 fp0 fn1 -> 4/5
    assert m.macro_f1(p, y) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_ece_perfect_and_overconfident():
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    assert m.ece(conf, np.array([True] * 4)) == 0.0
    assert m.ece(conf, np.array([True, False, True, False])) == pytest.approx(0.5)
    assert m.ece(conf, np.array([True, False, True, False]), mode="mass") == pytest.approx(0.5)


def test_risk_coverage_and_coverage_at_risk():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    cov, risk = m.risk_coverage(conf, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(risk) == pytest.approx([0, 0, 1 / 3, 0.25])
    assert m.coverage_at_risk(conf, correct, 0.0) == 0.5
    assert m.coverage_at_risk(conf, correct, 0.25) == 1.0
    assert m.aurc(conf, correct) == pytest.approx((0 + 0 + 1 / 3 + 0.25) / 4)


def test_qwk_perfect_and_mae():
    y = np.array([0, 1, 2, 3])
    assert m.qwk(y, y, 4) == pytest.approx(1.0)
    p = np.eye(4)[y]
    assert m.mae_expected(p, y) == 0.0


def test_bootstrap_ci_contains_point():
    correct = np.array([1, 1, 0, 1, 1, 0, 1, 1] * 5, dtype=bool)
    groups = [f"g{i // 2}" for i in range(len(correct))]
    lo, hi = m.bootstrap_ci(lambda idx: correct[idx].mean(), groups, n_boot=300)
    assert lo <= correct.mean() <= hi


def test_fit_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    logits, labels = [], []
    for _ in range(400):  # true accuracy 70 %, but logits claim ~99 %
        y = int(rng.random() < 0.3)
        logits.append(np.array([5.0, 0.0]))
        labels.append(y)
    t = fit_temperature(logits, labels)
    assert t > 3
    p = softmax_t(np.array([5.0, 0.0]), t)
    assert p[0] == pytest.approx(np.mean(np.array(labels) == 0), abs=0.005)


def test_ltt_threshold_cases():
    conf = np.linspace(0.5, 1.0, 1000)
    assert ltt_threshold(conf, np.ones(1000, dtype=bool), alpha=0.05) == pytest.approx(0.5)
    bad = np.zeros(1000, dtype=bool)
    assert ltt_threshold(conf, bad, alpha=0.05) is None
    tiny = np.array([0.99, 0.98])  # two items can't certify 1 % risk at delta 0.1
    assert ltt_threshold(tiny, np.array([True, True]), alpha=0.01) is None


def test_policy_roundtrip_and_ties(tmp_path):
    pol = AbstainPolicy(thresholds={"0.05": {"noul": 0.8, "choice": None}})
    assert not pol.abstain("noul", 0.8, 0.05)  # max_prob >= λ answers
    assert pol.abstain("noul", 0.79, 0.05)
    assert pol.abstain("choice", 0.99, 0.05)  # uncertifiable -> always abstain
    pol.save(tmp_path / "p.json")
    assert AbstainPolicy.load(tmp_path / "p.json") == pol
