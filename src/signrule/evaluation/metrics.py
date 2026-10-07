"""Metrics for typed decisions (CLAUDE.md §8, design-03 §5). Pure numpy, no I/O.

Conventions: `probs` is (n, K) per question, `y` the gold option index (n,). Confidence is the
max probability, correctness is argmax == y.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def correct_conf(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (probs.argmax(1) == y), probs.max(1)


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean()) if len(y) else float("nan")


def macro_f1(probs: np.ndarray, y: np.ndarray) -> float:
    """Macro-F1 over classes present in gold or predictions."""
    pred = probs.argmax(1)
    classes = np.union1d(np.unique(y), np.unique(pred))
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean()) if len(y) else float("nan")


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15, mode: str = "width") -> float:
    """Expected calibration error with equal-width or equal-mass bins."""
    n = len(conf)
    if n == 0:
        return float("nan")
    if mode == "width":
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
        groups = [np.where(idx == b)[0] for b in range(bins)]
    elif mode == "mass":
        order = np.argsort(conf, kind="stable")
        groups = [g for g in np.array_split(order, min(bins, n)) if len(g)]
    else:
        raise ValueError(mode)
    total = 0.0
    for g in groups:
        if len(g):
            total += len(g) / n * abs(correct[g].mean() - conf[g].mean())
    return float(total)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk when answering the k most confident items, k = 1..n."""
    order = np.argsort(-conf, kind="stable")
    errs = np.cumsum(~correct[order])
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), errs / k


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    if len(conf) == 0:
        return float("nan")
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Largest coverage whose empirical selective risk is <= alpha (0 if none)."""
    if len(conf) == 0:
        return float("nan")
    cov, risk = risk_coverage(conf, correct)
    ok = np.where(risk <= alpha + 1e-12)[0]
    return float(cov[ok].max()) if len(ok) else 0.0


def mae_expected(probs: np.ndarray, y: np.ndarray) -> float:
    """Score questions: |E[level] - gold|."""
    levels = np.arange(probs.shape[1])
    return float(np.abs(probs @ levels - y).mean()) if len(y) else float("nan")


def qwk(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    """Quadratic-weighted kappa between predicted and gold levels."""
    if len(y) == 0:
        return float("nan")
    o = np.zeros((k, k))
    for a, b in zip(y, pred, strict=True):
        o[a, b] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    denom = (w * e).sum()
    return float(1 - (w * o).sum() / denom) if denom else float("nan")


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    groups: Sequence[str],
    n_boot: int = 1000,
    seed: int = 13,
) -> tuple[float, float]:
    """95 % percentile CI, resampling whole groups (requests/entities) with replacement.

    `stat` receives the item indices of one resample.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        vals.append(stat(idx))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
