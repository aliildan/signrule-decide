"""Temperature scaling per question type and Learn-then-Test thresholds (plan-08, design-03 §6).

Fitted on validation only. `AbstainPolicy` is the single object eval and server both use.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom

QTYPES = ("choice", "noul", "score")
ALPHAS = (0.01, 0.02, 0.05)


def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits: list[np.ndarray], labels: list[int]) -> float:
    """T > 0 minimising NLL over items with possibly different option counts."""
    if not labels:
        return 1.0

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for z, y in zip(logits, labels, strict=True):
            p = softmax_t(z, t)
            total -= math.log(max(float(p[y]), 1e-12))
        return total / len(labels)

    res = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))


# Fixed, data-independent candidate thresholds (strict end dense, where decisions are made).
LAMBDA_GRID = tuple(
    [round(0.5 + 0.05 * i, 3) for i in range(9)]  # 0.50 .. 0.90
    + [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
)


def ltt_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> float | None:
    """Learn then Test with Bonferroni over a fixed grid: H0(λ): risk(λ) > α is rejected when the
    binomial-tail p-value of the observed errors among answered items is <= δ/|grid|. Returns the
    lowest rejected λ (maximal coverage with risk <= α at confidence 1-δ, family-wise), or None.

    Bonferroni rather than fixed-sequence testing: at the strict end very few items are answered,
    so a fixed sequence starting there would stop before reaching useful thresholds.
    """
    level = delta / len(grid)
    certified = []
    for lam in grid:
        answered = conf >= lam
        n = int(answered.sum())
        if n == 0:
            continue
        errors = int((~correct[answered]).sum())
        if float(binom.cdf(errors, n, alpha)) <= level:
            certified.append(lam)
    return min(certified) if certified else None


@dataclass
class AbstainPolicy:
    temperatures: dict[str, float] = field(default_factory=lambda: dict.fromkeys(QTYPES, 1.0))
    thresholds: dict[str, dict[str, float | None]] = field(
        default_factory=dict
    )  # alpha -> qtype -> λ
    parseable_gate: float = 0.5
    delta: float = 0.1

    def calibrate(self, qtype: str, logits: np.ndarray) -> np.ndarray:
        return softmax_t(logits, self.temperatures.get(qtype, 1.0))

    def abstain(self, qtype: str, conf: float, alpha: float) -> bool:
        lam = self.thresholds.get(str(alpha), {}).get(qtype)
        return lam is None or conf < lam

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> AbstainPolicy:
        return cls(**json.loads(path.read_text()))


def strictest(policies: list[AbstainPolicy]) -> AbstainPolicy:
    """For a jurisdiction no policy was fitted on: the softest temperature and the highest
    threshold of every policy (a threshold one policy could not certify stays uncertified)."""
    temps: dict[str, float] = {}
    thresholds: dict[str, dict[str, float | None]] = {}
    for pol in policies:
        for qt, t in pol.temperatures.items():
            temps[qt] = max(temps.get(qt, t), t)
        for alpha, per_type in pol.thresholds.items():
            out = thresholds.setdefault(alpha, {})
            for qt, lam in per_type.items():
                if qt in out and (out[qt] is None or lam is None):
                    out[qt] = None
                else:
                    out[qt] = lam if qt not in out else max(out[qt], lam)  # type: ignore[type-var]
    return AbstainPolicy(temperatures=temps, thresholds=thresholds)
