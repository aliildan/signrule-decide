"""Risk-coverage curves (aggregate points) -> results/figures/risk_coverage.json (paper figure).

    uv run python scripts/risk_coverage.py

Reads the private item sidecars of eval/run_all.py (runs/eval/<tag>/<model>.test.items.jsonl;
ids, labels and calibrated confidences only) and writes, per curve, the risk of the answered
structural questions (ambiguity excluded: experimental) at fixed coverage levels, answering the
most confident fraction first. Only these aggregate points are published.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CURVES = (
    ("at-c2", "at-random-to-at-gold", "noat-4b-v2", "Austria, SignRule-Decide 4B"),
    ("at-mmbert", "at-random-to-at-gold", "enc_mmBERT-base", "Austria, mmBERT-base"),
    ("dk-c2", "at+no-random-to-dk-gold", "noat-4b-v2", "Denmark (zero-shot), SignRule-Decide 4B"),
)
LEVELS = [i / 20 for i in range(1, 21)]


def curve(rows: list[dict]) -> list[tuple[float, float, float]]:
    """(coverage, risk, aurc contribution) at LEVELS; most confident answers first."""
    ranked = sorted(rows, key=lambda r: -r["conf"])
    wrong = [r["pred"] != r["y"] for r in ranked]
    out = []
    for level in LEVELS:
        k = max(1, round(level * len(ranked)))
        out.append((k / len(ranked), sum(wrong[:k]) / k, 0.0))
    return out


def aurc(rows: list[dict]) -> float:
    ranked = sorted(rows, key=lambda r: -r["conf"])
    errors, total = 0, 0.0
    for i, r in enumerate(ranked, 1):
        errors += r["pred"] != r["y"]
        total += errors / i
    return total / len(ranked)


def main() -> int:
    out = REPO / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    curves = {}
    for key, tag, model, label in CURVES:
        path = REPO / "runs" / "eval" / tag / f"{model}.test.items.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows = [r for r in rows if r["qid"] != "ambiguity"]
        curves[key] = {
            "label": label,
            "questions": len(rows),
            "aurc": round(aurc(rows), 5),
            "points": [
                {"coverage": round(cov, 4), "risk_percent": round(100 * risk, 3)}
                for cov, risk, _ in curve(rows)
            ],
        }
    # JSON only: results/ never holds .csv (publish_check); the paper build writes its CSVs
    (out / "risk_coverage.json").write_text(json.dumps(curves, indent=1) + "\n")
    print(
        json.dumps({k: {"aurc": v["aurc"], "questions": v["questions"]} for k, v in curves.items()})
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
