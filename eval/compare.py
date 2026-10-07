"""Paired comparison of two scored models (pre-registered H1 decision rule).

    uv run python eval/compare.py --split random --part test --a no-9b-base-v0 --b enc_mmBERT-base

Reads the private item sidecars written by eval/run_all.py (runs/eval/<jur>-<split>/), computes
per-question calibrated accuracy differences (A - B) on the items both scored, and a 95 % paired
bootstrap CI of the mean difference over questions, resampling text clusters. Writes an aggregate
JSON to results/<jur>-<split>/compare/.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from signrule.common.paths import REPO_ROOT


def load(path):  # noqa: ANN001, ANN201
    return {(r["rid"], r["qid"]): r for r in map(json.loads, path.read_text().splitlines())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jurisdiction", default="no")
    ap.add_argument("--split", default="random")
    ap.add_argument("--part", default="test")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-split", help="split of model B if different (ablation variants share ids)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    d = REPO_ROOT / "runs" / "eval" / f"{args.jurisdiction}-{args.split}"
    d_b = REPO_ROOT / "runs" / "eval" / f"{args.jurisdiction}-{args.b_split or args.split}"
    a = load(d / f"{args.a}.{args.part}.items.jsonl")
    b = load(d_b / f"{args.b}.{args.part}.items.jsonl")
    keys = sorted(set(a) & set(b))
    qids = sorted({k[1] for k in keys})
    groups = sorted({a[k]["group"] for k in keys})
    gidx = {g: i for i, g in enumerate(groups)}
    # per (group, question): summed correctness for A and B and counts
    ca = np.zeros((len(groups), len(qids)))
    cb = np.zeros_like(ca)
    n = np.zeros_like(ca)
    for k in keys:
        gi, qi = gidx[a[k]["group"]], qids.index(k[1])
        ca[gi, qi] += a[k]["pred"] == a[k]["y"]
        cb[gi, qi] += b[k]["pred"] == b[k]["y"]
        n[gi, qi] += 1

    def mean_diff(rows: np.ndarray) -> tuple[float, np.ndarray]:
        nn = n[rows].sum(0)
        diff = (ca[rows].sum(0) - cb[rows].sum(0)) / np.maximum(nn, 1)
        valid = nn > 0
        return float(diff[valid].mean()), diff

    point, per_q = mean_diff(np.arange(len(groups)))
    rng = np.random.default_rng(13)
    boots = [mean_diff(rng.integers(0, len(groups), len(groups)))[0] for _ in range(args.n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {
        "a": args.a,
        "b": args.b,
        "part": args.part,
        "split": args.split,
        "n_items": len(keys),
        "n_clusters": len(groups),
        "mean_accuracy_diff": point,
        "ci95": [float(lo), float(hi)],
        "a_beats_b": bool(lo > 0),
        "per_question_diff": {q: float(per_q[i]) for i, q in enumerate(qids)},
    }
    res = REPO_ROOT / "results" / f"{args.jurisdiction}-{args.split}" / "compare"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.a}__vs__{args.b}.{args.part}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{args.a} - {args.b} [{args.part}]: mean acc diff {point:+.4f} "
        f"CI95 [{lo:+.4f}, {hi:+.4f}] "
        f"over {len(qids)} questions, {len(groups)} clusters -> a_beats_b={out['a_beats_b']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
