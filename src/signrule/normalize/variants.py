"""Ablation variants of processed splits: same records, fields removed (plan-07 T5).

    python -m signrule.normalize.variants --jurisdiction no --split random --drop roles

Writes data/processed/<jur>/<split>_no<field>/{train,val,test}.jsonl. Nothing is generated: each
line is the original request with the listed state keys deleted; `_meta` (ids, groups) and labels
are unchanged, so comparisons with the full-input model are exactly paired. Variants are for
ablations only, never for a released model. Run `make data-check` afterwards to stamp them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from signrule.common.paths import PROCESSED_DIR


def drop_state_keys(src: Path, dst: Path, keys: list[str]) -> dict[str, int]:
    """Copy every split file from src to dst with `keys` removed from each state."""
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for part in ("train", "val", "test"):
        n = 0
        with (
            (src / f"{part}.jsonl").open(encoding="utf-8") as fin,
            (dst / f"{part}.jsonl").open("w", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                req = json.loads(line)
                if isinstance(req["state"], dict):
                    for k in keys:
                        req["state"].pop(k, None)
                req.setdefault("_meta", {})["variant"] = "drop:" + ",".join(keys)
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                n += 1
        counts[part] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jurisdiction", default="no")
    ap.add_argument("--split", default="random")
    ap.add_argument("--drop", required=True, help="comma-separated state keys to remove")
    a = ap.parse_args(argv)
    keys = [k for k in a.drop.split(",") if k]
    base = PROCESSED_DIR / a.jurisdiction
    name = f"{a.split}_no{''.join(keys)}"
    counts = drop_state_keys(base / a.split, base / name, keys)
    print(f"variant {name}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
