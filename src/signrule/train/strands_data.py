"""Processed requests -> Strands Decider training examples (plan-17 T2, design-05).

    uv run python -m signrule.train.strands_data build --name strands-4b-v1 \
        --jurisdictions no at --split random

One `strands_decider.data.format.Example` per labelled question: the request's state (canonical
key order), the question's instructions and its options exactly as Ollama's Strands renderer
builds them from the same request (noul defaults from `NOUL_DEFAULT_CRITERIA`), the label as an
index into the canonical option order. Masked questions are absent from the request, so they yield
no example. Writes `runs/<name>/data/{train,val}.jsonl` and `counts.json` (aggregates only); the
inputs must carry a valid data-check stamp. Nothing here imports strands_decider.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from signrule.common.paths import PROCESSED_DIR
from signrule.format.request import STATE_KEYS
from signrule.train.kev_wrapper import RUNS_DIR, verify_data_check

# Ollama decision/strands.go and strands_decider.prompting use these when a noul has no criteria.
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}


def canonical_state(state: dict[str, Any]) -> dict[str, Any]:
    """The state in STATE_KEYS order (the JSON rendering keeps key order)."""
    unknown = set(state) - set(STATE_KEYS)
    if unknown:
        raise ValueError(f"unknown state key(s) {sorted(unknown)}")
    return {k: state[k] for k in STATE_KEYS if k in state}


def options_for(q: dict[str, Any]) -> list[list[str]]:
    """[[name, description], ...] in canonical order, as the served model reads them."""
    if q["type"] == "noul":
        crit = {**NOUL_DEFAULT_CRITERIA, **(q.get("criteria") or {})}
        return [["false", crit["false"]], ["true", crit["true"]]]
    if q["type"] == "choice":
        return [[str(k), v or ""] for k, v in q["criteria"].items()]
    if q["type"] == "score":
        return [[str(i), v] for i, v in enumerate(q["criteria"])]
    raise ValueError(f"unknown question type {q['type']!r}")


def label_index(q: dict[str, Any], options: list[list[str]]) -> int:
    label = q["label"]
    if q["type"] == "noul":
        if not isinstance(label, bool):
            raise TypeError("noul label must be bool")
        return int(label)
    if q["type"] == "choice":
        return [name for name, _ in options].index(str(label))
    return int(label)


def to_examples(request: dict[str, Any]) -> list[dict[str, Any]]:
    """One Strands Example (as a dict) per labelled question of a processed request."""
    state = canonical_state(request["state"])
    jur = str(state.get("jurisdiction", "")).lower()
    out = []
    for qid, q in request["questions"].items():
        options = options_for(q)
        out.append(
            {
                "kind": q["type"],
                "state": state,
                "instructions": q["instructions"],
                "options": options,
                "label": label_index(q, options),
                "task": f"{jur}:{qid}",
                "weight": 1.0,
            }
        )
    return out


def build(name: str, jurisdictions: list[str], split: str) -> dict[str, Any]:
    out_dir = RUNS_DIR / name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {"jurisdictions": jurisdictions, "split": split, "parts": {}}
    for jur in jurisdictions:
        verify_data_check(jur)
    for part in ("train", "val"):
        records, by_task, kinds = 0, Counter(), Counter()
        with (out_dir / f"{part}.jsonl").open("w", encoding="utf-8") as f:
            for jur in jurisdictions:
                src = PROCESSED_DIR / jur / split / f"{part}.jsonl"
                for line in src.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    records += 1
                    for ex in to_examples(json.loads(line)):
                        by_task[ex["task"]] += 1
                        kinds[ex["kind"]] += 1
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts["parts"][part] = {
            "records": records,
            "examples": sum(by_task.values()),
            "by_kind": dict(kinds),
            "by_task": dict(sorted(by_task.items())),
        }
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--jurisdictions", nargs="+", default=["no", "at"])
    b.add_argument("--split", default="random")
    a = ap.parse_args(argv)
    counts = build(a.name, a.jurisdictions, a.split)
    for part, c in counts["parts"].items():
        print(f"{part}: {c['records']:,} records -> {c['examples']:,} examples {c['by_kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
