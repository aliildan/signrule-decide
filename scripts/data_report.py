"""How much register data the release model rests on -> results/data.json (aggregate counts only).

    uv run python scripts/data_report.py [--run noat-4b-v2]

Reads the pipelines' summary files (data/processed, counts only), the Danish fetch manifest, the
Norwegian cache statistics (runs/ingest) and the run's training file and metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from signrule.common.paths import PROCESSED_DIR, raw_dir

REPO = Path(__file__).resolve().parent.parent


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="noat-4b-v2")
    a = ap.parse_args(argv)

    no = _json(PROCESSED_DIR / "no" / "summary.json")
    no_stats = (REPO / "runs" / "ingest" / "no-fullmakt-stats-final.txt").read_text()
    responses = [int(n) for n in re.findall(r"== \w+: (\d+) cached responses", no_stats)]
    no_split = no["splits"]["random"]
    at = _json(PROCESSED_DIR / "at" / "summary_random.json")
    at_pilot = _json(PROCESSED_DIR / "at" / "pilot" / "stats.json")
    dk_manifest = _json(sorted(raw_dir("dk").glob("cvr_es_manifest_*.json"))[-1])
    dk_pool = _json(PROCESSED_DIR / "dk" / "pool" / "stats.json")
    metrics = _json(REPO / "runs" / a.run / "training_metrics.json")
    train_file = REPO / "runs" / f"{a.run}.train.jsonl"
    n_train = sum(1 for line in train_file.open(encoding="utf-8") if line.strip())
    epochs = int(_json(REPO / "runs" / a.run / "training_config.json")["args"]["epochs"])

    report = {
        "NO": {
            "entities_read": no["records"]["signatur_cached"],
            "api_responses": sum(responses),
            "entities_kept": no["records"]["kept"],
            "unique_groups": no["n_groups"],
            "train": no_split["train"],
            "val": no_split["val"],
            "test": no_split["test"],
        },
        "AT": {
            "extracts_read": sum(at["records"].values()),
            "companies_kept": at["records"]["kept"],
            "unique_patterns": at["n_groups"],
            "train": at["splits"]["train"]["groups"],
            "val": at["splits"]["val"]["groups"],
            "test": at["splits"]["test"]["groups"],
            "pilot": at_pilot["records"]["written"],
        },
        "DK": {
            "companies_read": dk_manifest["documents_seen"],
            "companies_kept": dk_pool["records"]["written"],
            "unique_texts": dk_pool.get("unique_texts"),
        },
        "training": {
            "run": a.run,
            "training_cases": n_train,
            "cases_per_epoch": metrics["records_seen"] // epochs,
            "epochs": epochs,
            "forward_tokens": metrics["forward_tokens"],
        },
    }
    report["total_companies_read"] = (
        report["NO"]["entities_read"]
        + report["AT"]["extracts_read"]
        + report["DK"]["companies_read"]
    )
    out = REPO / "results" / "data.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
