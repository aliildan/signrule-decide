"""Hardware, training cost and serving latency of the release model -> results/compute.json.

    uv run python scripts/compute_report.py [--run noat-4b-v2]

Reads the run's training_metrics.json (private runs/), the serving latency results and the local
hardware (nvidia-smi, /proc), and writes aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def gpu() -> dict[str, object]:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    name, mem = (x.strip() for x in out.split(","))
    return {"name": name, "memory_gb": round(int(mem) / 1024, 1), "count": 1}


def cpu_ram() -> dict[str, object]:
    info = Path("/proc/cpuinfo").read_text()
    model = re.search(r"model name\s*:\s*(.+)", info)
    mem_kb = int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1))  # type: ignore[union-attr]
    return {
        "cpu": model.group(1).strip() if model else platform.processor(),
        "ram_gb": round(mem_kb / 1024**2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="noat-4b-v2")
    ap.add_argument("--latency", default="results/serving/no-4b-base.latency.json")
    a = ap.parse_args(argv)
    m = json.loads((REPO / "runs" / a.run / "training_metrics.json").read_text())
    lat = json.loads((REPO / a.latency).read_text())
    single = next(r for r in lat["runs"] if r["concurrency"] == 1)
    report = {
        "hardware": {"gpu": gpu(), **cpu_ram()},
        "training": {
            "run": a.run,
            "wall_hours": round(m["wall_seconds"] / 3600, 1),
            "peak_gpu_memory_gb": round(m["peak_device_bytes"] / 1024**3, 1),
            "records_seen": m["records_seen"],
            "optimizer_steps": m["optimizer_steps"],
            "forward_tokens": m["forward_tokens"],
            "precision": m["dtype"],
            "weights": m["weights"],
        },
        "serving": {
            "measured_with": lat["model"],
            "note": "same 4B architecture, Norwegian requests",
            "questions_per_request": single["questions_per_request"],
            "p50_ms": round(single["p50_ms"]),
            "p95_ms": round(single["p95_ms"]),
            "throughput_rps": round(single["throughput_rps"], 1),
        },
    }
    out = REPO / "results" / "compute.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
