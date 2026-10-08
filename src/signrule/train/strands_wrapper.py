"""Train the Strands Decider-format model under the project's hard constraints (plan-17 T3).

    uv run --group strands python -m signrule.train.strands_wrapper train \
        --config configs/train/strands-4b-v1.yaml [--max-steps 60 --suffix smoke] [--dry-run]

Like `kev_wrapper`: trains only on processed register data that passed `data-check` (rebuilt into
Strands examples by `strands_data`), refuses every knob that would add generated rows, teacher or
reference-model targets or a released Strands checkpoint as the start (CLAUDE.md §2.1), and records
provenance (config, git commit, data hashes, versions) in `runs/<name>.signrule.json`. GPU memory
is sampled with nvidia-smi into `runs/<name>.gpu.csv`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from signrule.common.paths import REPO_ROOT
from signrule.train import strands_data
from signrule.train.kev_wrapper import RUNS_DIR, _git_commit, train_env, verify_data_check

# Each would add targets that are not register labels, or start from a checkpoint trained on
# generated data; the paths are the wrapper's.
FORBIDDEN_STRANDS_KEYS = {
    "kl_frozen_weight",
    "kl_frozen_reference",
    "kl_only_files",
    "kl_only_weight",
    "teacher_file",
    "teacher_weight",
    "init_from",
    "precompute_frozen_kl",
    "train_files",
    "val_files",
    "output_dir",
    "max_steps",
}
SAFE_DEFAULTS = {"kl_frozen_weight": 0.0, "teacher_weight": 0.0, "kl_only_files": []}
MONITOR_EVERY = 10  # Strands' own validation reads every 10th val row; our bench reads all


class StrandsRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    jurisdictions: list[str] = Field(default_factory=lambda: ["no", "at"])
    split: str = "random"
    strands: dict[str, Any]


def load_config(path: Path) -> StrandsRunConfig:
    cfg = StrandsRunConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
    bad = sorted(FORBIDDEN_STRANDS_KEYS & set(cfg.strands))
    if bad:
        raise ValueError(f"{path}: forbidden strands keys {bad} (§2.1; paths are the wrapper's)")
    if cfg.strands.get("head_type") != "pointer":
        raise ValueError("head_type must be 'pointer' (Ollama's StrandsDeciderForDecision)")
    return cfg


def monitor_subset(lines: list[str], every: int = MONITOR_EVERY) -> list[str]:
    return [line for i, line in enumerate(lines) if i % every == 0]


def trainer_config(cfg: StrandsRunConfig, run_dir: Path, max_steps: int = 0) -> dict[str, Any]:
    data = run_dir / "data"
    return {
        **cfg.strands,
        **SAFE_DEFAULTS,
        "train_files": [str(data / "train.jsonl")],
        "val_files": [str(data / "val_monitor.jsonl")],
        "output_dir": str(run_dir / "ckpt"),
        "max_steps": max_steps,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "missing"


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    name = f"{cfg.name}-{args.suffix}" if args.suffix else cfg.name
    run_dir = RUNS_DIR / name
    if (run_dir / "ckpt").exists():
        raise SystemExit(f"{run_dir}/ckpt exists; choose a new name or suffix")
    stamps = {jur: verify_data_check(jur) for jur in cfg.jurisdictions}
    counts = strands_data.build(name, cfg.jurisdictions, cfg.split)
    data = run_dir / "data"
    val_lines = (data / "val.jsonl").read_text(encoding="utf-8").splitlines()
    (data / "val_monitor.jsonl").write_text(
        "\n".join(monitor_subset(val_lines)) + "\n", encoding="utf-8"
    )
    tcfg = trainer_config(cfg, run_dir, max_steps=args.max_steps)
    tcfg_path = run_dir / "trainer_config.yaml"
    tcfg_path.write_text(yaml.safe_dump(tcfg, sort_keys=False), encoding="utf-8")
    argv = [sys.executable, "-m", "strands_decider.cli", "train", "--config", str(tcfg_path)]
    provenance: dict[str, Any] = {
        "config": cfg.model_dump(),
        "trainer_config": tcfg,
        "git_commit": _git_commit(),
        "data_check_sha256": stamps,
        "examples": {p: c["examples"] for p, c in counts["parts"].items()},
        "data_sha256": {p: _sha256(data / f"{p}.jsonl") for p in ("train", "val")},
        "versions": {d: _version(d) for d in ("strands-decider", "torch", "transformers", "peft")},
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    prov_path = RUNS_DIR / f"{name}.signrule.json"
    prov_path.write_text(json.dumps(provenance, indent=1))
    print(" ".join(argv[2:]))
    if args.dry_run:
        return 0
    with (RUNS_DIR / f"{name}.gpu.csv").open("w") as gpu_log:
        sampler = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-l", "10"],
            stdout=gpu_log,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        try:
            rc = subprocess.call(argv, cwd=REPO_ROOT, env=train_env())
        finally:
            sampler.terminate()
    provenance["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    provenance["returncode"] = rc
    prov_path.write_text(json.dumps(provenance, indent=1))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="strands_wrapper",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=0, help="hard step cap (smoke runs)")
    p.add_argument("--suffix", help="run name suffix, e.g. smoke")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
