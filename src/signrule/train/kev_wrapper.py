"""Train SignRule-Decide with Kev under the project's hard constraints (plan-07 T2).

    python -m signrule.train.kev_wrapper train --config configs/train/base.yaml [--dry-run]
    python -m signrule.train.kev_wrapper bench --run runs/<name> --split random --part val

The wrapper owns everything Kev doesn't know about: it trains only on processed register data
that passed `data-check`, forces every augmentation/replay knob off (CLAUDE.md §2.1), refuses
initialisation from checkpoints whose training data breaks the license chain (§2.5), and records
provenance (config, git commit, data hashes) next to the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from signrule.common.paths import PROCESSED_DIR, REPO_ROOT, stamped_files

RUNS_DIR = REPO_ROOT / "runs"

# Never configurable: each would add generated or third-party examples to training.
FORCED_FLAGS: dict[str, str] = {
    "--p_none": "0",
    "--p_none_distract": "0",
    "--p_distract": "0",
    "--p_none_pair": "0",
    "--synthetic_repeat": "1",
    "--public_frac": "1.0",
    "--replay": "0",
    "--perm_kl": "0",
    "--anchor_w": "0",
}
FORBIDDEN_KEYS = {
    "suite",
    "replay",
    "anchor",
    "anchor_w",
    "p_none",
    "p_none_distract",
    "p_distract",
    "p_none_pair",
    "synthetic_repeat",
    "public_frac",
    "perm_kl",
    "n_per_source",
    "train_sources",
}
# Checkpoints whose training data is not license-clean (docs/data_kev_audit.md).
BLOCKED_INIT_PREFIXES = ("jaredpalmer/kev", "nimble")


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base: str
    base_revision: str | None = None
    jurisdictions: list[str] = Field(default_factory=lambda: ["no"])
    split: str = "random"  # random | temporal | ablation variants such as random_noroles
    epochs: int = 2
    lr: float = 5e-5
    head_lr: float | None = None
    batch: int = 4
    accum: int = 2
    lora: int = 16
    lora_targets: Literal["all", "dense", "attn", "qv"] = "all"
    ord_w: float = 0.0
    max_state: int = 512
    seed: int = 13
    dup_weight: Literal["none", "sqrt"] = "none"
    init_from: str | None = None
    max_steps: int | None = None
    # state once per record, question branches from it (kev.shared_prefix, documented exact):
    # the row form repeats the state per question and ran out of memory on C″ (plan-13)
    shared_prefix: bool = False


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = FORBIDDEN_KEYS & set(raw)
    if bad:
        raise ValueError(f"{path}: forbidden keys {sorted(bad)} (no synthetic/replayed data, §2.1)")
    cfg = TrainConfig(**raw)
    if cfg.init_from and cfg.init_from.lower().startswith(BLOCKED_INIT_PREFIXES):
        raise ValueError(f"init_from {cfg.init_from!r} is blocked by the license audit (§2.5)")
    return cfg


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_data_check(jurisdiction: str) -> dict[str, str]:
    """Hashes recorded by a passing `data-check` must equal the current files."""
    out = PROCESSED_DIR / jurisdiction
    stamp = out / "check.json"
    if not stamp.exists():
        raise RuntimeError(f"no data-check stamp for {jurisdiction}; run `make data-check` first")
    recorded = json.loads(stamp.read_text())["files"]
    current = {str(p.relative_to(out)): _sha256(p) for p in stamped_files(out)}
    if recorded != current:
        raise RuntimeError(f"processed {jurisdiction} data changed since data-check; rerun it")
    return current


def build_train_file(cfg: TrainConfig, run_dir: Path) -> Path:
    """Concatenate the configured jurisdictions' train partitions (optionally sqrt-oversampled)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir.parent / f"{run_dir.name}.train.jsonl"
    n = 0
    with dest.open("w", encoding="utf-8") as f:
        for jur in cfg.jurisdictions:
            src = PROCESSED_DIR / jur / cfg.split / "train.jsonl"
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                reps = 1
                if cfg.dup_weight == "sqrt":
                    reps = max(
                        1, round(math.sqrt(json.loads(line)["_meta"].get("n_duplicates", 1)))
                    )
                for _ in range(reps):
                    f.write(line + "\n")
                    n += 1
    if n == 0:
        raise RuntimeError("empty training file")
    return dest


def build_kev_argv(cfg: TrainConfig, train_file: Path, out_dir: Path) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "kev.train",
        "--data",
        str(train_file),
        "--base",
        cfg.base,
        "--epochs",
        str(cfg.epochs),
        "--lr",
        str(cfg.lr),
        "--batch",
        str(cfg.batch),
        "--accum",
        str(cfg.accum),
        "--lora",
        str(cfg.lora),
        "--lora_targets",
        cfg.lora_targets,
        "--ord_w",
        str(cfg.ord_w),
        "--max_state",
        str(cfg.max_state),
        "--seed",
        str(cfg.seed),
        "--dtype",
        "bf16",
        "--weights_dtype",
        "bf16",
        "--checkpointing",
        "1",
        "--length_sort",
        "1",
        "--device",
        "cuda",
        "--out",
        str(out_dir),
    ]
    if cfg.head_lr:
        argv += ["--head_lr", str(cfg.head_lr)]
    if cfg.base_revision:
        argv += ["--base_revision", cfg.base_revision]
    if cfg.init_from:
        argv += ["--init_from", cfg.init_from]
    if cfg.max_steps:
        argv += ["--max_steps", str(cfg.max_steps)]
    if cfg.shared_prefix:
        argv += ["--shared_prefix", "1"]
    for flag, value in FORCED_FLAGS.items():
        argv += [flag, value]
    return argv


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_env() -> dict[str, str]:
    """The training process's environment: the CUDA caching allocator with expandable segments
    (no effect on the numbers). C″ attempt 1 died with 3.77 GiB reserved but unusable while a
    1.48 GiB block was requested; a caller's own setting wins."""
    return {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    data_hashes = {jur: verify_data_check(jur) for jur in cfg.jurisdictions}
    run_dir = RUNS_DIR / cfg.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose a new name")
    train_file = build_train_file(cfg, run_dir)
    argv = build_kev_argv(cfg, train_file, run_dir)
    run_dir.rmdir()  # kev.train refuses an existing directory; it recreates it
    provenance = {
        "config": cfg.model_dump(),
        "git_commit": _git_commit(),
        "data_sha256": data_hashes,
        "train_file_sha256": _sha256(train_file),
        "kev_argv": argv[1:],
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    prov_path = RUNS_DIR / f"{cfg.name}.signrule.json"
    prov_path.write_text(json.dumps(provenance, indent=1))
    print(" ".join(argv[2:]))
    if args.dry_run:
        return 0
    rc = subprocess.call(argv, cwd=REPO_ROOT, env=train_env())
    provenance["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    provenance["returncode"] = rc
    prov_path.write_text(json.dumps(provenance, indent=1))
    return rc


def cmd_bench(args: argparse.Namespace) -> int:
    run = Path(args.run)
    data = PROCESSED_DIR / args.jurisdiction / args.split / f"{args.part}.jsonl"
    out = run.parent / f"{run.name}-bench-{args.jurisdiction}-{args.split}-{args.part}"
    argv = [
        sys.executable,
        "-m",
        # serving context (states as long as the server accepts) unless the old 384-token
        # training context is asked for, to reproduce numbers from before 2026-10-07
        "kev.benchmark" if args.context == "train" else "signrule.train.kev_bench",
        "--run",
        str(run),
        "--data",
        str(data),
        "--out",
        str(out),
    ]
    if args.part == "test":
        argv.append("--allow-test")
    env_note = "raw logits (T=1)" if args.raw else "shipped temperature"
    print(f"benchmark {run.name} on {data.relative_to(PROCESSED_DIR)} ({env_note}) -> {out}")
    env = None
    if args.raw:
        env = {**os.environ, "KEV_TEMPERATURE": "1.0"}
    return subprocess.call(argv, cwd=REPO_ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kev_wrapper",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("bench")
    p.add_argument("--run", required=True)
    p.add_argument("--jurisdiction", default="no")
    p.add_argument("--split", default="random")
    p.add_argument("--part", default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--raw", action="store_true", help="KEV_TEMPERATURE=1.0 (for our own calibration)"
    )
    p.add_argument(
        "--context",
        choices=["serve", "train"],
        default="serve",
        help="serve: encode every record as the server does; train: kev.benchmark's 384-token "
        "training context (longer states skipped), as before 2026-10-07",
    )
    p.set_defaults(func=cmd_bench)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
