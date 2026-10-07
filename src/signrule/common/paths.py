"""Canonical project paths. Data lives under `data/` (a symlink to /data/signrule/data on the
training box) unless SIGNRULE_DATA_DIR overrides it."""

from __future__ import annotations

import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = Path(os.environ.get("SIGNRULE_DATA_DIR", REPO_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
GOLD_DIR = DATA_DIR / "gold"


EVAL_ONLY_SPLITS = frozenset({"gold", "pilot"})  # never training input; not in the data stamp


def stamped_files(out: Path) -> list[Path]:
    """Split files covered by the data-check stamp (training inputs and their val/test parts)."""
    return [p for p in sorted(out.glob("*/*.jsonl")) if p.parent.name not in EVAL_ONLY_SPLITS]


def raw_dir(jurisdiction: str) -> Path:
    return RAW_DIR / jurisdiction.lower()


VAULT_MARKER = ".signrule-vault"
log = logging.getLogger(__name__)


class VaultNotMounted(RuntimeError):
    """Raised when personal data would be written outside the encrypted vault."""


def _mounts() -> list[tuple[str, str]]:
    """(mount point, fs type) pairs from /proc/self/mounts."""
    out = []
    with open("/proc/self/mounts", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.append((parts[1].replace("\\040", " "), parts[2]))
    return out


def require_vault(path: Path | None = None) -> None:
    """Refuse unless `path` (default RAW_DIR) lies on the mounted gocryptfs vault.

    Escape hatch: SIGNRULE_ALLOW_UNENCRYPTED=1 (logged). See scripts/setup_data_root.sh.
    """
    if os.environ.get("SIGNRULE_ALLOW_UNENCRYPTED") == "1":
        log.warning("SIGNRULE_ALLOW_UNENCRYPTED=1: writing personal data outside the vault")
        return
    target = Path(path or RAW_DIR).resolve()
    best: tuple[str, str] | None = None
    for mnt, fstype in _mounts():
        m = Path(mnt)
        if (target == m or m in target.parents) and (best is None or len(mnt) > len(best[0])):
            best = (mnt, fstype)
    if best is None or not best[1].startswith("fuse.gocryptfs"):
        raise VaultNotMounted(
            f"{target} is not on the encrypted vault; run scripts/setup_data_root.sh mount"
        )
    if not (Path(best[0]) / VAULT_MARKER).exists():
        raise VaultNotMounted(f"vault mount {best[0]} lacks {VAULT_MARKER}; wrong vault?")
