"""Build the public repository from the private one (plan-10 T1).

    uv run python scripts/publish.py                      # build ../signrule-decide-public, no push
    SIGNRULE_PUBLISH=1 uv run python scripts/publish.py --push

Copies the tracked files on the public allow-list (publish_check.is_public) into a separate
working tree with its own history, scans every file there with publish_check, and commits one
release commit. The private history never reaches the public repo. Pushing needs --push and
SIGNRULE_PUBLISH=1. Model weights are not part of the repo (Hugging Face, plan-10 T3).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_check import check_file, is_public  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO.parent / "signrule-decide-public"
PUBLIC_REMOTE = "https://github.com/aliildan/signrule-decide.git"


def tracked_public(src: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=src, check=True, capture_output=True
    ).stdout.split(b"\0")
    paths = [p.decode() for p in out if p]
    return sorted(
        p for p in paths if is_public(p) and (src / p).is_file() and not (src / p).is_symlink()
    )


def build_tree(src: Path, out: Path) -> list[str]:
    """Scan, then copy the allow-listed tracked files of `src` into `out` (its .git is kept)."""
    files = tracked_public(src)
    violations = [v for f in files for v in check_file(f, (src / f).read_bytes())]
    if violations:
        for v in violations:
            print(f"publish: {v}", file=sys.stderr)
        raise SystemExit(f"publish: {len(violations)} violation(s); nothing written")
    if out.exists():
        for child in out.iterdir():
            if child.name != ".git":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dest)
    return files


def _git(out: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=out, check=True, capture_output=True, text=True
    ).stdout


def commit(out: Path, message: str) -> bool:
    if not (out / ".git").exists():
        _git(out, "init", "-q", "-b", "main")
        _git(out, "remote", "add", "origin", PUBLIC_REMOTE)
    _git(out, "add", "-A")
    if not _git(out, "status", "--porcelain").strip():
        return False
    _git(out, "commit", "-q", "-m", message)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--message", default="Release")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    files = build_tree(REPO, a.out)
    print(f"publish: {len(files)} files scanned and copied to {a.out}")
    changed = commit(a.out, a.message)
    print("publish: committed" if changed else "publish: nothing changed")
    if a.push:
        if os.environ.get("SIGNRULE_PUBLISH") != "1":
            raise SystemExit("publish: --push needs SIGNRULE_PUBLISH=1")
        _git(a.out, "push", "-u", "origin", "main")
        print(f"publish: pushed to {PUBLIC_REMOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
