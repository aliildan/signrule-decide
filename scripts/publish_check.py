"""Scan files for anything that must never be committed or published (CLAUDE.md §2, §9, §12).

Modes:
  --staged    files staged for commit (private dev branch). Used by scripts/hooks/pre-commit.
  --publish   tracked files on the public allow-list. Must pass before any publish step.
  PATH ...    explicit files (working-tree content).

Violations report path, line and rule name only, never the matched text: the match itself may
be personal data or a secret. Stdlib only, so git hooks can run it with any python3.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date

# What a stranger needs to install, test and reproduce (CLAUDE.md §9; extended 2026-10-07 with the
# project/lock files and tests/, whose fixtures are fictitious). Never .claude/, .github/, docs/.
PUBLIC_FILES = {
    "README.md",
    "LICENSE",
    "model_card.md",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    ".gitignore",
    ".python-version",
}
PUBLIC_DIRS = ("src/", "configs/", "scripts/", "server/", "eval/", "results/", "tests/")

FORBIDDEN_PATH = re.compile(
    r"^(?:CLAUDE\.md$|docs/|notes/|secrets/|data/|runs/)|(?:^|/)(?:data|secrets)/"
)
DATA_SUFFIXES = (
    ".env",
    ".gguf",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".onnx",
    ".parquet",
    ".arrow",
    ".jsonl",
    ".csv",
    ".tsv",
    ".gz",
    ".zip",
    ".xz",
    ".bz2",
    ".xlsx",
    ".pkl",
    ".npy",
)
FIXTURE_DIR = "tests/fixtures/"
FIXTURE_MARKER = b"_fixture"
MAX_BYTES = 2_000_000
ALLOW_MARKER = "publish-check: allow"

SECRET_PATTERNS = [
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:api[_-]?key|x-api-key|secret|password|passwd|token)[\"']?\s*[:=]\s*"
        r"[\"'][^\"'\s]{12,}[\"']"
    ),
]
FNR = re.compile(r"(?<![0-9A-Za-z])(\d{11})(?![0-9A-Za-z])")
CPR = re.compile(r"(?<![0-9A-Za-z])(\d{2})(\d{2})(\d{2})-(\d{4})(?![0-9A-Za-z])")
BIRTHDATE_VALUE = re.compile(
    r"(?i)[\"']?(?:f[oø]e?dsels(?:dato|nummer)|birth_?date|date_?of_?birth|geburtsdatum|"
    r"s[uü]nniaeg|s[uü]nnikuup[aä]ev|isikukood|cpr(?:_?nummer)?)[\"']?\s*[:=]\s*[\"']?\d"
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int  # 0 = whole file
    rule: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.rule}"


# Tests of the private Claude Code tooling (.claude/ is never published), so they would fail there.
PRIVATE_TESTS = {"tests/test_guard_hook.py", "tests/test_claude_config.py"}


def is_public(path: str) -> bool:
    if path in PRIVATE_TESTS:
        return False
    return path in PUBLIC_FILES or path.startswith(PUBLIC_DIRS)


def _fnr_valid(s: str) -> bool:
    d = [int(c) for c in s]
    k1 = 11 - sum(a * b for a, b in zip(d[:9], [3, 7, 6, 1, 8, 9, 4, 5, 2], strict=True)) % 11
    k1 = 0 if k1 == 11 else k1
    k2 = 11 - sum(a * b for a, b in zip(d[:10], [5, 4, 3, 2, 7, 6, 5, 4, 3, 2], strict=True)) % 11
    k2 = 0 if k2 == 11 else k2
    return k1 == d[9] and k2 == d[10]


def _cpr_plausible(dd: str, mm: str, yy: str) -> bool:
    try:
        date(1900 + int(yy), int(mm), int(dd))
    except ValueError:
        return False
    return True


def check_path(path: str) -> list[Violation]:
    out: list[Violation] = []
    if FORBIDDEN_PATH.search(path):
        out.append(Violation(path, 0, "forbidden-path"))
    in_fixtures = path.startswith(FIXTURE_DIR)
    if path.lower().endswith(DATA_SUFFIXES) and not in_fixtures:
        out.append(Violation(path, 0, "forbidden-suffix"))
    return out


def check_content(path: str, data: bytes) -> list[Violation]:
    out: list[Violation] = []
    if len(data) > MAX_BYTES and path != "LICENSE":
        out.append(Violation(path, 0, "too-large"))
    if path.startswith(FIXTURE_DIR) and not path.endswith(".md") and FIXTURE_MARKER not in data:
        out.append(Violation(path, 0, "fixture-unmarked"))
    if b"\x00" in data[:8192]:
        return out  # binary: path/suffix rules already applied
    text = data.decode("utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        if any(p.search(line) for p in SECRET_PATTERNS):
            out.append(Violation(path, lineno, "secret"))
        if any(_fnr_valid(m.group(1)) for m in FNR.finditer(line)):
            out.append(Violation(path, lineno, "pii-fnr"))
        if any(_cpr_plausible(*m.groups()[:3]) for m in CPR.finditer(line)):
            out.append(Violation(path, lineno, "pii-cpr"))
        if BIRTHDATE_VALUE.search(line):
            out.append(Violation(path, lineno, "pii-birthdate"))
    return out


def check_file(path: str, data: bytes) -> list[Violation]:
    return check_path(path) + check_content(path, data)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def staged_files() -> list[tuple[str, bytes]]:
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0")
    return [(n.decode(), _git("show", f":{n.decode()}")) for n in names if n]


def public_files() -> list[tuple[str, bytes]]:
    names = _git("ls-files", "-z").split(b"\0")
    paths = [n.decode() for n in names if n and is_public(n.decode())]
    out = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                out.append((p, f.read()))
        except FileNotFoundError:
            continue  # deleted in working tree
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--publish", action="store_true")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)

    if args.staged:
        files = staged_files()
    elif args.publish:
        files = public_files()
    else:
        files = []
        for p in args.paths:
            with open(p, "rb") as f:
                files.append((p, f.read()))

    violations = [v for path, data in files for v in check_file(path, data)]
    for v in violations:
        print(f"publish-check: {v}", file=sys.stderr)
    label = "staged" if args.staged else "public" if args.publish else "given"
    if violations:
        print(
            f"publish-check: FAILED, {len(violations)} violation(s) "
            f"in {len(files)} {label} file(s)",
            file=sys.stderr,
        )
        return 1
    print(f"publish-check: OK ({len(files)} {label} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
