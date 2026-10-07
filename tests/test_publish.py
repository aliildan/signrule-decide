"""scripts/publish.py: only allow-listed tracked files reach the public tree; violations abort."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("publish", "scripts/publish.py")
assert spec is not None and spec.loader is not None
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    src = tmp_path / "private"
    for rel, text in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    return src


def test_only_allow_listed_files_are_copied(tmp_path):
    src = _repo(
        tmp_path,
        {
            "README.md": "# fixture\n",
            "pyproject.toml": "[project]\nname = 'x'\n",
            "src/signrule/x.py": "X = 1\n",
            "tests/test_x.py": "def test_x():\n    pass\n",
            "CLAUDE.md": "private\n",
            "docs/plan.md": "private\n",
            ".claude/settings.json": "{}\n",
        },
    )
    out = tmp_path / "public"
    files = publish.build_tree(src, out)
    assert sorted(files) == ["README.md", "pyproject.toml", "src/signrule/x.py", "tests/test_x.py"]
    assert not (out / "CLAUDE.md").exists() and not (out / "docs").exists()
    assert (out / "src/signrule/x.py").read_text() == "X = 1\n"


def test_a_violation_aborts_before_anything_is_written(tmp_path):
    token = "ghp_" + "a" * 36  # split so this file itself passes the scan
    src = _repo(tmp_path, {"README.md": "# fixture\n", "src/leak.py": f"T = '{token}'\n"})
    out = tmp_path / "public"
    with pytest.raises(SystemExit, match="violation"):
        publish.build_tree(src, out)
    assert not out.exists()


def test_uncommitted_public_changes_block_publishing(tmp_path):
    src = _repo(
        tmp_path, {"README.md": "# fixture\n", "src/x.py": "X = 1\n", "docs/n.md": "private\n"}
    )
    subprocess.run(
        ["git", "-c", "user.name=f", "-c", "user.email=f@f", "commit", "-qm", "c"],
        cwd=src,
        check=True,
    )
    assert publish.dirty_public(src) == []
    (src / "src/x.py").write_text("X = 2\n")  # public file changed, not committed
    (src / "docs/n.md").write_text("changed\n")  # private file: irrelevant
    assert publish.dirty_public(src) == ["src/x.py"]
