"""Copy-paste safety of the public docs: shell blocks must run as pasted into any shell."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL_BLOCK = re.compile(r"^```(?:bash|sh|shell)\n(.*?)^```", re.M | re.S)


@pytest.mark.parametrize("doc", ["README.md", "model_card.md"])
def test_shell_blocks_have_no_comments(doc: str) -> None:
    # interactive zsh (the macOS default) passes "# ..." on as arguments
    blocks = SHELL_BLOCK.findall((ROOT / doc).read_text(encoding="utf-8"))
    assert blocks
    for block in blocks:
        assert "#" not in block, block
