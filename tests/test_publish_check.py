"""Tests for scripts/publish_check.py.

Secret- and PII-shaped strings are built at runtime so this file itself passes the check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publish_check", ROOT / "scripts/publish_check.py")
assert _spec and _spec.loader
pc = importlib.util.module_from_spec(_spec)
sys.modules["publish_check"] = pc
_spec.loader.exec_module(pc)


def make_fnr(prefix9: str) -> str | None:
    """Append mod-11 control digits to a 9-digit prefix (None if the prefix has no valid fnr)."""
    d = [int(c) for c in prefix9]
    k1 = 11 - sum(a * b for a, b in zip(d, [3, 7, 6, 1, 8, 9, 4, 5, 2], strict=True)) % 11
    k1 = 0 if k1 == 11 else k1
    if k1 == 10:
        return None
    d.append(k1)
    k2 = 11 - sum(a * b for a, b in zip(d, [5, 4, 3, 2, 7, 6, 5, 4, 3, 2], strict=True)) % 11
    k2 = 0 if k2 == 11 else k2
    if k2 == 10:
        return None
    return prefix9 + str(k1) + str(k2)


def valid_fnr() -> str:
    for i in range(100, 999):
        fnr = make_fnr(f"010190{i}")
        if fnr:
            return fnr
    raise AssertionError("no valid fnr found")


def rules(path: str, text: str) -> set[str]:
    return {v.rule for v in pc.check_file(path, text.encode())}


# ---- path rules -----------------------------------------------------------------------------


def test_private_paths_are_forbidden():
    for path in [
        "CLAUDE.md",
        "docs/decisions.md",
        "notes/x.md",
        "secrets/cvr.env",
        "data/raw/a.json",
        "runs/r1/config.yaml",
        "src/data/raw.json",
    ]:
        assert "forbidden-path" in rules(path, "x"), path


def test_data_and_weight_suffixes_are_forbidden_outside_fixtures():
    for path in [
        "src/a.jsonl",
        "eval/b.parquet",
        "server/m.safetensors",
        "server/m.gguf",
        "configs/x.env",
        "results/r.csv",
    ]:
        assert "forbidden-suffix" in rules(path, "x"), path


def test_fixture_data_files_are_allowed_when_marked():
    assert rules("tests/fixtures/no/a.json", '{"_fixture": true}') == set()


def test_unmarked_fixture_is_rejected():
    assert "fixture-unmarked" in rules("tests/fixtures/no/a.json", '{"a": 1}')


def test_plain_code_passes():
    assert rules("src/ingest/http.py", "def f():\n    return 1\n") == set()


# ---- content rules --------------------------------------------------------------------------


def test_detects_github_token():
    assert "secret" in rules("src/a.py", "TOKEN = '" + "ghp_" + "A1b2" * 9 + "'\n")


def test_detects_hf_token():
    assert "secret" in rules("src/a.py", "x = '" + "hf_" + "aB3d" * 9 + "'\n")


def test_detects_api_key_assignment():
    assert "secret" in rules("configs/a.yaml", "x_api_key: '" + "Zq8" * 8 + "'\n")


def test_detects_private_key_block():
    assert "secret" in rules("src/a.py", "-----BEGIN " + "RSA PRIVATE KEY-----\n")


def test_detects_valid_norwegian_fnr():
    assert "pii-fnr" in rules("src/a.py", f"id = '{valid_fnr()}'\n")


def test_ignores_11_digit_number_with_bad_checksum():
    fnr = valid_fnr()
    bad = fnr[:-1] + str((int(fnr[-1]) + 1) % 10)
    assert "pii-fnr" not in rules("src/a.py", f"n = {bad}\n")


def test_ignores_digits_inside_hex_hash():
    fnr = valid_fnr()
    assert "pii-fnr" not in rules("uv.lock", f'hash = "sha256:4376c{fnr}ad263"\n')


def test_ignores_orgnr_9_digits():
    assert rules("src/a.py", "orgnr = '910336819'\n") == set()


def test_detects_danish_cpr():
    assert "pii-cpr" in rules("src/a.py", "cpr = '" + "010190" + "-" + "1234" + "'\n")


def test_ignores_impossible_cpr_date():
    assert "pii-cpr" not in rules("src/a.py", "x = '" + "991399" + "-" + "1234" + "'\n")


def test_detects_birthdate_key_with_value():
    assert "pii-birthdate" in rules("src/a.py", '{"fodsels' + 'dato": "1990-01-01"}\n')


def test_birthdate_key_name_alone_is_fine():
    assert rules("src/a.py", 'DROP = {"fodselsdato", "fodselsnummer"}\n') == set()


def test_inline_allow_marker_skips_line():
    line = "x = '" + "ghp_" + "A1b2" * 9 + "'  # publish-check: allow\n"
    assert rules("src/a.py", line) == set()


def test_violations_never_contain_matched_value():
    fnr = valid_fnr()
    for v in pc.check_file("src/a.py", f"id = '{fnr}'\n".encode()):
        assert fnr not in str(v)


# ---- publish allow-list ---------------------------------------------------------------------


def test_public_allow_list():
    assert pc.is_public("src/ingest/http.py")
    assert pc.is_public("README.md")
    assert pc.is_public("results/random/model.json")
    # extended 2026-10-07 (decisions.md): what a stranger needs to install and test
    assert pc.is_public("tests/test_x.py")
    # tests of the private Claude Code tooling (.claude/) stay private
    assert not pc.is_public("tests/test_guard_hook.py")
    assert not pc.is_public("tests/test_claude_config.py")
    assert pc.is_public("pyproject.toml") and pc.is_public("uv.lock")
    assert pc.is_public("CITATION.cff")
    assert not pc.is_public(".claude/agents/x.md")
    assert not pc.is_public(".github/workflows/ci.yml")
    assert not pc.is_public("docs/ROADMAP.md")
    assert not pc.is_public("CLAUDE.md")
