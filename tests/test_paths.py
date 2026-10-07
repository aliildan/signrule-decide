"""Tests for the vault guard (plan-01 T4)."""

from __future__ import annotations

import pytest

from signrule.common import paths


def test_require_vault_refuses_when_unmounted(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNRULE_ALLOW_UNENCRYPTED", raising=False)
    monkeypatch.setattr(paths, "_mounts", lambda: [("/", "ext4")])
    with pytest.raises(paths.VaultNotMounted):
        paths.require_vault(tmp_path / "raw")


def test_require_vault_allows_with_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNRULE_ALLOW_UNENCRYPTED", "1")
    monkeypatch.setattr(paths, "_mounts", lambda: [("/", "ext4")])
    paths.require_vault(tmp_path / "raw")


def test_require_vault_accepts_marker_on_fuse(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNRULE_ALLOW_UNENCRYPTED", raising=False)
    mnt = tmp_path / "plain"
    (mnt / "raw").mkdir(parents=True)
    (mnt / paths.VAULT_MARKER).touch()
    monkeypatch.setattr(paths, "_mounts", lambda: [("/", "ext4"), (str(mnt), "fuse.gocryptfs")])
    paths.require_vault(mnt / "raw")


def test_require_vault_rejects_fuse_without_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNRULE_ALLOW_UNENCRYPTED", raising=False)
    mnt = tmp_path / "plain"
    (mnt / "raw").mkdir(parents=True)
    monkeypatch.setattr(paths, "_mounts", lambda: [(str(mnt), "fuse.gocryptfs")])
    with pytest.raises(paths.VaultNotMounted):
        paths.require_vault(mnt / "raw")
