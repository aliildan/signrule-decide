"""Tests for the Kev training wrapper's guards (plan-07 T2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signrule.train import kev_wrapper as w


def write_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"name": "t", "base": "Qwen/Qwen3.5-0.8B-Base", **extra}
    p = tmp_path / "cfg.yaml"
    p.write_text(json.dumps(cfg))  # JSON is valid YAML
    return p


def test_guard_forces_no_augmentation(tmp_path):
    cfg = w.load_config(write_cfg(tmp_path))
    argv = w.build_kev_argv(cfg, tmp_path / "train.jsonl", tmp_path / "out")
    pairs = dict(zip(argv[3::2], argv[4::2], strict=False))
    for flag in ("--p_none", "--p_none_distract", "--p_distract", "--p_none_pair", "--replay"):
        assert pairs[flag] == "0"
    assert pairs["--synthetic_repeat"] == "1"
    assert "--suite" not in argv and "--anchor" not in argv
    assert pairs["--data"].endswith("train.jsonl")


@pytest.mark.parametrize("key", ["replay", "suite", "p_none", "anchor", "public_frac"])
def test_guard_rejects_augmentation_or_suite_keys(tmp_path, key):
    with pytest.raises(ValueError, match="forbidden"):
        w.load_config(write_cfg(tmp_path, **{key: 1}))


def test_guard_rejects_unclean_init(tmp_path):
    with pytest.raises(ValueError, match="license"):
        w.load_config(write_cfg(tmp_path, init_from="jaredpalmer/kev-9b"))


def test_shared_prefix_is_off_by_default_and_passed_when_set(tmp_path):
    def flags(**extra):
        cfg = w.load_config(write_cfg(tmp_path, **extra))
        argv = w.build_kev_argv(cfg, tmp_path / "train.jsonl", tmp_path / "out")
        return dict(zip(argv[3::2], argv[4::2], strict=False))

    assert "--shared_prefix" not in flags()
    assert flags(shared_prefix=True)["--shared_prefix"] == "1"


def test_train_env_avoids_allocator_fragmentation_but_respects_the_caller(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert w.train_env()["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    assert w.train_env()["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"


def test_unknown_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        w.load_config(write_cfg(tmp_path, learning_rate=1))


def test_verify_data_check_detects_changes(tmp_path, monkeypatch):
    out = tmp_path / "no"
    (out / "random").mkdir(parents=True)
    f = out / "random" / "train.jsonl"
    f.write_text('{"state": "x", "questions": {}}\n')
    monkeypatch.setattr(w, "PROCESSED_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="data-check"):
        w.verify_data_check("no")
    (out / "check.json").write_text(json.dumps({"files": {"random/train.jsonl": w._sha256(f)}}))
    assert w.verify_data_check("no") == {"random/train.jsonl": w._sha256(f)}
    f.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        w.verify_data_check("no")


def test_sqrt_dup_weight_oversamples_real_rows(tmp_path, monkeypatch):
    src = tmp_path / "no" / "random"
    src.mkdir(parents=True)
    rows = [
        {"state": "a", "questions": {}, "_meta": {"n_duplicates": 9}},
        {"state": "b", "questions": {}, "_meta": {"n_duplicates": 1}},
    ]
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(w, "PROCESSED_DIR", tmp_path)
    cfg = w.TrainConfig(name="t", base="b", dup_weight="sqrt")
    dest = w.build_train_file(cfg, tmp_path / "runs" / "t")
    lines = dest.read_text().splitlines()
    assert len(lines) == 4  # 3 copies of the 9-duplicate row + 1


def test_bench_uses_the_serving_context_unless_asked(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(w.subprocess, "call", lambda argv, **kw: calls.append(argv) or 0)
    base = ["bench", "--run", str(tmp_path / "m"), "--jurisdiction", "at", "--split", "gold"]
    assert w.main([*base, "--part", "test"]) == 0
    assert w.main([*base, "--part", "test", "--context", "train"]) == 0
    assert calls[0][1:3] == ["-m", "signrule.train.kev_bench"]
    assert calls[1][1:3] == ["-m", "kev.benchmark"]


def test_kev_bench_shim_sets_the_serving_context(monkeypatch):
    import kev.benchmark as kb
    from kev.suite import SERVING_CONTEXT

    from signrule.train import kev_bench

    monkeypatch.setattr(kb, "main", lambda: kb.CONTEXT)
    assert kev_bench.main() == SERVING_CONTEXT
