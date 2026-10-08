"""Strands Decider training wrapper (plan-17 T3): the §2.1 guards and the config it hands over."""

from pathlib import Path

import pytest
import yaml

from signrule.train import strands_wrapper as sw

BASE = {
    "name": "strands-test",
    "jurisdictions": ["no", "at"],
    "split": "random",
    "strands": {"base_model": "Qwen/Qwen3.5-4B-Base", "head_type": "pointer", "epochs": 1},
}


def _write(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "key,value",
    [
        ("kl_frozen_weight", 0.3),
        ("kl_frozen_reference", "some/ckpt"),
        ("kl_only_files", ["x.jsonl"]),
        ("teacher_file", "t.jsonl"),
        ("teacher_weight", 1.0),
        ("init_from", "StrandsAgents/strands-decider-2B-hobson-v21"),
        ("precompute_frozen_kl", True),
        ("train_files", ["other.jsonl"]),
        ("output_dir", "elsewhere"),
    ],
)
def test_forbidden_strands_keys_are_refused(tmp_path: Path, key: str, value: object) -> None:
    cfg = {**BASE, "strands": {**BASE["strands"], key: value}}
    with pytest.raises(ValueError, match=key):
        sw.load_config(_write(tmp_path, cfg))


def test_pointer_head_is_required(tmp_path: Path) -> None:
    cfg = {**BASE, "strands": {**BASE["strands"], "head_type": "slot"}}
    with pytest.raises(ValueError, match="pointer"):
        sw.load_config(_write(tmp_path, cfg))


def test_trainer_config_gets_paths_and_safe_defaults(tmp_path: Path) -> None:
    cfg = sw.load_config(_write(tmp_path, BASE))
    run = tmp_path / "run"
    out = sw.trainer_config(cfg, run, max_steps=60)
    assert out["train_files"] == [str(run / "data" / "train.jsonl")]
    assert out["val_files"] == [str(run / "data" / "val_monitor.jsonl")]
    assert out["output_dir"] == str(run / "ckpt")
    assert out["max_steps"] == 60
    assert out["kl_frozen_weight"] == 0.0 and out["teacher_weight"] == 0.0
    assert out["head_type"] == "pointer" and out["epochs"] == 1


def test_monitor_subset_is_deterministic_and_small() -> None:
    lines = [f'{{"i": {i}}}' for i in range(1000)]
    a, b = sw.monitor_subset(lines, every=10), sw.monitor_subset(lines, every=10)
    assert a == b and len(a) == 100 and a[0] == lines[0]
