"""scripts/export_ollama.py (plan-17 T4): LoRA merge, tensor selection, Ollama configs."""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")  # CI installs no ML stack

spec = importlib.util.spec_from_file_location("export_ollama", "scripts/export_ollama.py")
assert spec is not None and spec.loader is not None
eo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eo)


def test_adapter_keys_map_to_base_tensor_names() -> None:
    key = "base_model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight"
    assert eo.base_name(key) == "model.language_model.layers.3.linear_attn.in_proj_qkv.weight"
    with pytest.raises(ValueError):
        eo.base_name("something.else.weight")


def test_kept_tensors_drop_vision_and_mtp() -> None:
    assert eo.keep("model.language_model.layers.0.mlp.up_proj.weight")
    assert eo.keep("model.language_model.embed_tokens.weight")
    assert not eo.keep("model.visual.blocks.0.mlp.linear_fc1.weight")
    assert not eo.keep("mtp.fc.weight")


def test_merge_matches_the_unmerged_lora_forward() -> None:
    torch.manual_seed(0)
    w = torch.randn(6, 4, dtype=torch.float32)
    a, b = torch.randn(2, 4), torch.randn(6, 2)
    x = torch.randn(3, 4)
    merged = eo.merge(w, a, b, scale=32 / 16, dtype=torch.float32)
    assert torch.allclose(x @ merged.T, x @ w.T + (x @ a.T @ b.T) * 2.0, atol=1e-5)


def test_ollama_config_is_flat_text_config() -> None:
    base = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 8,
            "tie_word_embeddings": True,
        },
        "vision_config": {"depth": 2},
    }
    cfg = eo.ollama_config(base)
    assert cfg["architectures"] == ["StrandsDeciderForDecision"]
    assert cfg["model_type"] == "qwen3_5_text" and cfg["hidden_size"] == 8
    assert "vision_config" not in cfg and "text_config" not in cfg


def test_decider_config_carries_window_and_temperatures() -> None:
    ck = {"head_type": "pointer", "pointer_dim": 256, "max_length": 8192, "temperature": 1.0}
    out = eo.decider_config(ck, max_length=16384, temperatures={"noul": 1.3, "choice": 0.9})
    assert out["max_length"] == 16384 and out["temperature"] == 1.0
    assert out["temperature_by_kind"] == {"noul": 1.3, "choice": 0.9}
    with pytest.raises(ValueError):
        eo.decider_config({**ck, "head_type": "slot"}, max_length=4096, temperatures={})
    with pytest.raises(ValueError):
        eo.decider_config(ck, max_length=4096, temperatures={"other": 1.0})
