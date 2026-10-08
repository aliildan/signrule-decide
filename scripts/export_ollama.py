"""Export a Strands Decider-format checkpoint as an Ollama-importable folder (plan-17 T4).

    uv run --group strands python scripts/export_ollama.py --ckpt runs/<name>/ckpt \
        --out runs/<name>/ollama [--temperatures runs/<name>/temperatures.json] [--max-length 16384]
    OLLAMA_HOST=127.0.0.1:11435 ollama create signrule-decide -f runs/<name>/ollama/Modelfile

What Ollama's StrandsDeciderForDecision loads (ollama mlxrunner/model/strands, create/):
- the backbone as full weights: the base model's language-model tensors under their Hugging Face
  names, with the LoRA merged in (W + B·A·alpha/r, in fp32, stored bf16). Names, shapes and the raw
  norm/conv1d layout stay exactly as in the base checkpoint (Ollama detects that layout); the vision
  tower and the MTP head are dropped;
- the pointer head in FP32 as top-level `norm.*`, `q.*`, `k.*`;
- `config.json`: the base text config, flat, `architectures: [StrandsDeciderForDecision]`;
- `strands_decider_config.json`: head type, pointer dim, the serving window and our per-type
  temperatures (`temperature_by_kind`), so Ollama returns calibrated probabilities;
- the tokenizer files and a Modelfile (Apache-2.0 licence text with the data attributions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ADAPTER_PREFIX = "base_model.model."
BASE_PREFIX = "model.language_model."
DROP_PREFIXES = ("model.visual.", "mtp.")
HEAD_TENSORS = ("norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias")
KINDS = ("noul", "choice", "score")
LICENSE_NOTE = """SignRule-Decide (Strands Decider format). Apache License 2.0.
Base model Qwen/Qwen3.5-4B-Base (Apache-2.0); trained with strands-decider (Apache-2.0).
Contains information derived from: Brønnøysundregistrene (NLOD); Firmenbuch - Bundesministerium
für Justiz / JustizOnline (HVD), CC BY 4.0. No register data is included.
Not legal advice; keep a person in the loop for binding decisions.
https://github.com/aliildan/signrule-decide"""


def base_name(adapter_key: str) -> str:
    """'base_model.model.<path>.lora_A.weight' -> 'model.language_model.<path>.weight'."""
    if not adapter_key.startswith(ADAPTER_PREFIX) or ".lora_" not in adapter_key:
        raise ValueError(f"unexpected adapter key {adapter_key!r}")
    path = adapter_key[len(ADAPTER_PREFIX) :].split(".lora_", 1)[0]
    return f"{BASE_PREFIX}{path}.weight"


def keep(name: str) -> bool:
    return not name.startswith(DROP_PREFIXES)


def merge(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    """W + scale · B·A, computed in fp32."""
    return (w.float() + (b.float() @ a.float()) * scale).to(dtype).contiguous()


def ollama_config(base: dict[str, Any]) -> dict[str, Any]:
    text = dict(base.get("text_config", base))
    text["architectures"] = ["StrandsDeciderForDecision"]
    text["model_type"] = "qwen3_5_text"
    text.pop("vision_config", None)
    text.pop("text_config", None)
    return text


def decider_config(
    ckpt_cfg: dict[str, Any], max_length: int, temperatures: dict[str, float]
) -> dict[str, Any]:
    if ckpt_cfg.get("head_type") != "pointer":
        raise ValueError("only the pointer head is served by Ollama")
    unknown = set(temperatures) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown question kinds {sorted(unknown)}")
    return {
        **ckpt_cfg,
        "max_length": max_length,
        "temperature": 1.0,
        "temperature_by_kind": {k: float(v) for k, v in temperatures.items()},
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(ckpt: Path, out: Path, max_length: int, temperatures: dict[str, float]) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    ck_cfg = json.loads((ckpt / "strands_decider_config.json").read_text())
    lora_cfg = json.loads((ckpt / "lora" / "adapter_config.json").read_text())
    if lora_cfg.get("use_rslora") or lora_cfg.get("use_dora") or lora_cfg.get("modules_to_save"):
        raise SystemExit("only plain LoRA adapters are supported")
    scale = lora_cfg["lora_alpha"] / lora_cfg["r"]
    base_id, rev = ck_cfg["base_model"], ck_cfg.get("base_revision") or "main"

    adapters: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(ckpt / "lora" / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
            part = "A" if ".lora_A." in key else "B"
            adapters.setdefault(base_name(key), {})[part] = f.get_tensor(key)

    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(
        Path(hf_hub_download(base_id, "model.safetensors.index.json", revision=rev)).read_text()
    )
    shards = sorted(set(index["weight_map"].values()))
    weight_map: dict[str, str] = {}
    merged: set[str] = set()
    for i, shard in enumerate(shards, start=1):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(hf_hub_download(base_id, shard, revision=rev), "pt") as f:
            for name in f.keys():  # noqa: SIM118
                if not keep(name):
                    continue
                t = f.get_tensor(name)
                if name in adapters:
                    ab = adapters[name]
                    t = merge(t, ab["A"], ab["B"], scale, t.dtype)
                    merged.add(name)
                tensors[name] = t.contiguous()
        fname = f"model-{i:05d}-of-{len(shards) + 1:05d}.safetensors"
        save_file(tensors, str(out / fname), metadata={"format": "pt"})
        weight_map.update({n: fname for n in tensors})
    missing = set(adapters) - merged
    if missing:
        raise SystemExit(
            f"{len(missing)} adapter modules had no base tensor, e.g. {sorted(missing)[:3]}"
        )

    head = torch.load(ckpt / "slot_head.pt", map_location="cpu", weights_only=True)
    if set(head) != set(HEAD_TENSORS):
        raise SystemExit(f"unexpected head tensors {sorted(head)}")
    fname = f"model-{len(shards) + 1:05d}-of-{len(shards) + 1:05d}.safetensors"
    save_file({k: head[k].float().contiguous() for k in HEAD_TENSORS}, str(out / fname))
    weight_map.update({k: fname for k in HEAD_TENSORS})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=1)
    )

    base_cfg = json.loads(Path(hf_hub_download(base_id, "config.json", revision=rev)).read_text())
    (out / "config.json").write_text(json.dumps(ollama_config(base_cfg), indent=2))
    dcfg = decider_config(ck_cfg, max_length, temperatures)
    (out / "strands_decider_config.json").write_text(json.dumps(dcfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (ckpt / name).exists():
            shutil.copy2(ckpt / name, out / name)
    (out / "LICENSE").write_text(LICENSE_NOTE + "\n")
    (out / "Modelfile").write_text(f'FROM {out.resolve()}\nLICENSE """{LICENSE_NOTE}"""\n')

    record = {
        "ckpt": str(ckpt),
        "base_model": base_id,
        "base_revision": rev,
        "lora_scale": scale,
        "merged_modules": len(merged),
        "max_length": max_length,
        "temperature_by_kind": dcfg["temperature_by_kind"],
        "files": {
            p.name: _sha256(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "export.json"
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "export.json").write_text(json.dumps(record, indent=1))
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=16384, help="serving window (tokens)")
    ap.add_argument("--temperatures", type=Path, help="JSON {noul|choice|score: T}")
    a = ap.parse_args(argv)
    temps = json.loads(a.temperatures.read_text()) if a.temperatures else {}
    rec = export(a.ckpt, a.out, a.max_length, temps)
    print(f"exported {rec['merged_modules']} merged modules -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
