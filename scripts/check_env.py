"""Verify that the training stack sees the RTX 5090 (sm_120) and can run bf16 kernels.

Usage: uv run python scripts/check_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    import torch.version

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device      {name}, sm_{major}{minor}, {total_gb:.1f} GiB")
    print(f"arch list   {' '.join(arch_list)}")

    if (major, minor) == (12, 0) and "sm_120" not in arch_list:
        print("FAIL: GPU is sm_120 but this torch build has no sm_120 kernels (use cu128+ wheels)")
        return 1

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).float()
    torch.cuda.synchronize()
    if not torch.isfinite(c).all():
        print("FAIL: bf16 matmul produced non-finite values")
        return 1
    print(f"bf16 matmul OK (bf16 supported: {torch.cuda.is_bf16_supported()})")

    for mod in ("transformers", "peft", "accelerate", "datasets", "sklearn"):
        try:
            m = __import__(mod)
            print(f"{mod:11s} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod:11s} not installed")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
