#!/usr/bin/env python3
"""Read-only backend check: what this machine can run natively, and which
optional quant providers are missing. Installs nothing, changes nothing.

    .venv/bin/python tools/backend_check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))

_WHY = {
    "llama_cpp": "whole-model GGUF boxes (optional path)",
    "torchao": "future lower-bit CUDA kernels (optional)",
    "bitsandbytes": "future 8/4-bit paths (optional)",
    "gemlite": "future packed-kernel paths (optional)",
    "vllm": "future server-side execution (optional)",
}
_PKG = {"llama_cpp": "llama-cpp-python"}


def main():
    print("backend check — read-only, nothing is installed or changed\n")
    try:
        import nn_quant
    except Exception as e:  # noqa: BLE001
        print("agent/nn_quant.py is not importable: %s" % e)
        return 1
    info = nn_quant.platform_info()
    caps = nn_quant.capabilities()

    print("platform   %s %s · python %s" % (info["platform"], info["arch"], info["python"]))
    torch_bits = [info.get("torch") or "not importable"]
    if info.get("cuda"):
        torch_bits.append("cuda " + str(info["cuda"]))
    if info.get("rocm"):
        torch_bits.append("rocm " + str(info["rocm"]))
    if info.get("mps"):
        torch_bits.append("mps")
    print("torch      %s" % " · ".join(torch_bits))
    print("device     %s%s" % (info.get("gpu") or info.get("platform", "?"),
                               " (cc %s)" % info["cc"] if info.get("cc") else ""))
    print("triton     %s" % (caps.get("triton") or "not installed"))
    if caps.get("available"):
        print("native     %s  (%s)" % (", ".join(caps.get("kernels") or []),
                                       caps.get("backend") or "?"))
    else:
        print("native     none — GGUF layers dequantize to f16 at load")

    print("\noptional providers:")
    for name, present in (caps.get("providers") or {}).items():
        mark = "present" if present else "absent "
        why = _WHY.get(name, "")
        print("  %s  %-14s %s" % (mark, name, why if not present else ""))
    missing = [n for n, p in (caps.get("providers") or {}).items() if not p and n in _WHY]

    print("\nhints:")
    if not caps.get("available"):
        if info.get("mps"):
            print("  · the apple gpu (metal) has no triton kernels here: GGUF")
            print("    layers dequantize to f16 at load — correct, just heavier.")
        elif not caps.get("triton"):
            print("  · use a PyTorch wheel that bundles Triton (official CUDA/ROCm")
            print("    wheels do) — the f16 fallback works meanwhile.")
        else:
            print("  · Triton is here but no GPU is visible to torch: native kernels")
            print("    run on CUDA/ROCm only. The f16 fallback is correct, just")
            print("    heavier (a Q4_K 20B model needs ~40 GB of RAM/VRAM loaded).")
    else:
        print("  · nothing to do for GGUF layers: %s run natively."
              % ", ".join(caps.get("kernels") or []))
    if missing:
        print("  · optional, only if a path needs it (never installed by nnstudio):")
        for name in missing:
            pkg = _PKG.get(name, name)
            print("      pip install %-20s # %s" % (pkg, _WHY[name]))
    print("\nsee docs/backend.md for the full story.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
