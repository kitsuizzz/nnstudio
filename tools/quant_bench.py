#!/usr/bin/env python3
"""What a design's weights actually cost per sweep.

Times the native quant kernels on real blocks of a GGUF — one per tensor
type, steady state — and adds up what the whole file must read for one
forward pass. That sum is the floor every decode step pays *before* any
compute, and it is usually the whole story on a big quantized model.

    .venv/bin/python tools/quant_bench.py models/Qwen3.8-27B-UD-IQ4_XS.gguf
    .venv/bin/python tools/quant_bench.py <file> --cap-mb 280

It needs a CUDA device with enough free VRAM for the largest block it
samples (the cap keeps that honest; the warm model is left alone).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

import torch  # noqa: E402
import nn_quant  # noqa: E402
import nn_torch as T  # noqa: E402
from gguf import GGUFReader  # noqa: E402


def bench_block(store, path, name, device):
    """(packed bytes, ms at one row) for one real block, steady state."""
    nat = store.native_blocks(path, name)
    if nat is None:
        return None
    blocks = store.blocks_param(path, name, device=device)
    meta = {"qname": nat["qname"], "shape": [int(s) for s in nat["shape"]]}
    _n, k = meta["shape"]
    x = torch.randn(1, k, device=device, dtype=torch.float16)
    for _ in range(2):                              # warm
        nn_quant.linear(x, blocks, meta)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        t0 = time.perf_counter()
        nn_quant.linear(x, blocks, meta)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return blocks.numel() * blocks.element_size(), best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="a .gguf to measure")
    ap.add_argument("--cap-mb", type=float, default=280.0,
                    help="largest sampled block, MB (fits the free VRAM)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device: the native kernels only run on a gpu")
        return 1
    if a.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print("device: %s · %.2f of %.2f GB free" % (
            torch.cuda.get_device_name(0), free / 2**30, total / 2**30))

    reader = GGUFReader(a.file)
    store = T.WeightStore()

    by_type = {}
    for t in reader.tensors:
        by_type[t.tensor_type.name] = by_type.get(t.tensor_type.name, 0) + int(t.n_bytes)

    # the largest matmul block per type that fits the cap (embeddings excluded:
    # they are a gather, not a gemm)
    picks = {}
    for t in reader.tensors:
        tp, nm = t.tensor_type.name, t.name
        nb = int(t.n_bytes)
        if tp == "F32" or "embd" in nm or nb > a.cap_mb * 2**20 or nb < 8 * 2**20:
            continue
        if nb > picks.get(tp, (0, None))[0]:
            picks[tp] = (nb, nm)

    print("\n%-9s %8s %10s %9s %10s" % ("type", "in model", "sampled", "ms/row", "rate"))
    rates = {}
    for tp, (nb, nm) in sorted(picks.items(), key=lambda kv: -by_type.get(kv[0], 0)):
        got = bench_block(store, a.file, nm, a.device)
        if got is None:
            print("%-9s %7.2fGB %9s %10s %10s   (not native — f16 after one load)"
                  % (tp, by_type[tp] / 2**30, "-", "-", "-"))
            continue
        packed, ms = got
        rate = (packed / 2**30) / (ms / 1000.0)
        rates[tp] = rate
        print("%-9s %7.2fGB %8.1fMB %8.2f %6.0f GB/s"
              % (tp, by_type[tp] / 2**30, packed / 2**20, ms, rate))

    # the model, type by type: what one sweep must read
    print("\none forward pass, by type:")
    total_ms, floor_ms, unknown = 0.0, 0.0, 0.0
    best = max(rates.values()) if rates else 0.0
    for tp, nb in sorted(by_type.items(), key=lambda kv: -kv[1]):
        if tp == "F32":
            continue
        can = nn_quant.can(tp)
        if not can:
            read = nb * 2                            # dequantized once, f16 at rest
            total_ms += 0.0                          # a raw read is no gemm: skip
            print("  %-8s %7.2fGB  f16 at rest (%4.1f MB read, no kernel)" % (tp, nb / 2**30, read / 2**20))
            continue
        rate = rates.get(tp)
        if rate is None:
            unknown += nb / 2**30
            print("  %-8s %7.2fGB  (not sampled)" % (tp, nb / 2**30))
            continue
        ms = (nb / 2**30) / rate * 1000.0
        total_ms += ms
        floor_ms += (nb / 2**30) / best * 1000.0
        print("  %-8s %7.2fGB  at %5.0f GB/s -> %6.1f ms" % (tp, nb / 2**30, rate, ms))
    if unknown:
        print("  (+ %.2f GB unsampled)" % unknown)
    print("\nweights alone:  %6.1f ms per sweep" % total_ms)
    if best:
        print("at the best measured rate (%.0f GB/s): %6.1f ms — the kernel headroom"
              % (best, floor_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
