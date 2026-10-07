#!/usr/bin/env python3
"""Native quantized-layer kernels for the canvas runner.

A frozen layer box can keep its weights at rest as raw GGUF blocks and run a
fused dequant-GEMM in Triton on the device: no f16 copy of the matrix is ever
materialized, so a Q4_K model costs ~4.5 bits/weight of memory instead of 16.
Training never uses this path — a trainable box dequantizes at load exactly as
before.

Everything here is optional and detection-only. Triton ships with PyTorch; if
it (or a GPU) is missing, `can()` reports no native support and callers keep
the dequantize-at-load fallback. No dependency is ever installed from here.

Block layouts follow gguf.quants and are verified tensor-by-tensor against
`gguf.quants.dequantize` in tests/nn_quant_test.py:

    Q8_0   block =  32 values: f16 d, 32 x i8                      (34 bytes)
    Q4_K   block = 256 values: f16 d, f16 dmin, 12 scale/min bytes,
                               128 nibbles                            (144 bytes)
    Q5_K   block = 256 values: Q4_K + a 32-byte high-bit plane        (176 bytes)
    Q6_K   block = 256 values: 128 low nibbles, 64 high 2-bit planes,
                               16 int8 scales, f16 d                  (210 bytes)
    Q2_K   block = 256 values: 16 scale/min nibble pairs, 64 2-bit bytes,
                               f16 d, f16 dmin                         (84 bytes)
    Q3_K   block = 256 values: inverted high-bit plane, 64 2-bit bytes,
                               12 packed 6-bit scales, f16 d          (110 bytes)
    IQ4_XS block = 256 values: f16 d, 2 + 4 scale bytes, nibbles through
                               the 16-value codebook                  (136 bytes)
    IQ4_NL block =  32 values: f16 d, 16 nibble bytes, same codebook  (18 bytes)

Both kernels compute C[M, N] = A[M, K] @ dequant(W[N, K]).T in one pass:
each program owns a [BM, N] tile of C and walks K in 32-value (Q8_0) or
256-value (Q4_K) steps, dequantizing the weight rows it needs in registers.
"""
import importlib.util
import os
import platform
import sys

import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001 — the module is importable without torch
    torch = None

try:
    import triton
    import triton.language as tl
    _TRITON = triton
except Exception:  # noqa: BLE001 — triton is optional (ships with torch)
    triton = tl = None
    _TRITON = None

#: native kernel registry: gguf type name -> (values per block, bytes per block)
_KERNELS = {
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "IQ4_XS": (256, 136),
    "IQ4_NL": (32, 18),
    "IQ3_S": (256, 110),
    "IQ3_XXS": (256, 98),
    "IQ2_S": (256, 82),
    "IQ2_XS": (256, 74),
}

#: the 16-value IQ4 codebook (gguf's IQ4_NL.kvalues)
_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113)
_KV_CACHE = None

#: dtype name aliases the registry also answers to (lower-case forms)
_ALIASES = {name.lower(): name for name in _KERNELS}


def block_elems(qname):
    """Values per block for a native type, else None."""
    name = _canon(qname)
    return _KERNELS[name][0] if name else None


def block_bytes(qname):
    """Stored bytes per block for a native type, else None."""
    name = _canon(qname)
    return _KERNELS[name][1] if name else None


def _canon(qname):
    if qname in _KERNELS:
        return qname
    return _ALIASES.get(str(qname or "").lower())


def gpu_ok():
    """A triton-capable torch device is present right now."""
    if _TRITON is None or torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a broken driver is "no"
        return False


def can(qname, k=None):
    """Can this gguf quant type run natively here — optionally for K columns?"""
    name = _canon(qname)
    if name is None or not gpu_ok():
        return False
    if k is not None and int(k) % _KERNELS[name][0]:
        return False
    return True


def native_gguf_tensor(hit):
    """A GGUF reader tensor -> raw blocks + type, or None when there is no
    native kernel for it. `hit.data` is already [rows, row_bytes] uint8 for a
    quantized type; `hit.shape` is the file's ne order, so reversed is the
    logical [out, in] the dequantized array would have."""
    qname = getattr(getattr(hit, "tensor_type", None), "name", None)
    name = _canon(qname)
    if name is None or not gpu_ok():
        return None
    data = np.asarray(hit.data)
    if data.dtype != np.uint8 or data.ndim != 2:
        return None
    logical = tuple(int(x) for x in reversed(tuple(hit.shape)))
    if len(logical) != 2:
        return None
    n, k = logical
    elems, nbytes = _KERNELS[name]
    if data.shape[0] != n or k % elems:
        return None
    if int(data.shape[1]) != (k // elems) * nbytes:
        return None                            # a layout we do not know
    blocks = np.array(data, dtype=np.uint8, copy=True).reshape(-1)
    return {"qname": name, "shape": [n, k], "blocks": blocks}


def capabilities():
    """The capability manifest the app shows: what runs natively, and which
    optional quant providers happen to be installed (detection only)."""
    out = {"available": False, "backend": None, "device": None,
           "kernels": [], "triton": None, "torch": None, "providers": {}}
    if torch is not None:
        out["torch"] = getattr(torch, "__version__", "?")
    if _TRITON is not None:
        out["triton"] = getattr(_TRITON, "__version__", "?")
    if gpu_ok():
        out["available"] = True
        out["kernels"] = sorted(_KERNELS)
        if getattr(torch.version, "hip", None):
            out["backend"] = "triton/rocm"
        else:
            out["backend"] = "triton/cuda"
        try:
            out["device"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            out["device"] = "cuda"
    for name in ("llama_cpp", "torchao", "bitsandbytes", "gemlite", "vllm"):
        try:
            # presence only — never import a provider just to ask its name
            out["providers"][name] = bool(importlib.util.find_spec(name))
        except Exception:  # noqa: BLE001 — absent or broken counts as absent
            out["providers"][name] = False
    return out


def cpu_flags():
    """The instruction sets a future ggml-style cpu kernel would report on."""
    flags = []
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        low = text.lower()
        for flag, label in (("avx2", "AVX2"), ("avx512f", "AVX512"),
                            ("avx_vnni", "AVX-VNNI"), ("f16c", "F16C"),
                            ("fma", "FMA")):
            if flag in low:
                flags.append(label)
    except Exception:  # noqa: BLE001 — not linux: no /proc
        pass
    return flags


def platform_info():
    """Platform / accelerator summary for the agent's capability manifest."""
    info = {"platform": platform.system(), "arch": platform.machine(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None) if torch else None,
            "cuda": None, "rocm": None, "mps": None,
            "gpu": None, "cpu_flags": cpu_flags()}
    if torch is None:
        return info
    try:
        if getattr(torch.version, "hip", None):
            info["rocm"] = torch.version.hip
        if getattr(torch.version, "cuda", None):
            info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            try:
                cap = torch.cuda.get_device_capability(0)
                info["cc"] = "%d.%d" % cap
            except Exception:  # noqa: BLE001
                pass
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps and mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    return info


if _TRITON is not None:

    @triton.jit
    def _f16_at(W, off, mask):
        """The little-endian f16 stored at byte offset `off`."""
        b0 = tl.load(W + off, mask=mask, other=0).to(tl.uint16)
        b1 = tl.load(W + off + 1, mask=mask, other=0).to(tl.uint16)
        return (b0 | (b1 << 8)).to(tl.float16, bitcast=True)

    @triton.jit
    def _q8_0_gemm(A, W, C, M, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, BQ: tl.constexpr):
        """C = A @ dequant(W).T for Q8_0 blocks: 2-byte f16 scale, 32 i8."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nblk = K // 32
        rn64 = rn.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nblk, BQ):
            for jj in tl.static_range(BQ):
                j = c + jj
                live = (j < nblk) & nm
                base = (rn64 * nblk + j) * 34
                d = _f16_at(W, base, live).to(tl.float32)
                q = tl.load(W + base[:, None] + 2 + tl.arange(0, 32)[None, :],
                            mask=live[:, None], other=0)
                qi = q.to(tl.int16)
                qi = tl.where(qi > 127, qi - 256, qi)          # i8 wrap
                w = (qi.to(tl.float32) * d[:, None]).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (j * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None] & (j < nblk), other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _q4_k_gemm(A, W, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """C = A @ dequant(W).T for Q4_K superblocks: 8 sub-blocks of 32, the
        sub-block scales/mins packed into 12 bytes (ggml's split 6-bit form)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            # one 144-byte superblock per weight row
            base = (rn64 * nsup + c) * 144
            d = _f16_at(W, base, nm).to(tl.float32)
            dmin = _f16_at(W, base + 2, nm).to(tl.float32)
            for s in tl.static_range(4):          # sub-blocks 0..3
                # sc = d[s] & 0x3F, min = m[s] & 0x3F  (bytes 4+s / 8+s)
                sc = tl.load(W + base + 4 + s, mask=nm, other=0).to(tl.int32)
                mn = tl.load(W + base + 8 + s, mask=nm, other=0).to(tl.int32)
                w = _q4_k_subblock(W, base, d, dmin, sc & 0x3F, mn & 0x3F,
                                   s, s // 2, s % 2, nm)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
            for s4 in tl.static_range(4):         # sub-blocks 4..7
                # sc = (m_d[s] & 0x0F) | ((d[s] >> 2) & 0x30); min likewise
                db = tl.load(W + base + 4 + s4, mask=nm, other=0).to(tl.int32)
                mb = tl.load(W + base + 8 + s4, mask=nm, other=0).to(tl.int32)
                mdb = tl.load(W + base + 12 + s4, mask=nm, other=0).to(tl.int32)
                sc = (mdb & 0x0F) | ((db >> 2) & 0x30)
                mn = (mdb >> 4) | ((mb >> 2) & 0x30)
                s = 4 + s4
                w = _q4_k_subblock(W, base, d, dmin, sc, mn, s, s // 2, s % 2, nm)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _q4_k_subblock(W, base, d, dmin, sc, mn, s, g, h, nm):
        """One 32-value sub-block: bytes 16 + g*32, nibble h, scaled by
        d*sc and shifted by -dmin*mn."""
        nib = tl.load(W + base[:, None] + 16 + g * 32
                      + tl.arange(0, 32)[None, :],
                      mask=nm[:, None], other=0).to(tl.int32)
        qv = (nib >> (4 * h)) & 0x0F
        return (d[:, None] * sc.to(tl.float32)[:, None] * qv.to(tl.float32)
                - dmin[:, None] * mn.to(tl.float32)[:, None]).to(tl.float16)

    @triton.jit
    def _u16_at(W, off, mask):
        """The little-endian uint16 stored at byte offset `off`."""
        b0 = tl.load(W + off, mask=mask, other=0).to(tl.int32)
        b1 = tl.load(W + off + 1, mask=mask, other=0).to(tl.int32)
        return b0 | (b1 << 8)

    @triton.jit
    def _s8(b):
        """A stored byte read as int8 (two's complement), in int32."""
        v = b.to(tl.int16)
        return tl.where(v > 127, v - 256, v).to(tl.int32)

    @triton.jit
    def _q5_k_gemm(A, W, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """Q5_K: like Q4_K, plus one high bit per value from a shared 32-byte
        qh plane; scales/mins come from the same 12-byte pack."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        qh_offs = tl.arange(0, 32)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 176
            d = _f16_at(W, base, nm).to(tl.float32)
            dmin = _f16_at(W, base + 2, nm).to(tl.float32)
            qh = tl.load(W + base[:, None] + 16 + qh_offs,
                         mask=nm[:, None], other=0).to(tl.int32)
            for s in tl.static_range(4):
                sc = tl.load(W + base + 4 + s, mask=nm, other=0).to(tl.int32) & 0x3F
                mn = tl.load(W + base + 8 + s, mask=nm, other=0).to(tl.int32) & 0x3F
                w = _q5_k_subblock(W, base, d, dmin, sc, mn, qh, s, s // 2, s % 2, nm)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
            for s4 in tl.static_range(4):
                s = 4 + s4
                db = tl.load(W + base + 4 + s4, mask=nm, other=0).to(tl.int32)
                mb = tl.load(W + base + 8 + s4, mask=nm, other=0).to(tl.int32)
                mdb = tl.load(W + base + 12 + s4, mask=nm, other=0).to(tl.int32)
                sc = (mdb & 0x0F) | ((db >> 2) & 0x30)
                mn = (mdb >> 4) | ((mb >> 2) & 0x30)
                w = _q5_k_subblock(W, base, d, dmin, sc, mn, qh, s, s // 2, s % 2, nm)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _q5_k_subblock(W, base, d, dmin, sc, mn, qh, s, g, h, nm):
        ql = tl.load(W + base[:, None] + 48 + g * 32 + tl.arange(0, 32)[None, :],
                     mask=nm[:, None], other=0).to(tl.int32)
        nib = (ql >> (4 * h)) & 0x0F
        hi = (qh >> s) & 1
        q5 = (nib | (hi << 4)).to(tl.float32)
        return (d[:, None] * sc.to(tl.float32)[:, None] * q5
                - dmin[:, None] * mn.to(tl.float32)[:, None]).to(tl.float16)

    @triton.jit
    def _q6_k_gemm(A, W, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """Q6_K: 8 sub-blocks of 32; values are 6 bits (low nibble from ql,
        high 2 from qh), scales are int8 per 16 values."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        j = tl.arange(0, 32)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 210
            d = _f16_at(W, base + 208, nm).to(tl.float32)
            for s in tl.static_range(8):
                g = s // 4
                ql = tl.load(W + base[:, None] + 64 * g + 32 * (s % 2) + j,
                             mask=nm[:, None], other=0).to(tl.int32)
                qh = tl.load(W + base[:, None] + 128 + 32 * g + j,
                             mask=nm[:, None], other=0).to(tl.int32)
                q = (((ql >> (4 * ((s >> 1) & 1))) & 0x0F)
                     | (((qh >> (2 * (s % 4))) & 0x03) << 4)) - 32
                sc0 = _s8(tl.load(W + base + 192 + 2 * s, mask=nm, other=0))
                sc1 = _s8(tl.load(W + base + 192 + 2 * s + 1, mask=nm, other=0))
                scf = tl.where(j < 16, sc0[:, None], sc1[:, None]).to(tl.float32)
                w = (d[:, None] * scf * q.to(tl.float32)).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _q2_k_gemm(A, W, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """Q2_K: 16 sub-blocks of 16; 4-bit scales and mins in one byte,
        2-bit values packed four to a byte."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 16)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 84
            d = _f16_at(W, base + 80, nm).to(tl.float32)
            dmin = _f16_at(W, base + 82, nm).to(tl.float32)
            for s in tl.static_range(16):
                sc = tl.load(W + base + s, mask=nm, other=0).to(tl.int32)
                qb = tl.load(W + base[:, None] + 16 + 32 * (s // 8)
                             + 16 * (s % 2) + t,
                             mask=nm[:, None], other=0).to(tl.int32)
                qv = (qb >> (2 * ((s >> 1) & 3))) & 0x03
                dl = (d * (sc & 0x0F).to(tl.float32))[:, None]
                ml = (dmin * (sc >> 4).to(tl.float32))[:, None]
                w = (dl * qv.to(tl.float32) - ml).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 16 + tl.arange(0, 16))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _q3_k_gemm(A, W, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """Q3_K: 16 sub-blocks of 16; 3-bit values (low 2 from qs, inverted
        high bit from hmask), 6-bit-biased int8 scales packed in 12 bytes."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 16)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 110
            d = _f16_at(W, base + 108, nm).to(tl.float32)
            for s in tl.static_range(16):
                qb = tl.load(W + base[:, None] + 32 + 32 * (s // 8)
                             + 16 * (s % 2) + t,
                             mask=nm[:, None], other=0).to(tl.int32)
                qv = (qb >> (2 * ((s >> 1) & 3))) & 0x03
                hb = tl.load(W + base[:, None] + 16 * (s % 2) + t,
                             mask=nm[:, None], other=0).to(tl.int32)
                hi = (hb >> (s // 2)) & 1
                q = qv - 4 * (1 - hi)
                lb = tl.load(W + base + 96 + (s % 8), mask=nm, other=0).to(tl.int32)
                hsc = tl.load(W + base + 104 + (s % 4), mask=nm, other=0).to(tl.int32)
                sc6 = ((lb >> (4 * (s // 8))) & 0x0F) | (((hsc >> (2 * (s // 4))) & 0x03) << 4)
                scf = (sc6 - 32).to(tl.float32)
                w = (d[:, None] * scf[:, None] * q.to(tl.float32)).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 16 + tl.arange(0, 16))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _iq4_xs_gemm(A, W, KV, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ4_XS: 8 sub-blocks of 32; 6-bit biased scales packed into
        4 + 2 bytes, nibbles mapped through the 16-value codebook."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        j = tl.arange(0, 32)[None, :]
        j16 = j % 16
        jsh = 4 * (j // 16)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 136
            d = _f16_at(W, base, nm).to(tl.float32)
            for s in tl.static_range(8):
                lb = tl.load(W + base + 4 + (s // 2), mask=nm, other=0).to(tl.int32)
                hb = _u16_at(W, base + 2, nm)
                ls = ((lb >> (4 * (s % 2))) & 0x0F) | (((hb >> (2 * s)) & 0x03) << 4)
                dl = (d * (ls - 32).to(tl.float32))[:, None]
                qb = tl.load(W + base[:, None] + 8 + 16 * s + j16,
                             mask=nm[:, None], other=0).to(tl.int32)
                nib = (qb >> jsh) & 0x0F
                kvg = tl.load(KV + nib).to(tl.float32)
                w = (dl * kvg).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _u32_at(W, off, mask):
        """The little-endian uint32 at byte offset `off` (int64 in Triton)."""
        b0 = tl.load(W + off, mask=mask, other=0).to(tl.int64)
        b1 = tl.load(W + off + 1, mask=mask, other=0).to(tl.int64)
        b2 = tl.load(W + off + 2, mask=mask, other=0).to(tl.int64)
        b3 = tl.load(W + off + 3, mask=mask, other=0).to(tl.int64)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    @triton.jit
    def _iq3_s_gemm(A, W, G, KS, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ3_S: 8 sub-blocks of 32; a 9-bit grid index (qs byte + qh bit)
        picks one 4-value grid row, a sign byte flips values, a nibble scales
        the sub-block (d * (1 + 2*sc))."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 32)[None, :]
        j = t // 8
        k = t % 8
        li = 2 * j + k // 4
        v = k % 4
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 110
            d = _f16_at(W, base, nm).to(tl.float32)
            for s in tl.static_range(8):
                sc = tl.load(W + base + 106 + s // 2, mask=nm, other=0).to(tl.int32)
                db = (d * (1 + 2 * ((sc >> (4 * (s % 2))) & 0x0F)).to(tl.float32))[:, None]
                qh = tl.load(W + base + 66 + s, mask=nm, other=0).to(tl.int32)
                qb = tl.load(W + base[:, None] + 2 + 8 * s + li,
                             mask=nm[:, None], other=0).to(tl.int32)
                entry = qb | (((qh[:, None] >> li) & 1) << 8)
                gval = tl.load(G + entry * 4 + v).to(tl.float32)
                sb = tl.load(W + base[:, None] + 74 + 4 * s + j,
                             mask=nm[:, None], other=0).to(tl.int32)
                sign = tl.where(((sb >> k) & 1) == 0, 1.0, -1.0)
                w = (db * gval * sign).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _iq3_xxs_gemm(A, W, G, KS, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ3_XXS: same 4-value grid rows, but one uint32 per sub-block holds
        both the 4-bit scale (top bits) and four 7-bit sign-codebook indices."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 32)[None, :]
        j = t // 8
        k = t % 8
        li = 2 * j + k // 4
        v = k % 4
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 98
            d = _f16_at(W, base, nm).to(tl.float32)
            for s in tl.static_range(8):
                u = _u32_at(W, base + 66 + 4 * s, nm)
                db = (d * (0.5 + (u >> 28).to(tl.float32)) * 0.5)[:, None]
                qb = tl.load(W + base[:, None] + 2 + 8 * s + li,
                             mask=nm[:, None], other=0).to(tl.int32)
                gval = tl.load(G + qb * 4 + v).to(tl.float32)
                sb = tl.load(KS + ((u[:, None] >> (7 * j)) & 0x7F)).to(tl.int32)
                sign = tl.where(((sb >> k) & 1) == 0, 1.0, -1.0)
                w = (db * gval * sign).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 32 + tl.arange(0, 32))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _iq2_s_gemm(A, W, G, KS, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ2_S: 8 sub-blocks of 32; 10-bit grid index (byte + 2 qh bits) into
        8-value rows, sign bits per byte, nibble scales (d*(0.5+sc)*0.25)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 16)[None, :]
        j = t // 8
        k = t % 8
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 82
            d = _f16_at(W, base, nm).to(tl.float32)
            for s in tl.static_range(16):
                sc = tl.load(W + base + 74 + s // 2, mask=nm, other=0).to(tl.int32)
                db = (d * (0.5 + ((sc >> (4 * (s % 2))) & 0x0F).to(tl.float32)) * 0.25)[:, None]
                i = 2 * s + j
                qh = tl.load(W + base[:, None] + 66 + i // 4,
                             mask=nm[:, None], other=0).to(tl.int32)
                qb = tl.load(W + base[:, None] + 2 + i,
                             mask=nm[:, None], other=0).to(tl.int32)
                entry = qb | (((qh >> (2 * (i % 4))) & 3) << 8)
                gval = tl.load(G + entry * 8 + k).to(tl.float32)
                sb = tl.load(W + base[:, None] + 34 + i,
                             mask=nm[:, None], other=0).to(tl.int32)
                sign = tl.where(((sb >> k) & 1) == 0, 1.0, -1.0)
                w = (db * gval * sign).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 16 + tl.arange(0, 16))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _iq2_xs_gemm(A, W, G, KS, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ2_XS: 16 sub-blocks of 16; uint16 indices into 8-value grid rows,
        the top bits selecting a 7-bit sign codebook entry."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nsup = K // 256
        rn64 = rn.to(tl.int64)
        t = tl.arange(0, 16)[None, :]
        j = t // 8
        k = t % 8
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nsup):
            base = (rn64 * nsup + c) * 74
            d = _f16_at(W, base, nm).to(tl.float32)
            for s in tl.static_range(16):
                sc = tl.load(W + base + 66 + s // 2, mask=nm, other=0).to(tl.int32)
                db = (d * (0.5 + ((sc >> (4 * (s % 2))) & 0x0F).to(tl.float32)) * 0.25)[:, None]
                u = _u16_at(W, base[:, None] + 2 + 2 * (2 * s + j),
                            nm[:, None])
                gval = tl.load(G + (u & 511) * 8 + k).to(tl.float32)
                sb = tl.load(KS + ((u >> 9) & 0x7F)).to(tl.int32)
                sign = tl.where(((sb >> k) & 1) == 0, 1.0, -1.0)
                w = (db * gval * sign).to(tl.float16)
                a = tl.load(A + rm[:, None].to(tl.int64) * K
                            + (c * 256 + s * 16 + tl.arange(0, 16))[None, :],
                            mask=mm[:, None], other=0.0)
                acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])

    @triton.jit
    def _iq4_nl_gemm(A, W, KV, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr):
        """IQ4_NL: 32-value blocks, 18 bytes each, nibbles through the
        same codebook (no sub-block scaling)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < N
        nblk = K // 32
        rn64 = rn.to(tl.int64)
        j = tl.arange(0, 32)[None, :]
        j16 = j % 16
        jsh = 4 * (j // 16)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for c in range(0, nblk):
            base = (rn64 * nblk + c) * 18
            d = _f16_at(W, base, nm).to(tl.float32)
            qb = tl.load(W + base[:, None] + 2 + j16,
                         mask=nm[:, None], other=0).to(tl.int32)
            nib = (qb >> jsh) & 0x0F
            w = (d[:, None] * tl.load(KV + nib).to(tl.float32)).to(tl.float16)
            a = tl.load(A + rm[:, None].to(tl.int64) * K
                        + (c * 32 + tl.arange(0, 32))[None, :],
                        mask=mm[:, None], other=0.0)
            acc += tl.dot(a, tl.trans(w))
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])


def dequant_host(blocks, meta):
    """The reference dequantization on the host — the safety net when a
    native box's blocks have been moved off the gpu (offload) or a kernel
    cannot run there."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize as _ref_dequantize
    qname = _canon(meta.get("qname"))
    if qname is None:
        raise RuntimeError("unknown quant type %r" % (meta.get("qname"),))
    elems, nbytes = _KERNELS[qname]
    n, k = int(meta["shape"][0]), int(meta["shape"][1])
    arr = blocks.detach().cpu().numpy().reshape(n, (k // elems) * nbytes)
    qtype = getattr(GGMLQuantizationType, qname)
    return torch.from_numpy(_ref_dequantize(arr, qtype)).to(torch.float32)


def _kv_tensor(device):
    """The IQ4 codebook, on the device (built once)."""
    global _KV_CACHE
    if _KV_CACHE is None or _KV_CACHE.device != device:
        _KV_CACHE = torch.tensor(_KVALUES, dtype=torch.int8, device=device)
    return _KV_CACHE


_GRID_CACHE = {}
_KS_CACHE = {}


def _grid_tensor(qname, device):
    """A grid-codebook table as floats, on the device (built once per type)."""
    key = (qname, str(device))
    hit = _GRID_CACHE.get(key)
    if hit is None:
        from gguf import quants as _gq
        cls = getattr(_gq, qname)
        cls.init_grid()
        g = np.array(np.asarray(cls.grid, dtype=np.float32), copy=True).reshape(-1)
        hit = torch.from_numpy(g).to(device)
        _GRID_CACHE[key] = hit
    return hit


def _ksigns_tensor(device):
    """The 128-entry sign codebook IQ2_XS/IQ3_XXS share."""
    hit = _KS_CACHE.get(str(device))
    if hit is None:
        from gguf import quants as _gq
        ks = np.frombuffer(_gq.IQ2_XXS.ksigns, dtype=np.uint8).copy()
        hit = torch.from_numpy(ks).to(device)
        _KS_CACHE[str(device)] = hit
    return hit


def embed(ids, blocks, meta):
    """Rows of an embedding table kept as blocks: only the rows the batch asks
    for are dequantized (the reference math, on the host), then handed back on
    the ids' device. A decode step touches a handful of rows; a prompt touches
    its tokens — never the whole table."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize as _ref_dequantize
    qname = _canon(meta.get("qname"))
    if qname is None:
        raise RuntimeError("unknown quant type %r" % (meta.get("qname"),))
    elems, nbytes = _KERNELS[qname]
    vocab, d = int(meta["shape"][0]), int(meta["shape"][1])
    table = blocks.reshape(vocab, (d // elems) * nbytes)
    flat = ids.reshape(-1).to(torch.long)
    rows = table[flat.to(table.device)].cpu().numpy()
    out = torch.from_numpy(_ref_dequantize(rows, getattr(GGMLQuantizationType, qname)))
    return out.to(device=ids.device, dtype=torch.float32).reshape(*ids.shape, d)


def linear(x, blocks, meta):
    """One layer, natively: x [..., K] -> [..., N] in float32.

    `blocks` is the flat uint8 storage, `meta` the {"qname", "shape"} the
    loader recorded. Raises when the kernel is not available — callers only
    choose this path after `can()`."""
    if not gpu_ok():
        raise RuntimeError("native quant kernels need triton and a gpu device")
    if x.device.type != "cuda":
        raise RuntimeError("native quant kernels run on the gpu, got %s" % x.device)
    qname = _canon(meta.get("qname"))
    if qname is None:
        raise RuntimeError("no native kernel for %r" % (meta.get("qname"),))
    n, k = int(meta["shape"][0]), int(meta["shape"][1])
    x = x.float()
    lead = tuple(x.shape[:-1])
    x2 = x.reshape(-1, k)
    m = int(x2.shape[0])
    if m == 0:
        return x.new_zeros(*lead, n)
    xh = x2.to(torch.float16).contiguous()
    if blocks.dtype != torch.uint8:
        blocks = blocks.to(torch.uint8)
    blocks = blocks.reshape(-1)
    if blocks.device != x.device or not blocks.is_contiguous():
        blocks = blocks.contiguous().to(x.device)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    bm, bn = 16, 64
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    if qname == "Q8_0":
        _q8_0_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn, BQ=4)
    elif qname == "Q4_K":
        _q4_k_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn)
    elif qname == "Q5_K":
        _q5_k_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn)
    elif qname == "Q6_K":
        _q6_k_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn)
    elif qname == "Q2_K":
        _q2_k_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn)
    elif qname == "Q3_K":
        _q3_k_gemm[grid](xh, blocks, out, m, n, k, BM=bm, BN=bn)
    elif qname == "IQ4_XS":
        _iq4_xs_gemm[grid](xh, blocks, _kv_tensor(x.device), out, m, n, k,
                           BM=bm, BN=bn)
    elif qname == "IQ4_NL":
        _iq4_nl_gemm[grid](xh, blocks, _kv_tensor(x.device), out, m, n, k,
                           BM=bm, BN=bn)
    elif qname in ("IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS"):
        kern = {"IQ3_S": _iq3_s_gemm, "IQ3_XXS": _iq3_xxs_gemm,
                "IQ2_S": _iq2_s_gemm, "IQ2_XS": _iq2_xs_gemm}[qname]
        kern[grid](xh, blocks, _grid_tensor(qname, x.device),
                   _ksigns_tensor(x.device), out, m, n, k, BM=bm, BN=bn,
                   num_stages=1)
    else:                                          # pragma: no cover
        raise RuntimeError("no native kernel for %s" % qname)
    return out.reshape(*lead, n)


if __name__ == "__main__":                         # a small self-report
    import json
    print(json.dumps({"platform": platform_info(), "capabilities": capabilities(),
                      "cpu_flags": cpu_flags()}, indent=2, default=str))
    if gpu_ok():
        rng = np.random.default_rng(0)
        for qname in sorted(_KERNELS):
            elems, nbytes = _KERNELS[qname]
            k = elems * 3
            w = rng.standard_normal((16, k)).astype(np.float32)
            x = rng.standard_normal((4, k)).astype(np.float32)
            print(qname, "k=%d" % k, "smoke:", can(qname, k))
