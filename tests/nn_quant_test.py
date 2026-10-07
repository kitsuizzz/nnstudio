#!/usr/bin/env python3
"""Native GGUF kernels (agent/nn_quant.py) vs the gguf reference.

The kernels are the only place the canvas re-implements ggml's block math, so
every supported type is checked tensor-by-tensor against
`gguf.quants.dequantize` plus a matmul in float64. Skips (green) where there
is no triton-capable device: the fallback path is what runs there.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))

import numpy as np
import torch

import nn_quant


def check_registry():
    assert nn_quant.can("nope") is False
    assert nn_quant.can("IQ1_S") is False               # not ported
    assert nn_quant.can("IQ2_XXS") is False             # not ported
    if nn_quant.gpu_ok():
        for name in ("Q8_0", "Q4_K", "Q5_K", "Q6_K", "Q2_K", "Q3_K",
                     "IQ4_XS", "IQ4_NL", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS"):
            assert nn_quant.can(name) is True, name
        assert nn_quant.can("q4_k") is True             # case-insensitive
        assert nn_quant.can("Q8_0", 96) is True
        assert nn_quant.can("Q8_0", 100) is False       # not a block multiple
        assert nn_quant.can("Q4_K", 512) is True
        assert nn_quant.can("Q4_K", 128) is False
    caps = nn_quant.capabilities()
    assert set(caps) >= {"available", "kernels", "providers", "triton"}
    assert isinstance(nn_quant.cpu_flags(), list)
    assert nn_quant.platform_info()["platform"]


def fake_hit(qname, blocks, k, n):
    hit = types.SimpleNamespace()
    hit.tensor_type = types.SimpleNamespace(name=qname)
    hit.shape = (k, n)                        # gguf ne order: reversed logical
    hit.data = blocks
    return hit


def check_loader_meta(blocks, qname, k, n):
    meta = nn_quant.native_gguf_tensor(fake_hit(qname, blocks, k, n))
    assert meta is not None, "loader refused a %s tensor" % qname
    assert meta["qname"] == qname
    assert meta["shape"] == [n, k]
    assert meta["blocks"].dtype == np.uint8
    assert meta["blocks"].size == blocks.size
    assert nn_quant.native_gguf_tensor(fake_hit(qname, blocks, k + 32, n)) is None
    assert nn_quant.native_gguf_tensor(fake_hit("IQ1_S", blocks, k, n)) is None


def check_kernel(qname, blocks, ref_w, x):
    n, k = ref_w.shape
    got = nn_quant.linear(
        torch.tensor(x, device="cuda"), torch.tensor(blocks).reshape(-1),
        {"qname": qname, "shape": [n, k]})
    ref = x.astype(np.float64) @ ref_w.astype(np.float64).T
    err = float(np.abs(got.cpu().numpy() - ref).max())
    scale = float(np.abs(ref).max()) + 1e-6
    assert err / scale < 0.02, "%s: max rel err %.4f" % (qname, err / scale)
    # odd leading dims ride along
    rows = x.shape[0]
    got2 = nn_quant.linear(
        torch.tensor(x.reshape(rows, 1, -1), device="cuda"),
        torch.tensor(blocks).reshape(-1),
        {"qname": qname, "shape": [n, k]})
    assert tuple(got2.shape) == (rows, 1, n), got2.shape


def synth_blocks(qname, n, k, rng):
    """Valid block bytes for a type: the reference dequantizer, not a
    quantizer, is the authority on layout — random stored scales/nibbles
    exercise every field the kernel reads. The f16 scale fields are made
    well-behaved (random bits can decode to NaN)."""
    elems, nbytes = nn_quant.block_elems(qname), nn_quant.block_bytes(qname)
    nblk = k // elems
    raw = rng.integers(0, 256, size=(n, nblk, nbytes), dtype=np.uint8)

    def put16(off):
        v = np.float16(0.001 + rng.random((n, nblk)) * 0.1).view(np.uint8)
        v = v.reshape(n, nblk, 2)
        raw[..., off] = v[..., 0]
        raw[..., off + 1] = v[..., 1]

    fields = {"Q8_0": (0,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,),
              "Q2_K": (80, 82), "Q3_K": (108,), "IQ4_XS": (0,), "IQ4_NL": (0,),
              "IQ3_S": (0,), "IQ3_XXS": (0,), "IQ2_S": (0,), "IQ2_XS": (0,)}
    for off in fields[qname]:
        put16(off)
    return raw.reshape(n, nblk * nbytes)


def main():
    check_registry()
    if not nn_quant.gpu_ok():
        print("ok (skipped: no triton + gpu here — the f16 fallback runs)")
        return
    try:
        from gguf import GGMLQuantizationType, GGUFReader
        from gguf.quants import dequantize, quantize
    except Exception:  # noqa: BLE001
        print("ok (skipped: gguf package not importable)")
        return

    rng = np.random.default_rng(7)
    cases = [
        ("Q8_0", 16, 256),
        ("Q8_0", 37, 96),        # N and K off the pretty multiples
        ("Q4_K", 16, 512),
        ("Q4_K", 37, 256),
        ("Q5_K", 16, 512),
        ("Q5_K", 37, 256),
        ("Q6_K", 16, 512),
        ("Q6_K", 37, 256),
        ("Q2_K", 16, 512),
        ("Q2_K", 37, 256),
        ("Q3_K", 16, 512),
        ("Q3_K", 37, 256),
        ("IQ4_XS", 16, 512),
        ("IQ4_XS", 37, 256),
        ("IQ4_NL", 16, 128),
        ("IQ4_NL", 37, 96),
        ("IQ3_S", 16, 512),
        ("IQ3_S", 37, 256),
        ("IQ3_XXS", 16, 512),
        ("IQ3_XXS", 37, 256),
        ("IQ2_S", 16, 512),
        ("IQ2_S", 37, 256),
        ("IQ2_XS", 16, 512),
        ("IQ2_XS", 37, 256),
    ]
    for qname, n, k in cases:
        qtype = getattr(GGMLQuantizationType, qname)
        if qname == "Q8_0":
            blocks = quantize((rng.standard_normal((n, k)) * 0.2).astype(np.float32),
                              qtype)
        else:
            blocks = synth_blocks(qname, n, k, rng)
        elems, nbytes = nn_quant.block_elems(qname), nn_quant.block_bytes(qname)
        assert blocks.shape == (n, (k // elems) * nbytes), blocks.shape
        ref_w = dequantize(blocks, qtype)
        assert ref_w.shape == (n, k)
        check_loader_meta(blocks, qname, k, n)
        check_kernel(qname, blocks, ref_w, rng.standard_normal((5, k)).astype(np.float32))
        print("  %s: %dx%d vs reference ok" % (qname, n, k))

    real = os.environ.get("NNSTUDIO_QUANT_GGUF",
                          "models/gemma-4-12b-it-UD-Q4_K_XL.gguf")
    if os.path.isfile(real):
        reader = GGUFReader(real)
        best = {}
        for hit in reader.tensors:                 # smallest 2-D tensor per type
            name = hit.tensor_type.name
            if nn_quant.block_elems(name) is None or len(hit.shape) != 2:
                continue
            numel = 1
            for d in hit.shape:
                numel *= int(d)
            if name not in best or numel < best[name][0]:
                best[name] = (numel, hit)
        for name in sorted(best):
            numel, hit = best[name]
            if numel > 10_000_000:                 # keep the suite light
                continue
            meta = nn_quant.native_gguf_tensor(hit)
            if meta is None:
                continue
            ref_w = dequantize(hit.data, hit.tensor_type)
            n, k = meta["shape"]
            check_kernel(name, meta["blocks"].reshape(n, -1), ref_w,
                         rng.standard_normal((2, k)).astype(np.float32))
            print("  %s: real %s tensor (%dx%d) ok"
                  % (name, os.path.basename(real), n, k))
        del reader

    check_engine_native(rng)
    print("ok")


def check_engine_native(rng):
    """The engine actually takes the native path: a supported type keeps
    blocks wherever the box runs (gpu fuses, cpu dequantizes host-side);
    `native=False` (training) and non-native types dequantize as before."""
    import tempfile

    import nn_torch as T
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import dequantize, quantize

    n_out, k = 16, 128
    ref_w = (rng.standard_normal((n_out, k)) * 0.2).astype(np.float32)
    blocks = quantize(ref_w, GGMLQuantizationType.Q8_0)
    x = rng.standard_normal((4, k)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.gguf")
        w = GGUFWriter(path, "llama")
        w.add_tensor("blk.weight", blocks, raw_dtype=GGMLQuantizationType.Q8_0)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        node = {"id": "l", "kind": "layer", "name": "l", "din": k, "dout": n_out,
                "t": "blk.weight", "frozen": True}
        doc = {"type": "canvas", "version": 1, "nodes": [node], "edges": []}
        g = T.compile_canvas(dict(doc), gguf_file=path, device="cuda")
        box = g.boxes["l"]
        assert box.w_blocks is not None and box.w is None, "native path not taken"
        with torch.no_grad():
            got = box(torch.tensor(x, device="cuda")).cpu().numpy()
        want = x.astype(np.float64) @ dequantize(blocks, GGMLQuantizationType.Q8_0).astype(np.float64).T
        err = float(np.abs(got - want).max()) / (float(np.abs(want).max()) + 1e-6)
        assert err < 0.02, "native forward grew apart: %.4f" % err
        rep = g.weight_report()
        assert rep["params"] == n_out * k, rep              # logical params
        assert rep["bytes"] == int(blocks.size), rep        # stored bytes

        g4 = T.compile_canvas(dict(doc), gguf_file=path, device="cuda", native=False)
        assert g4.boxes["l"].w_blocks is None and g4.boxes["l"].w is not None, \
            "training compiles with native=False — masters stay f16"

        trainable = T.compile_canvas({"type": "canvas", "version": 1,
                                      "nodes": [dict(node, frozen=False)], "edges": []},
                                     gguf_file=path, device="cuda")
        assert trainable.boxes["l"].w_blocks is not None and trainable.boxes["l"].w is None, \
            "inference keeps native blocks regardless of the freeze flag"
        g3 = T.compile_canvas(dict(doc), gguf_file=path, device="cpu")
        if nn_quant.gpu_ok():
            assert g3.boxes["l"].w_blocks is not None and g3.boxes["l"].w is None, \
                "off the gpu the blocks stay packed at rest"
            with torch.no_grad():
                got3 = g3.boxes["l"](torch.tensor(x)).numpy()
            err3 = float(np.abs(got3 - want).max()) / (float(np.abs(want).max()) + 1e-6)
            assert err3 < 0.02, "host-side dequant forward grew apart: %.4f" % err3
        else:
            assert g3.boxes["l"].w_blocks is None and g3.boxes["l"].w is not None, \
                "without kernels anywhere the fallback must load"

        # a quantized embedding table: packed at rest, rows gathered per pass
        vocab, de = 8, 64
        wemb = (rng.standard_normal((vocab, de)) * 0.2).astype(np.float32)
        eblocks = quantize(wemb, GGMLQuantizationType.Q8_0)
        p2 = os.path.join(d, "emb.gguf")
        w2 = GGUFWriter(p2, "llama")
        w2.add_tensor("tok.weight", eblocks, raw_dtype=GGMLQuantizationType.Q8_0)
        w2.write_header_to_file()
        w2.write_kv_data_to_file()
        w2.write_tensors_to_file()
        w2.close()
        edoc = {"type": "canvas", "version": 1, "nodes": [
            {"id": "e", "kind": "embed", "name": "e", "vocab": vocab, "dout": de,
             "t": "tok.weight"}], "edges": []}
        ge = T.compile_canvas(dict(edoc), gguf_file=p2, device="cuda")
        ebox = ge.boxes["e"]
        assert ebox.w_blocks is not None and ebox.w is None, "embed table dequantized"
        ids = torch.tensor([[1, 3, 3, 7], [0, 5, 2, 6]], device="cuda")
        with torch.no_grad():
            got = ebox(ids).cpu().numpy()
        want = dequantize(eblocks, GGMLQuantizationType.Q8_0)[ids.cpu().numpy()]
        assert got.shape == (2, 4, de), got.shape
        assert np.abs(got - want).max() < 1e-4, "embed gather mismatch"
    print("  engine: blocks kept per type (host fallback off the gpu too)")


if __name__ == "__main__":
    main()
