#!/usr/bin/env python3
"""The torch runner's tests, all offline and on synthetic designs:

  * a known-weight forward pass matches the arithmetic, every box captured
  * a nested box's ports, paths and values
  * the ops' shapes (reshape, transpose, concat, matmul, rope)
  * char-level training: the loss falls, samples draw, the checkpoint lands
  * frozen boxes are counted and skipped, through nested levels
  * the generate loop samples valid ids; integer feeds reach a layer
  * a GGUF tensor loads dequantized and oriented [out, in]
  * describe() counts trainable vs frozen; an unwired box is refused by name
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nn_torch as T  # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        import traceback
        FAILED.append(name)
        print("  FAIL " + name + " — " + str(e))
        if os.environ.get("STACK"):
            traceback.print_exc()


def n(kind, nid, x=0, y=0, **kw):
    out = {"id": nid, "kind": kind, "name": kw.pop("name", nid), "x": x, "y": y}
    out.update(kw)
    return out


def edge(a, b, fp=0, tp=0):
    return {"from": a, "to": b, "fromPort": fp, "toPort": tp}


def canvas(nodes, edges):
    return {"type": "canvas", "version": 1, "nodes": nodes, "edges": edges}


print("nn_torch: a known-weight forward pass")

KNOWN = [[1.0, -2.0, 0.5, 0.0], [0.0, 1.0, 1.0, -1.0], [-1.0, 0.0, 2.0, 1.0]]


def forward_known():
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "l", din=4, dout=3),
        n("act", "a", act="relu"),
        n("output", "o", din=3),
    ], [edge("i", "l"), edge("l", "a"), edge("a", "o")])
    g = T.compile_canvas(c)
    with torch.no_grad():
        g.boxes["l"].w.copy_(torch.tensor(KNOWN))
    x = torch.tensor([1.0, 1.0, 1.0, 1.0])
    got = g(feeds={"i": x})["o"].tolist()
    want = [max(0.0, sum(r[j] * float(x[j]) for j in range(4))) for r in KNOWN]
    assert all(abs(a - b) < 1e-5 for a, b in zip(got, want)), (got, want)
    res = T.run(c, {"i": [1.0, 1.0, 1.0, 1.0]}, input_id="i", output_id="o")
    assert set(res["values"]) == {"i", "l", "a", "o"}, res["values"].keys()
    assert res["values"]["o"]["shape"] == [3]
    assert res["values"]["a"]["min"] >= 0, "relu keeps the minimum at 0"


check("known weights, every box captured", forward_known)


def nested_forward():
    inner = {"type": "nested", "version": 1,
             "nodes": [n("input", "ni", dout=2), n("layer", "nl", din=2, dout=2),
                       n("output", "no", din=2)],
             "edges": [edge("ni", "nl"), edge("nl", "no")]}
    c = canvas([
        n("input", "i", dout=2),
        n("custom", "c", inner=inner),
        n("output", "o", din=2),
    ], [edge("i", "c"), edge("c", "o")])
    g = T.compile_canvas(c)
    with torch.no_grad():
        g.boxes["c"].inner.boxes["nl"].w.copy_(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    vals = g(feeds={"i": torch.tensor([3.0, 4.0])})
    got = vals["o"].tolist()                       # x @ W^T = [7, 4]
    assert abs(got[0] - 7.0) < 1e-5 and abs(got[1] - 4.0) < 1e-5, got
    paths = g.collect()
    assert "c/ni" in paths and "c/nl" in paths and "c/no" in paths, sorted(paths)
    assert "c" in paths, "the nested box carries its own output value"
    res = T.run(c, {"i": [3.0, 4.0]}, input_id="i", output_id="o")
    assert res["values"]["c/no"]["shape"] == [2]


check("nested ports, paths and values", nested_forward)


def ops_shapes():
    c = canvas([
        n("input", "i", dout=8),
        n("op", "r", op="reshape", dims="4x2"),
        n("op", "t", op="transpose"),
        n("input", "k", dout=12),
        n("op", "k2", op="reshape", dims="4x3"),
        n("op", "mm", op="matmul"),
        n("input", "p", dout=3),
        n("input", "q", dout=5),
        n("op", "cc", op="concat"),
        n("input", "g", dout=8),
        n("op", "rp2", op="reshape", dims="2x4"),
        n("op", "rp", op="rope"),
        n("output", "o", din=8),
    ], [edge("i", "r"), edge("r", "t"), edge("k", "k2"),
        edge("t", "mm", 0, 0), edge("k2", "mm", 0, 1),
        edge("p", "cc", 0, 0), edge("q", "cc", 0, 1),
        edge("g", "rp2"), edge("rp2", "rp"), edge("rp", "o")])
    feeds = {"i": [1, 2, 3, 4, 5, 6, 7, 8], "k": list(range(12)),
             "p": [1, 2, 3], "q": [4, 5, 6, 7, 8], "g": list(range(8))}
    v = T.run(c, feeds)["values"]
    assert v["r"]["shape"] == [4, 2], v["r"]
    assert v["t"]["shape"] == [2, 4], v["t"]
    assert v["k2"]["shape"] == [4, 3], v["k2"]
    assert v["mm"]["shape"] == [2, 3], v["mm"]
    assert v["cc"]["shape"] == [8], v["cc"]
    assert v["rp"]["shape"] == [2, 4], v["rp"]
    peek = v["rp"]["peek"]
    assert peek[:4] == [0.0, 1.0, 2.0, 3.0], peek          # position 0 untouched
    assert any(abs(a - b) > 0.1 for a, b in zip(peek[4:], [4, 5, 6, 7])), peek
    c["nodes"][1]["dims"] = "3x3"                          # 8 values ≠ 9
    try:
        T.run(c, feeds)
        raise AssertionError("no error for a bad reshape")
    except T.CanvasError as e:
        assert e.box == "r", e


check("op shapes (reshape, transpose, concat, matmul, rope)", ops_shapes)


def training_works():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    vocab = sorted(set(text))
    V = len(vocab)
    c = canvas([
        n("input", "tok", name="tokens", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=8),
        n("layer", "l1", din=8, dout=16),
        n("act", "act1", act="silu"),
        n("layer", "l2", din=16, dout=V),
        n("output", "logits", name="logits", din=V),
    ], [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "act1"),
        edge("act1", "l2"), edge("l2", "logits")])
    seen = []
    cfg = {"ctx": 8, "batch": 8, "steps": 60, "lr": 0.01, "seed": 3,
           "eval_every": 0, "sample_every": 20, "sample_len": 24, "optimizer": "adamw"}
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "ckpt.pt")
        res = T.train(c, text, "tok", "logits", cfg, device="cpu",
                      progress=lambda **kw: seen.append(kw), checkpoint_path=ckpt)
        assert res["loss"] < seen[1]["loss"], (res["loss"], seen[1]["loss"])
        assert res["samples"] and isinstance(res["samples"][-1]["text"], str)
        assert os.path.isfile(ckpt)
        blob = torch.load(ckpt, weights_only=False)
        assert blob["vocab"] == vocab and "state_dict" in blob
    assert res["trainable_params"] > 0
    assert any(k.get("trainable_params") for k in seen if k.get("trainable_params"))


check("char-level training: loss falls, samples, checkpoint", training_works)


def frozen_skipped():
    text = "abcdefg abcdefg abcdefg abcdefg abcdefg"
    V = len(set(text))
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=8),
        n("layer", "l1", din=8, dout=8, frozen=True),
        n("act", "act1", act="silu"),
        n("layer", "l2", din=8, dout=V),
        n("output", "logits", din=V),
    ], [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "act1"),
        edge("act1", "l2"), edge("l2", "logits")])
    d = T.describe(c)
    frozen, total = 8 * 8, V * 8 + 8 * 8 + 8 * V
    assert d["counts"]["params"] == total, (d["counts"], total)
    assert d["counts"]["frozen"] == frozen and d["counts"]["trainable"] == total - frozen
    with tempfile.TemporaryDirectory() as tmp:
        tr = T.Trainer(c, text, "tok", "logits",
                       {"ctx": 4, "batch": 4, "steps": 3, "lr": 0.01,
                        "eval_every": 0, "sample_every": 0})
        assert tr.trainable_params == total - frozen, (tr.trainable_params, total - frozen)
        assert not tr.graph.boxes["l1"].w.requires_grad
        assert tr.graph.boxes["l2"].w.requires_grad
        tr.save(os.path.join(tmp, "x.pt"))


check("freeze is read through and skipped", frozen_skipped)


def generate_samples():
    V = 6
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=4),
        n("layer", "l", din=4, dout=V),
        n("output", "logits", din=V),
    ], [edge("tok", "emb"), edge("emb", "l"), edge("l", "logits")])
    res = T.run(c, {"tok": [2]}, generate=5, input_id="tok", output_id="logits",
                temperature=0.9, sample_seed=7)
    assert len(res["generated"]) == 5 and all(0 <= i < V for i in res["generated"])


check("generate samples valid ids", generate_samples)


def integer_inputs_work():
    c = canvas([
        n("input", "i", dout=1),
        n("layer", "l", din=1, dout=2),
        n("output", "o", din=2),
    ], [edge("i", "l"), edge("l", "o")])
    res = T.run(c, {"i": [1]})                 # no decimals: arrives as int64
    assert res["values"]["o"]["shape"] == [2], res["values"]["o"]


check("an integer input feeds a layer", integer_inputs_work)


def gguf_tensor_loads():
    from gguf import GGUFWriter
    arr = np.arange(6, dtype=np.float32).reshape(3, 2)      # logical [out=3, in=2]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.gguf")
        w = GGUFWriter(path, "llama")
        w.add_tensor("test.weight", arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        t = T.load_gguf_tensor(path, "test.weight")
        assert tuple(t.shape) == (3, 2) and torch.allclose(t, torch.from_numpy(arr)), t
        c = canvas([
            n("input", "i", dout=2),
            n("layer", "l", din=2, dout=3, t="test.weight"),
            n("output", "o", din=3),
        ], [edge("i", "l"), edge("l", "o")])
        g = T.compile_canvas(c, gguf_file=path)
        assert torch.allclose(g.boxes["l"].w, torch.from_numpy(arr))
        assert abs(float(g.boxes["l"].w.detach()[0, 0]) - 0.0) < 1e-6


check("GGUF tensor: dequantized and oriented [out, in]", gguf_tensor_loads)


def missing_wire_says_so():
    c = canvas([
        n("input", "i", dout=2),
        n("layer", "l", din=2, dout=2),
        n("output", "o", din=2),
    ], [edge("l", "o")])
    try:
        T.run(c, {"i": [1.0, 2.0]})
        raise AssertionError("a layer with no input dot ran anyway")
    except T.CanvasError as e:
        assert e.box == "l", e


check("an unwired box is refused, by name", missing_wire_says_so)

if FAILED:
    print("nn_torch_test: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("nn_torch_test: all assertions passed")
