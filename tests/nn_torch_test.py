#!/usr/bin/env python3
"""The torch runner's tests, all offline and on synthetic designs:

  * a known-weight forward pass matches the arithmetic, every box captured
  * a nested box's ports, paths and values
  * the ops' shapes (reshape, transpose, concat, matmul, rope)
  * char-level training: the loss falls, samples draw, the checkpoint lands
  * frozen boxes are counted and skipped, through nested levels
  * the run's mode: inference holds no gradients; training keeps them and
    reads the dataset boxes
  * the generate loop samples valid ids; integer feeds reach a layer
  * a GGUF tensor loads dequantized and oriented [out, in]
  * describe() counts trainable vs frozen; an unwired box is refused by name
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
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


def run_modes():
    V = 4
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=8),
        n("layer", "l1", din=8, dout=8, frozen=True),
        n("layer", "l2", din=8, dout=V),
        n("output", "logits", din=V),
        n("dataset", "data", vocab="abcd", text="ab\tcd\nxy\tzq"),
        n("loss", "loss", din=V),
    ], [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "l2"),
        edge("l2", "logits"),
        edge("data", "loss", 0, 1), edge("logits", "loss", 0, 0)])
    # the default is an inference graph: nothing asks for gradients
    g = T.compile_canvas(c)
    assert not g.boxes["l2"].w.requires_grad
    assert not g.boxes["l1"].w.requires_grad
    # a training graph: the non-frozen weights keep gradients, frozen ones do not
    g = T.compile_canvas(c, train=True)
    assert g.boxes["l2"].w.requires_grad
    assert not g.boxes["l1"].w.requires_grad
    # the loss of a training run keeps its graph, and backprop reaches the
    # non-frozen weights only
    ds = [x for x in c["nodes"] if x.get("kind") == "dataset"][0]
    rows = T.dataset_rows(ds)
    ids, labels = T.dataset_batch(ds, rows, 0, 4)
    g.boxes["data"].batch = labels
    loss = g(feeds={"tok": ids})["loss"]
    assert loss.requires_grad, "a training run's loss must keep its graph"
    loss.backward()
    assert g.boxes["l2"].w.grad is not None, "gradients must reach the non-frozen weights"
    assert g.boxes["l1"].w.grad is None, "frozen weights take no gradient"
    # a training run reads the datasets: the loss runs on their rows
    res = T.run(c, {}, mode="training")
    assert res["values"]["loss"]["shape"] == [], res["values"]["loss"]
    assert res["values"]["loss"]["scalar"] is not None
    # an inference run of the same design still refuses its idle dataset
    try:
        T.run(c, {}, mode="inference")
        raise AssertionError("an inference run must not feed the dataset")
    except T.CanvasError:
        pass
    # without a dataset box, a training run is just a forward that keeps grads
    c2 = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=8),
        n("layer", "l2", din=8, dout=V),
        n("output", "logits", din=V),
    ], [edge("tok", "emb"), edge("emb", "l2"), edge("l2", "logits")])
    assert not T.compile_canvas(c2)(feeds={"tok": torch.tensor([0, 1])})["logits"].requires_grad
    assert T.compile_canvas(c2, train=True)(feeds={"tok": torch.tensor([0, 1])})["logits"].requires_grad
    res = T.run(c2, {"tok": [0, 1]}, mode="training")
    assert res["values"]["logits"]["shape"] == [2, V], res["values"]["logits"]


check("the run's mode: inference holds no grads, training keeps them", run_modes)


def generate_samples():
    V = 6
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=V, din=1, dout=4),
        n("layer", "l", din=4, dout=V),
        n("output", "logits", din=V),
    ], [edge("tok", "emb"), edge("emb", "l"), edge("l", "logits")])
    steps = []
    res = T.run(c, {"tok": [2]}, generate=5, input_id="tok", output_id="logits",
                temperature=0.9, sample_seed=7, on_step=steps.append)
    assert len(res["generated"]) == 5 and all(0 <= i < V for i in res["generated"])
    assert steps == [0, 1, 2, 3, 4, 5], \
        "one step per forward pass — prompt then each token: %r" % (steps,)


check("generate samples valid ids", generate_samples)


def text_pipeline_generates():
    """A tokenizer -> model -> de-tokenizer design generates through the same
    loop: the ids the tokenizer made of its text grow, and the finished text
    lands on the last forward."""
    import nn_tokenizers as nn_hf
    repo = "unsloth/Qwen3.8-27B-NVFP4"
    if not os.path.isfile(nn_hf.tokenizer_path(repo, download=False)):
        print("  skip text generation (no cached tokenizer for %s)" % repo)
        return
    c = canvas([
        n("tokenize", "tokz", repo=repo, text="The capital of France is"),
        n("embed", "emb", vocab=248320, din=1, dout=8),
        n("layer", "fc", din=8, dout=16),
        n("detokenize", "detok", repo=repo),
        n("output", "logits", din=16),
    ], [edge("tokz", "emb"), edge("emb", "fc"), edge("fc", "logits"),
        edge("tokz", "detok")])
    res = T.run(c, {}, generate=1, input_id="tokz", output_id="logits", sample_seed=1)
    assert len(res["generated"]) == 1 and 0 <= res["generated"][0] < 16, res["generated"]
    text = res["texts"].get("detok", "")
    assert text.startswith("The capital of France is"), repr(text)
    assert len(text) > len("The capital of France is"), \
        "the finished text carries the sampled token too: %r" % text
    assert res["values"]["logits"]["shape"] == [6, 16], \
        "the last forward runs the prompt plus the token it made: %s" % res["values"]["logits"]


check("a text pipeline generates: the tokenizer's ids grow, the text lands",
      text_pipeline_generates)


def generate_needs_ids():
    """Asked to generate from something that is no token ids, the run says so
    by name — it never grows a mystery tensor."""
    c = canvas([
        n("input", "i", dout=1),
        n("embed", "emb", vocab=16, din=1, dout=4),
        n("output", "logits", din=4),
    ], [edge("i", "emb"), edge("emb", "logits")])
    try:
        T.run(c, {"i": [1.5]}, generate=1, input_id="i", output_id="logits")
    except T.CanvasError as e:
        assert "token ids" in str(e), str(e)
        return
    raise AssertionError("generating from non-ids should refuse clearly")


check("generate refuses a feed that is not token ids", generate_needs_ids)


def greedy_at_zero():
    """Temperature 0 is greedy — the loop and the sample box take the argmax."""
    row = torch.tensor([0.1, 3.0, 0.2])
    assert T._pick_one(row, 0, 0, 1.0, None) == 1, "the generate loop is greedy"
    node = {"id": "s", "kind": "sample", "temperature": 0, "sample_seed": 1}
    assert T.sample_one(row, node).tolist() == [1], "the sample box is greedy"


check("temperature 0 picks the most likely id", greedy_at_zero)


def agent_boxes_run_no_maths():
    """An agent box is a view of an agent's task: a design that carries one
    still runs, and the box holds no values."""
    c = canvas([
        n("input", "i", dout=2),
        {"id": "ag", "kind": "agent", "name": "agent", "x": 120, "y": 0},
        n("layer", "l", din=2, dout=2),
        n("output", "o", din=2),
    ], [edge("i", "l"), edge("l", "o")])
    res = T.run(c, {"i": [1.0, 2.0]})
    assert res["values"]["o"]["shape"] == [2], res["values"]
    assert "ag" not in res["values"], res["values"].keys()


check("an agent box rides along inertly \u2014 the design still runs",
      agent_boxes_run_no_maths)


def integer_inputs_work():
    c = canvas([
        n("input", "i", dout=1),
        n("layer", "l", din=1, dout=2),
        n("output", "o", din=2),
    ], [edge("i", "l"), edge("l", "o")])
    res = T.run(c, {"i": [1]})                 # no decimals: arrives as int64
    assert res["values"]["o"]["shape"] == [2], res["values"]["o"]


check("an integer input feeds a layer", integer_inputs_work)


def zero_centered_norm_and_residual_add():
    c = canvas([
        n("input", "x", dout=4),
        n("input", "r", dout=4),
        n("norm", "pn", dout=4),
        n("norm", "zn", dout=4, zc=True),
        n("op", "z", op="zeros"),
        n("op", "add", op="add"),
        n("output", "o", din=4),
    ], [edge("x", "pn"), edge("x", "zn"), edge("x", "z", 0, 0),
       edge("pn", "add", 0, 0), edge("zn", "add", 0, 1),
       edge("r", "add", 0, 2), edge("z", "add", 0, 3),
       edge("add", "o")])
    x = [2.0, -1.0, 0.0, 4.0]
    m = (sum(v * v for v in x) / len(x)) ** 0.5       # rms over the mean
    g = T.compile_canvas(c)
    with torch.no_grad():
        g.boxes["pn"].w.fill_(0.0)          # plain norm, zero weight -> zeros
        g.boxes["zn"].w.fill_(0.0)          # zero-centered, zero delta -> identity
    vals = g(feeds={"x": torch.tensor(x),
                    "r": torch.tensor([1.0, 1.0, 1.0, 1.0])})
    want = [v / m + 1.0 for v in x]         # 0 + x/m + 1 + 0
    assert all(abs(a - b) < 1e-6 for a, b in zip(vals["o"].tolist(), want)), vals["o"].tolist()
    with torch.no_grad():
        g.boxes["zn"].w.fill_(2.0)          # (1 + 2) * x / rms
    vals = g(feeds={"x": torch.tensor(x),
                    "r": torch.tensor([0.0, 0.0, 0.0, 0.0])})
    want = [3.0 * v / m for v in x]
    assert all(abs(a - b) < 1e-6 for a, b in zip(vals["o"].tolist(), want)), vals["o"].tolist()


check("a zero-centered norm is the identity at zero; add sums its feeds", zero_centered_norm_and_residual_add)


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


def per_box_dtype_halves_the_weights():
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "l", din=4, dout=3, dtype="f16"),
        n("output", "o", din=3),
    ], [edge("i", "l"), edge("l", "o")])
    g = T.compile_canvas(c)
    assert g.boxes["l"].w.dtype == torch.float16, g.boxes["l"].w.dtype
    assert g.weight_report()["bytes"] == 3 * 4 * 2, g.weight_report()
    res = T.run(c, {"i": [1.0, 2.0, 3.0, 4.0]})
    assert res["values"]["o"]["shape"] == [3], res["values"]["o"]

    # quantised dtypes land at half once dequantised, declared f32 stays f32,
    # and the mixed chain still computes (in f32)
    cq = canvas([
        n("input", "i", dout=4),
        n("layer", "q", din=4, dout=2, dtype="q4_k"),
        n("norm", "rn", norm="rms", size=2, dtype="f32"),
        n("output", "o", din=2),
    ], [edge("i", "q"), edge("q", "rn"), edge("rn", "o")])
    gq = T.compile_canvas(cq)
    assert gq.boxes["q"].w.dtype == torch.float16, gq.boxes["q"].w.dtype
    assert gq.boxes["rn"].w.dtype == torch.float32, gq.boxes["rn"].w.dtype
    res = T.run(cq, {"i": [1.0, 2.0, 3.0, 4.0]})
    assert res["values"]["o"]["shape"] == [2], res["values"]["o"]

    ce = canvas([
        n("input", "ids", dout=4),
        n("embed", "e", vocab=5, dout=2, dtype="q6_k"),
        n("output", "eo", din=2),
    ], [edge("ids", "e"), edge("e", "eo")])
    ge = T.compile_canvas(ce)
    assert ge.boxes["e"].w.dtype == torch.float16, ge.boxes["e"].w.dtype
    res = T.run(ce, {"ids": [1.0, 2.0, 3.0, 4.0]})
    assert res["values"]["eo"]["shape"] == [4, 2], res["values"]["eo"]


check("per-box dtype: f32 / f16 / quantised storage, f32 maths",
      per_box_dtype_halves_the_weights)


def estimate_reads_the_design():
    inner = {"type": "nested", "version": 1,
             "nodes": [n("input", "ni", dout=4),
                       n("layer", "nl", din=4, dout=3, dtype="f16"),
                       n("output", "no", din=3)],
             "edges": [edge("ni", "nl"), edge("nl", "no")]}
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "q", din=4, dout=1024, dtype="q4_k"),        # quantised -> f16
        n("custom", "m", inner=inner),                          # nested, inherits
        n("layer", "sys", din=3, dout=2, place="system"),
        n("output", "o", din=2),
    ], [])
    est = T.estimate_canvas(c, "cuda")
    assert est["gpu_bytes"] == 4096 * 2 + 12 * 2, est
    assert est["system_bytes"] == 6 * 4, est                    # no dtype -> f32
    assert est["bytes"] == est["gpu_bytes"] + est["system_bytes"], est
    assert est["boxes"] == 3, est
    est2 = T.estimate_canvas(c, "cpu")
    assert est2["system_bytes"] == est2["bytes"] and est2["gpu_bytes"] == 0, est2
    assert T.estimate_node({"kind": "layer", "din": 3, "dout": 4, "dtype": "f32"}) == (12, 48)
    assert T.estimate_node({"kind": "norm"}) == (0, 0)


check("estimate: what a compile would hold, per device, from the JSON alone",
      estimate_reads_the_design)


def estimate_understands_packed_gguf():
    # a bound gguf layer: packed bytes on a gpu with a native kernel, f16 off it
    node = {"kind": "layer", "din": 4096, "dout": 4096, "dtype": "q4_k",
            "wbytes": 10_000_000, "params": 16_777_216}
    assert T.estimate_node(node, native=True) == (16_777_216, 10_000_000)
    assert T.estimate_node(node, native=False) == (16_777_216, 16_777_216 * 2)
    # a type with no kernel still dequantizes, wherever it runs
    assert T.estimate_node(dict(node, dtype="iq1_s"), native=True)[1] == 16_777_216 * 2
    # packed safetensors storage keeps its byte count without the flag
    assert T.estimate_node({"kind": "layer", "din": 4096, "dout": 4096,
                            "dtype": "u8", "wbytes": 1_000_000,
                            "params": 8_000_000})[1] == 1_000_000
    # embedding tables stay packed with kernels, f16 without
    assert T.estimate_node({"kind": "embed", "vocab": 1000, "dout": 64,
                            "dtype": "q6_k", "wbytes": 100_000}, native=True)[1] == 100_000
    assert T.estimate_node({"kind": "embed", "vocab": 1000, "dout": 64,
                            "dtype": "q6_k", "wbytes": 100_000}, native=False)[1] == 128_000
    # a gpu-placed canvas counts the packed bytes on a machine with kernels
    canvas = {"nodes": [{"id": "l", "kind": "layer", "din": 4096, "dout": 4096,
                         "dtype": "q4_k", "wbytes": 10_000_000, "place": "gpu"}]}
    est = T.estimate_canvas(canvas, "cuda")
    native_here = T.nn_quant is not None and T.nn_quant.gpu_ok()
    assert est["gpu_bytes"] == (10_000_000 if native_here else 4096 * 4096 * 2), est


check("estimate: native gguf storage counts packed, not a phantom f16 (50.9 GB bug)",
      estimate_understands_packed_gguf)


# ---------------------------------------------------------------- hf models
def hf_box_takes_text_gives_the_next_token():
    # a stubbed model in the engine's cache: the plumbing, without a download
    import types

    class Tok:
        def __call__(self, text, return_tensors=None):
            return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, ids, skip_special_tokens=False):
            return {3: "!", 4: "?"}.get(int(ids[0]), "?")

    class Model:
        def __call__(self, ids):
            logits = torch.zeros(1, ids.shape[1], 8)
            logits[0, -1, 3] = 9.0                 # always pick id 3 -> "!"
            return types.SimpleNamespace(logits=logits)

    T._HF_MODELS[("test/tiny", "float32", "cpu")] = (Tok(), Model())
    try:
        c = canvas([n("hf", "h", repo="test/tiny")], [])
        res = T.run(c, {"h": "Once upon a"})       # the prompt arrives as a feed
        assert res["texts"]["h"] == "!", res
        c2 = canvas([n("hf", "h2", repo="test/tiny", text="its own prompt")], [])
        res2 = T.run(c2, {})                       # …or the box's own text
        assert res2["texts"]["h2"] == "!", res2
        assert T.hf_loaded("test/tiny") and not T.hf_loaded("nope/nope")
        try:
            T.run(canvas([n("hf", "h3")], []), {})
            raise AssertionError("a repo-less hf box must refuse")
        except T.CanvasError as e:
            assert "repo" in str(e), e
            assert e.nid == "h3" or e.box == "h3", (e.box, e.nid)
    finally:
        T._HF_MODELS.clear()


check("an hf box: text (or ids) in, the next token's text out",
      hf_box_takes_text_gives_the_next_token)


def hf_file_reference_binds():
    # an `hf:<repo>/<file>` reference: the engine fetches it once — the file is
    # already in the cache here, so nothing touches the network
    cache = os.path.join(os.path.expanduser("~"), ".cache", "nnstudio", "hf",
                         "test", "tiny")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "model.safetensors")
    T.write_safetensors(path, {"q.weight": torch.ones(3, 2)})
    try:
        w = T.load_weight_tensor("hf:test/tiny/model.safetensors", "q.weight")
        assert tuple(w.shape) == (3, 2), w
        c = canvas([n("input", "i", dout=2),
                    n("layer", "l", din=2, dout=3, dtype="f16",
                      file="hf:test/tiny/model.safetensors", t="q.weight"),
                    n("output", "o", din=3)],
                   [edge("i", "l"), edge("l", "o")])
        res = T.run(c, {"i": [1.0, 2.0]})
        assert res["values"]["o"]["shape"] == [3], res["values"]["o"]
        assert abs(res["values"]["o"]["mean"] - 3.0) < 1e-6, res["values"]["o"]
        assert T.compile_canvas(c).boxes["l"].w.dtype == torch.float16, "stored at f16"
    finally:
        os.remove(path)


check("an hf: file reference binds like any safetensors — cache first",
      hf_file_reference_binds)


def head_ops_do_attention_shapes():
    # permute: a general axis order, not just the last two
    c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0\n4.0 5.0 6.0"),
                n("op", "p", op="permute", axes="1,0"),
                n("output", "o", din=6)], [edge("i", "p"), edge("p", "o")])
    res = T.run(c, {})
    o = res["values"]["o"]
    assert o["shape"] == [3, 2] and o["peek"] == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], o

    # reshape with a -1: the sequence dim is inferred
    c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0\n4.0 5.0 6.0"),
                n("op", "r", op="reshape", dims="-1 2"),
                n("output", "o", din=6)], [edge("i", "r"), edge("r", "o")])
    o = T.run(c, {})["values"]["o"]
    assert o["shape"] == [3, 2] and o["peek"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], o

    # repeat: kv heads -> q heads (GQA), interleaved
    c = canvas([n("input", "i", dout=2),
                n("op", "g", op="repeat", repeats="3", dim="0"),
                n("output", "o", din=2)], [edge("i", "g"), edge("g", "o")])
    o = T.run(c, {"i": [[1.0, 2.0]]})["values"]["o"]      # a 1 x 2 matrix
    assert o["shape"] == [3, 2] and o["peek"] == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], o

    # mask: a causal (optionally windowed) -inf fill, then softmax
    c = canvas([n("input", "i", dout=3, values="1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0"),
                n("op", "m", op="mask", mode="causal"),
                n("act", "sm", act="softmax"),
                n("output", "o", din=3)], [edge("i", "m"), edge("m", "sm"), edge("sm", "o")])
    o = T.run(c, {})["values"]["o"]
    assert o["shape"] == [3, 3], o
    row0 = o["peek"][:3]
    assert abs(row0[0] - 1.0) < 1e-6 and abs(row0[1]) < 1e-6 and abs(row0[2]) < 1e-6, row0
    # a window of 1: a position sees only itself
    c2 = canvas([n("input", "i", dout=3, values="1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0"),
                 n("op", "m", op="mask", mode="causal", window=1),
                 n("act", "sm", act="softmax"),
                 n("output", "o", din=3)],
                [edge("i", "m"), edge("m", "sm"), edge("sm", "o")])
    o2 = T.run(c2, {})["values"]["o"]
    assert abs(o2["peek"][3] - 0.0) < 1e-6 and abs(o2["peek"][4] - 1.0) < 1e-6, \
        "row 1 sees only itself: %r" % o2["peek"]

    # scale by a factor: 1/sqrt(head_dim), no weight tensor needed
    c = canvas([n("input", "i", dout=2),
                n("op", "s", op="scale", factor="0.5"),
                n("output", "o", din=2)], [edge("i", "s"), edge("s", "o")])
    o = T.run(c, {"i": [[2.0, 4.0]]})["values"]["o"]      # a 1 x 2 matrix
    assert o["peek"] == [1.0, 2.0], o


check("the head ops: permute, reshape -1, repeat, mask, scale factor",
      head_ops_do_attention_shapes)


def slice_zeros_and_partial_rope():
    # slice by index removes one dimension; by start/stop it keeps a range
    c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0\n4.0 5.0 6.0"),
                n("op", "s1", op="slice", axis=1, index=1),
                n("output", "o1", din=2)], [edge("i", "s1"), edge("s1", "o1")])
    o1 = T.run(c, {})["values"]["o1"]
    assert o1["shape"] == [2] and o1["peek"] == [2.0, 5.0], o1

    c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0\n4.0 5.0 6.0"),
                n("op", "s2", op="slice", axis=1, start=0, stop=2),
                n("output", "o2", din=4)], [edge("i", "s2"), edge("s2", "o2")])
    o2 = T.run(c, {})["values"]["o2"]
    assert o2["shape"] == [2, 2] and o2["peek"] == [1.0, 2.0, 4.0, 5.0], o2

    # slice index -1: the last slot along the axis \u2014 the last token's vector
    c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0\n4.0 5.0 6.0"),
                n("op", "s3", op="slice", axis=0, index=-1),
                n("output", "o3", din=3)], [edge("i", "s3"), edge("s3", "o3")])
    o3 = T.run(c, {})["values"]["o3"]
    assert o3["shape"] == [3] and o3["peek"] == [4.0, 5.0, 6.0], o3

    # zeros: the same shape, all zero (an elided branch's contribution)
    c = canvas([n("input", "i", dout=3, values="1.0 2.0 3.0"),
                n("op", "z", op="zeros"),
                n("output", "o", din=3)], [edge("i", "z"), edge("z", "o")])
    o = T.run(c, {})["values"]["o"]
    assert o["shape"] == [3] and o["peek"] == [0.0, 0.0, 0.0], o

    # partial rope: only the leading `dim` rotates, the tail is untouched
    import math
    c = canvas([n("input", "i", dout=8, values="1.0 0.0 0.0 0.0\n0.0 1.0 0.0 0.0"),
                n("op", "r", op="rope", theta=10000.0, dim=2),
                n("output", "o", din=8)], [edge("i", "r"), edge("r", "o")])
    o = T.run(c, {})["values"]["o"]
    assert o["shape"] == [2, 4], o
    # `peek` is rounded to 4 decimals by tensor_stats, so compare there
    assert o["peek"][4] == round(-math.sin(1.0), 4), o["peek"]
    assert o["peek"][5] == round(math.cos(1.0), 4), o["peek"]
    assert o["peek"][6:] == [0.0, 0.0], o["peek"]


check("slice, zeros and partial rope (Qwen3_5's non-rotated tail)",
      slice_zeros_and_partial_rope)


def stats_are_strict_json():
    """A stat that is NaN/±inf must leave the JSON valid: the browser refuses
    Infinity, and -Infinity dies at the minus sign."""
    st = T.tensor_stats(torch.tensor([float("-inf"), float("nan"), 2.5,
                                      float("inf")]))
    assert st["mean"] is None and st["min"] is None and st["max"] is None, st
    assert st["peek"][0] is None and st["peek"][1] is None, st["peek"]
    assert st["peek"][2] == 2.5 and st["peek"][3] is None, st["peek"]
    text = json.dumps(st, allow_nan=False)     # strict: raises on any survivor
    assert "Infinity" not in text and "NaN" not in text, text
    st2 = T.tensor_stats(torch.tensor([1.0, 2.0, 3.0]))   # finite: exactly as before
    assert st2["mean"] == 2.0 and st2["max"] == 3.0 and st2["peek"] == [1.0, 2.0, 3.0], st2


check("tensor stats stay strictly JSON, inf and nan included",
      stats_are_strict_json)


def zero_centered_norm_and_residual_add():
    # a zero-centered rms norm: the stored weight is the *delta* from one, so
    # a zero weight is the identity (Qwen3_5's norms); the plain norm scales
    c = canvas([n("input", "i", dout=4, values="1.0 1.0 2.0 2.0"),
                 n("norm", "z", norm="rms", eps=1e-6, size=4, zc=True,
                   values="0.0 0.0 0.0 0.0"),
                 n("output", "o", din=4)], [edge("i", "z"), edge("z", "o")])
    g = T.compile_canvas(c, device="cpu")
    with torch.no_grad():
        g.boxes["z"].w.zero_()                  # a zero weight is the identity
    o = T.run_graph(g, c, {})["values"]["o"]
    # x * rsqrt(mean(x^2) + eps) * (1 + 0)
    import math
    s = 1.0 / math.sqrt(2.5 + 1e-6)
    assert o["peek"] == [round(s, 4), round(s, 4), round(2 * s, 4), round(2 * s, 4)], o
    # and the same norm without zc scales by the weight: a zero weight is zero
    c2 = canvas([n("input", "i", dout=4, values="1.0 1.0 2.0 2.0"),
                  n("norm", "p", norm="rms", eps=1e-6, size=4),
                  n("output", "o", din=4)], [edge("i", "p"), edge("p", "o")])
    g2 = T.compile_canvas(c2, device="cpu")
    with torch.no_grad():
        g2.boxes["p"].w.zero_()
    o2 = T.run_graph(g2, c2, {})["values"]["o"]
    assert o2["peek"] == [0.0, 0.0, 0.0, 0.0], o2

    # a residual add takes every input as an addend: the live branch's output
    # plus the zero its elided twin would have contributed
    c = canvas([n("input", "a", dout=2, values="1.0 2.0"),
                 n("input", "b", dout=2, values="3.0 4.0"),
                 n("op", "z", op="zeros"),
                 n("op", "sum", op="add"),
                 n("output", "o", din=2)],
                [edge("a", "sum", 0, 0), edge("b", "sum", 0, 1), edge("a", "z"),
                 edge("z", "sum", 0, 2), edge("sum", "o")])
    o = T.run(c, {})["values"]["o"]
    assert o["shape"] == [2] and o["peek"] == [4.0, 6.0], o


check("zero-centered norms; a residual add sums its elided branch's zero",
      zero_centered_norm_and_residual_add)


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


# ---------------------------------------------------------------- weight files
def safetensors_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.safetensors")
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.ones(3, dtype=torch.float16)
        T.write_safetensors(path, {"l.weight": t, "l.bias": b}, metadata={"app": "test"})
        meta = T.file_tensors(path)
        assert meta["kind"] == "safetensors" and meta["count"] == 2, meta
        by = {x["name"]: x for x in meta["tensors"]}
        assert by["l.weight"]["shape"] == [3, 4] and by["l.weight"]["dtype"] == "F32", by
        assert by["l.bias"]["dtype"] == "F16" and by["l.bias"]["shape"] == [3], by
        assert meta["metadata"] == {"app": "test"}, meta["metadata"]
        assert torch.allclose(T.load_weight_tensor(path, "l.weight"), t)
        assert T.load_weight_tensor(path, "l.bias").dtype == torch.float32, \
            "weights arrive in the engine's working dtype"
        # a canvas bound to it runs on those weights, not the random init
        c = canvas([
            n("input", "i", dout=4),
            n("layer", "l", din=4, dout=3, file=path, t="l.weight,l.bias"),
            n("output", "o", din=3),
        ], [edge("i", "l"), edge("l", "o")])
        g = T.compile_canvas(c, device="cpu")
        assert torch.allclose(g.boxes["l"].w, t), "the bound weight is the real one"
        assert torch.allclose(g.boxes["l"].bias, b.float()), "…and its bias"
        res = T.run_graph(g, c, {"i": [1, 2, 3, 4]})
        want = t @ torch.tensor([1.0, 2, 3, 4]) + b.float()
        got = torch.tensor(res["values"]["o"]["peek"][:3])
        assert torch.allclose(got, want, atol=1e-4), (got, want)


check("safetensors: metadata, load, and a run on real weights", safetensors_roundtrip)


def pt_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.pt")
        t = torch.randn(5, 2)
        torch.save({"l.weight": t, "step": 3, "opt": {"lr": 0.1}}, path)
        meta = T.file_tensors(path)
        assert meta["kind"] == "pt", meta
        assert [x["name"] for x in meta["tensors"]] == ["l.weight"], meta
        assert meta["tensors"][0]["shape"] == [5, 2], meta
        assert torch.allclose(T.load_weight_tensor(path, "l.weight"), t)
        path2 = os.path.join(d, "w2.pt")
        torch.save({"state_dict": {"a.weight": t}}, path2)
        assert torch.allclose(T.load_weight_tensor(path2, "a.weight"), t), "state dict"


check("pt: metadata and load (state dicts included)", pt_roundtrip)


def export_roundtrip():
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "l", name="fc 1", din=4, dout=3),
        n("embed", "e", name="embedding 1", vocab=7, din=1, dout=2),
        n("output", "o", din=3),
    ], [edge("i", "l"), edge("l", "o")])
    import json as _json
    for fmt in ("safetensors", "pt"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out." + fmt)
            man = T.export_weights(c, path, fmt)
            assert man["keys"] == 2, man
            keys = {m["key"]: m for m in man["tensors"]}
            assert "fc 1.weight" in keys and "embedding 1.weight" in keys, keys
            assert keys["fc 1.weight"]["shape"] == [3, 4], keys
            meta = T.file_tensors(path)
            assert {x["name"] for x in meta["tensors"]} == set(keys), meta
            # point the boxes at their keys and run: exactly the exported values
            c2 = _json.loads(_json.dumps(c))
            for node in c2["nodes"]:
                mine = [m["key"] for m in man["tensors"] if m["id"] == node["id"]]
                if mine:
                    node["file"] = path
                    node["t"] = ",".join(mine)
            g1 = T.compile_canvas(c, device="cpu")
            g2 = T.compile_canvas(c2, device="cpu")
            assert torch.allclose(g1.boxes["l"].w, g2.boxes["l"].w), fmt
            assert torch.allclose(g1.boxes["e"].w, g2.boxes["e"].w), fmt


check("export weights: one file, a manifest, and a run from it", export_roundtrip)


# ---------------------------------------------------------------- text boxes
def tokenizer_roundtrip():
    c = canvas([
        n("tokenize", "t", vocab="ab c"),
        n("detokenize", "d", vocab="ab c"),
    ], [edge("t", "d")])
    res = T.run(c, {"t": "cab"})
    assert res["texts"]["d"] == "cab", res["texts"]
    assert "t" in res["values"], "the ids travel the wire as a tensor"
    assert T.run(c, {"t": "caxb"})["texts"]["d"] == "cab", "chars outside the vocab are skipped"
    # a box-level text runs with no feed at all
    c2 = canvas([
        n("tokenize", "t", vocab="ab", text="aab"),
        n("detokenize", "d", vocab="ab"),
    ], [edge("t", "d")])
    assert T.run(c2, {})["texts"]["d"] == "aab"
    try:
        T.run(canvas([n("tokenize", "t", vocab="ab"), n("detokenize", "d", vocab="ab")],
                     [edge("t", "d")]), {})
        raise AssertionError("ran without any text")
    except T.CanvasError as e:
        assert e.box == "t", e


check("tokenizer -> de-tokenizer: text in, text out", tokenizer_roundtrip)


def detokenizer_takes_ids():
    c = canvas([
        n("input", "i", dout=3),
        n("detokenize", "d", vocab="abc", din=3),
    ], [edge("i", "d")])
    assert T.run(c, {"i": [2, 0]})["texts"]["d"] == "ca", "ids go in as they are"
    assert T.run(c, {"i": []})["texts"]["d"] == "", "an empty sequence is empty text"
    try:                                       # logits are the sample box's job
        T.run(c, {"i": [0.1, 5.0, 0.2]})
        raise AssertionError("took logits")
    except T.CanvasError as e:
        assert e.box == "d" and "sample" in str(e), e
    try:
        T.run(canvas([n("input", "i", dout=2), n("detokenize", "d", din=2)], [edge("i", "d")]),
              {"i": [0, 1]})
        raise AssertionError("de-tokenized into nothing")
    except T.CanvasError as e:
        assert e.box == "d" and "vocabulary" in str(e), e


check("a de-tokenizer: ids only, and it says when the vocab is empty",
      detokenizer_takes_ids)


def sample_picks_one():
    def pick(seed, temp):
        c = canvas([
            n("input", "i", dout=3),
            n("sample", "s", temperature=temp, sample_seed=seed),
            n("detokenize", "d", vocab="abc"),
        ], [edge("i", "s"), edge("s", "d")])
        out = T.run(c, {"i": [0.0, 0.0, 0.0]})
        assert out["values"]["s"]["shape"] == [1], out["values"]["s"]
        return out["texts"]["d"]
    c = canvas([
        n("input", "i", dout=3),
        n("sample", "s", temperature=0),
        n("detokenize", "d", vocab="abc"),
    ], [edge("i", "s"), edge("s", "d")])
    assert T.run(c, {"i": [0.1, 5.0, 0.2]})["texts"]["d"] == "b", "temperature 0 is greedy"
    assert pick(1, 1.0) == pick(1, 1.0), "the same seed picks the same id"
    got = {pick(s, 2.0) for s in range(1, 9)}
    assert len(got) > 1 and got <= set("abc"), got
    c2 = canvas([
        n("input", "i", values="0.0 9.0 0.0\n0.1 0.2 8.0"),
        n("sample", "s", temperature=0),
        n("detokenize", "d", vocab="abc"),
    ], [edge("i", "s"), edge("s", "d")])
    res = T.run(c2, {})                        # rows [0,9,0] then [0.1,0.2,8]
    assert res["texts"]["d"] == "c", "the last position's row decides: " + str(res["texts"])
    try:
        T.run(c2, {"i": [0, 1, 2, 3, 4, 5]})
        raise AssertionError("sampled ids")
    except T.CanvasError as e:
        assert e.box == "s" and "logits" in str(e), e


check("a sample box: logits in, one id out (greedy, seeded, last position)",
      sample_picks_one)


def input_text_feeds_a_tokenizer():
    c = canvas([
        n("input", "i", input_mode="text", text="cab"),
        n("tokenize", "t", vocab="ab c"),
        n("detokenize", "d", vocab="ab c"),
    ], [edge("i", "t"), edge("t", "d")])
    assert T.run(c, {})["texts"]["d"] == "cab", "the input's own text travels the wire"
    assert T.run(c, {"i": "bca"})["texts"]["d"] == "bca", "…or what the run typed"
    c2 = canvas([
        n("input", "i", input_mode="text", text="cab"),
        n("tokenize", "t", vocab="ab c", text="aabb"),
        n("detokenize", "d", vocab="ab c"),
    ], [edge("i", "t"), edge("t", "d")])
    assert T.run(c2, {})["texts"]["d"] == "cab", "the wire beats the box's own text"
    bad = canvas([
        n("input", "i", dout=2),
        n("tokenize", "t", vocab="ab"),
    ], [edge("i", "t")])
    try:
        T.run(bad, {"i": [1, 2]})
        raise AssertionError("tokenized a tensor")
    except T.CanvasError as e:
        assert e.box == "t" and "text" in str(e), e


check("an input in text mode feeds the tokenizer over the wire",
      input_text_feeds_a_tokenizer)


# ---------------------------------------------------------------- data flow
def bound_tensors_read_through_nesting():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.safetensors")
        T.write_safetensors(path, {
            "deep.layer.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "deep.norm.weight": torch.ones(4),
        })
        innermost = {"type": "nested", "version": 1,
                     "nodes": [n("input", "ii", dout=2),
                               n("layer", "deep layer", din=2, dout=4, t="deep.layer.weight"),
                               n("norm", "deep norm", norm="rms", t="deep.norm.weight"),
                               n("output", "io", din=4)],
                     "edges": [edge("ii", "deep layer"), edge("deep layer", "deep norm"),
                               edge("deep norm", "io")]}
        middle = {"type": "nested", "version": 1,
                  "nodes": [n("input", "mi", dout=2),
                            n("custom", "inner box", inner=innermost),
                            n("output", "mo", din=4)],
                  "edges": [edge("mi", "inner box"), edge("inner box", "mo")]}
        c = canvas([
            n("input", "i", dout=2),
            n("custom", "model", inner=middle, file=path),   # the file lives at the top
            n("output", "o", din=4),
        ], [edge("i", "model"), edge("model", "o")])
        g = T.compile_canvas(c, device="cpu")
        inn = g.boxes["model"].inner.boxes["inner box"].inner
        assert torch.allclose(inn.boxes["deep layer"].w,
                              torch.arange(8, dtype=torch.float32).reshape(4, 2)), "layer bound"
        assert inn.boxes["deep norm"].w is not None and inn.boxes["deep norm"].w.numel() == 4, \
            "the file reads through two nested levels"
        res = T.run(c, {"i": [1.0, 2.0]})
        assert res["values"]["o"]["shape"] == [4], res["values"]["o"]


check("a bound file reads through nested boxes, any depth",
      bound_tensors_read_through_nesting)


def progress_reports_the_data_flow():
    c = canvas([
        n("input", "i", dout=2),
        n("layer", "l", din=2, dout=3),
        n("output", "o", din=3),
    ], [edge("i", "l"), edge("l", "o")])
    seen = []
    T.run(c, {"i": [1.0, 2.0]}, on_node=seen.append)
    assert seen == ["i", "l", "o"], seen
    innermost = {"type": "nested", "version": 1,
                 "nodes": [n("input", "ii", dout=2),
                           n("layer", "nl", din=2, dout=3),
                           n("output", "io", din=3)],
                 "edges": [edge("ii", "nl"), edge("nl", "io")]}
    c2 = canvas([
        n("input", "i2", dout=2),
        n("custom", "c2", inner=innermost),
        n("output", "o2", din=3),
    ], [edge("i2", "c2"), edge("c2", "o2")])
    seen2 = []
    T.run(c2, {"i2": [1.0, 2.0]}, on_node=seen2.append)
    assert seen2[0] == "i2" and seen2[-1] == "o2", seen2
    assert {"c2/ii", "c2/nl", "c2/io"} <= set(seen2), seen2
    assert seen2.index("c2/nl") < seen2.index("c2"), \
        "the inside reports first — a glowing block is one that finished"


check("a run reports every box as its value lands, nested boxes included",
      progress_reports_the_data_flow)


def hf_tokenizer_from_cache():
    try:
        import tokenizers  # noqa: F401
    except Exception:  # noqa: BLE001
        print("  skip huggingface tokenizer (pip install tokenizers)")
        return
    import json as _json
    import nn_tokenizers as HT
    for text, want in (("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
                       ("gpt2", "gpt2"),
                       ("https://huggingface.co/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
                       ("https://huggingface.co/Qwen/Qwen3-8B/tree/main", "Qwen/Qwen3-8B"),
                       ("https://huggingface.co/gpt2/resolve/main/tokenizer.json", "gpt2")):
        assert HT.hf_repo_id(text) == want, (text, HT.hf_repo_id(text))
    with tempfile.TemporaryDirectory() as d:
        os.environ["NNSTUDIO_TOKENIZER_CACHE"] = d
        try:
            repo_dir = os.path.join(d, "test__tiny")
            os.makedirs(repo_dir)
            doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
                   "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
                   "post_processor": None, "decoder": None,
                   "model": {"type": "WordLevel",
                             "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "cab": 3},
                             "unk_token": "[UNK]"}}
            with open(os.path.join(repo_dir, "tokenizer.json"), "w") as f:
                _json.dump(doc, f)
            meta = HT.tokenizer_meta("test/tiny")
            assert meta["model"] == "WordLevel" and meta["tokens"] == 4, meta
            c = canvas([
                n("tokenize", "t", repo="test/tiny"),
                n("detokenize", "d", repo="test/tiny"),
            ], [edge("t", "d")])
            res = T.run(c, {"t": "hello world"})
            assert res["values"]["t"]["shape"] == [2], res["values"]["t"]
            assert res["texts"]["d"] == "hello world", res["texts"]
            c2 = canvas([
                n("input", "i", dout=4),
                n("sample", "s", temperature=0),
                n("detokenize", "d", repo="test/tiny"),
            ], [edge("i", "s"), edge("s", "d")])
            assert T.run(c2, {"i": [0.0, 1.0, 9.0, 2.0]})["texts"]["d"] == "world", \
                "greedy sample -> id 2 -> the tokenizer's own vocab"
        finally:
            os.environ.pop("NNSTUDIO_TOKENIZER_CACHE", None)


check("a huggingface tokenizer: the repo loads from the cache, ids round-trip",
      hf_tokenizer_from_cache)


# ---------------------------------------------------------------- offload
def offload_clears_the_caches():
    T._TEXT_TOKENS["probe/tiny"] = object()      # whatever is cached, it goes
    T._READERS["/probe.gguf"] = (0, object())
    T.offload()
    assert T._TEXT_TOKENS == {} and T._READERS == {}, "every cache is empty"
    c = canvas([n("input", "i", dout=2), n("layer", "l", din=2, dout=2)],
               [edge("i", "l")])
    res = T.run(c, {"i": [1.0, 2.0]})
    assert res["values"]["l"]["shape"] == [2], "the engine still runs after"


check("offload: every reader and tokenizer is let go, and the engine still runs",
      offload_clears_the_caches)


# ---------------------------------------------------------------- matrix inputs
def input_python_expression():
    c = canvas([
        n("input", "i", dout=4, values="1.0 2.0 3.0 4.0"),
        n("layer", "l", din=4, dout=3),
        n("output", "o", din=3),
    ], [edge("i", "l"), edge("l", "o")])
    res = T.run(c, {})                           # no feeds: the box defines its own
    st = res["values"]["i"]
    assert st["shape"] == [4] and abs(st["mean"] - 2.5) < 1e-6, st
    g = T.compile_canvas(c, device="cpu")
    want = g.boxes["l"].w @ torch.tensor([1.0, 2.0, 3.0, 4.0])
    got = torch.tensor(res["values"]["o"]["peek"][:3])
    assert torch.allclose(got, want, atol=1e-4), (got, want)


check("a matrix input: the numbers typed on the box, built by it",
      input_python_expression)


def input_matrix_two_rows():
    c = canvas([
        n("input", "i", dout=4, values="1 2\n3 4"),
        n("output", "o", din=4),
    ], [edge("i", "o")])
    res = T.run(c, {})
    assert res["values"]["i"]["shape"] == [2, 2], res["values"]["i"]
    a = T.build_input(n("input", "i", dout=4, values="1 2\n3 4"))
    assert a.dtype == torch.int64, a           # whole numbers stay whole (ids)


check("a matrix input: two rows arrive 2-D", input_matrix_two_rows)


def input_matrix_decimal_rows():
    node = n("input", "i", dout=3, values="1.5, 2, 3")
    a = T.build_input(node)
    assert tuple(a.shape) == (3,) and a.dtype == torch.float32, a   # one row: a vector
    c = canvas([node, n("output", "o", din=3)], [edge("i", "o")])
    res = T.run(c, {})
    assert res["values"]["i"]["shape"] == [3], res["values"]["i"]


check("a matrix input: commas, decimals, one row is a vector", input_matrix_decimal_rows)


def input_feed_wins():
    node = n("input", "i", dout=2, values="9 9")        # the box's own matrix
    out = T.build_feeds(canvas([node], []), {"i": [1, 1]})
    assert list(out["i"]) == [1, 1], "what the caller passed wins"
    out2 = T.build_feeds(canvas([node], []), {})
    assert list(out2["i"]) == [9, 9], out2


check("an explicit feed beats the input's own matrix", input_feed_wins)


def input_errors_are_named():
    for node, word in ((n("input", "i", dout=2, values=""), "matrix is empty"),
                       (n("input", "i", dout=2, values="1 2\n3"), "do not line up"),
                       (n("input", "i", dout=2, values="1 x"), "not a number")):
        try:
            T.build_input(node)
            raise AssertionError("built anyway")
        except T.CanvasError as e:
            assert e.box == "i" and word in str(e), e


check("matrix inputs fail by name, with a reason", input_errors_are_named)


def errors_carry_their_box_id():
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "l", din=4, dout=2),
        n("output", "o", din=2),
    ], [edge("i", "l"), edge("l", "o")])
    try:                                       # no feed for the input
        T.run(c, {})
        raise AssertionError("ran anyway")
    except T.CanvasError as e:
        assert e.nid == "i" and e.box == "i", (e.box, e.nid)
    bad = canvas([
        n("input", "i2", dout=2),
        n("tokenize", "t", vocab="ab"),
    ], [edge("i2", "t")])
    try:                                       # a tensor on the tokenizer's wire
        T.run(bad, {"i2": [1.0, 2.0]})
        raise AssertionError("ran anyway")
    except T.CanvasError as e:
        assert e.nid == "t", (e.box, e.nid)
    innermost = {"type": "nested", "version": 1,
                 "nodes": [n("norm", "dn", norm="rms")], "edges": []}
    middle = {"type": "nested", "version": 1,
              "nodes": [n("custom", "inner", inner=innermost)], "edges": []}
    deep = canvas([n("custom", "model", inner=middle)], [])
    try:                                       # compile-time, nested: the innermost box
        T.compile_canvas(deep, device="cpu")
        raise AssertionError("compiled anyway")
    except T.CanvasError as e:
        assert e.nid == "dn" and e.box == "dn", (e.box, e.nid)


check("an error carries the id of the box it is about, any depth",
      errors_carry_their_box_id)


# ---------------------------------------------------------------- placement
def placement_keeps_boxes_where_told():
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "l", din=4, dout=3, place="system"),
        n("layer", "l2", din=3, dout=2),
        n("output", "o", din=2),
    ], [edge("i", "l"), edge("l", "l2"), edge("l2", "o")])
    g = T.compile_canvas(c, device="cpu")
    assert g.boxes["l"].device == "cpu" and g.boxes["l2"].device == "cpu"
    res = T.run_graph(g, c, {"i": [1, 2, 3, 4]})
    assert res["values"]["o"]["shape"] == [2], res["values"]["o"]
    rep = g.weight_report()
    assert rep["params"] == 4 * 3 + 3 * 2, rep
    assert rep["bytes"] == rep["params"] * 4, rep          # the engine defaults to f32
    assert rep["boxes"] == 2, rep


check("placement: system-placed boxes sit on the cpu, forward still runs",
      placement_keeps_boxes_where_told)


def placement_reads_through_nested():
    inner = {"type": "nested", "version": 1,
             "nodes": [n("input", "ni", dout=4),
                       n("layer", "nl", din=4, dout=3, place="system"),
                       n("output", "no", din=3)],
             "edges": [edge("ni", "nl"), edge("nl", "no")]}
    c = canvas([
        n("input", "i", dout=4),
        n("custom", "c", inner=inner),
        n("output", "o", din=3),
    ], [edge("i", "c"), edge("c", "o")])
    g = T.compile_canvas(c, device="cpu")
    assert g.boxes["c"].inner.boxes["nl"].place == "system"
    res = T.run_graph(g, c, {"i": [1, 2, 3, 4]})
    assert res["values"]["o"]["shape"] == [3], res["values"]["o"]
    rep = g.weight_report()
    assert rep["params"] == 12 and rep["boxes"] == 1, rep   # nested weight, counted at its place


check("placement reads through nested boxes", placement_reads_through_nested)


def placement_splits_devices():
    if not torch.cuda.is_available():
        print("  skip placement across gpu/cpu (no cuda here)")
        return
    c = canvas([
        n("input", "i", dout=4),
        n("layer", "gpu_l", din=4, dout=3),
        n("layer", "sys_l", din=3, dout=2, place="system"),
        n("output", "o", din=2),
    ], [edge("i", "gpu_l"), edge("gpu_l", "sys_l"), edge("sys_l", "o")])
    g = T.compile_canvas(c, device="cuda")
    assert g.boxes["gpu_l"].w.device.type == "cuda", g.boxes["gpu_l"].w.device
    assert g.boxes["sys_l"].w.device.type == "cpu", g.boxes["sys_l"].w.device
    res = T.run_graph(g, c, {"i": [1, 2, 3, 4]})
    assert res["values"]["o"]["shape"] == [2], res["values"]["o"]
    assert res["values"]["o"]["mean"] == res["values"]["o"]["mean"], "no NaN"
    rep = g.weight_report()
    assert rep["on_gpu_bytes"] == 4 * 3 * 4, rep           # gpu_l: 12 params, f32
    assert rep["on_system_bytes"] == 3 * 2 * 4, rep        # sys_l: 6 params, f32
    assert rep["bytes"] == rep["on_gpu_bytes"] + rep["on_system_bytes"]


check("placement splits a graph across gpu and cpu mid-forward", placement_splits_devices)


def storage_dtypes_compile_and_run():
    # every dtype the picker offers: the torch names the engine knows, and a
    # box built on each — ints and bools store, and f32 maths still runs
    want = {"f64": torch.float64, "f32": torch.float32, "f16": torch.float16,
            "bf16": torch.bfloat16, "i64": torch.int64, "i32": torch.int32,
            "i16": torch.int16, "i8": torch.int8, "u8": torch.uint8,
            "bool": torch.bool}
    for name, dt in want.items():
        assert T._DTYPE_NAMES.get(name) is dt, (name, T._DTYPE_NAMES.get(name))
        assert T._DTYPE_BYTES.get(name), name
    for name in ("f8_e4m3", "f8_e5m2"):
        if name in T._DTYPE_NAMES:
            assert T._DTYPE_BYTES.get(name) == 1, name
    for dtype in ("f64", "u8", "i8", "i32"):
        c = canvas([n("input", "i", dout=4, values="1.0 1.0 2.0 2.0"),
                    n("layer", "l", din=4, dout=2, dtype=dtype),
                    n("output", "o", din=2)], [edge("i", "l"), edge("l", "o")])
        res = T.run(c, {})
        assert res["values"]["o"]["shape"] == [2], (dtype, res["values"]["o"])
        assert res["values"]["o"]["mean"] == res["values"]["o"]["mean"], dtype
    # an integer weight cannot ask for gradients — and does not blow up
    c = canvas([n("input", "i", dout=2, values="1.0 2.0"),
                n("layer", "l", din=2, dout=2, dtype="u8"),
                n("output", "o", din=2)], [edge("i", "l"), edge("l", "o")])
    g = T.compile_canvas(c)
    assert g.boxes["l"].w.requires_grad is False, "an int8 weight carries no grad"


check("torch storage dtypes: f64, fp8 and the ints compile and run",
      storage_dtypes_compile_and_run)


def quantized_weight_files():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "q.safetensors")
        # ---- fp8 per-channel: w = fp8 * weight_scale --------------------
        w = torch.tensor([[0.5, -1.25, 2.0, 0.75, -0.5, 1.0],
                          [-1.0, 0.25, -0.75, 1.5, 0.125, -2.0]], dtype=torch.float32)
        scale = torch.tensor([[0.01], [0.02]], dtype=torch.float32)
        q8 = (w / scale).to(torch.float8_e4m3fn)
        T.write_safetensors(path, {"a.weight": q8.contiguous(),
                                   "a.weight_scale": scale,
                                   "a.bias": torch.zeros(2)})
        assert T.load_weight_tensor(path, "a.weight", None).dtype == torch.float8_e4m3fn, \
            "fp8 comes back as stored"
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        c = canvas([n("input", "i", dout=6, values="1.0 2.0 3.0 4.0 5.0 6.0"),
                    n("layer", "a", din=6, dout=2, t="a.weight", file=path, dtype="f8_e4m3"),
                    n("output", "o", din=2)], [edge("i", "a"), edge("a", "o")])
        g = T.compile_canvas(c, device="cpu")
        assert g.boxes["a"].w_scale is not None, "the fp8 channel scale rides along"
        got = g(feeds={"i": x})["o"].float()
        want = (q8.to(torch.float32) * scale) @ x
        assert torch.allclose(got.reshape(-1), want, atol=1e-3), (got, want)

        # ---- packed NVFP4: low nibble first, per-16 scales, stored divisor
        codes = torch.tensor([list(range(16)), list(reversed(range(16)))],
                             dtype=torch.uint8)
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
        bscale = torch.tensor([[0.5], [2.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
        T.write_safetensors(path, {"b.weight_packed": packed,
                                   "b.weight_scale": bscale,
                                   "b.weight_global_scale": torch.tensor([100.0])})
        x2 = torch.arange(1.0, 17.0).reshape(1, 16)
        c2 = canvas([n("input", "i", dout=16),
                     n("layer", "b", din=16, dout=2, t="b.weight_packed",
                       file=path, dtype="u8"),
                     n("output", "o", din=2)], [edge("i", "b"), edge("b", "o")])
        g2 = T.compile_canvas(c2, device="cpu")
        assert g2.boxes["b"].w_packed and g2.boxes["b"].w_global == 100.0, \
            "the packed weight, its block scales and the stored divisor are bound"
        got2 = g2(feeds={"i": x2})["o"].float()
        w2 = (lut[codes.long()].to(torch.float32)
              * bscale.to(torch.float32)) / 100.0
        want2 = w2 @ x2.reshape(-1)
        assert torch.allclose(got2.reshape(-1), want2, atol=1e-3), (got2, want2)


check("quantized weight files: fp8 scales, packed E2M1 with its divisor",
      quantized_weight_files)


def flow_paths_address_each_level():
    c = canvas([
        n("input", "i", dout=2),
        {"id": "m", "kind": "custom", "name": "model", "x": 100, "y": 0,
         "inner": {"nodes": [n("input", "bi", dout=2),
                             n("layer", "l", din=2, dout=2),
                             n("output", "bo", din=2)],
                   "edges": [edge("bi", "l"), edge("l", "bo")]}},
        n("output", "o", din=2)], [edge("i", "m"), edge("m", "o")])
    seq = []
    res = T.run(c, {"i": [1.0, 2.0]}, on_node=seq.append)
    assert seq == ["i", "m/bi", "m/l", "m/bo", "m", "o"], \
        "full paths, and a nested box is marked only when it is done: %r" % seq
    assert res["values"]["m/l"]["shape"] == [2], res["values"]["m/l"]


check("the flow stream carries full paths, nested boxes after their inside",
      flow_paths_address_each_level)


def gemma_primitives():
    """The three things Gemma 4 needs beyond llama: a tanh-approximate GELU,
    transformers' *proportional* RoPE (the whole head dim rotates, but only the
    leading pairs carry a frequency), and a scaled word embedding."""
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    tanh = T._activate("gelu_pytorch_tanh", x)
    assert torch.allclose(tanh, torch.nn.functional.gelu(x, approximate="tanh"))
    assert not torch.allclose(tanh, torch.nn.functional.gelu(x)), \
        "the tanh approximation is its own activation"

    # proportional rope: denominator is the full head dim, pairs past
    # int(partial * head_dim / 2) have inv_freq 0 (an identity rotation)
    d, theta, part = 8, 10000.0, 0.25
    x = torch.arange(1.0, 1.0 + 2 * d).reshape(2, d)
    inv = torch.tensor([theta ** (-2.0 * k / d) for k in range(d // 2)])
    inv[int(part * d // 2):] = 0.0
    ang = torch.arange(2).float().unsqueeze(1) * inv.unsqueeze(0)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    want = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    got = T._rope(x, theta, None, part)
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    assert torch.allclose(T._rope(x, theta, None, None), want) is False, \
        "a plain rope over the same width is a different rotation"
    # the trailing pairs (angles 2,3 here) come back untouched
    tail = got[:, d // 2 + 2:]
    assert torch.allclose(tail, x[:, d // 2 + 2:]), tail

    # a scaled embedding: the table times sqrt(d_model)
    c = canvas([
        n("input", "i", dout=1),
        n("embed", "e", vocab=3, din=1, dout=4, embed_scale=2.5),
        n("output", "o", din=4),
    ], [edge("i", "e"), edge("e", "o")])
    g = T.compile_canvas(c, device="cpu")
    with torch.no_grad():
        g.boxes["e"].w.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    o = T.run_graph(g, c, {"i": [2]})["values"]["o"]
    assert o["peek"] == [20.0, 22.5, 25.0, 27.5], o


check("gemma primitives: tanh gelu, proportional rope, scaled embedding",
      gemma_primitives)


def causal_conv_is_time_major():
    torch.manual_seed(0)
    c, k, t = 6, 4, 5
    w = torch.randn(c, 1, k)
    x = torch.randn(t, c)
    b = type("B", (), {})()
    b.w, b.node, b.name = w, {"act": None}, "conv"
    y = T._causal_conv1d(b, x)
    assert tuple(y.shape) == (t, c), y.shape
    xp = torch.nn.functional.pad(x, (0, 0, k - 1, 0))
    want = torch.zeros(t, c)
    for s in range(t):
        for j in range(k):
            want[s] += xp[s + j] * w[:, 0, j]
    want = want * torch.sigmoid(want)            # the short conv is silu
    assert torch.allclose(y, want, atol=1e-5), (y, want)
    yb = T._causal_conv1d(b, x.unsqueeze(0))     # a batch rides along
    assert tuple(yb.shape) == (1, t, c) and torch.allclose(yb[0], y, atol=1e-5)
    b2 = type("B", (), {})()
    b2.w, b2.node, b2.name = w, {"act": "silu"}, "conv"
    ys = T._causal_conv1d(b2, x)
    assert torch.allclose(ys, want, atol=1e-5), "explicit act ignored"


check("conv1d: time-major [T, C], causal over time, act only when set",
      causal_conv_is_time_major)


def gated_delta_rule_matches_reference():
    """The scan primitive is the gated delta rule. HF checkpoints bind the raw
    `A_log` (the spec marks it `a_log`, so the decay is
    `-exp(A_log) * softplus`); a GGUF binds llama.cpp's `ssm_a`, which already
    holds `-exp(A_log)` and must not be exponentiated again. Both must match
    the library's recurrent formula (`torch_recurrent_gated_delta_rule`)."""
    torch.manual_seed(3)
    B, L = 1, 6
    nk, dk, nv, dv = 2, 3, 4, 4                  # nv // nk == 2: repeat runs
    key_dim, value_dim = nk * dk, nv * dv
    conv = torch.randn(B, L, key_dim * 2 + value_dim)
    b_in = torch.randn(B, L, nv)
    a_in = torch.randn(B, L, nv)
    A_log = torch.randn(nv)
    dt = torch.randn(nv)

    def reference(decay_scale):
        q = conv[..., :key_dim].reshape(B, L, nk, dk)
        k = conv[..., key_dim:2 * key_dim].reshape(B, L, nk, dk)
        v = conv[..., 2 * key_dim:].reshape(B, L, nv, dv)
        q = q.repeat_interleave(nv // nk, dim=2)
        k = k.repeat_interleave(nv // nk, dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))     # [B, heads, T, dim]
        q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        q = q / (dk ** 0.5)
        g = decay_scale * torch.nn.functional.softplus(a_in + dt)
        beta = torch.sigmoid(b_in).transpose(1, 2)
        decay = g.transpose(1, 2).exp()
        state = torch.zeros(B, nv, dk, dv)
        out = torch.zeros_like(v)
        for t in range(L):
            k_t, q_t, v_t = k[:, :, t], q[:, :, t], v[:, :, t]
            state = state * decay[:, :, t].unsqueeze(-1).unsqueeze(-1)
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        return out.transpose(1, 2).contiguous()

    def box(w, marked):
        b = type("B", (), {})()
        b.name = "scan"
        b.node = {"op": "scan", "a_log": marked, "k_heads": nk, "k_dim": dk,
                  "heads": nv, "head_dim": dv}
        b.w = w
        b.extras = [dt]
        return b

    want = reference(-A_log.exp())
    got = T._gated_delta_rule(box(A_log, True), [conv, b_in, a_in])
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    got2 = T._gated_delta_rule(box(-A_log.exp(), False), [conv, b_in, a_in])
    assert torch.allclose(got2, want, atol=1e-5), (got2 - want).abs().max()


check("scan: gated delta rule \u2014 raw A_log and GGUF ssm_a agree",
      gated_delta_rule_matches_reference)

def shared_namespaces_load_one_copy():
    from gguf import GGUFWriter
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.gguf")
        w = GGUFWriter(path, "llama")
        w.add_tensor("shared.weight", arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        def node(nid, t):
            return n("layer", nid, din=4, dout=3, t=t, dtype="f32")

        ns = path + "::shared.weight"
        c = canvas([node("a", ns), node("b", ns),
                    node("c", "shared.weight"),
                    n("output", "o", din=3)],
                   [edge("a", "o"), edge("b", "o"), edge("c", "o")])
        g = T.compile_canvas(c, gguf_file=path)
        a, b, d3 = g.boxes["a"], g.boxes["b"], g.boxes["c"]
        assert a.w is b.w, "one namespace: the same Parameter object in both boxes"
        assert d3.w is a.w, "a plain name and its namespace share the same copy"
        rep = g.weight_report()
        assert rep["bytes"] == 12 * 4, "48 bytes of weight, counted once: %r" % rep
        assert rep["shared"] == 1, rep
        est = T.estimate_canvas({"nodes": [node("a", ns), node("b", ns),
                                           node("c", "shared.weight")]}, "cpu")
        assert est["bytes"] == 48 and est["shared"] == 1, \
            "the pre-flight counts one copy too: %r" % est


check("one namespace, one copy: shared weights load once", shared_namespaces_load_one_copy)

def design_training_and_checkpoints():
    ds = {"id": "d", "kind": "dataset", "name": "rows", "x": -400, "y": 0,
          "text": "a\tc\nb\td", "vocab": "abcd"}
    c = canvas([ds,
                n("input", "i", dout=1),
                n("embed", "e", vocab=4, dout=8),
                n("layer", "l", din=8, dout=4),
                {"id": "L", "kind": "loss", "name": "loss", "x": 400, "y": 0}],
               [{"from": "i", "to": "e"}, {"from": "e", "to": "l"},
                {"from": "l", "to": "L", "fromPort": 0, "toPort": 0},
                {"from": "d", "to": "L", "fromPort": 0, "toPort": 1}])
    import shutil, tempfile
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, {"batch": 2, "lr": 0.05, "name": "toy",
                                 "checkpoints": d, "save_every": 10**9})
        events = []
        res = tr.run(steps=30, on_event=events.append)
        losses = [e["loss"] for e in events if e.get("type") == "loss"]
        assert len(losses) == 30 and losses[-1] < losses[0], \
            "the loss falls: %.4f -> %.4f" % (losses[0], losses[-1])
        path = tr.save()
        assert os.path.isfile(os.path.join(path, "state.pt")), path
        want = {k: v.detach().clone() for k, v in tr.trainable[:1]}
        tr.step = 0
        for _, p in tr.trainable[:1]:
            p.data.add_(1.0)                       # wreck it on purpose
        tr2 = T.DesignTrainer(c, {"batch": 2, "lr": 0.05, "checkpoints": d,
                                  "name": "toy", "save_every": 10**9},
                              checkpoint=os.path.join(d, "toy"))
        k, v = tr2.trainable[0]
        assert torch.allclose(v, want[k]), "resume restores the trained weights"
        assert tr2.step == 30, "…and the step: %s" % tr2.step
        cmds = iter(["save", "detach"])
        r = tr2.run(steps=4, command=lambda: next(cmds, None))
        assert r["state"] == "detached" and os.path.isfile(
            os.path.join(r["checkpoint"], "state.pt")), r
    finally:
        shutil.rmtree(d, ignore_errors=True)


check("a design trains: the loss falls, checkpoints land, resume continues",
      design_training_and_checkpoints)

def dataset_csv_reads_like_a_frame():
    node = {"id": "d", "kind": "dataset", "name": "rows", "format": "csv",
            "text": 'prompt,completion,note\n"a, b","say ""hi""",x\nq,"multi\nline",y'}
    rows = T.dataset_rows(node)
    assert rows == [{"prompt": "a, b", "completion": 'say "hi"',
                     "cells": ["a, b", 'say "hi"', "x"]},
                    {"prompt": "q", "completion": "multi\nline",
                     "cells": ["q", "multi\nline", "y"]}], rows
    named = {"id": "d2", "kind": "dataset", "format": "csv", "vocab": "hiter",
             "text": "question,answer\nhi,there"}
    assert T.dataset_rows(named) == [{"prompt": "hi", "completion": "there",
                                       "cells": ["hi", "there"]}], \
        "columns map by name (question/answer), and every column rides along"
    fell = {"id": "d3", "kind": "dataset", "format": "csv", "text": "x,y\n1,2"}
    assert T.dataset_rows(fell) == [{"prompt": "1", "completion": "2",
                                     "cells": ["1", "2"]}], \
        "unknown names fall back to the first two columns"
    ids, labels = T.dataset_batch(named, T.dataset_rows(named), 0, 2)
    lab = labels[0].tolist()
    assert lab[0] == -100 and lab[-1] != -100, "the prompt stays masked: %s" % lab
    assert ids[0].tolist() == [0, 1, 2, 0, 3, 4, 3], ids   # hi there, in ids


check("a csv dataset reads like a dataframe: headers map, quotes survive",
      dataset_csv_reads_like_a_frame)

# ------------------------------------------------- the train node is the hub --

def _hub_canvas(with_test=True):
    nodes = [
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=8, din=1, dout=4),
        n("layer", "l1", din=4, dout=4),
        n("layer", "l2", din=4, dout=8),
        n("output", "logits", din=8),
        n("dataset", "data", vocab="cdxyzabr",
          text="c\tx\ta\nd\ty\tb\nc\tz\ta"),
    ]
    if with_test:
        nodes.append(n("dataset", "test", vocab="cdxyzabr",
                       text="c\tq\ta\nd\tr\tb"))
    nodes.append(n("train", "hub", name="train", dataset="data",
                   test_dataset="test" if with_test else None,
                   x_col="A", y_col="C", classes=["a", "b"],
                   target="logits", paradigm="supervised",
                   batch=2, steps=4, lr=0.05, eval_every=2, save_every=1))
    return canvas(nodes, [edge("tok", "emb"), edge("emb", "l1"),
                          edge("l1", "l2"), edge("l2", "logits")])


def hub_columns_and_classes():
    c = _hub_canvas()
    ds = [x for x in c["nodes"] if x["id"] == "data"][0]
    rows = T.dataset_rows(ds)
    assert all(len(r["cells"]) == 3 for r in rows), rows
    # x=A, y=C, classes: the B column is ignored, the labels are 1-D indices
    ids, labels = T.dataset_batch(ds, rows, 0, 3, 0, 2, ["a", "b"])
    assert labels.dim() == 1 and labels.tolist() == [0, 1, 0], labels.tolist()
    assert ids.shape[1] == 1, "every X is one char, so the batch is width 1"
    try:
        bad = dict(ds, name="bad", text="c\tx\tfish")
        T.dataset_batch(bad, T.dataset_rows(bad), 0, 1, 0, 2, ["a", "b"])
        raise AssertionError("no error for a class outside the hub's")
    except T.CanvasError as e:
        assert "fish" in str(e) and "bad" in str(e), e
    assert T.col_index("A") == 0 and T.col_index("Z") == 25 and T.col_index("AA") == 26
    assert T.col_index("") is None and T.col_index("1") is None


check("the hub picks the X/Y columns; classes become 1-D label indices",
      hub_columns_and_classes)


def hub_left_pads_the_class_input():
    ds = {"id": "d", "kind": "dataset", "name": "rows", "x": 0, "y": 0,
          "vocab": "abxyzwoneut", "text": "a\tz\tone\nxy\tw\ttwo"}
    rows = T.dataset_rows(ds)
    ids, labels = T.dataset_batch(ds, rows, 0, 2, 0, 2, ["one", "two"])
    assert ids.shape == (2, 2), ids.shape
    enc_a = T._ds_encode(ds, "a")
    enc_xy = T._ds_encode(ds, "xy")
    assert ids[0].tolist()[-1] == enc_a[0], "the short X sits at the sequence's end"
    assert ids[1].tolist() == enc_xy, "the long X fills its row"
    assert labels.tolist() == [0, 1], labels.tolist()


check("class mode left-pads X so the class falls at the sequence's end",
      hub_left_pads_the_class_input)


def class_labels_derive_from_the_column():
    # the switch, no list: the values y takes are the classes — the train
    # split first, then the test, so a label only the test carries counts
    ds = {"id": "d", "kind": "dataset", "name": "rows", "x": 0, "y": 0,
          "vocab": "cdxyzabr", "text": "c\tx\ta\nd\ty\tb\nc\tz\ta"}
    test = {"id": "t", "kind": "dataset", "name": "test", "x": 0, "y": 0,
            "vocab": "cdxyzabr", "text": "c\tq\tb\nd\tr\tc"}
    assert T.dataset_classes([ds, test], [2]) == ["a", "b", "c"], "train, then test"
    assert T.dataset_classes([ds], [2], ["b", "a"]) == ["b", "a"], \
        "a legacy list fixes the order its checkpoints know"
    assert T.dataset_classes([], [2]) is None, "nothing to pick from"
    # the run: the switch alone, and the probe reports accuracy
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=8, din=1, dout=4),
        n("layer", "l1", din=4, dout=8),
        n("output", "logits", din=8, train=True, dataset="data",
          test_dataset="test", y_class=True, x_col="A", y_col="C", batch=2,
          eval_every=0, save_every=0),
        n("dataset", "data", vocab="cdxyzabr", text="c\tx\ta\nd\ty\tb\nc\tz\ta"),
        n("dataset", "test", vocab="cdxyzabr", text="c\tq\tb\nd\tr\tc"),
    ], [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "logits")])
    res = T.run(c, {}, mode="training")
    probe = res.get("probe") or {}
    assert probe.get("loss") is not None and probe.get("acc") is not None, probe
    # the trainer: the same switch, no list anywhere on the canvas
    import shutil
    d = tempfile.mkdtemp()
    try:
        cfg = {"name": "cls", "checkpoints": d, "dataset": "data",
               "test_dataset": "test", "x_col": "A", "y_col": "C",
               "y_class": True, "target": "logits", "batch": 2, "lr": 0.05,
               "eval_every": 1, "save_every": 0}
        tr = T.DesignTrainer(c, cfg)
        assert tr.label_mode and tr.class_mode, "the switch drives it"
        assert tr.classes == ["a", "b", "c"], tr.classes
        events = []
        tr.run(steps=4, on_event=events.append)
        ev = [e for e in events if e.get("tacc") is not None]
        assert ev, "the test split scores too"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # a vector y and the switch are refused together
    c2 = canvas([
        n("input", "vec", dout=2),
        n("layer", "l1", din=2, dout=8),
        n("output", "logits", din=8, train=True, dataset="rows", y_class=True,
          x_col="x0 x1", y_col="y0"),
        n("dataset", "rows", format="csv", text="x0,x1,y0\n0,1,a"),
    ], [edge("vec", "l1"), edge("l1", "logits")])
    try:
        T.run(c2, {}, mode="training")
        raise AssertionError("a class label rode on a vector target")
    except T.CanvasError as e:
        assert "single y column" in str(e), e


check("a class label derives from its column: the switch, no list",
      class_labels_derive_from_the_column)


def training_wires_on_the_input():
    # a train config wired into the input names the settings; the input's ONE
    # data dot carries the matrix, and the config picks the columns from it
    data = "\n".join(["x0,x1,x2,y0,y1"] +
                     ["%d,%d,%d,%d,%d" % (a, b, c, 2 * a - b, c)
                      for a, b, c in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]])

    def wired(cfg_fields, data_node=None, attached="rows"):
        nodes = [
            n("input", "vec", dout=3),
            n("layer", "l1", din=3, dout=8),
            n("layer", "l2", din=8, dout=2),
            n("output", "pred", din=2, train=True, batch=8, save_every=0,
              eval_every=0),
            n("dataset", "rows", format="csv", text=data),
            n("config", "cfg", paradigm="supervised", batch=8, lr=0.02,
              **cfg_fields),
        ]
        edges = [edge("cfg", "vec", 0, 0), edge(attached, "vec", 0, 1),
                 edge("vec", "l1"), edge("l1", "l2"), edge("l2", "pred")]
        if attached != "rows":
            nodes.append(n("select", attached, cols="x0 x1 x2"))
            edges.append(edge("rows", attached))
        return canvas(nodes, edges)

    c = wired({"x_col": "x0 x1 x2", "y_col": "y0 y1"})
    hub = T.train_hub(c)
    assert hub.get("_config") and hub.get("target") == "pred", hub
    src2 = T.train_sources(c, hub)
    assert src2["wired"], src2
    assert src2["x"]["cols"] == [0, 1, 2] and src2["y"]["cols"] == [3, 4], src2
    assert src2["x"]["node"]["id"] == "rows" == src2["y"]["node"]["id"], \
        "the two sides read the one attached sheet"
    # the run scores the pair, with a probe
    res = T.run(c, {}, mode="training")
    probe = res.get("probe") or {}
    assert probe.get("loss") is not None, probe
    # the trainer reads the same wires, and the net fits the pattern
    import shutil
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, {"batch": 8, "lr": 0.02, "save_every": 10 ** 9,
                                 "eval_every": 0, "name": "wired",
                                 "checkpoints": d})
        assert tr.vector and tr.x_col == [0, 1, 2] and tr.y_col == [3, 4], \
            (tr.vector, tr.x_col, tr.y_col)
        assert not tr.x_only and tr.dataset_id == "rows", (tr.x_only, tr.dataset_id)
        events = []
        tr.run(steps=300, on_event=events.append)
        losses = [e["loss"] for e in events if e.get("type") == "loss"]
        assert len(losses) == 300 and losses[-1] < losses[0], (losses[0], losses[-1])
        with torch.no_grad():
            xb, yb = tr._batch()
            assert tuple(xb.shape) == (8, 3) and tuple(yb.shape) == (8, 2), \
                (xb.shape, yb.shape)
            pred = tr.graph(feeds={tr.input_id: xb})["pred"]
            err = float((pred[-yb.shape[0]:].float() - yb).abs().mean())
        assert err < 0.2, "the fitted error: %.3f" % err
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # with no x_col/y_col the first two columns are A and B, as ever
    bare = wired({})
    srcb = T.train_sources(bare, T.train_hub(bare))
    assert srcb["x"]["cols"] == [0] and srcb["y"]["cols"] == [1], srcb
    # a select on the dot: the config's columns resolve within what it hands out
    sel = wired({"x_col": "x1", "y_col": "x0"}, attached="xs")
    src3 = T.train_sources(sel, T.train_hub(sel))
    assert src3["x"]["cols"] == [1] and src3["y"]["cols"] == [0], src3
    # a name that is not among the handed-out columns is named
    bad = wired({"x_col": "y0", "y_col": "y1"}, attached="xs")
    try:
        T.train_sources(bad, T.train_hub(bad))
        raise AssertionError("a column outside the matrix was accepted")
    except T.CanvasError as e:
        assert "not among the columns" in str(e), e
    # a bare data dot, a dataset-less select, and a ghost selection name it too
    none = canvas([
        n("input", "vec", dout=2),
        n("layer", "l1", din=2, dout=2),
        n("output", "pred", din=2, train=True),
        n("dataset", "rows", format="csv", text="a,b\n0,1\n1,0"),
        n("config", "cfg", paradigm="supervised"),
    ], [edge("cfg", "vec", 0, 0), edge("vec", "l1"), edge("l1", "pred")])
    try:
        T.run(none, {}, mode="training")
        raise AssertionError("a bare data dot trained anyway")
    except T.CanvasError as e:
        assert "wire a dataset" in str(e) and "data dot" in str(e), e
    c2 = canvas([
        n("input", "vec", dout=2),
        n("layer", "l1", din=2, dout=2),
        n("output", "pred", din=2, train=True),
        n("select", "xs", cols="a"),
        n("config", "cfg", paradigm="supervised"),
    ], [edge("cfg", "vec", 0, 0), edge("xs", "vec", 0, 1),
        edge("vec", "l1"), edge("l1", "pred")])
    try:
        T.run(c2, {}, mode="training")
        raise AssertionError("a dataset-less matrix was accepted")
    except T.CanvasError as e:
        assert "carries no dataset" in str(e), e
    c3 = canvas([
        n("input", "vec", dout=2),
        n("layer", "l1", din=2, dout=2),
        n("output", "pred", din=2, train=True),
        n("dataset", "rows", format="csv", text="a,b\n0,1\n1,0"),
        n("select", "xs", cols="ghost0"),
        n("config", "cfg", paradigm="supervised"),
    ], [edge("cfg", "vec", 0, 0), edge("xs", "vec", 0, 1),
        edge("rows", "xs"), edge("vec", "l1"), edge("l1", "pred")])
    try:
        T.run(c3, {}, mode="training")
        raise AssertionError("a ghost column was accepted")
    except T.CanvasError as e:
        assert "not in the dataset's sheet" in str(e), e


check("the training wires live on the input: the config, and ONE data dot",
      training_wires_on_the_input)


def a_model_trains_through_its_node():
    # the new shape: a nested model node holds the architecture; the train
    # config names it, the data wires into the config, and the batch enters the
    # model's first inner input — the probe scores its first out-port
    data = "\n".join(["x0,x1,x2,y0,y1"] +
                     ["%d,%d,%d,%d,%d" % (a, b, c, 2 * a - b, c)
                      for a, b, c in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]])
    c = canvas([
        n("custom", "model", name="model", inner={
            "nodes": [n("input", "min", dout=3),
                      n("layer", "h", din=3, dout=8),
                      n("layer", "o", din=8, dout=2),
                      n("output", "mout", din=2)],
            "edges": [edge("min", "h"), edge("h", "o"), edge("o", "mout")]}),
        n("dataset", "rows", format="csv", text=data),
        n("config", "cfg", paradigm="supervised", model="model", batch=8,
          lr=0.02, x_col="x0 x1 x2", y_col="y0 y1", save_every=0, eval_every=0),
    ], [edge("rows", "cfg", 0, 0)])
    hub = T.train_hub(c)
    assert hub.get("_model") and hub.get("target") == "model", hub
    src2 = T.train_sources(c, hub)
    assert src2["wired"] and src2["x"]["cols"] == [0, 1, 2] \
        and src2["y"]["cols"] == [3, 4], src2
    assert src2["x"]["node"]["id"] == "rows" == src2["y"]["node"]["id"], src2
    # the run reads the rows through the model, and the probe scores its output
    res = T.run(c, {}, mode="training")
    probe = res.get("probe") or {}
    assert probe.get("loss") is not None, probe
    # the trainer: the batch enters the model, the loss falls, the net fits
    import shutil
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, {"batch": 8, "lr": 0.02, "save_every": 10 ** 9,
                                 "eval_every": 0, "name": "m", "checkpoints": d})
        assert tr.input_id == "model" and tr.target_id == "model", \
            (tr.input_id, tr.target_id)
        events = []
        tr.run(steps=300, on_event=events.append)
        losses = [e["loss"] for e in events if e.get("type") == "loss"]
        assert len(losses) == 300 and losses[-1] < losses[0], (losses[0], losses[-1])
        with torch.no_grad():
            xb, yb = tr._batch()
            pred = T._target_val(tr.graph(feeds={tr.input_id: xb}), "model")
            err = float((pred[-yb.shape[0]:].float() - yb).abs().mean())
        assert err < 0.2, "the fitted error: %.3f" % err
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # inference flows through the model too: a source into its first port
    c5 = canvas([
        n("input", "vec", dout=3),
        n("custom", "model", name="model", inner={
            "nodes": [n("input", "min", dout=3),
                      n("layer", "h", din=3, dout=2),
                      n("output", "mout", din=2)],
            "edges": [edge("min", "h"), edge("h", "mout")]}),
        n("output", "view", din=2),
    ], [edge("vec", "model", 0, 0), edge("model", "view", 0, 0)])
    res2 = T.run(c5, {"vec": [[1.0, 0.0, 0.0]]})
    assert res2["values"]["view"]["shape"] == [1, 2], res2["values"]["view"]
    # a model with no output box inside names the problem
    c2 = canvas([
        n("custom", "model", name="model", inner={
            "nodes": [n("input", "min", dout=2)], "edges": []}),
        n("dataset", "rows", format="csv", text="a,b\n0,1\n1,0"),
        n("config", "cfg", paradigm="supervised", model="model", batch=2),
    ], [edge("rows", "cfg", 0, 0)])
    try:
        T.DesignTrainer(c2, {"batch": 2})
        raise AssertionError("a model with no outputs trained anyway")
    except T.CanvasError as e:
        assert "no output box" in str(e), e
    # a model that is off the canvas names itself too
    c3 = canvas([
        n("dataset", "rows", format="csv", text="a,b\n0,1"),
        n("config", "cfg", paradigm="supervised", model="ghost", batch=2),
    ], [edge("rows", "cfg", 0, 0)])
    try:
        T.DesignTrainer(c3, {"batch": 2})
        raise AssertionError("a ghost model trained anyway")
    except T.CanvasError as e:
        assert "not on the canvas" in str(e), e
    # the data dot on the config is where a bare config looks
    c4 = canvas([
        n("custom", "model", name="model", inner={
            "nodes": [n("input", "min", dout=1), n("output", "mout", din=1)],
            "edges": [edge("min", "mout")]}),
        n("config", "cfg", paradigm="supervised", model="model", batch=2),
    ], [])
    try:
        T.run(c4, {}, mode="training")
        raise AssertionError("a config with no data trained anyway")
    except T.CanvasError as e:
        assert "wire a dataset" in str(e) and "data dot" in str(e), e


check("a model node holds the architecture: the config names it, its ports score",
      a_model_trains_through_its_node)


def unsupervised_trains_on_x_alone():
    # unsupervised: the one matrix's x is the whole story; the labels are the
    # input itself — next-token over its text, or the numbers it rebuilds
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=4, din=1, dout=8),
        n("layer", "l2", din=8, dout=4),
        n("output", "logits", din=4, train=True),
        n("dataset", "rows", vocab="abcd", text="ab\ncd\nab"),
        n("config", "cfg", paradigm="unsupervised", batch=2, x_col="A"),
    ], [edge("cfg", "tok", 0, 0), edge("rows", "tok", 0, 1),
        edge("tok", "emb"), edge("emb", "l2"), edge("l2", "logits")])
    res = T.run(c, {}, mode="training")
    probe = res.get("probe") or {}
    assert probe.get("loss") is not None and probe.get("acc") is None, probe
    import shutil
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, {"batch": 2, "lr": 0.05, "save_every": 10 ** 9,
                                 "eval_every": 0, "name": "unsup",
                                 "checkpoints": d})
        assert tr.x_only and tr.x_col == [0], (tr.x_only, tr.x_col)
        events = []
        tr.run(steps=40, on_event=events.append)
        losses = [e["loss"] for e in events if e.get("type") == "loss"]
        assert len(losses) == 40 and losses[-1] < losses[0], (losses[0], losses[-1])
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # a numeric x learns to rebuild itself: the labels are the matrix
    v = canvas([
        n("input", "vec", dout=2),
        n("layer", "l1", din=2, dout=2),
        n("output", "back", din=2, train=True),
        n("dataset", "rows", format="csv", text="x0,x1\n0,0\n1,0\n0,1\n1,1"),
        n("config", "cfg", paradigm="unsupervised", batch=4, x_col="x0 x1"),
    ], [edge("cfg", "vec", 0, 0), edge("rows", "vec", 0, 1),
        edge("vec", "l1"), edge("l1", "back")])
    res2 = T.run(v, {}, mode="training")
    probe2 = res2.get("probe") or {}
    assert probe2.get("loss") is not None, probe2
    hub = T.train_hub(v)
    srcv = T.train_sources(v, hub)
    xb, yb = T.dataset_batch(srcv["x"]["node"], T.dataset_rows(srcv["x"]["node"]),
                             0, 4, srcv["x"]["cols"], [1], x_only=True)
    assert xb.shape == yb.shape and xb.dtype.is_floating_point, (xb.shape, yb.shape)
    # the class switch is a supervised thing
    v["nodes"] = [x for x in v["nodes"] if x["id"] != "cfg"] + \
        [n("config", "cfg2", paradigm="supervised", y_class=True, batch=2)]
    v["edges"] = [e for e in v["edges"] if e.get("from") != "cfg"] + \
        [{"from": "cfg2", "to": "vec", "fromPort": 0, "toPort": 0}]
    try:
        T.DesignTrainer(v, {"paradigm": "unsupervised", "y_class": True, "batch": 2})
        raise AssertionError("no error for the class switch in unsupervised")
    except T.CanvasError as e:
        assert "no labels" in str(e), e


check("unsupervised trains on X alone \u2014 the model learns to rebuild it",
      unsupervised_trains_on_x_alone)


def training_setup_resolves_the_signal():
    c = _hub_canvas()
    hub = [x for x in c["nodes"] if x["kind"] == "train"][0]
    assert T.training_setup(c, hub) == {"loss_id": None, "target_id": "logits",
                                         "carrier_id": "data", "implicit": True}
    c2 = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=4, din=1, dout=4),
        n("layer", "l2", din=4, dout=4),
        n("output", "logits", din=4),
        n("dataset", "data", vocab="abcd", text="ab\tcd"),
        n("loss", "loss", din=4),
        n("train", "hub", name="train", dataset="data", target="logits"),
    ], [edge("tok", "emb"), edge("emb", "l2"), edge("l2", "logits"),
       edge("data", "loss", 0, 1), edge("logits", "loss", 0, 0)])
    hub2 = [x for x in c2["nodes"] if x["kind"] == "train"][0]
    s = T.training_setup(c2, hub2)
    assert s["loss_id"] == "loss" and s["implicit"] is False, \
        "a wired dataset->loss box outranks the hub's target"
    assert s["carrier_id"] == "data", s


check("the signal: a wired loss box wins, else the hub's target is implicit",
      training_setup_resolves_the_signal)


def the_trainer_scores_both_splits():
    import shutil
    c = _hub_canvas()
    hub = [x for x in c["nodes"] if x["kind"] == "train"][0]
    d = tempfile.mkdtemp()
    try:
        cfg = {"name": "hubrun", "checkpoints": d}
        for key in ("dataset", "test_dataset", "x_col", "y_col", "y_class",
                    "classes", "target", "paradigm", "batch", "lr",
                    "eval_every", "save_every"):
            if hub.get(key) not in (None, "", []):
                cfg[key] = hub[key]
        tr = T.DesignTrainer(c, cfg)
        assert tr.loss_id is None and tr.target_id == "logits" and tr.implicit
        assert [b.node.get("id") for b in tr.dataset_boxes] == ["data", "test"], \
            "every dataset box on the canvas wears a batch"
        events = []
        tr.run(steps=4, on_event=lambda e: events.append(e))
        losses = [e for e in events if e["type"] == "loss"]
        assert len(losses) == 4, len(losses)
        assert all(e.get("acc") is not None for e in losses), \
            "class mode reports accuracy with every loss"
        ev = [e for e in losses if e["step"] == 2][0]
        assert ev.get("tloss") is not None and ev.get("tacc") is not None, \
            "the test split rides the event at eval_every: %s" % ev
        ck = [e for e in events if e["type"] == "checkpoint"]
        assert ck and os.path.isfile(os.path.join(ck[-1]["path"], "state.pt"))
        path = tr.save()
        tr2 = T.DesignTrainer(c, cfg)
        tr2.load(path)
        assert tr2.step == 4, "resume restores the step"
        assert tr2._tpos == tr._tpos, "…and the test-split row position"
        g = T.compile_canvas(c, device="cpu", native=False, train=True)
        before = [p.clone() for _, p in g.named_parameters() if p.requires_grad]
        T.apply_checkpoint(g, path, "cpu")
        after = [p for _, p in g.named_parameters() if p.requires_grad]
        assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
            "apply_checkpoint puts the trained weights on a fresh graph"
    finally:
        shutil.rmtree(d, ignore_errors=True)


check("the trainer: hub columns, acc on both splits, resume, warm start",
      the_trainer_scores_both_splits)


def the_run_menu_probes_the_setup():
    c = _hub_canvas()
    res = T.run(c, {}, mode="training")
    probe = res.get("probe")
    assert probe is not None, "a training run reports its probe"
    assert probe["loss"] is not None and probe["acc"] is not None, probe
    assert probe["tloss"] is not None and probe["tacc"] is not None, \
        "the idle test split is scored by the probe, not the run"
    # no test split: the run stands, the probe just says so
    c2 = canvas([x for x in _hub_canvas()["nodes"] if x["id"] != "test"],
                [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "l2"),
                 edge("l2", "logits")])
    for x in c2["nodes"]:
        if x["kind"] == "train":
            x["test_dataset"] = None
    res2 = T.run(c2, {}, mode="training")
    assert res2["probe"]["loss"] is not None and res2["probe"]["tloss"] is None
    # without a train node the old feeding path stands, and there is no probe
    c3 = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=4, din=1, dout=4),
        n("layer", "l2", din=4, dout=4),
        n("output", "logits", din=4),
        n("dataset", "data", vocab="abcd", text="ab\tcd"),
    ], [edge("tok", "emb"), edge("emb", "l2"), edge("l2", "logits")])
    res3 = T.run(c3, {}, mode="training")
    assert "probe" not in res3 or res3["probe"]["loss"] is None
    # the hub points at a dataset that is not on the canvas
    c4 = canvas([
        n("input", "tok", dout=1),
        n("dataset", "other", name="rows", vocab="abcd", text="ab\tcd"),
        n("train", "hub", name="train", dataset="ghost"),
    ], [])
    try:
        T.run(c4, {}, mode="training")
        raise AssertionError("no error for a hub pointing at a missing dataset")
    except T.CanvasError as e:
        assert "no rows to read" in str(e) and "dataset" in str(e), e
    # the hub's dataset has no rows: the error names the box
    c5 = canvas([
        n("input", "tok", dout=1),
        n("dataset", "rows", name="rows", vocab="abcd", text=""),
        n("train", "hub", name="train", dataset="rows"),
    ], [])
    try:
        T.run(c5, {}, mode="training")
        raise AssertionError("no error for an empty hub dataset")
    except T.CanvasError as e:
        assert "rows" in str(e), e
    # an inference run of a design with a hub is refused on its dataset, as before
    try:
        T.run(c4, {}, mode="inference")
        raise AssertionError("an inference run must not feed the dataset")
    except T.CanvasError:
        pass


check("the run menu: the hub feeds the run, the probe scores both splits",
      the_run_menu_probes_the_setup)


def training_under_an_output():
    # the output box's training tab is a hub too: the fields ride the output
    c = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=8, din=1, dout=4),
        n("layer", "l1", din=4, dout=4),
        n("layer", "l2", din=4, dout=4),
        n("dataset", "data", vocab="cdxyzabr",
          text="c\tx\ta\nd\ty\tb\nc\tz\ta"),
        n("output", "logits", din=4, train=True, dataset="data",
          x_col="A", y_col="C", classes=["a", "b"], paradigm="supervised",
          batch=2, steps=4, lr=0.05, eval_every=0, save_every=0),
    ], [edge("tok", "emb"), edge("emb", "l1"), edge("l1", "l2"), edge("l2", "logits")])
    hub = T.train_hub(c)
    assert hub.get("kind") == "output" and hub.get("target") == "logits", hub
    res = T.run(c, {}, mode="training")
    probe = res.get("probe")
    assert probe is not None and probe["loss"] is not None, probe
    # an inference run of the same design: the bound dataset stands down
    res0 = T.run(c, {"tok": torch.tensor([0, 1])}, mode="inference")
    assert res0["values"]["logits"]["shape"] == [2, 4], res0["values"]["logits"]
    # a dataset wired into the output's dot feeds the run, picker or no picker
    c2 = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=8, din=1, dout=4),
        n("layer", "l2", din=4, dout=4),
        n("dataset", "rows", vocab="cdxyzabr", text="c\tx\ta\nd\ty\tb"),
        n("output", "logits", din=4, train=True, x_col="A", y_col="C",
          classes=["a", "b"], batch=2, steps=2, lr=0.05, eval_every=0, save_every=0),
    ], [edge("tok", "emb"), edge("emb", "l2"), edge("l2", "logits"),
        edge("rows", "logits", 0, 1)])
    hub2 = T.train_hub(c2)
    assert hub2.get("dataset") == "rows", hub2
    res2 = T.run(c2, {}, mode="training")
    assert res2.get("probe") and res2["probe"]["loss"] is not None, res2.get("probe")
    # …and an inference run of the same design stands the idle dataset down:
    # the training dot is no part of an inference run
    res3 = T.run(c2, {"tok": torch.tensor([0, 1])}, mode="inference")
    assert res3["values"]["logits"]["shape"] == [2, 4], res3["values"]["logits"]
    # an enabled output outranks the train node
    c3 = canvas([
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=8, din=1, dout=4),
        n("dataset", "data", vocab="abc", text="a\tb"),
        n("output", "logits", din=4, train=True, dataset="data"),
        n("train", "hub", name="train", dataset="data"),
    ], [edge("tok", "emb"), edge("emb", "logits")])
    assert T.train_hub(c3).get("kind") == "output", "the output host wins"


check("training under an output: its tab is a hub, its dot feeds, inference stands down",
      training_under_an_output)

# ------------------------------------------------- the rl paradigm -------------

def _rl_canvas(hub_kw=None, with_test=True):
    nodes = [
        n("input", "tok", dout=1),
        n("embed", "emb", vocab=3, din=1, dout=8),
        n("layer", "l1", din=8, dout=8),
        n("layer", "l2", din=8, dout=3),
        n("output", "logits", din=3),
        n("dataset", "data", vocab="go!", text="go\tg!\ngoo\tg!o\nrog\t!go"),
    ]
    if with_test:
        nodes.append(n("dataset", "test", vocab="go!", text="go\tg!\nrog\t!go"))
    nodes.append(n("reward", "scorer", name="reward", reward="python",
                   expr="output.count('!') / max(len(output), 1)"))
    hub = {"dataset": "data", "test_dataset": "test" if with_test else None,
           "x_col": "A", "y_col": "B", "paradigm": "rl", "target": "logits",
           "reward": "scorer", "batch": 2, "n_samples": 4, "max_tokens": 3,
           "temperature": 1.0, "lr": 0.05, "steps": 60,
           "save_every": 20, "eval_every": 20}
    if hub_kw:
        for k, v in hub_kw.items():
            if v is None:
                hub.pop(k, None)
            else:
                hub[k] = v
    nodes.append(n("train", "hub", name="train", **hub))
    return canvas(nodes, [edge("tok", "emb"), edge("emb", "l1"),
                          edge("l1", "l2"), edge("l2", "logits")])


def _rl_cfg(c, d, name="rltoy"):
    hub = [x for x in c["nodes"] if x["kind"] == "train"][0]
    cfg = {"name": name, "checkpoints": d}
    for key in ("dataset", "test_dataset", "x_col", "y_col", "y_class",
                "classes", "target", "paradigm", "reward", "lr", "optimizer",
                "weight_decay", "batch", "steps", "clip", "save_every",
                "eval_every", "temperature", "max_tokens", "n_samples"):
        if hub.get(key) not in (None, "", []):
            cfg[key] = hub[key]
    return cfg


def rl_x_only_batch():
    ds = {"id": "d", "kind": "dataset", "name": "rows", "x": 0, "y": 0,
          "vocab": "go!", "text": "go\ngoo"}
    rows = T.dataset_rows(ds)
    ids, labels = T.dataset_batch(ds, rows, 0, 2, 0, 0, x_only=True)
    # right-padded to the widest row (id 0); the labels ride the ids
    assert ids.dim() == 2 and ids.tolist() == [[0, 1, 0], [0, 1, 1]], ids.tolist()
    assert ids.tolist() == labels.tolist(), "x-only: the labels are the ids"
    # the y column is simply not read
    ids2, labels2 = T.dataset_batch(ds, rows, 0, 2, 0, 5, x_only=True)
    assert ids.tolist() == ids2.tolist() and labels.tolist() == labels2.tolist()


check("the x-only batch: 2-D ids, right-padded, the labels ride the ids",
      rl_x_only_batch)


def rl_ds_decode():
    ds = {"id": "d", "kind": "dataset", "name": "rows", "x": 0, "y": 0,
          "vocab": "go!", "text": "go\ngoo"}
    assert T._ds_decode(ds, [0, 1]) == "go", "the char vocab maps back"
    assert T._ds_decode(ds, [0, 99, 1]) == "g?o", \
        "an out-of-range id becomes ? rather than crashing the decode"


check("_ds_decode: ids back to text, out-of-range ids become ?", rl_ds_decode)


def rl_trainer_rewards_climb():
    import shutil
    c = _rl_canvas()
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, _rl_cfg(c, d))
        assert tr.loss_id is None and tr.target_id == "logits"
        assert tr.reward_id == "scorer"
        assert tr.baseline is None, "the baseline is born with the run"
        assert tr.test_box is None, "the test split is idle in rl"
        events = []
        tr.run(steps=60, on_event=lambda e: events.append(e))
        losses = [e for e in events if e["type"] == "loss"]
        assert len(losses) == 60, len(losses)
        for e in losses:
            assert e.get("acc") is None, "rl events carry no accuracy"
            assert e.get("reward") is not None and e.get("baseline") is not None, e
            assert e.get("sample"), "every rl loss event carries a sample"
            assert set(e["sample"]) <= set("go!"), \
                "the sample stays inside the vocabulary: %r" % e["sample"]
        first = sum(e["reward"] for e in losses[:5]) / 5
        last = sum(e["reward"] for e in losses[-5:]) / 5
        assert last > first, "the reward climbed (first-5 %.3f, last-5 %.3f)" % (first, last)
        assert tr.baseline is not None and tr.baseline > 0
        # the checkpoint carries the baseline for the resume
        path = tr.save()
        state = torch.load(os.path.join(path, "state.pt"), map_location="cpu")
        assert state.get("baseline") is not None, "the baseline rides the checkpoint"
        assert abs(state["baseline"] - tr.baseline) < 1e-12
    finally:
        shutil.rmtree(d, ignore_errors=True)


check("the rl trainer: one row a step, rewards climb, the baseline rides along",
      rl_trainer_rewards_climb)


def rl_trainer_resumes():
    import shutil
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(_rl_canvas(), _rl_cfg(_rl_canvas(), d))
        tr.run(steps=10)
        path = tr.save()
        base = tr.baseline
        tr2 = T.DesignTrainer(_rl_canvas(), _rl_cfg(_rl_canvas(), d))
        tr2.load(path)
        assert tr2.step == 10, "resume restores the step"
        assert tr2.baseline == base, "…and the decaying baseline"
        assert tr2._pos == (10 % len(tr2.rows)), \
            "one row per step: the position is step mod rows"
        tr2.run(steps=5)
        assert tr2.step == 15, "…and training continues from there"
    finally:
        shutil.rmtree(d, ignore_errors=True)


check("the rl trainer: resume restores the step, the baseline, the row",
      rl_trainer_resumes)


def rl_run_menu_probe():
    res = T.run(_rl_canvas(), {}, mode="training")
    probe = res.get("probe")
    assert probe is not None, "a training run reports its probe"
    assert probe.get("reward") is not None, probe
    assert isinstance(probe.get("sample"), str) and probe["sample"], probe
    assert probe.get("reason") is None, "a healthy reward box reports no reason: %s" % probe
    for key in ("loss", "acc", "tloss", "tacc"):
        assert probe.get(key) is None, "rl probes carry no supervised numbers: %s" % key


check("the run menu: the rl probe scores one sample and reports the reward",
      rl_run_menu_probe)


def rl_setup_errors():
    import shutil
    d = tempfile.mkdtemp()
    try:
        def expect(c, *needles):
            try:
                T.DesignTrainer(c, _rl_cfg(c, d))
                raise AssertionError("no error for %s" % (needles,))
            except T.CanvasError as e:
                for s in needles:
                    assert s in str(e), "expected %r in %r" % (s, e)

        # no reward bound: the signal is missing
        expect(_rl_canvas(hub_kw={"reward": None}), "reward box")
        # no target: nothing to sample from
        expect(_rl_canvas(hub_kw={"target": None}), "sample from")
        # the bound reward box is off the canvas
        expect(_rl_canvas(hub_kw={"reward": "ghost"}), "not on the canvas")
        # a dataset->loss wire is a supervised leftover in rl
        c = _rl_canvas()
        c["nodes"].append(n("loss", "loss", din=3))
        c["edges"].append(edge("logits", "loss"))   # positional: logits first
        c["edges"].append(edge("data", "loss"))
        expect(c, "supervised leftover")
        # the run menu's guard says the same
        try:
            T.run(c, {}, mode="training")
            raise AssertionError("the run accepts a wired loss in rl")
        except T.CanvasError as e:
            assert "supervised leftover" in str(e), e
        # supervised designs are untouched: a wired loss still wins there
        cs = _rl_canvas(hub_kw={"paradigm": "supervised"})
        cs["nodes"].append(n("loss", "loss", din=3))
        cs["edges"].append(edge("logits", "loss"))
        cs["edges"].append(edge("data", "loss"))
        tr = T.DesignTrainer(cs, _rl_cfg(cs, d))
        assert tr.loss_id == "loss", "a wired dataset->loss box still wins in supervised"
        tr.run(steps=1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


check("rl setup: the reward is the signal — no reward, no target, a wired loss",
      rl_setup_errors)

# ------------------------------------------------- the image input -------------
print("nn_torch: the image input's pixels")


def image_tensor_convert():
    import base64
    # a 32x32 image: left half red, right half blue
    px = bytearray()
    for _y in range(32):
        for x in range(32):
            px += bytes((255, 0, 0)) if x < 16 else bytes((0, 0, 255))
    node = n("input", "img", input_mode="image", image_size=32)
    img = T.image_tensor(node, {"w": 32, "h": 32,
                                "rgb": base64.b64encode(bytes(px)).decode()})
    assert tuple(img.shape) == (32, 32, 3), img.shape          # the picture itself
    assert abs(float(img[0, 0, 0]) - 1.0) < 1e-6, float(img[0, 0, 0])
    assert abs(float(img[0, 0, 2])) < 1e-6, float(img[0, 0, 2])
    assert abs(float(img[0, 31, 2]) - 1.0) < 1e-6, float(img[0, 31, 2])
    # the run's own conversion: a dict feed becomes the picture
    c = canvas([node, n("output", "out", din=3)], [edge("img", "out")])
    feeds = T.build_feeds(c, {"img": {"w": 32, "h": 32,
                                      "rgb": base64.b64encode(bytes(px)).decode()}})
    assert torch.is_tensor(feeds["img"]) and tuple(feeds["img"].shape) == (32, 32, 3)
    # a picture at another size is resized to the box's own
    small = T.image_tensor(node, {"w": 8, "h": 8,
                                  "rgb": base64.b64encode(bytes([255, 0, 0]) * 64).decode()})
    assert tuple(small.shape) == (32, 32, 3), small.shape
    # a node that carries only a blob ref refuses, with the reason
    c2 = canvas([n("input", "img2", input_mode="image", image="blob:abc.png"),
                 n("output", "out", din=3)], [edge("img2", "out")])
    try:
        T.build_feeds(c2, {})
        raise AssertionError("a blob-only image must refuse")
    except T.CanvasError as e:
        assert "ride with the run" in str(e), e


check("image inputs: the picture itself travels (h x w x 3, 0..1)",
      image_tensor_convert)


def image_reaches_an_output_box():
    import base64
    node = n("input", "img", input_mode="image", image_size=8, image="blob:pic.png")
    c = canvas([node, n("output", "out", din=3)], [edge("img", "out")])
    px = bytes([10, 20, 30]) * 64
    res = T.run_graph(T.compile_canvas(c), c, {"img": {"w": 8, "h": 8,
                     "rgb": base64.b64encode(px).decode()}})
    st = (res.get("values") or {}).get("out") or {}
    assert st.get("image"), "the output's value carries the picture: %r" % list(st)
    assert st["image"]["w"] == 8 and st["image"]["h"] == 8, st["image"]
    assert st["image"]["blob"] == "blob:pic.png", st["image"]
    outs = (res.get("outputs") or [{}])[0].get("stats") or {}
    assert outs.get("image"), "…and the output's own stats say so too"
    # a plain tensor through the same wire is not marked as a picture
    c2 = canvas([n("input", "v", values="1 2", dout=2),
                 n("output", "out", din=2)], [edge("v", "out")])
    r2 = T.run_graph(T.compile_canvas(c2), c2, {"v": [1.0, 2.0]})
    assert not ((r2.get("values") or {}).get("out") or {}).get("image"), "no image there"


check("an image arrives at the output box marked as a picture", image_reaches_an_output_box)


def _write_tiny_mmproj(path):
    """A one-block qwen3vl tower, small enough to run in a test."""
    from gguf import GGUFWriter
    rng = np.random.default_rng(7)
    w = GGUFWriter(path, "clip")
    w.add_string("clip.projector_type", "qwen3vl_merger")
    # shapes are the logical (numpy) ones the loader hands torch: [out, in]
    # for a linear, [n_pos, d] for the position table, [d, c, kh, kw] for a conv
    tens = {
        "v.patch_embd.weight": (8, 3, 16, 16),
        "v.patch_embd.weight.1": (8, 3, 16, 16),
        "v.patch_embd.bias": (8,),
        "v.position_embd.weight": (4, 8),
        "v.blk.0.attn_qkv.weight": (24, 8), "v.blk.0.attn_qkv.bias": (24,),
        "v.blk.0.attn_out.weight": (8, 8), "v.blk.0.attn_out.bias": (8,),
        "v.blk.0.ffn_up.weight": (16, 8), "v.blk.0.ffn_up.bias": (16,),
        "v.blk.0.ffn_down.weight": (8, 16), "v.blk.0.ffn_down.bias": (8,),
        "v.blk.0.ln1.weight": (8,), "v.blk.0.ln1.bias": (8,),
        "v.blk.0.ln2.weight": (8,), "v.blk.0.ln2.bias": (8,),
        "v.post_ln.weight": (8,), "v.post_ln.bias": (8,),
        "mm.0.weight": (32, 32), "mm.0.bias": (32,),
        "mm.2.weight": (12, 32), "mm.2.bias": (12,),
    }
    for name, shape in tens.items():
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.2)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def vision_takes_a_picture():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mm.gguf")
        _write_tiny_mmproj(path)
        node = {"kind": "vision", "name": "tower", "file": path,
                "projector": "qwen3vl_merger", "d_model": 8, "heads": 2, "blocks": 1,
                "patch": 16, "merge": 2, "image_size": 32, "proj_dim": 12, "eps": 1e-6,
                "image_mean": "0.5 0.5 0.5", "image_std": "0.5 0.5 0.5"}
        tow = T.VisionTower(node, device="cpu")
        img = torch.rand(32, 32, 3, generator=torch.Generator().manual_seed(3))
        a = tow(img)                                 # the picture itself
        assert tuple(a.shape) == (1, 12), a.shape    # 4 patches -> 1 merged group
        # the same picture, normalized and patchified the way the tower does
        norm = (img - 0.5) / 0.5
        pat = norm.permute(2, 0, 1).reshape(3, 2, 16, 2, 16).permute(1, 3, 0, 2, 4)
        b = tow(pat.reshape(4, 768))                 # pre-cut patch rows still work
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
        # an image at another size is resized to the tower's own on the way in
        small = tow(torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(4)))
        assert tuple(small.shape) == (1, 12), small.shape
        # an RGB pixel list [n, 3] (a square) is the same picture
        c = tow(img.reshape(-1, 3))
        assert torch.allclose(a, c, atol=1e-5), float((a - c).abs().max())


check("a vision tower takes the picture itself, and its patch rows",
      vision_takes_a_picture)


def stats_carry_dtype():
    c = canvas([n("input", "ids", dout=4, values="0 1 2 3"),
                n("output", "out", din=4)], [edge("ids", "out")])
    res = T.run_graph(T.compile_canvas(c), c, {})
    st = (res.get("values") or {}).get("out") or {}
    assert st.get("dtype") == "i64", st            # the value's own storage type
    c2 = canvas([n("input", "f", dout=4, values="0.5 0.5 0.5 0.5"),
                 n("output", "out2", din=4)], [edge("f", "out2")])
    res2 = T.run_graph(T.compile_canvas(c2), c2, {})
    st2 = (res2.get("values") or {}).get("out2") or {}
    assert st2.get("dtype") == "f32", st2
    c3 = canvas([n("input", "i", dout=6, values="0 1 2 3 4 5"),
                 n("output", "out3", din=6)], [edge("i", "out3")])
    r3 = T.run_graph(T.compile_canvas(c3), c3, {})
    assert (r3.get("values") or {}).get("out3", {}).get("shape") == [6], r3


check("every value's stats carry its dtype (and the shape stays)", stats_carry_dtype)


def image_matrix_determination():
    # an RGB matrix in 0..1 handed to the run: an output box determines it is
    # a picture and the backend sends a thumbnail it can draw
    c = canvas([n("input", "m"), n("output", "out", din=3)], [edge("m", "out")])
    res = T.run_graph(T.compile_canvas(c), c, {"m": torch.full((4, 4, 3), 0.5)})
    vals = res.get("values") or {}
    st = vals.get("out") or {}
    assert st.get("image"), "the output reads as a picture: %r" % list(st)
    assert (st["image"]["w"], st["image"]["h"]) == (4, 4), st["image"]
    th = st["image"].get("thumb") or {}
    assert (th.get("w"), th.get("h")) == (4, 4) and th.get("rgb"), th
    import base64 as b64
    assert len(b64.b64decode(th["rgb"])) == 4 * 4 * 3, "an RGB thumbnail"
    assert not (vals.get("m") or {}).get("image"), \
        "…but the box that made it is not re-labelled"
    # [n, 3] with n a square: an RGB pixel list, also a picture
    c2 = canvas([n("input", "m"), n("output", "out", din=3)], [edge("m", "out")])
    r2 = T.run_graph(T.compile_canvas(c2), c2, {"m": torch.full((16, 3), 0.25)})
    assert ((r2.get("values") or {}).get("out") or {}).get("image"), "a square RGB list"
    # not a square: a matrix, not a picture
    c3 = canvas([n("input", "m"), n("output", "out", din=3)], [edge("m", "out")])
    r3 = T.run_graph(T.compile_canvas(c3), c3, {"m": torch.full((12, 3), 0.25)})
    assert not ((r3.get("values") or {}).get("out") or {}).get("image"), "a ragged list"
    # values outside 0..1 are not pixels
    c4 = canvas([n("input", "m"), n("output", "out", din=3)], [edge("m", "out")])
    r4 = T.run_graph(T.compile_canvas(c4), c4, {"m": torch.full((4, 4, 3), 2.0)})
    assert not ((r4.get("values") or {}).get("out") or {}).get("image"), "2.0 is not a pixel"


check("a matrix that reads as a picture is drawn as one (thumbnail and all)",
      image_matrix_determination)


def heatmap_draws_a_vector():
    c = canvas([
        n("input", "vec", dout=96),
        n("heatmap", "heat", din=96),
        n("output", "pic", din=96),
    ], [edge("vec", "heat"), edge("heat", "pic")])
    x = torch.linspace(-1.0, 1.0, 96)            # 96 = 8 x 12: the exact pair
    img = T.compile_canvas(c)(feeds={"vec": x})["heat"]
    assert tuple(img.shape) == (12, 8, 3), img.shape
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0, "pixels live in 0..1"
    assert bool(torch.isfinite(img).all()), "a picture never carries nan or inf"
    iv = T.image_view(img)
    assert iv and [iv["w"], iv["h"]] == [8, 12], "it reads as a picture"
    # the ends of the ramp: the smallest value reads blue, the largest red
    first, last = img.reshape(-1, 3)[0], img.reshape(-1, 3)[-1]
    assert float(first[2]) > float(first[0]), ("the low end is blue", first.tolist())
    assert float(last[0]) > float(last[2]), ("the high end is red", last.tolist())
    # similar vectors draw similar pictures: the map is median-centred and
    # robust-scaled, so one value moving moves one cell — not the picture
    def moved(a, b):
        return float((a - b).abs().mean())
    bump = x.clone()
    bump[int(x.argmax())] += 0.1                 # the element min/max used to rescale
    b_img = T._heatmap(bump)
    assert moved(img, b_img) < 0.02, "one cell moves, not the picture"
    assert float((b_img.reshape(-1, 3)[:-1] - img.reshape(-1, 3)[:-1]).abs().max()) == 0.0, \
        "…and every other cell is untouched"
    assert moved(img, T._heatmap(x + 0.1)) < 1e-4, "a shift is invisible"
    assert moved(img, T._heatmap(x * 1.05)) < 1e-4, "…and so is a scale"
    far = x.clone()
    far[int(x.argmin())] = 50.0                  # an outlier clips: it does not squash the rest
    assert moved(img, T._heatmap(far)) < 0.02, \
        "an outlier saturates its own cell, not the picture"
    narrow = 0.5 + torch.randn(5120) * 0.002     # a tight band: used to rescale wholesale
    n1 = T._heatmap(narrow)
    n2 = narrow.clone()
    n2[int(narrow.argmax())] += 0.1
    assert moved(n1, T._heatmap(n2)) < 0.02, "a tight band is not rescaled by one bump"
    # float dust around a constant: one nearly flat colour, stable between draws
    d1 = torch.ones(5120) + torch.randn(5120) * 1e-7
    d2 = torch.ones(5120) + torch.randn(5120) * 1e-7
    f1 = T._heatmap(d1)
    assert float((f1.reshape(-1, 3)[:5120] - f1.reshape(-1, 3)[0]).abs().max()) < 0.1, \
        "dust reads as one (nearly) flat colour"
    assert moved(f1, T._heatmap(d2)) < 0.06, "…and two dust draws stay close"
    # a prime length still draws: a near-square grid with a neutral tail
    c2 = canvas([
        n("input", "vec", dout=127),
        n("heatmap", "heat", din=127),
        n("output", "pic", din=127),
    ], [edge("vec", "heat"), edge("heat", "pic")])
    g2 = T.compile_canvas(c2)
    img2 = g2(feeds={"vec": torch.arange(127).float()})["heat"]
    assert tuple(img2.shape) == (11, 12, 3), img2.shape   # 127 -> 11x12, five pads
    assert bool(torch.isfinite(img2).all()) and T.image_view(img2) is not None
    img3 = g2(feeds={"vec": torch.ones(127)})["heat"]     # a constant vector: no divide
    flat = img3.reshape(-1, 3)[:127]                      # the five pads read gray
    assert bool((flat == flat[0]).all()), "one flat colour, no nan"
    assert bool(torch.isfinite(img3).all())
    # the run reports it with the picture attached \u2014 a viewable node
    res = T.run(c, {"vec": x}, mode="inference")
    st = (res.get("values") or {}).get("heat")
    assert st and st.get("image") and st["shape"] == [12, 8, 3], st
    assert ((st.get("image") or {}).get("thumb") or {}).get("rgb"), "the thumbnail rides"
    json.dumps(res)                              # the boundary: nothing NaN or inf


check("the heatmap: a vector in, one picture out", heatmap_draws_a_vector)


def vector_columns_train():
    data = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    text = "\n".join(["x0,x1,x2,y0,y1"] +
                      ["%d,%d,%d,%d,%d" % (a, b, c, 2 * a - b, c)
                       for a, b, c in data])
    c = canvas([
        n("input", "vec", dout=3),
        n("layer", "l1", din=3, dout=8),
        n("layer", "l2", din=8, dout=2),
        n("dataset", "rows", format="csv", text=text),
        n("output", "pred", din=2, train=True, dataset="rows",
          x_col="x0 x1 x2", y_col="y0 y1", batch=8, lr=0.02,
          eval_every=0, save_every=0),
    ], [edge("vec", "l1"), edge("l1", "l2"), edge("l2", "pred")])
    ds = [x for x in c["nodes"] if x["id"] == "rows"][0]
    # the names come from the frame, and a sequence of them is a vector
    assert T.dataset_cols(ds) == ["x0", "x1", "x2", "y0", "y1"], T.dataset_cols(ds)
    assert T.col_indexes("y1 x0", T.dataset_cols(ds)) == [4, 0], "names, case-blind"
    assert T.col_indexes("A", []) == [0], "letters still work"
    assert T.col_indexes("ghost0", T.dataset_cols(ds)) is None, \
        "a named stranger (a name with digits) is nothing"
    rws = T.dataset_rows(ds)
    ids, labels = T.dataset_batch(ds, rws, 0, 8, [0, 1, 2], [3, 4])
    assert tuple(ids.shape) == (8, 3) and tuple(labels.shape) == (8, 2), \
        (ids.shape, labels.shape)
    assert ids.dtype.is_floating_point and labels.dtype.is_floating_point
    # the run's probe scores it with mse, and reports no accuracy
    res = T.run(c, {}, mode="training")
    probe = res.get("probe") or {}
    assert probe.get("loss") is not None and probe.get("acc") is None, probe
    # a real run: the loss falls, and the net fits the pattern
    import shutil
    d = tempfile.mkdtemp()
    try:
        tr = T.DesignTrainer(c, {"dataset": "rows", "target": "pred",
                                 "x_col": "x0 x1 x2", "y_col": "y0 y1",
                                 "batch": 8, "lr": 0.02, "save_every": 10 ** 9,
                                 "eval_every": 0, "name": "vec",
                                 "checkpoints": d})
        assert tr.vector and tr.x_col == [0, 1, 2] and tr.y_col == [3, 4], \
            (tr.x_col, tr.y_col)
        events = []
        tr.run(steps=300, on_event=events.append)
        losses = [e["loss"] for e in events if e.get("type") == "loss"]
        assert len(losses) == 300 and losses[-1] < losses[0], (losses[0], losses[-1])
        with torch.no_grad():
            xb, yb = tr._batch()
            pred = tr.graph(feeds={tr.input_id: xb})["pred"]
            err = float((pred[-yb.shape[0]:].float() - yb).abs().mean())
        assert err < 0.2, "the fitted error: %.3f" % err
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # a ghost column names itself
    c2 = canvas(
        [x for x in c["nodes"] if x["id"] != "pred"] +
        [n("output", "pred", din=2, train=True, dataset="rows",
           x_col="ghost0", y_col="y0 y1", batch=8)],
        [edge("vec", "l1"), edge("l1", "l2"), edge("l2", "pred")])
    try:
        T.run(c2, {}, mode="training")
        raise AssertionError("a ghost column was accepted")
    except T.CanvasError as e:
        assert "not in the dataset" in str(e), e
    # a text cell in a vector column names the column
    c3 = canvas([
        n("input", "vec", dout=3),
        n("layer", "l1", din=3, dout=2),
        n("dataset", "rows", format="csv", text="x0,x1,x2,y0,y1\n0,0,0,0,oops"),
        n("output", "pred", din=2, train=True, dataset="rows",
          x_col="x0 x1 x2", y_col="y0 y1", batch=1),
    ], [edge("vec", "l1"), edge("l1", "pred")])
    try:
        T.run(c3, {}, mode="training")
        raise AssertionError("a text cell passed as a number")
    except T.CanvasError as e:
        assert "not a number" in str(e) and "oops" in str(e), e


check("a sequence of columns is a vector: names resolve, batches are matrices, "
      "the loss is mse", vector_columns_train)


def vision_refuses_unknown_projectors():
    try:
        T.VisionTower({"kind": "vision", "name": "tower", "file": "x.gguf",
                       "projector": "gemma4uv", "d_model": 8, "heads": 2, "blocks": 1})
        raise AssertionError("an unknown projector must refuse")
    except T.CanvasError as e:
        assert "gemma4uv" in str(e), e


check("a vision tower names the projector it cannot run", vision_refuses_unknown_projectors)



def text_reaches_its_output():
    c = canvas([n("input", "i", input_mode="text"), n("output", "o")],
               [edge("i", "o")])
    r = T.run_graph(T.compile_canvas(c), c, {"i": "hello there"})
    assert r["values"].get("o") == {"text": "hello there"}, r["values"].get("o")
    assert r["values"].get("i") == {"text": "hello there"}, "the input shows its own text"
    assert r.get("texts", {}).get("o") == "hello there", r.get("texts")
    outs = [o for o in r.get("outputs", []) if o["id"] == "o"]
    assert outs and outs[0]["stats"] == {"text": "hello there"}, outs


check("text that reaches an output is the output's value", text_reaches_its_output)



def timing_counters():
    c = canvas([n("input", "i", dout=1),
                n("embed", "e", vocab=8, din=1, dout=4),
                n("layer", "l", din=4, dout=8),
                n("output", "o", din=8)],
               [edge("i", "e"), edge("e", "l"), edge("l", "o")])
    res = T.run(c, {"i": [1]}, input_id="i", output_id="o", generate=3, temperature=0)
    t = res["timing"]
    assert t["sweeps"] == 4, t                  # the prompt pass, then one per token
    assert len(res["generated"] or []) == 3, res["generated"]
    assert t["input_to_output_ms"] >= 0 and t["per_sweep_ms"] >= 0, t
    r2 = T.run(c, {"i": [1]}, input_id="i", output_id="o")
    assert r2["timing"]["sweeps"] == 1, r2["timing"]
    assert isinstance(r2["timing"]["input_to_output_ms"], int), r2["timing"]


check("a run clocks its sweeps: input to output, one per token when generating",
      timing_counters)

if FAILED:
    print("nn_torch_test: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("nn_torch_test: all assertions passed")
