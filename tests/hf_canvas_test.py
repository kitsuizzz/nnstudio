#!/usr/bin/env python3
"""nnstudio's HF -> canvas builder: a synthetic repo, no network, no torch.

A repo is `config.json` + `model.safetensors`; the builder reads only their
headers, binds every box to the repo's own tensor name and the shard it lives
in, and hands the backend `file: "hf:<repo>/<shard>"` (or a local path)."""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nn_engine  # noqa: E402
import nn_arc     # noqa: E402

_arcdir = tempfile.mkdtemp(prefix="nnstudio-hf-arc-")
os.environ["NNSTUDIO_ARC_DB"] = os.path.join(_arcdir, "hf_arc.db")


def f32(n):
    return struct.pack("<%df" % n, *([0.0] * n))


def f16(n):
    return struct.pack("<%de" % n, *([0.0] * n))


def write_safetensors(path, tensors):
    """{name: (dims, dtype, raw)} — header + data, stdlib only."""
    header, offset = {}, 0
    for name, (dims, dtype, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(dims),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for dims, dtype, raw in tensors.values():
            f.write(raw)


def fixture_dir(d, model_type="fixture"):
    cfg = {"model_type": model_type, "hidden_size": 16, "num_hidden_layers": 1,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
           "intermediate_size": 32, "vocab_size": 64, "rms_norm_eps": 1e-5,
           "rope_theta": 10000.0, "max_position_embeddings": 128}
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    tensors = {
        "model.embed_tokens.weight": ([64, 16], "F16", f16(64 * 16)),
        "model.layers.0.input_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.q_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.self_attn.q_proj.bias": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.k_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.v_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.o_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.post_attention_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.mlp.gate_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.up_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.down_proj.weight": ([16, 32], "F16", f16(16 * 32)),
        "model.norm.weight": ([16], "F32", f32(16)),
        "lm_head.weight": ([64, 16], "F16", f16(64 * 16)),
        "model.rotary_emb.inv_freq": ([4], "F32", f32(4)),
    }
    write_safetensors(os.path.join(d, "model.safetensors"), tensors)
    return tensors


def walk(nodes, out=None):
    out = out if out is not None else []
    for n in nodes or []:
        out.append(n)
        if n.get("kind") == "custom" and n.get("inner"):
            walk(n["inner"]["nodes"], out)
    return out


with tempfile.TemporaryDirectory() as d:
    tensors = fixture_dir(d)
    out = nn_engine.hf_canvas(d)
    c = out["canvas"]
    assert out["ok"] and c["type"] == "canvas" and c["version"] == 1, "a canvas"
    assert c["model"]["fidelity"] == "approximate", \
        "no spec claims a 'fixture' family: %r" % c["model"]
    assert out["summary"]["arch"] == "fixture" and out["summary"]["blocks"] == 1, out["summary"]
    assert [n["kind"] for n in c["nodes"]] == ["input", "custom", "output"], \
        "the model box between its tokens and logits"
    mbox = next(n for n in c["nodes"] if n["kind"] == "custom")
    assert mbox["name"] == os.path.basename(d), "named after the repo/folder: %r" % mbox["name"]
    assert c["model"]["layers"] == 1 and c["model"]["kv_heads"] == 2, c["model"]

    nodes = walk(mbox["inner"]["nodes"])
    by_name = {n.get("name"): n for n in nodes}
    emb = by_name["token embedding"]
    assert (emb["kind"], emb["din"], emb["dout"], emb["vocab"]) == ("embed", 1, 16, 64), emb
    assert emb["t"] == "model.embed_tokens.weight", emb["t"]
    assert emb["dtype"] == "f16", "the repo's dtype travels: %r" % emb["dtype"]
    assert emb["file"].endswith("model.safetensors"), emb["file"]

    labels = [n["name"] for n in mbox["inner"]["nodes"] if n.get("kind") == "custom"]
    assert labels == ["blk 0 \u00b7 attn mlp"], "block labels: %r" % labels
    att = next(n for n in walk(by_name["blk 0 \u00b7 attn mlp"]["inner"]["nodes"])
               if n.get("name") == "attention")
    anodes = {n.get("name"): n for n in att["inner"]["nodes"]}
    q = anodes["q proj"]
    assert (q["din"], q["dout"]) == (16, 16), q
    assert q["t"] == ("model.layers.0.self_attn.q_proj.weight, "
                      "model.layers.0.self_attn.q_proj.bias"), q["t"]
    assert q["params"] == 16 * 16 + 16, "the bias rides with its weight: %r" % q["params"]
    k = anodes["k proj"]
    assert (k["din"], k["dout"]) == (16, 8), "kv heads shape the projection: %r" % (k,)
    assert anodes["v proj"]["dout"] == 8 and anodes["out proj"]["dout"] == 16, anodes.keys()

    hn = by_name["output_norm"]
    assert hn["t"] == "model.norm.weight" and hn["dtype"] == "f32", hn
    head = by_name["lm head"]
    assert (head["din"], head["dout"]) == (16, 64) and head["t"] == "lm_head.weight", head

    loose = [n for n in nodes if n.get("loose")]
    assert any(n.get("t") == "model.rotary_emb.inv_freq" for n in loose), \
        "a tensor with no role is kept, loose: %r" % [n.get("t") for n in loose]

    tensored = [n for n in nodes if n.get("t")]
    assert all(n.get("file") for n in tensored), "every bound box names its shard"
    assert all(n.get("file").endswith("model.safetensors") for n in tensored), \
        "…the same local file, here"

# ---- a family with a verified spec: the exact, native-boxes-only canvas ----
with tempfile.TemporaryDirectory() as d:
    fixture_dir(d, model_type="llama")
    out = nn_engine.hf_canvas(d)
    c = out["canvas"]
    assert c["model"]["fidelity"] == "exact" and c["model"]["spec"] == "llama", c["model"]
    assert out["summary"]["fidelity"] == "exact" and out["summary"]["unbound"] == 1, out["summary"]
    assert [n["kind"] for n in c["nodes"]] == ["input", "custom", "output"], \
        "the model box between its tokens and logits"
    mbox = next(n for n in c["nodes"] if n["kind"] == "custom")
    assert mbox["name"] == os.path.basename(d), mbox["name"]
    inner = mbox["inner"]["nodes"]
    labels = [n["name"] for n in inner if n.get("kind") == "custom"]
    assert labels == ["blk 0 \u00b7 llama"], "the spec names the block: %r" % labels
    blk = next(n for n in inner if n.get("kind") == "custom")
    bnodes = {n["name"]: n for n in blk["inner"]["nodes"]}

    # the attention core is native boxes: heads split, rope, GQA repeat, mask
    for want in ("q heads", "q [h]", "rope", "k\u1d40", "causal mask", "scores\u00b7v"):
        assert want in bnodes, "attention atom %r missing: %r" % (want, sorted(bnodes))
    assert bnodes["k repeat"]["repeats"] == 2 and bnodes["k repeat"]["op"] == "repeat", \
        bnodes["k repeat"]
    assert bnodes["causal mask"]["op"] == "mask", bnodes["causal mask"]
    assert abs(float(bnodes["scale 1/\u221ad"]["factor"]) - 0.5) < 1e-9, \
        "1/sqrt(head_dim) = 1/2: %r" % bnodes["scale 1/\u221ad"]
    # optional tensors are simply not here for this repo
    assert "q norm" not in bnodes and "k norm" not in bnodes, sorted(bnodes)

    # every bound box reads the repo's own name, and the bias rides along
    q = bnodes["q proj"]
    assert q["t"] == ("model.layers.0.self_attn.q_proj.weight, "
                      "model.layers.0.self_attn.q_proj.bias"), q["t"]
    assert q["params"] == 16 * 16 + 16, q["params"]
    assert q["file"].endswith("model.safetensors") and q["dtype"] == "f16", q
    every = [n for n in walk(c["nodes"]) if n.get("t")]
    assert all(n.get("file") for n in every) and len(every) == 12, \
        "every weight box names its shard: %d" % len(every)
    audit = nn_engine.audit_canvas(c)
    assert audit["ok"], "the llama canvas is fully wired: %r" % audit["issues"][:4]

# ---- qwen2: the rope theta lives in rope_parameters, not top-level ----
with tempfile.TemporaryDirectory() as d:
    cfg = {"model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 1,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
           "intermediate_size": 32, "vocab_size": 64, "rms_norm_eps": 1e-6,
           "rope_parameters": {"rope_theta": 123456.0, "rope_type": "default"},
           "max_position_embeddings": 128}
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    tensors = {
        "model.embed_tokens.weight": ([64, 16], "F16", f16(64 * 16)),
        "model.layers.0.input_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.q_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.self_attn.q_proj.bias": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.k_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.k_proj.bias": ([8], "F32", f32(8)),
        "model.layers.0.self_attn.v_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.v_proj.bias": ([8], "F32", f32(8)),
        "model.layers.0.self_attn.o_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.post_attention_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.mlp.gate_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.up_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.down_proj.weight": ([16, 32], "F16", f16(16 * 32)),
        "model.norm.weight": ([16], "F32", f32(16)),
        "lm_head.weight": ([64, 16], "F16", f16(64 * 16)),
    }
    write_safetensors(os.path.join(d, "model.safetensors"), tensors)
    out = nn_engine.hf_canvas(d)
    c = out["canvas"]
    assert c["model"]["fidelity"] == "exact" and c["model"]["spec"] == "qwen2", c["model"]
    assert out["summary"]["unbound"] == 0, "qwen2 keeps no rotary buffer: %r" % out["summary"]
    blk = next(n for n in c["nodes"] if n["kind"] == "custom")["inner"]
    blk = next(n for n in blk["nodes"] if n.get("kind") == "custom")
    assert blk["name"] == "blk 0 \u00b7 qwen2", blk["name"]
    bnodes = {n["name"]: n for n in blk["inner"]["nodes"]}
    assert abs(float(bnodes["rope"]["theta"]) - 123456.0) < 1e-9, \
        "the theta comes from rope_parameters, not the default: %r" % bnodes["rope"]
    for want in ("q proj", "k proj", "v proj"):
        assert bnodes[want]["t"].endswith("bias") and bnodes[want]["t"].count(",") == 1, \
            "the bias rides with its weight: %r" % bnodes[want]
    assert "q norm" not in bnodes and "k norm" not in bnodes, \
        "qwen2 has no q/k norms: %r" % sorted(bnodes)
    every = [n for n in walk(c["nodes"]) if n.get("t")]
    assert all(n.get("file") for n in every) and len(every) == 12, \
        "every weight box names its shard: %d" % len(every)
    assert out["summary"]["tensors"] == 15, "the three biases count as tensors"
    audit = nn_engine.audit_canvas(c)
    assert audit["ok"], "the qwen2 canvas is fully wired: %r" % audit["issues"][:4]

# ---- hf_arc: a researched repo is served straight from the database ----
with tempfile.TemporaryDirectory() as d:
    assert nn_arc.canonical("https://huggingface.co/org/repo/tree/main") == "org/repo"
    assert nn_arc.canonical(d) == d, "local folders pass through untouched"
    fixture_dir(d, model_type="llama")
    first = nn_engine.hf_canvas(d)
    assert first["canvas"]["model"].get("arc") is False, "not cached yet"
    nn_arc.put(d, first["canvas"], transformers="test", note="the row")
    out = nn_engine.hf_canvas(d)
    assert out.get("arc") is True and out["summary"].get("arc") is True, out.get("summary")
    assert out["canvas"]["model"]["arc"] is True, out["canvas"]["model"]
    assert out["canvas"]["model"]["fidelity"] == "exact", out["canvas"]["model"]
    assert [(n["kind"], n["name"]) for n in out["canvas"]["nodes"]] == \
        [(n["kind"], n["name"]) for n in first["canvas"]["nodes"]], \
        "the stored canvas comes back as it was drawn"
    fresh = nn_engine.hf_canvas(d, use_arc=False)
    assert fresh.get("arc") is None and fresh["canvas"]["model"].get("arc") is False, \
        "the job's own proof can bypass the cache"
    row = nn_arc.get(d)
    assert row["model"]["spec"] == "llama" and row["nodes"], row["model"]
    assert nn_arc.rows()[0]["repo_id"] == d and nn_arc.rows()[0]["note"] == "the row"
    assert nn_arc.delete(d) and nn_arc.get(d) is None, "and a row can be dropped"
    again = nn_engine.hf_canvas(d)
    assert again.get("arc") is None, "gone: the builder takes over again"

# ---- arc_canvas: every model nested, the top box the pipeline ----
with tempfile.TemporaryDirectory() as d:
    a = os.path.join(d, "text-encoder")
    b = os.path.join(d, "dit")
    os.makedirs(a)
    os.makedirs(b)
    fixture_dir(a, model_type="llama")
    fixture_dir(b, model_type="llama")
    pipeline = {"models": [{"name": "text", "source": a},
                           {"name": "dit", "source": b}],
                "edges": [{"from": "text", "to": "dit", "fromPort": 0, "toPort": 0}]}
    out = nn_engine.arc_canvas(a, pipeline=pipeline)
    c = out["canvas"]
    assert out["models"] == ["text", "dit"], out["models"]
    assert c["model"]["arch"] == "pipeline" and c["model"]["fidelity"] == "exact"
    assert c["model"]["pipeline"] == ["text", "dit"]
    kinds = [(n["kind"], n["name"]) for n in c["nodes"]]
    assert kinds == [("custom", "text"), ("custom", "dit"),
                     ("input", "in text"), ("output", "out dit")], kinds
    edges = {(e["from"], e["to"]) for e in c["edges"]}
    text, dit = c["nodes"][0], c["nodes"][1]
    assert (text["id"], dit["id"]) in edges, "the pipeline edge is a real wire"
    assert out["summary"]["models"] == 2 and out["summary"]["boxes"] > 10, out["summary"]
    # a model that is not exact cannot land: no spec for the family
    bad = os.path.join(d, "mystery")
    os.makedirs(bad)
    fixture_dir(bad, model_type="mystery")
    try:
        nn_engine.arc_canvas(bad, pipeline={"models": [{"name": "x", "source": bad}]},
                             use_arc=True)
        raise AssertionError("a family without a spec must not assemble")
    except nn_engine.ArcError as e:
        assert "mystery" in str(e), e

# ---- spec elision: a box the wiring never reads is not drawn ----
spec = {
    "family": "toy",
    "block": {
        "nodes": [
            {"id": "in", "kind": "input", "name": "in", "pos": [0, 0], "dout": "d_model"},
            {"id": "zero", "kind": "op", "name": "elided branch \u2192 0",
             "pos": [0, 0], "op": "zeros"},
            {"id": "proj", "kind": "op", "name": "q proj", "pos": [0, 0],
             "op": "multiply", "if": "heads == 0"},
            {"id": "repeat", "kind": "op", "name": "k repeat", "pos": [0, 0],
             "op": "repeat", "repeats": 2, "dim": 0},
            {"id": "add", "kind": "op", "name": "add", "pos": [0, 0], "op": "add"},
            {"id": "out", "kind": "output", "name": "out", "pos": [0, 0], "din": "d_model"},
        ],
        "edges": [["in", "proj"], ["proj", "repeat"], ["proj", "add"],
                  ["in", "add"], ["add", "out"]],
    },
}
doc = nn_engine._spec_inner(spec, {"d_model": 8, "heads": 2}, {}, None)
names = [n.get("name") for n in doc["nodes"]]
assert "q proj" not in names, "the elided branch is gone: %r" % names
assert "k repeat" not in names, "its stranded consumer is swept too: %r" % names
assert "elided branch \u2192 0" in names and "add" in names, names
by = {n["id"]: n.get("name") for n in doc["nodes"]}
edges = {(by[e["from"]], by[e["to"]]) for e in doc["edges"]}
assert ("elided branch \u2192 0", "add") in edges, \
    "the live consumer still takes a real zero: %r" % edges

# ---- the audit: a dirty canvas reports every problem, block by block ----
dirty = {
    "type": "canvas", "version": 1, "model": {"fidelity": "exact"},
    "nodes": [
        {"id": "i", "kind": "input", "name": "tokens", "x": 0, "y": 0, "dout": 1},
        {"id": "m", "kind": "custom", "name": "model", "x": 0, "y": 0, "inner": {
            "type": "nested", "version": 1, "nodes": [
                {"id": "bi", "kind": "input", "name": "in", "x": 0, "y": 0, "dout": 1},
                {"id": "b1", "kind": "layer", "name": "dead fc",
                 "x": 0, "y": 0, "din": 1, "dout": 1},
                {"id": "b2", "kind": "layer", "name": "out fc",
                 "x": 0, "y": 0, "din": 1, "dout": 1},
                {"id": "bo", "kind": "output", "name": "out", "x": 0, "y": 0, "din": 1}],
            "edges": [{"from": "bi", "to": "b2"}]}},
        {"id": "o", "kind": "output", "name": "logits", "x": 0, "y": 0, "din": 1}],
    "edges": [{"from": "i", "to": "m"}, {"from": "m", "to": "o"},
              {"from": "m", "to": "ghost"}]}
audit = nn_engine.audit_canvas(dirty)
assert not audit["ok"] and audit["boxes"] == 7, audit
assert audit["count"] == 5, audit["issues"]
assert audit["blocks"] == {"(top)": 1, "model": 4}, audit["blocks"]
assert any(i["severity"] == "error" and "not here" in i["what"]
           for i in audit["issues"]), audit["issues"]
assert {i["box"] for i in audit["issues"] if i["where"] == "model"} == \
    {"dead fc", "out fc", "out"}, audit["issues"]

# ---- dtype names: the safetensors spellings reach the picker's catalog ----
assert nn_engine._HF_DTYPE_NAMES["F8_E4M3"] == "f8_e4m3", "fp8 e4m3"
assert nn_engine._HF_DTYPE_NAMES["F8_E5M2"] == "f8_e5m2", "fp8 e5m2"
assert nn_engine._HF_DTYPE_NAMES["U8"] == "u8", "uint8"
assert nn_engine._HF_DTYPE_NAMES["BOOL"] == "bool", "bool"

print("hf_canvas: all assertions passed")
