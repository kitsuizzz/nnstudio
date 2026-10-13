"""nnstudio's GGUF -> canvas builder: pure, so it runs on synthetic tensors."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_engine import build_canvas  # noqa: E402

META = {"arch": "test", "name": "fixture model", "blocks": 2, "d_model": 512,
        "heads": 8, "kv_heads": 2,
        "head_dim": 64, "rope": 10000.0, "eps": 1e-6, "ctx": 8192,
        "experts": None, "topk": None}
def T(name, dims, dtype="f16"):
    """A fixture tensor with the info a real GGUF reader hands over."""
    params = 1
    for d in dims:
        params *= d
    bpp = {"q4_k": 0.5625, "q8_0": 1.0625, "f16": 2.0, "f32": 4.0}[dtype]
    return (name, dims, {"dtype": dtype, "params": params, "bytes": int(params * bpp)})


TENSORS = [
    T("token_embd.weight", [512, 4096], "f16"),
    T("output_norm.weight", [512], "f32"),
    T("output.weight", [512, 4096], "f16"),
    # an attention block
    T("blk.0.attn_norm.weight", [512], "f32"),
    T("blk.0.attn_q.weight", [512, 1024], "q4_k"), T("blk.0.attn_k.weight", [512, 128], "q8_0"),
    T("blk.0.attn_v.weight", [512, 128], "f16"), T("blk.0.attn_output.weight", [1024, 512], "f16"),
    T("blk.0.attn_q_norm.weight", [64], "f32"), T("blk.0.attn_k_norm.weight", [64], "f32"),
    T("blk.0.attn_gate.weight", [512, 1024], "f16"),
    T("blk.0.post_attention_norm.weight", [512], "f32"),
    T("blk.0.ffn_norm.weight", [512], "f32"),
    T("blk.0.ffn_gate.weight", [512, 1408], "f16"), T("blk.0.ffn_up.weight", [512, 1408], "f16"),
    T("blk.0.ffn_down.weight", [1408, 512], "f16"),
    T("blk.0.post_ffw_norm.weight", [512], "f32"), T("blk.0.layer_output_scale.weight", [1], "f16"),
    # an SSM block with something we do not know
    T("blk.1.attn_norm.weight", [512], "f32"), T("blk.1.ssm_a", [16], "f32"),
    T("blk.1.ssm_conv1d.weight", [4, 1024], "f16"), T("blk.1.ssm_out.weight", [1024, 512], "f16"),
    T("blk.1.ffn_norm.weight", [512], "f32"),
    T("blk.1.ffn_gate.weight", [512, 1408], "f16"), T("blk.1.ffn_up.weight", [512, 1408], "f16"),
    T("blk.1.ffn_down.weight", [1408, 512], "f16"), T("blk.1.mystery.weight", [512, 512], "f16"),
]

c = build_canvas(META, TENSORS)
assert c["type"] == "canvas" and c["version"] == 1, "a canvas design"
# the whole model is one dedicated nested box, named after the model
assert [n["kind"] for n in c["nodes"]] == ["input", "custom", "output"], \
    "the top level is the model box between its tokens and logits"
outer = {n["name"]: n for n in c["nodes"]}
mbox = outer["fixture model"]
assert mbox["kind"] == "custom" and mbox["inner"]["type"] == "nested", "the box is the model"
assert any(e["from"] == outer["tokens"]["id"] and e["to"] == mbox["id"] for e in c["edges"]), \
    "tokens feeds the box"
assert any(e["from"] == mbox["id"] and e["to"] == outer["logits"]["id"] for e in c["edges"]), \
    "the box feeds logits"
assert outer["tokens"]["dout"] == 1 and outer["logits"]["din"] == 4096, \
    "the outer boundaries carry the model's own sizes"
inner = mbox["inner"]
top = {n["name"]: n for n in inner["nodes"]}
assert top["tokens"]["kind"] == "input" and top["tokens"]["dout"] == 1, \
    "the token input is one id (shape 1), not a vocab-wide vector"
emb = top["token embedding"]
assert (emb["kind"], emb["din"], emb["dout"], emb["vocab"]) == ("embed", 1, 512, 4096), \
    "the embedding is a table lookup: one id in, a 512 vector out of a 4096-row table"
assert emb["t"] == "token_embd.weight", "bound to its tensor"
labels = [n["name"] for n in inner["nodes"] if n["kind"] == "custom"]
assert labels == ["blk 0 \u00b7 attn mlp", "blk 1 \u00b7 ssm mlp"], "block labels: %r" % labels
assert top["tokens"]["id"] and any(e["from"] == top["tokens"]["id"] and e["to"] == emb["id"] for e in inner["edges"]), "input -> embedding"
assert top["lm head"]["dout"] == 4096 and top["logits"]["din"] == 4096, "the LM head reads out to the vocab"
assert top["output_norm"]["kind"] == "norm" and top["output_norm"]["norm"] == "rms", "the final norm"
assert top["output_norm"]["t"] == "output_norm.weight", "…bound"
# the box's dots are exactly the boundary boxes it holds
assert [n["kind"] for n in inner["nodes"] if n["kind"] in ("input", "output")] == \
    ["input", "output"], "one in dot, one out dot"

b0 = top["blk 0 \u00b7 attn mlp"]["inner"]
assert b0["type"] == "nested" and b0["version"] == 1, "a block's own design says nested"
kinds0 = [(n["kind"], n["name"]) for n in b0["nodes"]]
assert ("custom", "attention") in kinds0 and ("norm", "attn_norm") in kinds0, "attention block internals: %r" % kinds0
att = next(n for n in b0["nodes"] if n["name"] == "attention")
assert att["inner"]["type"] == "nested", "attention is a nested box"
akinds = [(n["kind"], n["name"]) for n in att["inner"]["nodes"]]
for want in [("layer", "q proj"), ("layer", "k proj"), ("layer", "v proj"), ("layer", "out proj"),
             ("op", "rope"), ("op", "k\u1d40"), ("op", "q\u00b7k\u1d40 scores"), ("op", "scores\u00b7v"),
             ("act", "softmax"), ("norm", "q norm"), ("norm", "k norm")]:
    assert want in akinds, "attention atom %r missing from %r" % (want, akinds)
abind = {n["name"]: n.get("t") for n in att["inner"]["nodes"] if n.get("t")}
assert abind.get("q proj") == "blk.0.attn_q.weight", "q bound: %r" % abind
assert abind.get("q norm") == "blk.0.attn_q_norm.weight", "q norm bound: %r" % abind
assert abind.get("out proj") == "blk.0.attn_output.weight", "out proj bound: %r" % abind
assert next(n for n in att["inner"]["nodes"] if n["name"] == "scores\u00b7v")["op"] == "matmul", "the contraction atom"
assert ("act", "sigmoid") in akinds and ("op", "gate") in akinds, "the gate path is there too"
# no atom floats: inside an attention every box must be fed, and feed something
def orphans(doc):
    ids = {n["id"] for n in doc["nodes"]}
    ins = {e["to"] for e in doc["edges"]}
    outs = {e["from"] for e in doc["edges"]}
    loose = []
    for n in doc["nodes"]:
        if n.get("loose"):                     # unmapped tensors float by design
            continue
        if n["kind"] != "input" and n["id"] not in ins:
            loose.append(n["name"] + " has no input")
        if n["kind"] != "output" and n["id"] not in outs:
            loose.append(n["name"] + " has no output")
    return loose
assert orphans(att["inner"]) == [], "attention atoms are all wired: %r" % orphans(att["inner"])
gp = next(n for n in att["inner"]["nodes"] if n["name"] == "gate proj")
fed = [e["from"] for e in att["inner"]["edges"] if e["to"] == gp["id"]]
assert len(fed) == 1 and fed[0] == next(n for n in att["inner"]["nodes"] if n["kind"] == "input")["id"], \
    "the gate projection is fed from the attention input"

gate = next(n for n in b0["nodes"] if n["name"] == "ffn_gate")
down = next(n for n in b0["nodes"] if n["name"] == "ffn_down")
assert (gate["din"], gate["dout"]) == (512, 1408) and (down["din"], down["dout"]) == (1408, 512), "the MLP sizes"
assert any(n["kind"] == "act" and n["act"] == "silu" for n in b0["nodes"]), "SwiGLU's silu"
assert sum(1 for n in b0["nodes"] if n["kind"] == "op" and n["op"] == "add") == 2, "two residual adds"
# gemma's sandwich norm and its per-layer output scalar are part of the flow
pf = next(n for n in b0["nodes"] if n["name"] == "post_ffw_norm")
assert pf["kind"] == "norm" and pf["t"] == "blk.0.post_ffw_norm.weight", "the post-FFN norm is in the chain"
scales = [n for n in b0["nodes"] if n["name"] == "layer scale"]
assert len(scales) == 2 and all(n["op"] == "scale" for n in scales), "the layer scale sits on both sublayers"
assert all(n["t"] == "blk.0.layer_output_scale.weight" for n in scales), "…bound to its scalar tensor"
assert all(n["x"] >= 0 and n["y"] >= 0 for n in b0["nodes"]), "inner positions laid out"
# every inner inner input/output box gives the nested box its dots
assert [n["kind"] for n in b0["nodes"] if n["name"] in ("in", "out")] == ["input", "output"], "the block's own dots"

b1 = top["blk 1 \u00b7 ssm mlp"]["inner"]
ssm = next(n for n in b1["nodes"] if n["name"] == "state space")
skinds = [(n["kind"], n["name"]) for n in ssm["inner"]["nodes"]]
for want in [("op", "conv"), ("op", "scan"), ("layer", "out proj")]:
    assert want in skinds, "state-space atom %r missing from %r" % (want, skinds)
conv_node = next(n for n in ssm["inner"]["nodes"] if n["name"] == "conv")
assert conv_node.get("act") == "silu", "the short conv carries its silu activation"
assert not any(n["kind"] == "act" for n in ssm["inner"]["nodes"]), \
    "…as one atom, not a duplicate silu box"
sb = {n["name"]: n.get("t") for n in ssm["inner"]["nodes"] if n.get("t")}
assert sb.get("conv") == "blk.1.ssm_conv1d.weight" and "blk.1.ssm_a" in (sb.get("scan") or ""), \
    "ssm tensors bound to the atoms: %r" % sb
assert (sb.get("out proj") or "") == "blk.1.ssm_out.weight", "the out projection binds ssm_out"
myst = next(n for n in b1["nodes"] if n["name"] == "mystery")
assert myst["t"] == "blk.1.mystery.weight" and myst["kind"] == "layer", "unknown tensors land bound, not dropped"
assert not any(e["from"] == myst["id"] or e["to"] == myst["id"] for e in b1["edges"]), "…and unwired"

# the inside is folded: rows of the chain read left to right, and a long model
# is a compact block of rows rather than one enormous column
long = build_canvas({"arch": "long", "name": "long", "blocks": 64, "d_model": 8},
                    [T("token_embd.weight", [8, 16]), T("output_norm.weight", [8], "f32")])
lin = next(n for n in long["nodes"] if n["kind"] == "custom")["inner"]
lblocks = [n for n in lin["nodes"] if n["kind"] == "custom"]
lys = sorted({n["y"] for n in lblocks})
lxs = sorted({n["x"] for n in lblocks})
assert 2 <= len(lys) <= 12, "the chain folds into rows: %d rows" % len(lys)
assert len(lxs) <= 12, "\u2026and the rows have columns: %d" % len(lxs)
assert max(lys) + 60 <= 1400, "the fold stays compact: %d tall" % (max(lys) + 60)
for y in lys[:2]:
    row = [n["x"] for n in lblocks if n["y"] == y]
    assert row == sorted(row), "every row reads left to right: %r" % row
assert all(n["x"] >= 0 and n["y"] >= 0 for n in lin["nodes"]), "fold positions laid out"

# a model with no per-block tensors still builds (every block pass-through)
bare = build_canvas({"arch": "bare", "blocks": 1, "d_model": 64}, [("token_embd.weight", [64, 8])])
bare_inner = next(n for n in bare["nodes"] if n["kind"] == "custom")["inner"]
blk = next(n for n in bare_inner["nodes"] if n["kind"] == "custom")
assert [n["kind"] for n in blk["inner"]["nodes"]] == ["input", "output"], "a bare block is pass-through"
assert len(blk["inner"]["edges"]) == 1, "…wired in to out"
# MoE: expert tensors become one mixture box, no dense path
moe = build_canvas({"arch": "moe", "blocks": 1, "d_model": 64, "experts": 8, "topk": 2},
                   [("blk.0.ffn_gate_inp.weight", [64, 8]),
                    ("blk.0.ffn_gate_exps.weight", [64, 8, 32]),
                    ("blk.0.ffn_down_exps.weight", [32, 8, 64])])
moe_box = next(n for n in moe["nodes"] if n["kind"] == "custom")["inner"]
mb = next(n for n in moe_box["nodes"] if n["kind"] == "custom")["inner"]
m = next(n for n in mb["nodes"] if n["name"] == "mixture of experts")
mk = [(n["kind"], n["name"]) for n in m["inner"]["nodes"]]
for want in [("layer", "router"), ("act", "softmax"), ("op", "top 2"),
             ("layer", "expert up"), ("act", "silu"), ("layer", "expert down"), ("op", "weights")]:
    assert want in mk, "MoE atom %r missing from %r" % (want, mk)
mi = {n["name"]: n.get("t") for n in m["inner"]["nodes"] if n.get("t")}
assert mi.get("router") == "blk.0.ffn_gate_inp.weight", "router bound: %r" % mi
assert "blk.0.ffn_gate_exps.weight" in (mi.get("expert up") or ""), "experts bound: %r" % mi
assert "blk.0.ffn_down_exps.weight" in (mi.get("expert down") or ""), "…both halves: %r" % mi
print("gguf_canvas: all assertions passed")

# the same guard over every doc of the whole built canvas
def all_docs(canvas):
    docs = [canvas]
    for n in canvas["nodes"]:
        if n.get("kind") == "custom" and n.get("inner"):
            docs.extend(all_docs(n["inner"]))
    return docs

for doc in all_docs(c):
    loose = orphans(doc)
    assert loose == [], "no floating boxes anywhere: %r" % loose

# every bound box carries the tensor's dtype, element count and byte size
qp = next(n for n in att["inner"]["nodes"] if n["name"] == "q proj")
assert (qp["dtype"], qp["params"], qp["wbytes"]) == ("q4_k", 512 * 1024, int(512 * 1024 * 0.5625)), \
    "a bound box knows its dtype and size: %r" % {k: qp.get(k) for k in ("dtype", "params", "wbytes")}
kn = next(n for n in att["inner"]["nodes"] if n["name"] == "k norm")
assert (kn["dtype"], kn["params"]) == ("f32", 64), "a 1-D norm weighs its channels"

# the canvas carries the model's own numbers for the estimate
m = c["model"]
assert (m["layers"], m["kv_heads"], m["head_dim"], m["ctx"]) == (2, 2, 64, 8192), \
    "the model block has the shapes the KV cache needs: %r" % m
assert m["attn_layers"] == 1, "only the attention block caches K/V: %r" % m
assert m["params"] == sum(e[2]["params"] for e in TENSORS), "…and the file's exact param count"
assert m["bytes"] == sum(e[2]["bytes"] for e in TENSORS), "…and its exact byte size"
assert m["dtypes"].get("q4_k", 0) > 0 and m["dtypes"].get("f32", 0) > 0, \
    "…with a breakdown by dtype: %r" % m["dtypes"]

# a fused layer with no attn_output must not invent an output projection
fused = build_canvas({"arch": "t", "blocks": 1, "d_model": 8},
                     [T("blk.0.attn_qkv.weight", [8, 24], "q4_k"),
                      T("blk.0.attn_gate.weight", [8, 24], "q4_k")])
fused_model = next(n for n in fused["nodes"] if n["kind"] == "custom")["inner"]
fused_inner = next(n for n in fused_model["nodes"] if n["kind"] == "custom")["inner"]
fnames = [n["name"] for n in fused_inner["nodes"]]
assert "out proj" not in fnames, "a fused attention without attn_output invents nothing: %r" % fnames
assert fused["model"]["params"] == 8 * 24 * 2, "…and its params are only what the file holds"
assert fused["model"]["attn_layers"] == 1, "a fused qkv layer is attention: %r" % fused["model"]

# the next-token (MTP) head is a branch, and nothing in it floats
mtp_c = build_canvas({"arch": "mtp", "blocks": 2, "d_model": 8, "heads": 2, "kv_heads": 2,
                      "head_dim": 4, "ctx": 4096},
                     [T("token_embd.weight", [8, 16]), T("output_norm.weight", [8], "f32"),
                      T("output.weight", [8, 16]),
                      T("blk.0.attn_norm.weight", [8], "f32"), T("blk.0.ffn_norm.weight", [8], "f32"),
                      T("blk.0.ffn_gate.weight", [8, 16]), T("blk.0.ffn_up.weight", [8, 16]),
                      T("blk.0.ffn_down.weight", [16, 8]),
                      T("blk.1.nextn.hnorm.weight", [8], "f32"),
                      T("blk.1.nextn.enorm.weight", [8], "f32"),
                      T("blk.1.nextn.eh_proj.weight", [16, 8]),
                      T("blk.1.nextn.shared_head_norm.weight", [8], "f32"),
                      T("blk.1.ffn_norm.weight", [8], "f32"),
                      T("blk.1.ffn_gate.weight", [8, 16]), T("blk.1.ffn_up.weight", [8, 16]),
                      T("blk.1.ffn_down.weight", [16, 8])])
mtp_model = next(n for n in mtp_c["nodes"] if n["kind"] == "custom")
mtp_top = {n["name"]: n for n in mtp_model["inner"]["nodes"] if n["kind"] == "custom"}
assert list(mtp_top) == ["blk 0 \u00b7 mlp", "blk 1 \u00b7 mtp"], "the MTP block is its own branch: %r" % list(mtp_top)
mtp_inner = mtp_top["blk 1 \u00b7 mtp"]["inner"]
mtp_names = [n["name"] for n in mtp_inner["nodes"]]
for want in ["h", "e", "hnorm", "enorm", "concat", "eh proj", "shared head norm"]:
    assert want in mtp_names, "the next-token head has %r: %r" % (want, mtp_names)
assert "blk.1.nextn.eh_proj.weight" == next(n for n in mtp_inner["nodes"] if n["name"] == "eh proj")["t"]
assert [n["kind"] for n in mtp_inner["nodes"]].count("input") == 2, "it takes h and e"
mtp_outer = {n["name"]: n for n in mtp_c["nodes"]}
assert mtp_outer["mtp hidden"]["kind"] == "output" and mtp_outer["mtp hidden"]["din"] == 8, \
    "its own logits box, mirrored outside the model box"
mouts = sorted([n for n in mtp_model["inner"]["nodes"] if n["kind"] == "output"],
               key=lambda n: (n["y"], n["x"]))
assert [n["name"] for n in mouts] == ["logits", "mtp hidden"], "the box's two output dots"
mtp_node = next(n for n in mtp_model["inner"]["nodes"]
                if n["kind"] == "custom" and n["name"].startswith("blk 1"))
e_wire = next(e for e in mtp_model["inner"]["edges"]
              if e["to"] == mtp_node["id"] and e["toPort"] == 1)
assert e_wire.get("via"), "the next-token embedding's long wire bends down the lane: %r" % e_wire
assert any(e["from"] == mtp_model["id"] and e["to"] == mtp_outer["mtp hidden"]["id"]
           and e["fromPort"] == 1 for e in mtp_c["edges"]), \
    "the mtp hidden state leaves the box's second dot"
assert not [n for d in all_docs(mtp_c) for n in d["nodes"] if n.get("loose")], "nothing is left loose"
assert mtp_c["model"]["attn_layers"] == 0, \
    "no block in this fixture holds attention, MTP included: %r" % mtp_c["model"]
assert mtp_c["model"]["params"] == sum(e[2]["params"] for e in [
    T("token_embd.weight", [8, 16]), T("output_norm.weight", [8], "f32"), T("output.weight", [8, 16]),
    T("blk.0.attn_norm.weight", [8], "f32"), T("blk.0.ffn_norm.weight", [8], "f32"),
    T("blk.0.ffn_gate.weight", [8, 16]), T("blk.0.ffn_up.weight", [8, 16]), T("blk.0.ffn_down.weight", [16, 8]),
    T("blk.1.nextn.hnorm.weight", [8], "f32"), T("blk.1.nextn.enorm.weight", [8], "f32"),
    T("blk.1.nextn.eh_proj.weight", [16, 8]), T("blk.1.nextn.shared_head_norm.weight", [8], "f32"),
    T("blk.1.ffn_norm.weight", [8], "f32"), T("blk.1.ffn_gate.weight", [8, 16]),
    T("blk.1.ffn_up.weight", [8, 16]), T("blk.1.ffn_down.weight", [16, 8])]), \
    "the shared head is not counted twice"

# tied embeddings: the head reuses token_embd and is counted once, not twice
tied = build_canvas({"arch": "t", "blocks": 1, "d_model": 8},
                    [T("token_embd.weight", [8, 16], "q4_k"),
                     T("output_norm.weight", [8], "f32"),
                     T("blk.0.attn_norm.weight", [8], "f32")])
tied_inner = next(n for n in tied["nodes"] if n["kind"] == "custom")["inner"]
tied_top = {n["name"]: n for n in tied_inner["nodes"] if n["kind"] != "custom"}
assert tied_top["lm head"]["t"] == "token_embd.weight", "a tied head reuses the embedding table"
assert tied_top["lm head"].get("shared") is True, "…and is marked shared"
assert tied_top["lm head"]["params"] == 0 and tied_top["lm head"]["wbytes"] == 0, \
    "…so it weighs nothing itself"
assert tied_top["token embedding"]["params"] == 8 * 16, "the table is counted on the embedding"
box_p = sum(n.get("params", 0) for d in all_docs(tied) for n in d["nodes"])
assert box_p == tied["model"]["params"], "the boxes add up to the file exactly: %d vs %d" % (
    box_p, tied["model"]["params"])

# no tensor is ever dropped: every name in the file is bound to some box
for c2 in (c, mtp_c, tied):
    bound = set()
    for d in all_docs(c2):
        for n in d["nodes"]:
            for t in (n.get("t") or "").split(","):
                if t.strip():
                    bound.add(t.strip())
    names = {e[0] for e in (TENSORS if c2 is c else [])} or None
    if names is not None:
        assert names <= bound, "every tensor is bound somewhere: %r" % (names - bound)

# the weight registry: one namespace per file tensor, and bindings use it
reg = build_canvas({"arch": "qwen35", "blocks": 1, "d_model": 8, "heads": 2, "kv_heads": 2,
                    "head_dim": 4, "ctx": 16, "file": "m.gguf"},
                   [T("token_embd.weight", [8, 16]), T("output_norm.weight", [8], "f32"),
                    T("output.weight", [8, 16]),
                    T("blk.0.attn_norm.weight", [8], "f32"), T("blk.0.ffn_norm.weight", [8], "f32"),
                    T("blk.0.ffn_gate.weight", [8, 16]), T("blk.0.ffn_up.weight", [8, 16]),
                    T("blk.0.ffn_down.weight", [16, 8])])
W = reg.get("weights") or {}
assert "m.gguf::blk.0.ffn_gate.weight" in W, "namespaces are file::tensor: %r" % list(W)[:3]
assert W["m.gguf::blk.0.ffn_gate.weight"]["shape"] == [8, 16], W["m.gguf::blk.0.ffn_gate.weight"]
assert W["m.gguf::blk.0.ffn_gate.weight"].get("bytes"), "the registry carries metadata: %r" % W["m.gguf::blk.0.ffn_gate.weight"]
refs = [n["t"] for n in all_docs(reg) for n in n["nodes"] if n.get("t")]
assert refs and all(x.startswith("m.gguf::") for x in refs), "bindings carry the namespace: %r" % refs[:3]

# ---- an mmproj's vision tower: the cluster the loader adds -------------------
import tempfile                                                  # noqa: E402
import numpy as np                                               # noqa: E402
from nn_engine import _vision_cluster, clip_meta_and_tensors     # noqa: E402


def _write_mmproj(path, projector="qwen3vl_merger"):
    from gguf import GGUFWriter
    w = GGUFWriter(path, "clip")
    w.add_string("general.name", "fixture vision")
    w.add_string("clip.projector_type", projector)
    w.add_uint32("clip.vision.embedding_length", 8)
    w.add_uint32("clip.vision.block_count", 2)
    w.add_uint32("clip.vision.attention.head_count", 2)
    w.add_uint32("clip.vision.image_size", 32)
    w.add_uint32("clip.vision.patch_size", 16)
    w.add_uint32("clip.vision.projection_dim", 12)
    w.add_uint32("clip.vision.spatial_merge_size", 2)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    w.add_tensor("v.patch_embd.weight", np.zeros((16, 16, 3, 8), dtype=np.float16))
    w.add_tensor("mm.0.weight", np.zeros((32, 32), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


with tempfile.TemporaryDirectory() as td:
    mp = os.path.join(td, "mmproj-fixture-F16.gguf")
    _write_mmproj(mp)
    meta, tensors = clip_meta_and_tensors(mp)
    assert meta["projector"] == "qwen3vl_merger", meta
    assert (meta["d_model"], meta["blocks"], meta["proj_dim"]) == (8, 2, 12), meta
    assert meta["image_mean"] == [0.5, 0.5, 0.5] and meta["merge"] == 2, meta
    base = build_canvas({"arch": "t", "blocks": 0, "d_model": 8, "heads": 2,
                         "file": "t.gguf"}, [])
    canvas = _vision_cluster(base, mp)
    by_kind = {}
    for node in canvas["nodes"]:
        by_kind.setdefault(node["kind"], []).append(node)
    img = [x for x in canvas["nodes"] if x.get("input_mode") == "image"][0]
    assert img["kind"] == "input", img
    assert (img["image_size"], img["dout"]) == (32, 3), img      # a picture arrives
    tow = by_kind["vision"][0]
    assert tow["file"] == mp and tow["projector"] == "qwen3vl_merger", tow
    assert (tow["d_model"], tow["blocks"], tow["proj_dim"]) == (8, 2, 12), tow
    assert (tow["din"], tow["dout"]) == (3, 12), tow
    assert tow["image_mean"] == "0.5 0.5 0.5" and tow["image_std"] == "0.5 0.5 0.5", tow
    assert tow["params"] == sum(t[2]["params"] for t in tensors), tow
    assert tow["wbytes"] == sum(t[2]["bytes"] for t in tensors), tow
    assert tow["dtype"] == "f16", tow
    emb = [x for x in canvas["nodes"]
           if x["kind"] == "output" and x["name"] == "image embeds"]
    assert emb and emb[0]["din"] == 12, emb
    wired = {(e["from"], e["to"]) for e in canvas["edges"]}
    assert (img["id"], tow["id"]) in wired, wired
    assert (tow["id"], emb[0]["id"]) in wired, wired
    assert canvas["model"]["vision"]["projector"] == "qwen3vl_merger", canvas["model"]
    reg2 = canvas.get("weights") or {}
    key = "%s::v.patch_embd.weight" % mp
    assert key in reg2 and reg2[key]["dtype"] == "f16", list(reg2)[:3]
    # an unsupported projector refuses, naming the type it cannot run
    mp2 = os.path.join(td, "mmproj-other.gguf")
    _write_mmproj(mp2, projector="gemma4uv")
    try:
        _vision_cluster(base, mp2)
        raise AssertionError("an unsupported projector must refuse")
    except ValueError as e:
        assert "gemma4uv" in str(e), e

print("gguf_canvas: the vision tower cluster rides along")

# ---- the same folder holds both kinds: each .gguf is recognized on its own --
from nn_engine import find_mmproj, gguf_canvas, gguf_kind, local_models  # noqa: E402


def _write_tiny_model(path):
    from gguf import GGUFWriter
    w = GGUFWriter(path, "llama")
    w.add_string("general.name", "fixture model")
    w.add_uint32("llama.block_count", 2)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


with tempfile.TemporaryDirectory() as td:
    a, b = os.path.join(td, "a"), os.path.join(td, "b")
    os.makedirs(a)
    os.makedirs(b)
    model = os.path.join(a, "tiny-model.gguf")
    _write_tiny_model(model)
    mate = os.path.join(a, "vision-F16.gguf")      # no "mmproj" in the name
    _write_mmproj(mate)
    far = os.path.join(b, "mmproj-tiny-model-F16.gguf")
    _write_mmproj(far)
    blah = os.path.join(b, "mmproj-nothing-F16.gguf")
    _write_mmproj(blah)

    kinds = {m["name"]: m["kind"] for m in local_models(td)}
    assert kinds == {"tiny-model.gguf": "gguf", "vision-F16.gguf": "mmproj",
                     "mmproj-tiny-model-F16.gguf": "mmproj",
                     "mmproj-nothing-F16.gguf": "mmproj"}, kinds
    assert gguf_kind(mate) == "mmproj", "the header decides, not the folder or the name"
    # the pairing: the one beside it, then the best name match
    assert find_mmproj(model, [mate]) == mate, "the encoder in the same folder"
    assert find_mmproj(model, [mate, far]) == mate, "…wins over one further off"
    assert find_mmproj(model, [far]) == far, "a name match across folders"
    assert find_mmproj(model, [blah]) is None, "nothing shared: no pairing"
    assert find_mmproj(model, []) is None, "no encoder, no tower"
    # a vision encoder is never loaded as a text model
    try:
        gguf_canvas(mate)
        raise AssertionError("an mmproj must not load as a model")
    except ValueError as e:
        assert "vision encoder" in str(e), e

print("gguf_canvas: a folder's .gguf files are recognized by their own headers")
