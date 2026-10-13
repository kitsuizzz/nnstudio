#!/usr/bin/env python3
"""nnstudio's engine: turn a GGUF's header into a Training canvas.

The block structure is what the tensors say it is, every box is bound to the
tensor(s) it came from, and the wire sizes come from the tensor shapes. Only
the header and the tensor list are read (mmap) — no weights are loaded.

`build_canvas` is pure, so the unit test feeds it synthetic tensors;
`gguf_canvas` reads a real file. Extracted from the next-token app's engine:
the server here needs no model loading, no tokenizer, no torch — just a
GGUF header and its tensor list.
"""
import math
import os
import re
import json
import struct
import urllib.error
import urllib.request

import nn_arc        # the researched-repo database (`hf_arc`) + job file state

GRID = 40          # the canvas breadboard: every box and wire bend sits on it


def is_gguf(model: str) -> bool:
    return (model or "").strip().lower().endswith(".gguf")


ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_mmproj_path(path: str) -> bool:
    """True for vision-encoder files — they are not text models."""
    return "mmproj" in os.path.basename(path or "").lower()


def _resolve_model(model: str) -> str:
    """Keep old flat model paths working after the move into models/: when the
    given path is missing but models/<basename> exists, use that. A relative
    path resolves against the repo wherever the app was started from."""
    model = (model or "").strip()
    if not is_gguf(model):
        return model
    if os.path.isfile(model):
        return model
    cand = os.path.join(ENGINE_DIR, model)          # 'models/mmproj/x.gguf'
    if os.path.isfile(cand):
        return cand
    if not _is_mmproj_path(model):                  # legacy flat paths
        cand = os.path.join(ENGINE_DIR, "models", os.path.basename(model))
        if os.path.isfile(cand):
            return cand
    return model



# ---------------------------------------------------------------- GGUF -> canvas
# A Training canvas can be built straight from a GGUF: the block structure is
# what its tensors say it is, every box is bound to the tensor(s) it came from,
# and the wire sizes come from the tensor shapes. Only the header and the tensor
# list are read (mmap) — no weights are loaded. `build_canvas` is pure, so the
# unit test feeds it synthetic tensors; `gguf_canvas` reads a real file.
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_GGML_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
               8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
               13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
               18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s",
               23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64",
               28: "bf16", 29: "f64", 30: "iq1_m", 34: "tq1_0", 35: "tq2_0"}
_ATTN_KEYS = ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output", "attn_gate")
_NORM_KEYS = ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm")
_FFN_KEYS = ("ffn_gate", "ffn_up", "ffn_down")


def _tensor_lin(dims):
    """(in, out) of a linear weight — GGUF order: ne0 is the input side."""
    if not dims:
        return (None, None)
    if len(dims) == 1:
        return (int(dims[0]), int(dims[0]))
    return (int(dims[0]), int(dims[1]))


def _tensor_embed(dims):
    """(vocab, d_model) of an embedding/head table — GGUF stores [d_model, vocab]."""
    if not dims:
        return (None, None)
    if len(dims) == 1:
        return (int(dims[0]), int(dims[0]))
    return (int(dims[1]), int(dims[0]))


def _gguf_scalar(field):
    contents = getattr(field, "contents", None)
    value = contents() if callable(contents) else contents
    if value is None:
        value = getattr(field, "value", None)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def gguf_meta_and_tensors(path):
    """A GGUF's hyperparameters and tensor list, header only (mmap)."""
    from gguf import GGUFReader
    reader = GGUFReader(path)

    def kv(*suffixes):
        for key, field in reader.fields.items():
            if any(key.endswith(s) for s in suffixes):
                return _gguf_scalar(field)
        return None

    def num(*suffixes):
        try:
            v = kv(*suffixes)
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def num_max(*suffixes):             # per-layer lists: the largest is the safe one
        try:
            v = kv(*suffixes)
            if isinstance(v, (list, tuple)):
                return max((int(x) for x in v if x is not None), default=None)
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def fnum(*suffixes):
        try:
            v = kv(*suffixes)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    meta = {
        "arch": str(kv("general.architecture") or "unknown"),
        "name": str(kv("general.name") or "").strip() or None,
        "blocks": num("block_count") or 0,
        "d_model": num("embedding_length"),
        "heads": num("attention.head_count"),
        "kv_heads": num_max("attention.head_count_kv") or num("attention.head_count"),
        "head_dim": num("attention.key_length"),
        "rope": fnum("rope.freq_base"),
        "rope_dim": num("rope.dimension_count"),
        "rope_swa": fnum("rope.freq_base_swa"),
        "rope_dim_swa": num("rope.dimension_count_swa"),
        "ctx": num("context_length"),
        "eps": fnum("attention.layer_norm_rms_epsilon") or fnum("attention.layer_norm_epsilon"),
        "experts": num("expert_count"),
        "topk": num("expert_used_count"),
    }
    tensors = []
    for t in reader.tensors:
        dims = [int(d) for d in t.shape]
        params = 1
        for d in dims:
            params *= d
        try:
            nbytes = int(t.n_bytes)
        except Exception:  # noqa: BLE001 — older gguf-py: ask the data view
            nbytes = int(getattr(t.data, "nbytes", 0))
        tensors.append((t.name, dims, {
            "dtype": _GGML_TYPES.get(int(t.tensor_type), "?"),
            "params": params, "bytes": nbytes}))
    return meta, tensors


def clip_meta_and_tensors(path):
    """An mmproj's (vision encoder) clip hyperparameters and tensor list,
    header only — the same shape `gguf_meta_and_tensors` answers."""
    from gguf import GGUFReader
    reader = GGUFReader(path)

    def kv(*suffixes):
        for key, field in reader.fields.items():
            if any(key.endswith(s) for s in suffixes):
                return _gguf_scalar(field)
        return None

    def num(*suffixes):
        try:
            v = kv(*suffixes)
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def fnum(*suffixes):
        try:
            v = kv(*suffixes)
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def vec(*suffixes):
        v = kv(*suffixes)
        if v is None:
            return None
        if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None
        return [float(v)]

    meta = {
        "arch": str(kv("general.architecture") or "clip"),
        "name": str(kv("general.name") or "").strip() or None,
        "projector": str(kv("projector_type") or "").strip().lower() or None,
        "d_model": num("vision.embedding_length"),
        "heads": num("vision.attention.head_count"),
        "blocks": num("vision.block_count"),
        "ffn": num("vision.feed_forward_length"),
        "image_size": num("vision.image_size"),
        "patch_size": num("vision.patch_size"),
        "proj_dim": num("vision.projection_dim"),
        "merge": num("vision.spatial_merge_size") or 2,
        "eps": fnum("vision.attention.layer_norm_epsilon"),
        "image_mean": vec("vision.image_mean"),
        "image_std": vec("vision.image_std"),
    }
    tensors = []
    for t in reader.tensors:
        dims = [int(d) for d in t.shape]
        params = 1
        for d in dims:
            params *= d
        try:
            nbytes = int(t.n_bytes)
        except Exception:  # noqa: BLE001 — older gguf-py: ask the data view
            nbytes = int(getattr(t.data, "nbytes", 0))
        tensors.append((t.name, dims, {
            "dtype": _GGML_TYPES.get(int(t.tensor_type), "?"),
            "params": params, "bytes": nbytes}))
    return meta, tensors


def _snap_layout(doc, step=GRID):
    """Put every box — and every wire bend — on the canvas grid, so a loaded
    model lines up with the lines the UI draws."""
    for n in doc.get("nodes", []):
        n["x"] = int(math.floor(n.get("x", 0) / step + 0.5)) * step
        n["y"] = int(math.floor(n.get("y", 0) / step + 0.5)) * step
        inner = n.get("inner")
        if isinstance(inner, dict):
            _snap_layout(inner, step)
    for e in doc.get("edges", []):
        v = e.get("via")
        if isinstance(v, dict):
            v["x"] = int(math.floor(v.get("x", 0) / step + 0.5)) * step
            v["y"] = int(math.floor(v.get("y", 0) / step + 0.5)) * step


def _doc_maker(doc_type="nested"):
    """A little doc with its own ids: nodes, edges, add-a-box, join-two."""
    doc = {"type": doc_type, "version": 1, "nodes": [], "edges": []}
    counter = [0]

    def mk(kind, name, x, y, **kw):
        counter[0] += 1
        node = {"id": "gg%03d" % counter[0], "kind": kind, "name": name,
                "x": int(x), "y": int(y)}
        for k, v in kw.items():
            if v is not None:
                node[k] = v
        doc["nodes"].append(node)
        return node

    def link(a, b, fp=0, tp=0):
        doc["edges"].append({"from": a["id"], "to": b["id"],
                             "fromPort": int(fp), "toPort": int(tp)})

    return doc, mk, link


def _edge_via(doc, a, b, fp, tp, x, y):
    """Give one wire a bend point (the UI renders the curve through it)."""
    for e in doc.get("edges", []):
        if e["from"] == a and e["to"] == b and \
           e.get("fromPort", 0) == fp and e.get("toPort", 0) == tp:
            e["via"] = {"x": int(x), "y": int(y)}
            return True
    return False


def canvas_counts(doc):
    """(boxes, wires) in a nested canvas — every level counted."""
    n, e = len(doc.get("nodes", [])), len(doc.get("edges", []))
    for x in doc.get("nodes", []):
        if x.get("kind") == "custom" and x.get("inner"):
            a, b = canvas_counts(x["inner"])
            n += a
            e += b
    return n, e


def _has_attention(roles):
    """Does this block carry a real attention core -- its own K/V to cache?"""
    keys = set(roles)
    attn = [k for k in _ATTN_KEYS if k in keys]
    real = any(k in keys for k in ("attn_q", "attn_k", "attn_v", "attn_output"))
    fused = [k for k in ("attn_qkv", "attn_gate") if k in keys]
    if fused and not real and any(k.startswith("ssm_") for k in keys):
        attn = [k for k in attn if k not in fused]     # a fused state-space layer
    return bool(attn)


def _blk_role(suffix):
    """The role of a block tensor: its name without the .weight/.bias tail."""
    for tail in (".weight", ".bias"):
        if suffix.endswith(tail):
            return suffix[:-len(tail)]
    return suffix


def _role_dims(t, role):
    entry = t.get(role)
    return entry["dims"] if entry else None


def _role_name(t, role):
    entry = t.get(role)
    return entry["name"] if entry else None



def _block_label(i, t):
    if any(k.startswith("nextn.") for k in t):
        return "blk %d \u00b7 mtp" % i
    tags = []
    real_attn = any(k in t for k in ("attn_q", "attn_k", "attn_v", "attn_output"))
    fused = any(k in t for k in ("attn_qkv", "attn_gate"))
    has_ssm = any(k.startswith("ssm_") for k in t)
    if real_attn or (fused and not has_ssm):
        tags.append("attn")                   # a fused layer with a state space keeps it there
    if has_ssm:
        tags.append("ssm")
    if any(k.endswith("_exps") for k in t):
        tags.append("moe")
    elif all(k in t for k in _FFN_KEYS):
        tags.append("mlp")
    return "blk %d%s" % (i, " \u00b7 " + " ".join(tags) if tags else "")


def _attention_inner(meta, i, t, d_model):
    """The inside of an attention box: projections, rope, q·k, softmax, ·v."""
    doc, mk, link = _doc_maker("nested")
    eps = meta.get("eps")
    name = lambda k: _role_name(t, k)
    iin = mk("input", "in", 0, 0, dout=d_model)
    qg = None                                 # a q-fused sigmoid gate, when present
    if "attn_qkv" in t:                       # a fused projection (some first layers)
        qi, qo = _tensor_lin(_role_dims(t, "attn_qkv"))
        qkv = mk("layer", "qkv proj", 220, 0, din=qi, dout=qo, t=name("attn_qkv"))
        rp = mk("op", "rope", 440, 0, op="rope", theta=meta.get("rope") or 10000.0, dim=meta.get("rope_dim"))
        qk = mk("op", "q\u00b7k\u1d40 scores", 660, 0, op="matmul")
        sm = mk("act", "softmax", 880, 0, act="softmax", gain=1.0 / (int(meta.get("head_dim") or 256) ** 0.5), causal=1)
        pv = mk("op", "scores\u00b7v", 1100, 0, op="matmul")
        link(iin, qkv)
        link(qkv, rp)
        link(rp, qk, 0, 0)
        link(qkv, qk, 0, 1)
        link(qk, sm)
        link(sm, pv, 0, 0)
        link(qkv, pv, 0, 1)
        last, x = pv, 1320
    else:
        qi, qo = _tensor_lin(_role_dims(t, "attn_q"))
        ki, ko = _tensor_lin(_role_dims(t, "attn_k"))
        vi, vo = _tensor_lin(_role_dims(t, "attn_v"))
        qp = mk("layer", "q proj", 220, -140, din=qi, dout=qo, t=name("attn_q"))
        kp = mk("layer", "k proj", 220, 0, din=ki, dout=ko, t=name("attn_k"))
        if "attn_v" in t:
            vp = mk("layer", "v proj", 220, 140, din=vi, dout=vo, t=name("attn_v"))
            link(iin, vp)
            vsrc = vp
        else:
            # gemma-4's sliding layers carry no v projection: V *is* the k
            # projection, through a weightless rms norm (llama.cpp:
            # `Vcur = wv ? wv·x : Kcur` then `ggml_rms_norm(Vcur, eps)`)
            qn0 = _role_dims(t, "attn_q_norm") if "attn_q_norm" in t else ()
            hd = int(qn0[0]) if qn0 else (int(meta.get("head_dim") or 0) or int(ko))
            vsrc = mk("norm", "v norm", 220, 140, norm="rms", eps=eps, size=hd)
            link(kp, vsrc)
        link(iin, qp); link(iin, kp)
        last_q, last_k = qp, kp
        q_dim = int(meta.get("heads") or 0) * int(meta.get("head_dim") or 0)
        if q_dim and qo == 2 * q_dim:         # q and its gate, fused per head
            Hh = int(meta.get("heads") or 0)
            Dd = int(meta.get("head_dim") or 0)
            rs = mk("op", "q \u2192 heads", 420, -140, op="reshape",
                    dims="%d, %d" % (Hh, 2 * Dd), keep=1)
            link(qp, rs)
            qs = mk("op", "q only", 530, -140, op="slice", axis=-1, start=0, stop=Dd)
            link(rs, qs)
            qf = mk("op", "q flat", 640, -140, op="reshape", dims=str(Hh * Dd), keep=1)
            link(qs, qf)
            last_q = qf
            if "attn_gate" not in t:          # no separate gate: this half is it
                gh = mk("op", "q gate", 530, -70, op="slice", axis=-1,
                        start=Dd, stop=2 * Dd)
                link(rs, gh)
                gf = mk("op", "gate flat", 640, -70, op="reshape",
                        dims=str(Hh * Dd), keep=1)
                link(gh, gf)
                sg = mk("act", "sigmoid", 750, -70, act="sigmoid")
                link(gf, sg)
                qg = sg
        if "attn_q_norm" in t:
            qn = mk("norm", "q norm", 440, -140, norm="rms", eps=eps, t=name("attn_q_norm"))
            link(last_q, qn); last_q = qn
        if "attn_k_norm" in t:
            kn = mk("norm", "k norm", 440, 0, norm="rms", eps=eps, t=name("attn_k_norm"))
            link(last_k, kn); last_k = kn
        H = int(meta.get("heads") or 0)
        KV = int(meta.get("kv_heads") or H) or H
        D = int(meta.get("head_dim") or 0)
        if "attn_q_norm" in t:
            # per-layer geometry from the file itself: the q/k norms are one
            # head wide (gemma-4 mixes layouts — full layers 16x256/8x256,
            # sliding ones 16x512/1x512 — and the file's global heads are
            # only one of them)
            dims = _role_dims(t, "attn_q_norm")
            if dims:
                D = int(dims[0])
        q_width = (qo // 2) if qg is not None else qo
        if D and q_width % D == 0 and ko % D == 0:
            H = q_width // D
            KV = ko // D
        # per-layer rope: a gemma-4 full layer and its sliding sibling use
        # different (freq base, dim) pairs — each layer takes the pair whose
        # rope dim matches its own head width
        rbase, rdim = meta.get("rope") or 10000.0, meta.get("rope_dim")
        if D and meta.get("rope_dim_swa") is not None and D == int(meta.get("rope_dim_swa")) \
                and D != rdim:
            rbase = meta.get("rope_swa") or rbase
            rdim = meta.get("rope_dim_swa")
        if rdim and D and int(rdim) > D:
            rdim = D
        if H and D and KV and H % KV == 0 and q_width == H * D and ko == KV * D:

            def split_heads(src, px, py, tag, n_heads):
                rs = mk("op", tag + " \u2192 heads", px, py, op="reshape",
                        dims="%d, %d" % (n_heads, D), keep=1)
                link(src, rs)
                pm = mk("op", tag + " head-major", px + 110, py, op="permute",
                        axes="1, 0, 2")
                link(rs, pm)
                return pm

            rep = max(1, H // max(1, KV))
            hq = split_heads(last_q, 640, -140, "q", H)
            hk = split_heads(last_k, 640, 0, "k", KV)
            hv = split_heads(vsrc, 640, 140, "v", KV)
            if rep > 1:                       # GQA: each kv head serves a group
                kr = mk("op", "k repeat", 770, 0, op="repeat",
                        repeats=str(rep), dim=0)
                link(hk, kr); hk = kr
                vr = mk("op", "v repeat", 770, 140, op="repeat",
                        repeats=str(rep), dim=0)
                link(hv, vr); hv = vr
            rq = mk("op", "rope", 880, -140, op="rope", theta=rbase, dim=rdim)
            rk = mk("op", "rope", 880, 0, op="rope", theta=rbase, dim=rdim)
            kt = mk("op", "k\u1d40", 990, 0, op="transpose")
            qk = mk("op", "q\u00b7k\u1d40 scores", 1100, -70, op="matmul")
            sm = mk("act", "softmax", 1210, -70, act="softmax", gain=1.0 / (int(meta.get("head_dim") or 256) ** 0.5), causal=1)
            pv = mk("op", "scores\u00b7v", 1320, -70, op="matmul")
            pm = mk("op", "merge head-major", 1430, -70, op="permute",
                    axes="1, 0, 2")
            mg = mk("op", "merge", 1540, -70, op="reshape",
                    dims=str(H * D), keep=1)
            link(hq, rq); link(hk, rk)
            link(rk, kt)
            link(rq, qk, 0, 0); link(kt, qk, 0, 1)
            link(qk, sm)
            link(sm, pv, 0, 0); link(hv, pv, 0, 1)
            link(pv, pm); link(pm, mg)
            last, x = mg, 1650
        else:
            rq = mk("op", "rope", 660, -140, op="rope", theta=rbase, dim=rdim)
            rk = mk("op", "rope", 660, 0, op="rope", theta=rbase, dim=rdim)
            kt = mk("op", "k\u1d40", 880, 0, op="transpose")     # q @ k\u1d40: the K side turns
            qk = mk("op", "q\u00b7k\u1d40 scores", 1100, -70, op="matmul")
            sm = mk("act", "softmax", 1320, -70, act="softmax", gain=1.0 / (int(meta.get("head_dim") or 256) ** 0.5), causal=1)
            pv = mk("op", "scores\u00b7v", 1540, -70, op="matmul")
            link(last_q, rq); link(last_k, rk)
            link(rk, kt)
            link(rq, qk, 0, 0); link(kt, qk, 0, 1)
            link(qk, sm); link(sm, pv, 0, 0); link(vsrc, pv, 0, 1)
            last, x = pv, 1760
    if "attn_gate" in t or qg is not None:    # a gated attention output
        if "attn_gate" in t:                  # an explicit gate projection wins
            gi, go = _tensor_lin(_role_dims(t, "attn_gate"))
            gp = mk("layer", "gate proj", 1100, 160, din=gi, dout=go, t=name("attn_gate"))
            sg = mk("act", "sigmoid", 1320, 160, act="sigmoid")
            link(iin, gp)                     # the gate is projected from the attention input
            link(gp, sg)
        else:
            sg = qg                           # the gate rode out of attn_q
        mu = mk("op", "gate", x, -20, op="multiply")
        link(last, mu, 0, 0)
        link(sg, mu, 0, 1)
        last, x = mu, x + 220
    if "attn_output" in t:                    # a fused layer may have none: never invent one
        oi, oo = _tensor_lin(_role_dims(t, "attn_output"))
        op = mk("layer", "out proj", x, -70, din=oi, dout=oo, t=name("attn_output"))
        link(last, op)
        last, x = op, x + 220
    iout = mk("output", "out", x, -70, din=d_model)
    link(last, iout)
    return doc


def _ssm_inner(meta, i, t, d_model):
    """The inside of a state-space box. A gated linear layer keeps its fused
    qkv projection and its gate here too — in those layers the qkv output feeds
    the convolution and the gate shapes the output projection, they are not a
    separate attention."""
    doc, mk, link = _doc_maker("nested")
    eps = meta.get("eps")
    name = lambda k: _role_name(t, k)
    iin = mk("input", "in", 0, 0, dout=d_model)
    x = 220
    if "attn_qkv" in t:
        qi, qo = _tensor_lin(_role_dims(t, "attn_qkv"))
        qkv = mk("layer", "qkv proj", x, 0, din=qi, dout=qo, t=name("attn_qkv"))
        link(iin, qkv)
        prev, x = qkv, x + 220
    else:
        prev = iin
    alpha = beta = None
    if "attn_qkv" in t:                       # beta/alpha project the block input
        if "ssm_alpha" in t:
            ai, ao = _tensor_lin(_role_dims(t, "ssm_alpha"))
            alpha = mk("layer", "alpha proj", x, 160, din=ai, dout=ao, t=name("ssm_alpha"))
            link(iin, alpha)
        if "ssm_beta" in t:
            bi, bo = _tensor_lin(_role_dims(t, "ssm_beta"))
            beta = mk("layer", "beta proj", x, 320, din=bi, dout=bo, t=name("ssm_beta"))
            link(iin, beta)
    cv = mk("op", "conv", x, 0, op="conv1d", act="silu", t=name("ssm_conv1d"))
    link(prev, cv)
    ac = cv                                    # silu rides in the conv node itself
    state = [name(k) for k in ("ssm_a", "ssm_dt") if _role_dims(t, k)]
    scan_dims = {}
    if "attn_qkv" in t and "attn_gate" in t and "ssm_alpha" in t:
        _, heads = _tensor_lin(_role_dims(t, "ssm_alpha"))
        _, gate_dim = _tensor_lin(_role_dims(t, "attn_gate"))
        if heads and gate_dim and gate_dim % heads == 0:
            hd = gate_dim // heads
            key_total = (qo - gate_dim) // 2
            if key_total > 0 and key_total % hd == 0:
                scan_dims = {"heads": heads, "head_dim": hd,
                             "k_heads": key_total // hd, "k_dim": hd}
    sc = mk("op", "scan", x + 440, 0, op="scan", t=", ".join(state) or None,
            v_tiled=bool(meta.get("v_tiled")), **scan_dims)
    link(ac, sc)                              # 0: the convolved qkv
    if beta is not None:
        link(beta, sc, 0, 1)                  # 1: beta logits
    if alpha is not None:
        link(alpha, sc, 0, 2)                 # 2: alpha (dt) logits
    prev, x = sc, x + 660
    if "ssm_norm" in t:
        nm = mk("norm", "ssm norm", x, 0, norm="rms", eps=eps, t=name("ssm_norm"))
        link(prev, nm)
        prev, x = nm, x + 220
    if "attn_gate" in t:                      # the gate on the state-space output
        gi, go = _tensor_lin(_role_dims(t, "attn_gate"))
        gp = mk("layer", "gate proj", x, 160, din=gi, dout=go, t=name("attn_gate"))
        sg = mk("act", "silu", x + 220, 160, act="silu")   # gated norm: x * silu(z)
        src = prev
        if scan_dims and go and go > 0:       # the scan hands back [T, heads, dim]
            rsh = mk("op", "flatten", x + 110, -80, op="reshape",
                     dims=str(int(go)), keep=1)
            link(prev, rsh)
            src = rsh
        mu = mk("op", "gate", x, 0, op="multiply")
        link(iin, gp)
        link(gp, sg)
        link(src, mu, 0, 0)
        link(sg, mu, 0, 1)
        prev, x = mu, x + 440
    if "ssm_out" in t:
        oi, oo = _tensor_lin(_role_dims(t, "ssm_out"))
        op = mk("layer", "out proj", x, 0, din=oi, dout=oo, t=name("ssm_out"))
        link(prev, op)
        prev, x = op, x + 220
    iout = mk("output", "out", x, 0, din=d_model)
    link(prev, iout)
    return doc


def _moe_inner(meta, i, t, d_model):
    """The inside of a MoE box: a router, a top-k pick and an expert path."""
    doc, mk, link = _doc_maker("nested")
    exps = sorted(k for k in t if k.endswith("_exps"))
    ups = [k for k in exps if "up" in k or "gate" in k]
    downs = [k for k in exps if "down" in k]
    ui, uo = _tensor_lin(_role_dims(t, ups[0])) if ups else (None, None)
    di, do = _tensor_lin(_role_dims(t, downs[0])) if downs else (None, None)
    ri, ro = _tensor_lin(_role_dims(t, "ffn_gate_inp"))
    topk = meta.get("topk")
    iin = mk("input", "in", 0, 0, dout=d_model)
    rt = mk("layer", "router", 220, -120, din=ri, dout=ro, t=_role_name(t, "ffn_gate_inp"))
    sm = mk("act", "softmax", 440, -120, act="softmax")   # the router keeps its own softmax
    tk = mk("op", ("top %d" % topk) if topk else "top k", 660, -120, op="topk")
    up = mk("layer", "expert up", 220, 120, din=ui, dout=uo,
            t=", ".join(_role_name(t, k) for k in ups) or None)
    ac = mk("act", "silu", 440, 120, act="silu")
    dn = mk("layer", "expert down", 660, 120, din=di, dout=do,
            t=", ".join(_role_name(t, k) for k in downs) or None)
    mu = mk("op", "weights", 880, 0, op="multiply")
    iout = mk("output", "out", 1100, 0, din=d_model)
    link(iin, rt); link(rt, sm); link(sm, tk)
    link(iin, up); link(up, ac); link(ac, dn)
    link(tk, mu, 0, 0); link(dn, mu, 0, 1)
    link(mu, iout)
    return doc


def _mtp_prelude(doc, mk, link, meta, t, d_model):
    """h (hidden) and e (the next token's embedding) -> hnorm/enorm -> concat
    -> eh_proj, the merged state the next-token block then runs on."""
    eps = meta.get("eps")
    name = lambda k: _role_name(t, k)
    hin = mk("input", "h", 0, 0, dout=d_model)
    ein = mk("input", "e", 0, 160, dout=d_model)
    hn = mk("norm", "hnorm", 240, 0, norm="rms", eps=eps, t=name("nextn.hnorm"))
    en = mk("norm", "enorm", 240, 160, norm="rms", eps=eps, t=name("nextn.enorm"))
    cc = mk("op", "concat", 480, 80, op="concat", din=d_model, dout=2 * d_model)
    ei, eo = _tensor_lin(_role_dims(t, "nextn.eh_proj"))
    ep = mk("layer", "eh proj", 720, 80, din=ei or 2 * d_model, dout=eo or d_model,
            t=name("nextn.eh_proj"))
    link(hin, hn)
    link(ein, en)
    link(hn, cc, 0, 0)
    link(en, cc, 0, 1)
    link(cc, ep)
    return ep


def _block_inner(meta, i, t, d_model, mtp=False):
    """One block's own network: norm -> core -> residual -> FFN -> residual."""
    doc, mk, link = _doc_maker("nested")
    eps = meta.get("eps")
    attn_keys = [k for k in _ATTN_KEYS if k in t]
    head_norms = [k for k in ("attn_q_norm", "attn_k_norm") if k in t]
    ssm_keys = sorted(k for k in t if k.startswith("ssm_"))
    # a gated linear/state-space layer carries fused qkv + gate tensors: they
    # belong to the state-space path, not to a separate attention box
    real_attn = any(k in t for k in ("attn_q", "attn_k", "attn_v", "attn_output"))
    fused_keys = [k for k in ("attn_qkv", "attn_gate") if k in t]
    if fused_keys and not real_attn and ssm_keys:
        attn_keys = [k for k in attn_keys if k not in fused_keys]
        head_norms = []
    exp_keys = sorted(k for k in t if k.endswith("_exps") or k == "ffn_gate_inp")
    dense = all(k in t for k in _FFN_KEYS) and not exp_keys
    has_norm = "attn_norm" in t
    has_post = "post_attention_norm" in t
    has_fnorm = "ffn_norm" in t
    has_postffw = "post_ffw_norm" in t
    has_scale = "layer_output_scale" in t
    used = set(attn_keys + head_norms + ssm_keys + exp_keys + fused_keys +
               [k for k in _NORM_KEYS if k in t] +
               [k for k in t if k.startswith("nextn.")] +
               [k for k in ("layer_output_scale",) if k in t])
    if dense:
        used |= set(_FFN_KEYS)
    rest = sorted(k for k in t if k not in used)

    x = 0
    if mtp:
        ii = _mtp_prelude(doc, mk, link, meta, t, d_model)
        prev, x = ii, 940
    else:
        ii = mk("input", "in", x, 0, dout=d_model)
        prev, x = ii, x + 260
    if has_norm:
        nn = mk("norm", "attn_norm", x, 0, norm="rms", eps=eps, t=_role_name(t, "attn_norm"))
        link(prev, nn)
        prev, x = nn, x + 260
    made_core = False
    if attn_keys:
        an = mk("custom", "attention", x, 0, inner=_attention_inner(meta, i, t, d_model))
        link(prev, an)
        prev, x = an, x + 300
        made_core = True
    if ssm_keys:
        sn = mk("custom", "state space", x, 0, inner=_ssm_inner(meta, i, t, d_model))
        link(prev, sn)
        prev, x = sn, x + 260
        made_core = True
    if exp_keys:
        mn = mk("custom", "mixture of experts", x, 0, inner=_moe_inner(meta, i, t, d_model))
        link(prev, mn)
        prev, x = mn, x + 300
        made_core = True
    if has_post and has_fnorm:                # gemma's sandwich: post-norm the core
        pn = mk("norm", "post_attention_norm", x, 0, norm="rms", eps=eps,
                t=_role_name(t, "post_attention_norm"))
        link(prev, pn)
        prev, x = pn, x + 260
    if has_scale:                             # gemma's per-layer output scalar
        sc = mk("op", "layer scale", x, 0, op="scale", t=_role_name(t, "layer_output_scale"))
        link(prev, sc)
        prev, x = sc, x + 220
    if made_core or has_post or has_norm:
        add = mk("op", "add", x, 0, op="add")
        link(prev, add, 0, 0)
        link(ii, add, 0, 1)
        prev, x = add, x + 260
    base = prev
    pre_norm = "ffn_norm" if has_fnorm else ("post_attention_norm" if has_post else None)
    if pre_norm:                              # the pre-MLP norm (qwen: post_attention_norm)
        fn = mk("norm", pre_norm, x, 0, norm="rms", eps=eps, t=_role_name(t, pre_norm))
        link(base, fn)
        prev, x = fn, x + 260
    if dense:
        gi, go = _tensor_lin(_role_dims(t, "ffn_gate"))
        ui, uo = _tensor_lin(_role_dims(t, "ffn_up"))
        di, do = _tensor_lin(_role_dims(t, "ffn_down"))
        gate = mk("layer", "ffn_gate", x, 0, din=gi, dout=go, t=_role_name(t, "ffn_gate"))
        up = mk("layer", "ffn_up", x, 150, din=ui, dout=uo, t=_role_name(t, "ffn_up"))
        act = mk("act", "ffn_act", x + 260, 0, act="silu")
        mul = mk("op", "mul", x + 520, 0, op="multiply")
        down = mk("layer", "ffn_down", x + 780, 0, din=di, dout=do, t=_role_name(t, "ffn_down"))
        link(prev, gate)
        link(prev, up)
        link(gate, act)
        link(act, mul, 0, 0)
        link(up, mul, 0, 1)
        link(mul, down)
        prev, x = down, x + 1040
    if has_postffw:                           # gemma's post feed-forward norm
        pf = mk("norm", "post_ffw_norm", x, 0, norm="rms", eps=eps, t=_role_name(t, "post_ffw_norm"))
        link(prev, pf)
        prev, x = pf, x + 260
    if has_scale:
        sc2 = mk("op", "layer scale", x, 0, op="scale", t=_role_name(t, "layer_output_scale"))
        link(prev, sc2)
        prev, x = sc2, x + 220
    if prev is not base:
        add2 = mk("op", "add", x, 0, op="add")
        link(prev, add2, 0, 0)
        link(base, add2, 0, 1)
        prev, x = add2, x + 260
    if rest:                                   # tensors we have no role for: shown, bound,
        for j, k in enumerate(rest):           # deliberately loose (not part of the flow)
            dims = _role_dims(t, k)
            size = int(dims[0]) if dims else None
            mk("layer", k, x + 40, 280 + 90 * j,
               din=size, dout=(int(dims[1]) if dims and len(dims) > 1 else size),
               t=_role_name(t, k), loose=True)
    if "nextn.shared_head_norm" in t:
        sh = mk("norm", "shared head norm", x, 0, norm="rms", eps=eps,
                t=_role_name(t, "nextn.shared_head_norm"))
        link(prev, sh)
        prev, x = sh, x + 260
    iout = mk("output", "out", x, 0, din=d_model)
    link(prev, iout)
    return doc


def build_canvas(meta, tensors):
    """The canvas JSON (the format the Training UI loads) for a model.
    Pure: meta is the hyperparameters, tensors is [(name, [dims][, info])]
    where info (optional) carries the tensor's dtype, element count and bytes."""
    meta = meta or {}
    info_by_name = {}
    by_name = {}
    blocks = {}
    for entry in (tensors or []):
        name, dims = entry[0], entry[1]
        if len(entry) > 2 and entry[2]:
            info_by_name[name] = entry[2]
        by_name[name] = dims
        m = _BLK_RE.match(name)
        if m:
            blocks.setdefault(int(m.group(1)), {})[_blk_role(m.group(2))] = {
                "dims": dims, "name": name}

    # the model is one dedicated nested box: everything below builds its
    # *inside*, with `tokens` and `logits` as its own boundary boxes
    inn, mk, link = _doc_maker("nested")
    d_model = meta.get("d_model")
    emb_name = "token_embd.weight" if "token_embd.weight" in by_name else None
    if emb_name:
        vocab, emb_out = _tensor_embed(by_name[emb_name])
    else:
        vocab, emb_out = (d_model, d_model)
    head_name = "output.weight" if "output.weight" in by_name else emb_name
    if head_name:
        head_in, head_out = _tensor_lin(by_name[head_name])   # d_model -> vocab
    else:
        head_in, head_out = (d_model, vocab)

    # the inside is folded into rows: each row reads left to right and the
    # wire to the next row sweeps back through the gap below it, so a 60-block
    # model is a compact block of rows instead of one 12,000-unit column. The
    # free lane on the left carries the next-token head's long wire.
    LANE, COLW, ROWH = 160, 200, 110

    # a token id is a single integer: the table is *indexed* by it (a gather),
    # not multiplied -- so the input carries 1 and the embedding box returns the
    # vector; the LM head, which really is a matmul, stays a fully connected box
    tin = mk("input", "tokens", 0, 0, dout=1 if emb_name else d_model)
    main = [tin]
    prev = tin
    if emb_name:
        emb = mk("embed", "token embedding", 0, 0, vocab=vocab, din=1,
                 dout=emb_out if emb_out is not None else d_model, t=emb_name)
        link(prev, emb)
        prev = emb
        main.append(emb)

    n_blocks = int(meta.get("blocks") or 0)
    if not n_blocks and blocks:
        n_blocks = max(blocks) + 1
    mtp_blocks = [i for i in range(n_blocks)
                  if any(k.startswith("nextn.") for k in blocks.get(i, {}))]
    last_main = None
    for i in range(n_blocks):
        if i in mtp_blocks:
            continue
        doc = _block_inner(meta, i, blocks.get(i, {}), d_model)
        blk = mk("custom", _block_label(i, blocks.get(i, {})), 0, 0, inner=doc)
        link(prev, blk)
        prev = blk
        last_main = blk
        main.append(blk)

    if "output_norm.weight" in by_name:
        on = mk("norm", "output_norm", 0, 0, norm="rms", eps=meta.get("eps"),
                t="output_norm.weight")
        link(prev, on)
        prev = on
        main.append(on)
    if head_name:
        head = mk("layer", "lm head", 0, 0,
                  din=head_in if head_in is not None else d_model,
                  dout=head_out if head_out is not None else vocab, t=head_name)
        link(prev, head)
        prev = head
        main.append(head)
    out = mk("output", "logits", 0, 0, din=head_out if head_out is not None else vocab)
    link(prev, out)
    main.append(out)

    # the next-token head is a branch -- h from the last main block, e from the
    # embedding -- and it gets its own row, so its two long wires ride free space
    branch = []
    mtp = None
    for i in mtp_blocks:
        doc = _block_inner(meta, i, blocks.get(i, {}), d_model, mtp=True)
        mtp = mk("custom", _block_label(i, blocks.get(i, {})), 0, 0, inner=doc)
        link(last_main or prev, mtp, 0, 0)
        link(emb if emb_name else tin, mtp, 0, 1)
        branch.append(mtp)
        mtp_out = mk("output", "mtp hidden", 0, 0, din=d_model)
        link(mtp, mtp_out)
        branch.append(mtp_out)

    # fold: the column count whose block comes closest to the viewport's shape
    def _rows(n, c):
        return (n + c - 1) // c
    best = None
    for c in range(2, max(2, len(main)) + 1):
        rows = _rows(len(main), c) + (1 if branch else 0)
        score = abs((LANE + c * COLW) / float(rows * ROWH) - 1.7)
        if best is None or score < best[0]:
            best = (score, c)
    cols = best[1] if best else 2
    for idx, node in enumerate(main):
        r, c = divmod(idx, cols)
        node["x"] = LANE + c * COLW
        node["y"] = r * ROWH
    end_y = _rows(len(main), cols) * ROWH
    if branch:
        for j, node in enumerate(branch):
            node["x"] = LANE + j * COLW
            node["y"] = end_y
        end_y += ROWH

    # the next-token embedding comes from the top of the model: send it down
    # the lane, so it never crosses a block
    if emb_name and mtp is not None:
        _edge_via(inn, emb["id"], mtp["id"], 0, 1, LANE // 2, ROWH - 20)

    def count(doc):
        n = len(doc["nodes"])
        e = len(doc["edges"])
        for node in doc["nodes"]:
            if node.get("kind") == "custom" and node.get("inner"):
                cn, ce = count(node["inner"])
                n += cn
                e += ce
        return n, e

    # every box that came from a tensor knows its dtype, its element count and
    # its exact bytes — that is what the canvas' VRAM estimate is built on. A
    # tensor is counted once: when a second box reuses one (a tied LM head
    # reusing token_embd, an output scale applied twice), it is marked shared
    # and weighs nothing, so the estimate still equals the file.
    counted = set()

    def first_op(doc, op):
        for n in doc.get("nodes", []):
            if n.get("op") == op:
                return n
            if n.get("kind") == "custom" and n.get("inner"):
                hit = first_op(n["inner"], op)
                if hit:
                    return hit
        return None

    for name in list(info_by_name):
        if "rope_freq" in name:                       # a rope table: put it on the rope op
            rope = first_op({"nodes": inn["nodes"]}, "rope")
            if rope:
                rope["t"] = ((rope.get("t") + ", " + name) if rope.get("t") else name)

    def stamp(doc_node):
        t = doc_node.get("t")
        if t:
            names = [x.strip() for x in t.split(",") if x.strip() in info_by_name]
            if names:
                fresh = [x for x in names if x not in counted]
                infos = [info_by_name[x] for x in fresh]
                doc_node["params"] = sum(i.get("params", 0) for i in infos)
                doc_node["wbytes"] = sum(i.get("bytes", 0) for i in infos)
                kinds = {info_by_name[x].get("dtype", "?") for x in names}
                doc_node["dtype"] = kinds.pop() if len(kinds) == 1 else "mixed"
                if not fresh:
                    doc_node["shared"] = True       # this tensor is counted elsewhere
                counted.update(names)
        if doc_node.get("kind") == "custom" and doc_node.get("inner"):
            for inner in doc_node["inner"]["nodes"]:
                stamp(inner)

    for node in inn["nodes"]:
        stamp(node)

    # whatever the builder had no role for is shown, bound, at the top level —
    # never silently dropped
    bound = set()
    def collect(doc):
        for n in doc.get("nodes", []):
            for t in (n.get("t") or "").split(","):
                if t.strip():
                    bound.add(t.strip())
            if n.get("kind") == "custom" and n.get("inner"):
                collect(n["inner"])
    collect({"nodes": inn["nodes"]})
    stray = [name for name in info_by_name if name not in bound]
    for i, name in enumerate(stray):
        mk("layer", _blk_role(name.split(".", 1)[-1]), LANE, end_y + 60 + 90 * i,
           din=info_by_name[name].get("params"), dout=info_by_name[name].get("params"),
           t=name, loose=True)

    # the canvas: the model as one dedicated nested box, wired to the same
    # tokens and logits it holds inside — so the design is whole at both levels
    top, tmk, tlink = _doc_maker()
    box_name = meta.get("name") or meta.get("basename") or meta.get("arch") or "model"
    inner_ins = sorted((n for n in inn["nodes"] if n["kind"] == "input"),
                       key=lambda n: (n["y"], n["x"]))
    outer_ins = [tmk("input", io["name"], 0, 160 * port, dout=io.get("dout"))
                 for port, io in enumerate(inner_ins)]
    box = tmk("custom", str(box_name), 320, 0, inner=inn)
    for port, io in enumerate(outer_ins):
        tlink(io, box, 0, port)
    inner_outs = sorted((n for n in inn["nodes"] if n["kind"] == "output"),
                        key=lambda n: (n["y"], n["x"]))
    for port, io in enumerate(inner_outs):
        outer_out = tmk("output", io["name"], 640, 160 * port, din=io.get("din"))
        tlink(box, outer_out, port, 0)

    _snap_layout(top)                          # the breadboard: everything on-grid

    total_nodes, total_edges = count({"nodes": top["nodes"], "edges": top["edges"]})
    model_params = sum(i.get("params", 0) for i in info_by_name.values())
    model_bytes = sum(i.get("bytes", 0) for i in info_by_name.values())
    dtypes = {}
    for i in info_by_name.values():
        dtypes[i.get("dtype", "?")] = dtypes.get(i.get("dtype", "?"), 0) + i.get("bytes", 0)
    summary = {"arch": meta.get("arch"), "blocks": n_blocks, "d_model": d_model,
               "tensors": len(tensors or []), "boxes": total_nodes, "wires": total_edges}
    attn_layers = sum(1 for i in range(n_blocks) if _has_attention(blocks.get(i, {})))
    model = {"arch": meta.get("arch"), "layers": n_blocks, "attn_layers": attn_layers,
             "kv_heads": meta.get("kv_heads"), "head_dim": meta.get("head_dim"),
             "ctx": meta.get("ctx"), "params": model_params, "bytes": model_bytes,
             "dtypes": dtypes}
    # the weight registry: every tensor this file owns, namespaced
    # `file::tensor` — the canvas draws its metadata from here and validates
    # its bindings against it; the loader never touches a tensor no box uses
    weights = {}
    if meta.get("file"):
        fkey = str(meta["file"])
        for name, dims, info in (tensors or []):
            weights["%s::%s" % (fkey, name)] = {
                "tensor": name,
                "shape": [int(d) for d in dims],
                "dtype": info.get("dtype"),
                "params": info.get("params"),
                "bytes": info.get("bytes"),
            }
        # bindings travel as namespaces too, so a box and the registry agree
        prefix = fkey + "::"

        def _ns(nodes):
            for n in nodes or []:
                if n.get("t"):
                    n["t"] = ", ".join(
                        prefix + x.strip() for x in str(n["t"]).split(",") if x.strip())
                if isinstance(n.get("inner"), dict):
                    _ns((n.get("inner") or {}).get("nodes"))

        _ns(top["nodes"])
    return {"type": "canvas", "version": 1, "model": model, "weights": weights,
            "nodes": top["nodes"], "edges": top["edges"], "summary": summary}


_VISION_PROJECTORS = ("qwen3vl_merger",)


def _vision_cluster(canvas, mmproj):
    """Add an mmproj's vision tower to a text canvas: an image input, the
    tower box (the engine runs the whole tower) and an embeddings output —
    the merge into the text model's token stream is the next step. The
    tower's tensors join the canvas' weight registry, namespaced by file."""
    path = _resolve_model(mmproj)
    if not path or not os.path.isfile(path):
        raise ValueError("mmproj file not found: %s" % (mmproj,))
    meta, tensors = clip_meta_and_tensors(path)
    proj = str(meta.get("projector") or "")
    if proj not in _VISION_PROJECTORS:
        raise ValueError("%s: the projector type '%s' is not supported yet (%s is)"
                         % (os.path.basename(path), proj or "?",
                            ", ".join(_VISION_PROJECTORS)))
    d_model = int(meta.get("d_model") or 0)
    heads = int(meta.get("heads") or 0)
    blocks = int(meta.get("blocks") or 0)
    patch = int(meta.get("patch_size") or 16)
    size = int(meta.get("image_size") or 0)
    proj_dim = int(meta.get("proj_dim") or 0)
    merge = int(meta.get("merge") or 2)
    if not (d_model and heads and blocks and size and proj_dim):
        raise ValueError("%s: the mmproj's vision hyperparameters are incomplete"
                         % os.path.basename(path))
    if size % patch:
        raise ValueError("%s: image size %d is not a multiple of patch size %d"
                         % (os.path.basename(path), size, patch))
    mean = meta.get("image_mean") or [0.5, 0.5, 0.5]
    std = meta.get("image_std") or [0.5, 0.5, 0.5]
    nodes = canvas.setdefault("nodes", [])
    edges = canvas.setdefault("edges", [])
    y0 = max([int(n.get("y") or 0) for n in nodes] + [0]) + 400
    tower = {"id": "vision", "kind": "vision", "name": "%s vision" % (meta.get("name") or "vision"),
             "x": 320, "y": y0, "file": path, "projector": proj,
             "d_model": d_model, "heads": heads, "blocks": blocks,
             "patch": patch, "merge": merge, "image_size": size,
             "proj_dim": proj_dim, "eps": meta.get("eps") or 1e-6}
    image = {"id": "image", "kind": "input", "name": "image", "input_mode": "image",
             "x": 0, "y": y0, "dout": 3 * patch * patch,
             "image_size": size, "patch_size": patch,
             "image_mean": " ".join("%g" % v for v in mean),
             "image_std": " ".join("%g" % v for v in std)}
    sink = {"id": "image embeds", "kind": "output", "name": "image embeds",
            "x": 640, "y": y0, "din": proj_dim}
    nodes += [image, tower, sink]
    edges += [{"from": "image", "to": "vision", "fromPort": 0, "toPort": 0},
              {"from": "vision", "to": "image embeds", "fromPort": 0, "toPort": 0}]
    # the tower's own weights: stamped on the box (the estimate reads them)
    # and registered by namespace, so the app can show what the file holds
    tower["params"] = sum(i.get("params", 0) for _, _, i in tensors)
    tower["wbytes"] = sum(i.get("bytes", 0) for _, _, i in tensors)
    kinds = {i.get("dtype") for _, _, i in tensors}
    tower["dtype"] = kinds.pop() if len(kinds) == 1 else "mixed"
    weights = canvas.setdefault("weights", {})
    for tname, dims, info in tensors:
        weights["%s::%s" % (path, tname)] = {
            "tensor": tname, "shape": [int(d) for d in dims],
            "dtype": info.get("dtype"), "params": info.get("params"),
            "bytes": info.get("bytes")}
    canvas.setdefault("model", {})["vision"] = {
        "file": path, "projector": proj, "name": meta.get("name"),
        "image_size": size, "patch_size": patch, "projection_dim": proj_dim,
        "blocks": blocks}
    _snap_layout(canvas)
    return canvas


def gguf_canvas(model, mmproj=None):
    """Build a canvas design from a GGUF file (header + tensor list only).
    With an `mmproj`, the file's vision tower rides along: an image input,
    the tower box and an embeddings output beside the text model."""
    path = _resolve_model(model)
    if not path or not os.path.isfile(path):
        raise ValueError("GGUF file not found: %s" % (model,))
    if not is_gguf(path):
        raise ValueError("not a GGUF model: %s" % (os.path.basename(path),))
    meta, tensors = gguf_meta_and_tensors(path)
    meta["v_tiled"] = True        # llama.cpp's converter stores V heads tiled
    meta["file"] = path           # the namespace every binding reads from
    canvas = build_canvas(meta, tensors)
    canvas["name"] = os.path.basename(path)
    if mmproj:
        _vision_cluster(canvas, mmproj)
    return {"ok": True, "canvas": canvas, "summary": canvas.pop("summary")}


# ---------------------------------------------------------------- HF -> canvas
# A huggingface repo's transformer, taken apart into the same native blocks a
# GGUF is: `config.json` says how big each piece is, and the safetensors
# *headers* (eight bytes, then JSON) say which tensor belongs where and in
# which shard. No weights are downloaded here — every box carries
# `file: "hf:<repo>/<shard>"` and the backend fetches a shard the first time a
# box bound to it runs. Local folders with the same layout work too.
_HF_LAYER_ROLES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_HF_TOP_ROLES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
_HF_DTYPE_NAMES = {"F64": "f64", "F32": "f32", "F16": "f16", "BF16": "bf16",
                   "F8_E4M3": "f8_e4m3", "F8_E5M2": "f8_e5m2",
                   "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8",
                   "U8": "u8", "BOOL": "bool"}


def _is_packed(name, ent):
    """A 4-bit tensor stored as bytes: `*_packed` U8 holds two weights per
    stored element (the compressed-tensors convention)."""
    return (str(name).endswith("_packed")
            and str(ent.get("dtype") or "").upper() == "U8")


def _hf_translate(name):
    """The role name the shared builder knows (a bias keeps the same role with
    a `.bias` tail), or None when the tensor has no role."""
    tail, base = "", name
    if name.endswith(".bias"):
        base, tail = name[:-len(".bias")] + ".weight", ".bias"
    hit = _HF_TOP_ROLES.get(base)
    if hit is None:
        m = re.match(r"^model\.layers\.(\d+)\.(.+)$", base)
        if m:
            role = _HF_LAYER_ROLES.get(m.group(2))
            if role:
                hit = "blk.%s.%s" % (m.group(1), role)
    if hit is not None and tail:
        hit = (hit[:-len(".weight")] if hit.endswith(".weight") else hit) + tail
    return hit


def _safetensors_entries(header):
    out = {}
    for name, ent in (header or {}).items():
        if name == "__metadata__" or not isinstance(ent, dict):
            continue
        off = ent.get("data_offsets") or [0, 0]
        out[name] = {"dtype": str(ent.get("dtype") or "").upper(),
                     "dims": [int(x) for x in (ent.get("shape") or [])],
                     "bytes": int(off[1]) - int(off[0])}
    return out


def _safetensors_header(path):
    """{name: {dtype, dims, bytes}} from a local file — header only."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return {}
        n = struct.unpack("<Q", raw)[0]
        try:
            return _safetensors_entries(json.loads(f.read(n).decode("utf-8")))
        except Exception:  # noqa: BLE001 — a bad header is an empty one
            return {}


def _hf_fetch(repo, filename, start=None, end=None, timeout=60):
    """Bytes of one repo file (optionally a byte range), HF_TOKEN if set."""
    req = urllib.request.Request("https://huggingface.co/%s/resolve/main/%s"
                                 % (repo.strip("/"), filename))
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if start is not None:
        req.add_header("Range", "bytes=%d-%d" % (start, end if end is not None else ""))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _hf_repo_shards(repo, sub="", timeout=60):
    """The repo's safetensors shard names (relative to `sub`): the index,
    then the api list, then the one default file."""
    prefix = (sub.strip("/") + "/") if sub else ""
    try:
        doc = json.loads(_hf_fetch(repo, prefix + "model.safetensors.index.json",
                                   timeout=timeout))
        files = sorted({str(v).split("/")[-1] for v in (doc.get("weight_map") or {}).values()})
        if files:
            return files
    except Exception:  # noqa: BLE001 — no index: one file, or list the repo
        pass
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/%s" % repo.strip("/"))
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8"))
        files = sorted(str(s.get("rfilename") or "") for s in (doc.get("siblings") or []))
        if prefix:
            files = [f[len(prefix):] for f in files if f.startswith(prefix)]
        files = [f for f in files if f.endswith(".safetensors") and "/" not in f]
        if files:
            return files
    except Exception:  # noqa: BLE001
        pass
    return ["model.safetensors"]


def _hf_shard_header(repo, filename, timeout=60):
    """One remote shard's safetensors header, read by range — no weights."""
    head = _hf_fetch(repo, filename, 0, 7, timeout=timeout)
    if len(head) != 8:
        raise ValueError("not a safetensors shard: %s" % filename)
    n = struct.unpack("<Q", head)[0]
    raw = _hf_fetch(repo, filename, 8, 8 + n - 1, timeout=timeout)
    return _safetensors_entries(json.loads(raw.decode("utf-8")))


def _hf_num(cfg, *keys):
    for k in keys:
        v = cfg.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def hf_meta_and_tensors(source, timeout=60):
    """(meta, tensors, extra) for a repo id/url — or a local folder with
    `config.json` and `*.safetensors`. `tensors` is in the shared builder's
    shape (GGUF ne-order, role names); `extra` carries the original names,
    the shard each tensor lives in, and the biases."""
    source = (source or "").strip()
    if source.startswith("http"):
        source = source.rstrip("/")
        if "huggingface.co/" in source:
            source = source.split("huggingface.co/", 1)[1]
        if source.endswith("/tree/main"):
            source = source[:-len("/tree/main")]
    sub = ""
    if not os.path.isdir(source) and source.count("/") >= 2:
        head, rest = source.split("/", 1)                 # owner/repo/sub/dir...
        repo, _, sub = rest.partition("/")
        source, sub = head + "/" + repo, sub.strip("/")
    if os.path.isdir(source):
        with open(os.path.join(source, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        shards = sorted(n for n in os.listdir(source) if n.endswith(".safetensors"))
        header = lambda n: _safetensors_header(os.path.join(source, n))   # noqa: E731
        file_of = lambda n: os.path.join(source, n)                       # noqa: E731
        name = os.path.basename(os.path.abspath(source))
    else:
        prefix = (sub + "/") if sub else ""
        cfg = json.loads(_hf_fetch(source, prefix + "config.json", timeout=timeout))
        shards = _hf_repo_shards(source, sub=sub, timeout=timeout)
        header = lambda n: _hf_shard_header(source, prefix + n, timeout=timeout)   # noqa: E731
        file_of = lambda n: "hf:%s/%s" % (source.strip("/"), prefix + n)           # noqa: E731
        name = sub.split("/")[-1] if sub else source.strip("/").split("/")[-1]

    weights, biases = {}, {}                    # role name -> (orig, ent, file)
    for shard in shards:
        for orig, ent in header(shard).items():
            tn = _hf_translate(orig) or orig
            if tn.endswith(".bias"):
                biases[tn[:-len(".bias")] + ".weight"] = (orig, ent, file_of(shard))
            else:
                weights[tn] = (orig, ent, file_of(shard))

    tensors, info, back, by_orig = [], {}, {}, {}
    for tn, (orig, ent, file) in weights.items():
        dims = list(reversed(ent["dims"])) if ent["dims"] else []   # torch -> gguf ne
        params = 1
        for d in dims:
            params *= d
        if _is_packed(orig, ent):
            params *= 2                          # two 4-bit weights per stored byte
        info[tn] = {"params": params, "bytes": ent["bytes"],
                    "dtype": _HF_DTYPE_NAMES.get(ent["dtype"], "?"),
                    "orig": orig, "file": file}
        back[tn] = orig
        tensors.append((tn, dims, info[tn]))
        by_orig[orig] = {"dims": [int(x) for x in (ent["dims"] or [])],
                         "dtype": _HF_DTYPE_NAMES.get(ent["dtype"], "?"),
                         "bytes": ent["bytes"], "params": params, "file": file}
    for _tn, (orig, ent, file) in biases.items():
        params = 1
        for d in ent["dims"]:
            params *= d
        by_orig[orig] = {"dims": [int(x) for x in ent["dims"]],
                         "dtype": _HF_DTYPE_NAMES.get(ent["dtype"], "?"),
                         "bytes": ent["bytes"], "params": params, "file": file}

    d_model = int(_hf_num(cfg, "hidden_size", "n_embd") or 0)
    heads = int(_hf_num(cfg, "num_attention_heads", "n_head") or 0)
    meta = {
        "arch": str(cfg.get("model_type") or "hf"),
        "name": name,
        "blocks": int(_hf_num(cfg, "num_hidden_layers", "n_layer") or 0),
        "d_model": d_model,
        "heads": heads,
        "kv_heads": int(_hf_num(cfg, "num_key_value_heads") or heads),
        "head_dim": int(_hf_num(cfg, "head_dim") or (d_model // heads if heads else 0)),
        # v5 nests the theta under rope_parameters; older configs keep it flat
        "rope": _hf_num(cfg, "rope_theta")
        or (cfg.get("rope_parameters") or {}).get("rope_theta") or 10000.0,
        "ctx": int(_hf_num(cfg, "max_position_embeddings", "n_positions", "n_ctx") or 0),
        "eps": _hf_num(cfg, "rms_norm_eps", "layer_norm_eps") or 1e-5,
        "experts": int(_hf_num(cfg, "num_local_experts", "num_experts") or 0) or None,
        "topk": int(_hf_num(cfg, "num_experts_per_tok") or 0) or None,
    }
    return meta, tensors, {"info": info, "back": back, "biases": biases,
                           "cfg": cfg, "by_orig": by_orig}


# ---------------------------------------------------------------- family specs
# A spec is the *verified* wiring of one architecture family: drafted from the
# installed transformers source by the reverse-model agent (see
# tools/reverse_model.py and .opencode/commands/reverse-model.md) and gated by
# tests/spec_conformance_test.py, which compares stage by stage against the
# library's own eager run. `spec_canvas` turns a spec into the same nested,
# native-boxes-only canvas the GGUF builder makes — every leaf a real box,
# every weight bound to its repo tensor and shard.
SPECS_DIR = os.path.join(ENGINE_DIR, "specs")
_SPEC_CACHE = None


def load_specs():
    """Every family spec, keyed by family — read once per process."""
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        out = {}
        try:
            names = sorted(os.listdir(SPECS_DIR))
        except OSError:
            names = []
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(SPECS_DIR, name), encoding="utf-8") as f:
                    spec = json.load(f)
                out[str(spec.get("family") or name[:-5])] = spec
            except Exception:  # noqa: BLE001 — a broken spec is no spec
                continue
        _SPEC_CACHE = out
    return _SPEC_CACHE


def spec_for(model_type):
    """The spec that claims this model_type, if one exists."""
    for spec in load_specs().values():
        if model_type in (spec.get("match", {}).get("model_type") or []):
            return spec
    return None


class SpecMismatch(Exception):
    """The repo's config/tensors do not fill this spec — fall back to the
    approximate skeleton (and say so)."""


def _spec_expr(value, env, what="value"):
    """A spec value: numbers pass through, arithmetic over config names is
    evaluated, anything else stays a string (an activation, a mode).

    Quoted literals inside the expression are masked to placeholder
    identifiers so the word-subset test can see the real names, then the
    placeholders are bound back to the literals in the eval env:
    ``layer_types[i] == 'full_attention'`` stays per-layer dynamic.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in env:
        return env[text]
    masked, bound = text, {}
    for n, lit in enumerate(re.findall(r"'[^']*'|\"[^\"]*\"", text)):
        masked = masked.replace(lit, "__lit%d__" % n, 1)
        bound["__lit%d__" % n] = lit[1:-1]
    names = set(re.findall(r"[A-Za-z_]\w*", masked))
    if names and names <= set(env) | set(bound):
        try:
            return eval(compile(masked, "<spec>", "eval"), {"__builtins__": {}},
                        {**env, **bound})
        except Exception as e:  # noqa: BLE001
            raise SpecMismatch("%s: %s" % (what, e))
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+(\.\d*)?([eE]-?\d+)?", text):
        return float(text)
    return value


def _spec_value(value, env):
    if isinstance(value, list):
        return [_spec_value(v, env) for v in value]
    return _spec_expr(value, env)


def _cfg_get(cfg, key):
    """A config.json value by key — a dotted key (`rope_parameters.rope_theta`)
    walks nested dicts, the way v5 configs nest their rope settings."""
    node = cfg
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _spec_config(spec, cfg):
    """Resolve a spec's config slots from the repo's config.json.

    Multimodal wrappers (``base_config_key``) nest the text model under
    ``text_config`` — descend into it so the same slots fill either shape.
    """
    env = {}
    slots = spec.get("config") or {}
    for name, slot in slots.items():
        v = None
        for k in (slot.get("keys") or []):
            v = _cfg_get(cfg, k)
            if v is None and isinstance(cfg.get("text_config"), dict):
                v = _cfg_get(cfg["text_config"], k)
            if v is not None:
                break
        if v is None and "from" in slot and slot["from"] in env:
            v = env[slot["from"]]
        if v is None and "default" in slot:
            v = slot["default"]
        if v is not None:
            env[name] = v
    for name, slot in slots.items():
        if name not in env and slot.get("expr"):
            env[name] = _spec_expr(slot["expr"], env, "config %s" % name)
    if "layer_prefix" not in env:
        # multimodal wrappers live under language_model (and norm/
        # embed_tokens); plain text models straight under model.
        env["layer_prefix"] = ("model.language_model.layers.%d."
                                if isinstance(cfg.get("text_config"), dict)
                                else "model.layers.%d.")
    for name in ("d_model", "blocks", "heads", "kv_heads", "head_dim"):
        if not env.get(name):
            raise SpecMismatch("config is missing %s" % name)
    env["d_model"] = int(env["d_model"])
    env["blocks"] = int(env["blocks"])
    env["heads"] = int(env["heads"])
    env["kv_heads"] = int(env["kv_heads"] or env["heads"])
    env["head_dim"] = int(env["head_dim"] or (env["d_model"] // env["heads"]))
    env["repeat"] = max(1, env["heads"] // max(1, env["kv_heads"]))
    return env


_SPEC_SIZE_KEYS = ("din", "dout", "vocab", "size")


def _spec_attrs(node, env):
    """Every attr of a spec node, resolved."""
    out = {}
    for key, val in node.items():
        if key in ("id", "kind", "name", "pos", "if", "param", "params", "optional"):
            continue
        if key == "op":                          # an op *name* is never an expression
            out[key] = val
            continue
        if key in _SPEC_SIZE_KEYS:
            out[key] = int(_spec_expr(val, env, "%s.%s" % (node["id"], key)))
        else:
            out[key] = _spec_value(val, env)
        if key in ("repeats", "dim", "window") and out[key] is not None:
            try:
                out[key] = int(out[key])
            except (TypeError, ValueError):
                pass
    return out


def _spec_inner(spec, env, by_orig, layer):
    """One block's nested design: the spec's nodes and wires, binding and
    elision applied. Every leaf box is one of the app's own.

    `i` is this layer's index in the env: a branch's `if` may test it
    against the config's `layer_types`, so a hybrid model keeps the branch
    that is alive in *this* layer and elides the other. Elision has two
    faces: a single missing optional node is a wire (what fed it feeds
    what it fed); an elided *branch* is not — the live consumer it would
    have fed (the residual add) takes a real zero from the block's `zero`
    source instead, so the sum keeps exactly what the live branch adds.
    """
    ienv = dict(env)
    if layer is not None:
        ienv["i"] = layer
    prefix = (env.get("layer_prefix") or "model.layers.%d.") if layer is not None else ""
    doc, mk, _link = _doc_maker("nested")
    ref, elided, elided_if = {}, set(), set()
    for node in (spec.get("block", {}).get("nodes") or []):
        if node.get("if") is not None and not _spec_expr(node["if"], ienv, node["id"]):
            elided_if.add(node["id"])
            elided.add(node["id"])
            continue
        param = node.get("param")
        # `param` may list candidates — a quantized repo names the same weight
        # `<x>.weight_packed` on some layers and `<x>.weight` on others; the
        # first candidate the repo actually holds is the one this box binds.
        cands = param if isinstance(param, list) else ([param] if param else [])
        name = None
        for p in cands:
            cand = ((prefix % layer if "%d" in prefix else prefix) + p
                    if (p and layer is not None) else p)
            if cand and cand in by_orig:
                name = cand
                break
        names = None
        if cands and name is None:
            if node.get("optional"):
                elided.add(node["id"])
                continue
            raise SpecMismatch("the repo has no tensor %s" % (cands,))
        if name:
            names = [name]
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            if base + ".bias" in by_orig:
                names.append(base + ".bias")
        # `params` binds several tensors at once: the first is the box's own
        # weight, the rest ride along (the gated delta rule's A_log + dt_bias)
        if node.get("params"):
            names = []
            for p in node["params"]:
                full = ((prefix % layer if "%d" in prefix else prefix) + p
                        if layer is not None else p)
                if full not in by_orig:
                    raise SpecMismatch("the repo has no tensor %s" % full)
                names.append(full)
            name = names[0]
        attrs = _spec_attrs(node, ienv)
        if names:
            attrs["t"] = ", ".join(names)
            attrs["file"] = by_orig[name]["file"]
            attrs["dtype"] = by_orig[name]["dtype"]
            # the block's *logical* weights, not the storage element count: a
            # packed 4-bit tensor holds two weights per byte, so its numel is
            # half the block's real params. The stored bytes stay exact.
            logical = None
            kind = node.get("kind") or "op"
            if kind == "layer":
                din, dout = attrs.get("din"), attrs.get("dout")
                if isinstance(din, (int, float)) and isinstance(dout, (int, float)):
                    logical = int(din) * int(dout)
            elif kind == "embed":
                vocab, dout = attrs.get("vocab"), attrs.get("dout")
                if isinstance(vocab, (int, float)) and isinstance(dout, (int, float)):
                    logical = int(vocab) * int(dout)
            elif kind == "norm" and isinstance(attrs.get("size"), (int, float)):
                logical = int(attrs["size"])
            if logical is None:
                logical = sum(by_orig[x]["params"] for x in names)
            elif len(names) > 1:                # a bias stores its own weights
                logical += sum(by_orig[x]["params"] for x in names[1:])
            attrs["params"] = logical
            attrs["wbytes"] = sum(by_orig[x]["bytes"] for x in names)
        pos = node.get("pos") or [0, 0]
        ref[node["id"]] = mk(node.get("kind") or "op", node.get("name") or node["id"],
                             pos[0], pos[1], **attrs)
    raw = [list(e) + [0] * (4 - len(e)) for e in (spec.get("block", {}).get("edges") or [])]
    zero_id = next((n["id"] for n in (spec.get("block", {}).get("nodes") or [])
                    if n.get("kind") == "op" and n.get("op") == "zeros"), None)
    if elided_if:
        # an elided branch is not a wire: every live consumer it fed takes
        # an actual zero from the block's zero source
        for a, b, fp, tp in [e for e in raw if e[0] in elided_if and e[1] in ref]:
            if zero_id is None or zero_id not in ref:
                raise SpecMismatch("an elided branch feeds %s but the block "
                                    "declares no zero source" % b)
            raw.append([zero_id, b, 0, tp])
    elif zero_id is not None:
        elided.add(zero_id)                     # nothing elided: the source elides too
    # a single missing optional node is a wire: whatever fed it feeds whatever it fed
    changed = True
    while changed:
        changed = False
        for nid in list(elided - elided_if):
            ins = [e for e in raw if e[1] == nid]
            outs = [e for e in raw if e[0] == nid]
            if not ins or not outs:
                continue
            raw = [e for e in raw if e[0] != nid and e[1] != nid]
            for i in ins:
                for o in outs:
                    raw.append([i[0], o[1], i[2], o[3]])
            elided.discard(nid)
            changed = True
    if elided:
        raw = [e for e in raw if e[0] not in elided and e[1] not in elided]
    # a box the wiring never reads is not drawn: eliding a branch upstream can
    # strand one (in a linear layer, the attention internals it belonged to
    # are gone, so its zero-fed repeats would hang). Sweep to a fixed point.
    changed = True
    while changed:
        changed = False
        fed = {e[0] for e in raw}
        for nid in [n for n in ref if n not in fed
                    and ref[n].get("kind") not in ("input", "output")]:
            node = ref.pop(nid)
            doc["nodes"] = [x for x in doc["nodes"] if x is not node]
            raw = [e for e in raw if e[0] != nid and e[1] != nid]
            changed = True
    for a, b, fp, tp in raw:
        if a not in ref or b not in ref:
            raise SpecMismatch("an edge points outside the block: %s -> %s" % (a, b))
        doc["edges"].append({"from": ref[a]["id"], "to": ref[b]["id"],
                             "fromPort": int(fp), "toPort": int(tp)})
    return doc


def spec_canvas(spec, cfg, by_orig, name):
    """The whole model as native boxes: tokens -> embedding -> a nested block
    per layer -> final norm -> head -> logits, every weight bound to its repo
    tensor (and shard), the wiring exactly what the spec verified."""
    env = _spec_config(spec, cfg)
    weights = spec.get("weights") or {}

    def want(field, required=True):
        # a spec may list candidates: the multimodal wrappers nest the text
        # model under language_model, a plain one sits straight under model
        p = weights.get(field)
        cands = p if isinstance(p, (list, tuple)) else [p]
        for c in cands:
            if c and c in by_orig:
                return c
        if required:
            raise SpecMismatch("the repo has no %s tensor (%s)" % (field, p))
        return None

    embed = want("embed")
    final = want("final_norm")
    head = want("head", required=False) or embed        # tied embeddings share the table

    def bound(field, p):
        return {field: by_orig[p]["file"], "dtype": by_orig[p]["dtype"],
                "params": by_orig[p]["params"], "wbytes": by_orig[p]["bytes"]}

    inn, mk, link = _doc_maker("nested")
    tin = mk("input", "tokens", 0, 0, dout=1)
    emb = mk("embed", "token embedding", 0, 0, vocab=by_orig[embed]["dims"][0],
             din=1, dout=by_orig[embed]["dims"][-1], t=embed, **bound("file", embed))
    if spec.get("embed_scale") is not None:
        # a scaled word embedding (Gemma): the table times an expression over
        # the config slots, e.g. sqrt(d_model) — applied on the embed box itself
        emb["embed_scale"] = _spec_expr(spec["embed_scale"], env, "embed_scale")
    link(tin, emb)
    prev, main = emb, [tin, emb]
    for i in range(env["blocks"]):
        label = (spec.get("block", {}).get("label") or "blk {i}").replace("{i}", str(i))
        blk = mk("custom", label, 0, 0, inner=_spec_inner(spec, env, by_orig, i))
        link(prev, blk)
        prev = blk
        main.append(blk)
    on = mk("norm", "output_norm", 0, 0, norm="rms", eps=env["eps"],
            size=by_orig[final]["dims"][-1], t=final, **bound("file", final))
    if spec.get("final_norm_zc"):
        on["zc"] = True                          # this family's final norm is (1 + w)
    link(prev, on)
    prev = on
    main.append(on)
    hdims = by_orig[head]["dims"]                           # torch order [out, in]
    hm = mk("layer", "lm head", 0, 0, din=hdims[-1], dout=hdims[0], t=head,
            **bound("file", head))
    link(prev, hm)
    prev = hm
    main.append(hm)
    out = mk("output", "logits", 0, 0, din=hdims[0])
    link(prev, out)
    main.append(out)

    # fold the chain into rows so a 60-block model stays navigable
    COLS, DX, DY = 6, 320, 460
    for idx, node in enumerate(main):
        node["x"], node["y"] = (idx % COLS) * DX, (idx // COLS) * DY
    _snap_layout(inn)

    top, tmk, tlink = _doc_maker("canvas")
    t_in = tmk("input", "tokens", 0, 0, dout=1)
    box = tmk("custom", str(name or spec.get("family") or "model"), 400, 0, inner=inn)
    t_out = tmk("output", "logits", 800, 0, din=hdims[0])
    tlink(t_in, box)
    tlink(box, t_out)
    _snap_layout(top)

    boxes, wires = canvas_counts(top)
    bound = set()

    def collect(doc):
        for x in doc.get("nodes", []):
            for tn in (x.get("t") or "").split(","):
                if tn.strip():
                    bound.add(tn.strip())
            if x.get("kind") == "custom" and x.get("inner"):
                collect(x["inner"])

    collect(top)
    unbound = sorted(set(by_orig) - bound)
    params = sum(v["params"] for v in by_orig.values())
    bytes_ = sum(v["bytes"] for v in by_orig.values())
    dtypes = {}
    for v in by_orig.values():
        dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + v["bytes"]
    summary = {"arch": cfg.get("model_type"), "blocks": env["blocks"], "d_model": env["d_model"],
               "tensors": len(by_orig), "boxes": boxes, "wires": wires,
               "spec": spec.get("family"), "fidelity": "exact",
               "unbound": len(unbound)}
    model = {"arch": cfg.get("model_type"), "layers": env["blocks"],
             "attn_layers": env["blocks"], "kv_heads": env["kv_heads"],
             "head_dim": env["head_dim"],
             "ctx": int(_hf_num(cfg, "max_position_embeddings", "n_positions", "n_ctx") or 0),
             "params": params, "bytes": bytes_, "dtypes": dtypes,
             "d_model": env["d_model"], "tensors": len(by_orig),
             "spec": spec.get("family"), "fidelity": "exact",
             "unbound": unbound[:40], "unbound_total": len(unbound)}
    return {"type": "canvas", "version": 1, "model": model,
            "nodes": top["nodes"], "edges": top["edges"], "summary": summary}


def _arc_summary(canvas):
    """The summary line for a canvas read back from the hf_arc database."""
    m = canvas.get("model") or {}
    n, e = canvas_counts(canvas)
    unbound = m.get("unbound_total")
    if unbound is None:
        unbound = len(m.get("unbound") or [])
    return {"arch": m.get("arch"), "blocks": m.get("layers"),
            "d_model": m.get("d_model"), "tensors": m.get("tensors"),
            "boxes": n, "wires": e, "spec": m.get("spec"),
            "fidelity": m.get("fidelity") or "exact", "arc": True,
            "unbound": unbound}


class ArcError(Exception):
    """This repo cannot be assembled into the exact, native-boxes canvas yet
    (no spec, a spec that does not fit, or a model that is only approximate)."""


def _tokenizer_cached(repo):
    """Is the repo's tokenizer.json at hand — the cache the text boxes read?"""
    cache = os.environ.get("NNSTUDIO_TOKENIZER_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "nnstudio", "tokenizers")
    name = str(repo or "").strip().replace("/", "__")
    path = os.path.join(cache, name, "tokenizer.json")
    return bool(name) and os.path.isfile(path) and os.path.getsize(path) > 0


def _wrap_text_pipeline(canvas, repo):
    """A causal-lm design read back from hf_arc, served text in / text out:
    the plain tokens input becomes a tokenizer (its own text is the prompt)
    and a de-tokenizer rides beside the model's ids. Only when the shape is
    the plain tokens -> model -> logits chain and the repo's tokenizer is at
    hand; the stored row itself never changes."""
    if canvas.get("model", {}).get("text"):
        return canvas
    if not _tokenizer_cached(repo):
        return canvas
    nodes = canvas.get("nodes") or []
    ins = [n for n in nodes if n.get("kind") == "input"]
    mdls = [n for n in nodes if n.get("kind") == "custom" and n.get("inner")]
    outs = [n for n in nodes if n.get("kind") == "output"]
    if not (len(ins) == 1 and len(mdls) == 1 and len(outs) == 1):
        return canvas
    if ins[0].get("dout") not in (1, None):
        return canvas
    if any(n.get("kind") in ("tokenize", "detokenize", "hf") for n in nodes):
        return canvas
    edges = canvas.get("edges") or []
    chain = {e.get("from"): e.get("to") for e in edges}
    if (chain.get(ins[0]["id"]) != mdls[0]["id"]
            or chain.get(mdls[0]["id"]) != outs[0]["id"]):
        return canvas
    tok = ins[0]
    tok.update({"kind": "tokenize", "name": "tokenize", "repo": repo, "text": ""})
    for k in ("input_mode", "input_rows", "input_code"):
        tok.pop(k, None)
    taken = {n.get("id") for n in nodes}
    detok_id, k = "detok", 2
    while detok_id in taken:
        detok_id = "detok %d" % k
        k += 1
    nodes.append({"id": detok_id, "kind": "detokenize", "name": "text",
                  "x": outs[0].get("x", 800), "y": (outs[0].get("y") or 0) + 240,
                  "repo": repo})
    edges.append({"from": tok["id"], "to": detok_id, "fromPort": 0, "toPort": 0})
    canvas.setdefault("model", {})["text"] = True
    return canvas


def hf_canvas(source, timeout=60, use_arc=True):
    """Build a canvas design from a huggingface repo / Hub subfolder (or a
    local folder with the same layout) — config and safetensors headers only,
    no weights.

    The **hf_arc database** wins when this repo has been researched: its
    stored canvas comes back as-is, marked `arc: true`. Otherwise a **family
    spec** (specs/*.json) builds the verified, native-boxes-only design; a
    family without one gets the role-based skeleton, marked approximate."""
    source = nn_arc.canonical(source)
    if use_arc:
        cached = nn_arc.get(source)
        if cached is not None:
            model = cached.setdefault("model", {})
            model["arc"] = True
            model.setdefault("fidelity", "exact")
            cached["name"] = source
            _wrap_text_pipeline(cached, source)     # text models: prompt in, text out
            return {"ok": True, "canvas": cached, "arc": True,
                    "summary": _arc_summary(cached)}
    meta, tensors, extra = hf_meta_and_tensors(source, timeout=timeout)
    spec = spec_for(meta.get("arch"))
    if spec is not None:
        try:
            canvas = spec_canvas(spec, extra["cfg"], extra["by_orig"], meta["name"])
            canvas["name"] = meta["name"]
            canvas.setdefault("model", {})["arc"] = False
            return {"ok": True, "canvas": canvas, "summary": canvas.pop("summary")}
        except SpecMismatch as e:
            meta["spec_note"] = "the %s spec did not fit: %s" % (spec.get("family"), e)
    canvas = build_canvas(meta, tensors)
    info, back, biases = extra["info"], extra["back"], extra["biases"]

    def fix(doc):
        for n in doc.get("nodes", []):
            t = n.get("t")
            if t:
                names = []
                for tn in [x.strip() for x in t.split(",") if x.strip()]:
                    rec = info.get(tn)
                    names.append(rec["orig"] if rec else back.get(tn, tn))
                    if rec:
                        if rec.get("file"):
                            n["file"] = rec["file"]      # the shard this box reads
                    b = biases.get(tn)
                    if b:                                # a weight's bias rides with it
                        b_orig, b_ent, _f = b
                        names.append(b_orig)
                        bp = 1
                        for d in b_ent["dims"]:
                            bp *= d
                        n["params"] = (n.get("params") or 0) + bp
                        n["wbytes"] = (n.get("wbytes") or 0) + b_ent["bytes"]
                n["t"] = ", ".join(names)
            if n.get("kind") == "custom" and n.get("inner"):
                fix(n["inner"])

    fix(canvas)
    canvas.setdefault("model", {})["fidelity"] = "approximate"
    canvas["model"]["arc"] = False
    if meta.get("spec_note"):
        canvas["model"]["spec_note"] = meta["spec_note"]
    canvas["name"] = meta["name"]
    return {"ok": True, "canvas": canvas, "summary": canvas.pop("summary")}


def _canvas_model_box(canvas):
    """The nested box that holds one repo/model's insides."""
    for n in canvas.get("nodes") or []:
        if n.get("kind") == "custom" and n.get("inner"):
            return n
    return None


def arc_canvas(source, pipeline=None, timeout=60, use_arc=False):
    """A whole repo as one canvas, models nested and wired — the thing hf_arc
    stores. `pipeline` names the models and their edges:

        {"models": [{"name": "vae", "source": "org/repo/vae"}, ...],
         "edges":  [{"from": "text", "to": "dit", "fromPort": 0, "toPort": 0}]}

    Without one the repo is a single model and its canvas comes back as-is.
    Every model must be exact (spec-built, bound to the repo's real tensors);
    otherwise this raises ArcError. A model's outer input/output ports are the
    inner canvas's own input/output boxes, so an edge between models is a real
    wire from one nested box to the next."""
    items = []
    if isinstance(pipeline, dict) and pipeline.get("models"):
        for m in pipeline["models"]:
            if not isinstance(m, dict):
                continue
            src = nn_arc.canonical(str(m.get("source") or source).strip())
            if src:
                items.append({"name": str(m.get("name") or src.split("/")[-1]),
                              "source": src})
    if not items:
        items = [{"name": None, "source": nn_arc.canonical(source)}]

    built = []
    for m in items:
        out = hf_canvas(m["source"], timeout=timeout, use_arc=use_arc)
        c = out.get("canvas") or {}
        cm = c.get("model") or {}
        if cm.get("fidelity") != "exact" or cm.get("arc"):
            raise ArcError("%s: %s" % (m["source"], cm.get("spec_note") or
                                       "no exact spec for %s" % cm.get("arch")))
        box = _canvas_model_box(c)
        if box is None:
            raise ArcError("%s: the canvas has no model box" % m["source"])
        built.append((m, c, box, out.get("summary") or {}))

    if len(built) == 1:
        m, c, box, summary = built[0]
        c.setdefault("model", {})["pipeline"] = [m["name"] or m["source"]]
        c["name"] = nn_arc.canonical(source)
        return {"ok": True, "canvas": c, "summary": summary,
                "models": [m["name"] or m["source"]]}

    top, tmk, tlink = _doc_maker("canvas")
    boxes, ins, outs = {}, {}, {}
    for i, (m, c, box, _summary) in enumerate(built):
        name, k = m["name"], 2
        while name in boxes:                       # two models may share a name
            name = "%s %d" % (m["name"], k)
            k += 1
        boxes[name] = tmk("custom", name, 420, 340 * i,
                          inner=json.loads(json.dumps(box["inner"])))
        ins[name] = [n for n in (c.get("nodes") or []) if n.get("kind") == "input"]
        outs[name] = [n for n in (c.get("nodes") or []) if n.get("kind") == "output"]

    edges = []
    for e in (pipeline or {}).get("edges") or []:
        if not isinstance(e, dict):
            continue
        a, b = str(e.get("from")), str(e.get("to"))
        if a not in boxes or b not in boxes:
            raise ArcError("the pipeline edge %s -> %s names an unknown model" % (a, b))
        edges.append((a, b, int(e.get("fromPort") or 0), int(e.get("toPort") or 0)))

    fed_in, fed_out = set(), set()
    for a, b, fp, tp in edges:
        fed_in.add((b, tp))
        fed_out.add((a, fp))
        tlink(boxes[a], boxes[b], fp, tp)
    for i, name in enumerate(boxes):
        for p in range(len(ins[name])):
            if (name, p) not in fed_in:
                src = tmk("input", "in %s" % name, 0, 340 * i, dout=1)
                tlink(src, boxes[name], 0, p)
        for p in range(len(outs[name])):
            if (name, p) not in fed_out:
                dst = tmk("output", "out %s" % name, 860, 340 * i, din=1)
                tlink(boxes[name], dst, p, 0)
    _snap_layout(top)

    params = bytes_ = tensors = layers = 0
    dtypes, unbound, meta_models, specs = {}, [], [], set()
    for name, (m, c, box, _s) in zip(boxes, built):
        cm = c.get("model") or {}
        meta_models.append({"name": name, "source": m["source"], "arch": cm.get("arch"),
                            "spec": cm.get("spec"), "layers": cm.get("layers"),
                            "params": cm.get("params"), "bytes": cm.get("bytes")})
        params += cm.get("params") or 0
        bytes_ += cm.get("bytes") or 0
        tensors += cm.get("tensors") or 0
        layers += cm.get("layers") or 0
        for k, v in (cm.get("dtypes") or {}).items():
            dtypes[k] = dtypes.get(k, 0) + v
        unbound += cm.get("unbound") or []
        if cm.get("spec"):
            specs.add(cm["spec"])
    n, e = canvas_counts({"nodes": top["nodes"], "edges": top["edges"]})
    model = {"arch": "pipeline", "fidelity": "exact", "models": meta_models,
             "layers": layers, "params": params, "bytes": bytes_, "dtypes": dtypes,
             "tensors": tensors, "unbound": unbound[:40],
             "spec": "+".join(sorted(specs)) or None,
             "pipeline": [x["name"] for x in meta_models]}
    summary = {"arch": "pipeline", "blocks": layers, "models": len(meta_models),
               "tensors": tensors, "boxes": n, "wires": e, "fidelity": "exact",
               "unbound": len(unbound)}
    canvas = {"type": "canvas", "version": 1, "model": model,
              "nodes": top["nodes"], "edges": top["edges"],
              "name": nn_arc.canonical(source)}
    return {"ok": True, "canvas": canvas, "summary": summary,
            "models": [x["name"] for x in meta_models]}


# ---------------------------------------------------------------- the audit
# What is wrong with each box's connections, block by block. The numeric gate
# cannot see a drawing problem — a box fed by nothing still runs if nothing
# reads it — so this is the structural check the job runs on its own build and
# the reverse agent runs while drafting (specs must draw every wire they mean).


def _inner_ports(node):
    """(inputs, outputs) a nested box exposes — its inner io boxes, in order."""
    inner = node.get("inner") or {}
    ins = sum(1 for n in inner.get("nodes") or [] if n.get("kind") == "input")
    outs = sum(1 for n in inner.get("nodes") or [] if n.get("kind") == "output")
    return ins, outs


def _needs_in(node):
    """Does this box need at least one wire in? (`input` is the only source;
    a nested box with no inner input needs none either.)"""
    if node.get("kind") in ("input", "agent"):
        return False                           # a source, or a view of an agent's work
    if node.get("kind") == "custom":
        return _inner_ports(node)[0] > 0
    return True


def audit_canvas(canvas):
    """Per-box connection problems in a nested canvas.

    Walks every level and reports, deterministically: wires that point at a
    box that is not there, wires on ports that do not exist, boxes no wire
    leaves (they compute nothing), boxes no wire enters (they never run), and
    duplicate ids. Every issue names the block it lives in and the box, so the
    reverse agent can fix the spec that drew it and the app can point at it.
    """
    issues = []
    boxes = [0]

    def issue(sev, where, node, what, chain):
        issues.append({"severity": sev, "where": where or "(top)",
                       "box": node.get("name") if node else None,
                       "nid": node.get("id") if node else None,
                       "chain": list(chain), "what": what})

    def walk(doc, chain, where):
        nodes = doc.get("nodes") or []
        edges = doc.get("edges") or []
        boxes[0] += len(nodes)
        by_id, dup = {}, set()
        for n in nodes:
            nid = n.get("id")
            if nid in by_id:
                dup.add(str(nid))
            by_id[nid] = n
        for nid in sorted(dup):
            issue("error", where, by_id.get(nid), "duplicate box id", chain)
        outgoing, incoming, in_ports = set(), set(), {}
        for e in edges:
            a, b = e.get("from"), e.get("to")
            if a not in by_id or b not in by_id:
                issue("error", where, by_id.get(b) or by_id.get(a),
                      "a wire points at a box that is not here (%s \u2192 %s)"
                      % (a, b), chain)
                continue
            outgoing.add(a)
            incoming.add(b)
            an, bn = by_id[a], by_id[b]
            fp, tp = int(e.get("fromPort") or 0), int(e.get("toPort") or 0)
            ao = _inner_ports(an)[1] if an.get("kind") == "custom" else 1
            if fp >= ao:
                issue("error", where, an,
                      "a wire leaves port %d, but this box has %d output(s)"
                      % (fp, ao), chain)
            if bn.get("kind") == "custom":
                bi = _inner_ports(bn)[0]
                if tp >= bi:
                    issue("error", where, bn,
                          "a wire enters port %d, but the box has %d input(s)"
                          % (tp, bi), chain)
            else:
                in_ports.setdefault(b, set()).add(tp)
        for nid, ports in in_ports.items():
            n = by_id[nid]
            if n.get("loose") or not ports:
                continue
            if sorted(ports) != list(range(len(ports))):
                issue("warning", where, n,
                      "the wires do not fill its inputs in order (ports %s)"
                      % ", ".join(str(p) for p in sorted(ports)), chain)
        for n in nodes:
            if n.get("loose"):
                continue                           # a kept tensor, not drawn as wired
            kind = n.get("kind")
            if kind not in ("input", "output", "agent") and n.get("id") not in outgoing:
                issue("warning", where, n,
                      "feeds nothing \u2014 no wire leaves this box", chain)
            if _needs_in(n) and n.get("id") not in incoming:
                issue("warning", where, n,
                      "gets no data \u2014 no wire enters this box", chain)
        for n in nodes:
            if n.get("kind") == "custom" and n.get("inner"):
                name = str(n.get("name") or n.get("id"))
                walk(n["inner"], chain + [n.get("id")],
                     (where + " / " + name) if where else name)

    walk(canvas, [], "")
    blocks = {}
    for is_ in issues:
        blocks[is_["where"]] = blocks.get(is_["where"], 0) + 1
    return {"ok": not issues, "boxes": boxes[0], "count": len(issues),
            "issues": issues, "blocks": blocks}
