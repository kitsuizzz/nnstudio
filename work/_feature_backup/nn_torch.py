#!/usr/bin/env python3
"""nnstudio's runner: a canvas design, executed in torch.

The canvas JSON is a dataflow graph, so it compiles straight into a module
tree: every box is a module, every wire is an argument, and a nested box is a
sub-graph whose input/output boxes are its ports (top-to-bottom, exactly the
order the UI reads its dots in). On top of that:

  * `run`   — one forward pass; every box's output is captured (shape + stats)
              so the UI can put the value on the card. An optional generate
              loop samples ids from an output box.
  * `train` — backprop through every weight that is not frozen (the existing
              `frozen` flag, read through nested boxes), AdamW/SGD, char-level
              next-token cross-entropy, eval, samples, a checkpoint.
  * GGUF   — a bound box reads its tensor(s) through `load_gguf_tensor`:
              quantized blocks are dequantized with gguf.quants.dequantize,
              and `tensor.data` is already the logical array in torch's
              orientation ([out, in] for a linear, [vocab, d] for a table).

The same module runs locally (nn_server) and on Modal (nnstudio_modal) —
nothing here knows where it runs.
"""
import math
import os
import re
import threading
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize


class CanvasError(Exception):
    """A design that cannot execute — with the box it is about."""

    def __init__(self, box, reason):
        self.box = box or ""
        self.reason = reason
        super().__init__((self.box + ": " if self.box else "") + reason)


# ---------------------------------------------------------------- helpers
def _seed_for(seed, tag):
    return (int(seed) + zlib.crc32(str(tag).encode("utf-8"))) & 0x7FFFFFFF


_READERS = {}
_READER_LOCK = threading.Lock()


def _reader(path):
    """One GGUFReader per file (mtime-keyed), so a model is read once, not
    once per tensor."""
    key = os.path.abspath(path)
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    with _READER_LOCK:
        hit = _READERS.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    reader = GGUFReader(path)
    with _READER_LOCK:
        _READERS[key] = (mtime, reader)
    return reader


def load_gguf_tensor(path, name, want_dtype=torch.float32):
    """One tensor out of a GGUF file, dequantized and ready for torch.

    gguf-py's `tensor.shape` is the file's ne order, but `tensor.data` (and
    what dequantize returns) is the logical numpy array — [out, in] for a
    linear, [vocab, d_model] for an embedding — which is exactly torch's
    orientation. The values are copied off the mmap (writable, trainable)."""
    if not path or not os.path.isfile(path):
        raise CanvasError(name, "the GGUF file is not on disk: %s" % (path or "?"))
    reader = _reader(path)
    hit = next((t for t in reader.tensors if t.name == name), None)
    if hit is None:
        raise CanvasError(name, "tensor not found in %s" % os.path.basename(path))
    try:
        raw = gguf_dequantize(hit.data, hit.tensor_type)
    except Exception:  # noqa: BLE001 — f32/f16/bf16 already are plain arrays
        raw = np.asarray(hit.data)
    raw = np.array(raw, dtype=np.float32, copy=True)
    return torch.from_numpy(raw).to(want_dtype)


def parse_dims(text):
    """'128x40', '128 40', '128,40' -> [128, 40]; empty/none -> []."""
    if text is None:
        return []
    return [int(p) for p in re.split(r"[^0-9]+", str(text)) if p]


# ---------------------------------------------------------------- stats
def tensor_stats(t, peek=8):
    """The little summary a card pill knows: shape, three numbers, a peek."""
    if t is None:
        return None
    if isinstance(t, dict):                     # a nested box's port values
        t = next((v for v in t.values() if v is not None), None)
        if t is None:
            return None
    with torch.no_grad():
        x = t.detach().float()
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return {"shape": list(x.shape), "mean": None, "min": None, "max": None,
                    "scalar": None, "peek": []}
        return {
            "shape": list(x.shape),
            "mean": round(float(x.mean()), 4),
            "min": round(float(x.min()), 4),
            "max": round(float(x.max()), 4),
            "scalar": round(float(flat[0]), 4) if flat.numel() == 1 else None,
            "peek": [round(v, 4) for v in flat[:peek].tolist()],
        }


# ---------------------------------------------------------------- boxes
class Box(nn.Module):
    """One canvas box. Weight-bearing kinds own parameters; a nested box owns
    a Graph. `frozen`/`off` are inherited from the enclosing levels."""

    def __init__(self, node, seed=0, parent_path="", frozen=False, off=False,
                 loader=None, dtype=torch.float32):
        super().__init__()
        self.node = node
        self.kind = node.get("kind")
        self.name = node.get("name") or self.kind
        self.off = off or bool(node.get("off"))
        self.frozen = frozen or bool(node.get("frozen"))
        self.loose = bool(node.get("loose"))
        self.inner = None
        self.w = None
        self.bias = None
        self.last = None
        gen = torch.Generator(device="cpu").manual_seed(_seed_for(seed, node.get("id") or self.name))
        grad = not self.frozen

        if self.kind == "layer":
            din, dout = node.get("din"), node.get("dout")
            if din is None or dout is None:
                raise CanvasError(self.name, "needs both an in and an out size")
            bound = loader(node) if loader else None
            if bound is not None:
                self.w = nn.Parameter(bound["weight"].reshape(int(dout), int(din)).to(dtype),
                                      requires_grad=grad)
                if bound.get("bias") is not None:
                    self.bias = nn.Parameter(bound["bias"].reshape(-1).to(dtype),
                                             requires_grad=grad)
            else:
                self.w = nn.Parameter(torch.empty(int(dout), int(din), dtype=dtype),
                                      requires_grad=grad)
                k = 1.0 / math.sqrt(max(1, int(din)))
                with torch.no_grad():
                    self.w.uniform_(-math.sqrt(3) * k, math.sqrt(3) * k, generator=gen)
        elif self.kind == "embed":
            vocab, dout = node.get("vocab"), node.get("dout")
            if vocab is None or dout is None:
                raise CanvasError(self.name, "needs the table's vocab and its out size")
            bound = loader(node) if loader else None
            if bound is not None:
                self.w = nn.Parameter(bound["weight"].reshape(int(vocab), int(dout)).to(dtype),
                                      requires_grad=grad)
            else:
                self.w = nn.Parameter(torch.empty(int(vocab), int(dout), dtype=dtype),
                                      requires_grad=grad)
                with torch.no_grad():
                    self.w.normal_(0.0, 0.02, generator=gen)
        elif self.kind == "norm":
            bound = loader(node) if loader else None
            if bound is not None:
                w = bound["weight"].reshape(-1)
            elif node.get("size") is not None:
                w = torch.ones(int(node["size"]), dtype=dtype)
            else:
                raise CanvasError(self.name, "its size is not known — wire it or bind its tensor")
            self.w = nn.Parameter(w.to(dtype), requires_grad=grad)
        elif self.kind == "op" and node.get("op") == "scale":
            bound = loader(node) if loader else None
            w = bound["weight"].reshape(-1)[:1] if bound is not None else torch.ones(1)
            self.w = nn.Parameter(w.to(dtype), requires_grad=grad)
        elif self.kind == "custom":
            self.inner = Graph(node.get("inner") or {"nodes": [], "edges": []}, seed=seed,
                               path=parent_path + "/" + (node.get("id") or ""),
                               frozen=self.frozen, off=self.off, dtype=dtype,
                               gguf_file=node.get("file"))

    def forward(self, *xs):
        if self.off:
            self.last = None
            return None
        k = self.kind
        if k in ("layer", "norm", "act", "op"):
            # a feed straight from JSON can arrive as int64; arithmetic wants
            # a float (embed keeps its ids)
            xs = tuple(x.float() if torch.is_tensor(x) and not torch.is_floating_point(x) else x
                       for x in xs)
        if k in ("input", "output"):
            out = xs[0] if xs else None
        elif k == "layer":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = F.linear(xs[0], self.w, self.bias)
        elif k == "embed":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = F.embedding(xs[0].long(), self.w)
        elif k == "norm":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            x = xs[0]
            if self.node.get("norm") == "layer":
                out = F.layer_norm(x, (x.shape[-1],), self.w, None,
                                   float(self.node.get("eps") or 1e-5))
            else:
                v = x.float()
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) +
                                    float(self.node.get("eps") or 1e-6))
                out = v.to(x.dtype) * self.w
        elif k == "act":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = _activate(self.node.get("act") or "none", xs[0], self.name)
        elif k == "op":
            out = _operate(self, xs)
        elif k == "custom":
            if self.inner is None:
                raise CanvasError(self.name, "the nested box has no inside")
            vals = self.inner(*xs)
            out = {i: vals.get(oid) for i, oid in enumerate(self.inner.output_ids)}
            self.last = out
            return out
        else:
            raise CanvasError(self.name, "unknown box kind '%s'" % k)
        self.last = out
        return out


def _activate(name, x, box=""):
    name = (name or "none").lower()
    if name in ("", "none"):
        return x
    if name == "relu":
        return F.relu(x)
    if name == "leaky relu":
        return F.leaky_relu(x, 0.01)
    if name == "gelu":
        return F.gelu(x)
    if name == "silu":
        return F.silu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "softmax":
        return F.softmax(x.float(), dim=-1).to(x.dtype)
    raise CanvasError(box, "unknown activation '%s'" % name)


def _operate(box, xs):
    op = box.node.get("op") or "transpose"
    a = xs[0] if xs else None
    b = xs[1] if len(xs) > 1 else None
    if op == "transpose":
        return a.transpose(-1, -2)
    if op == "matmul":
        return torch.matmul(a, b)
    if op == "dot product":
        return (a * b).sum()
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return a / b
    if op == "concat":
        return torch.cat([a, b], dim=-1)
    if op == "reshape":
        dims = parse_dims(box.node.get("dims"))
        if not dims:
            return a
        if a.numel() != int(np.prod(dims)):
            raise CanvasError(box.name, "reshape: %d values cannot become %s"
                              % (a.numel(), "×".join(str(d) for d in dims)))
        return a.reshape(dims)
    if op == "scale":
        return a * box.w
    if op == "rope":
        return _rope(a, float(box.node.get("theta") or 10000.0))
    if op == "topk":
        k = 1
        m = re.search(r"(\d+)", str(box.name))
        if m:
            k = int(m.group(1))
        vals, idx = torch.topk(a.float(), min(k, a.shape[-1]), dim=-1)
        out = torch.zeros_like(a.float())
        out.scatter_(-1, idx, vals)
        return out.to(a.dtype)
    if op in ("conv1d", "scan"):
        raise CanvasError(box.name, "'%s' is not executable by the torch engine yet "
                                    "— run the model as-is (llama.cpp) or freeze it" % op)
    raise CanvasError(box.name, "unknown operation '%s'" % op)


def _rope(x, theta):
    """Rotate half-split pairs over the last dim; positions run along the
    last-but-one dim (a lone vector sits at position 0)."""
    d = x.shape[-1]
    if d % 2 or d < 2:
        return x
    positions = torch.arange(x.shape[-2], device=x.device) if x.dim() >= 2 \
        else torch.zeros(1, device=x.device)
    half = d // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device).float() * 2.0 / d)
    ang = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)      # [T, half]
    x1, x2 = x[..., :half], x[..., half:]
    while ang.dim() < x1.dim():
        ang = ang.unsqueeze(0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


# ---------------------------------------------------------------- the graph
class Graph(nn.Module):
    """One design: a whole canvas, or the inside of a nested box."""

    def __init__(self, doc, seed=0, path="", frozen=False, off=False,
                 dtype=torch.float32, gguf_file=None):
        super().__init__()
        self.path = path
        self.gguf_file = gguf_file
        nodes = [dict(n) for n in (doc.get("nodes") or [])]
        edges = list(doc.get("edges") or [])
        self.edges = edges
        self.boxes = nn.ModuleDict()
        self.last_vals = None

        by_pos = lambda seq: sorted(seq, key=lambda n: (n.get("y") or 0, n.get("x") or 0))
        self.input_ids = [n["id"] for n in by_pos(n for n in nodes if n.get("kind") == "input")]
        self.output_ids = [n["id"] for n in by_pos(n for n in nodes if n.get("kind") == "output")]

        incoming = {}
        for e in edges:
            incoming.setdefault(e["to"], []).append(e)
        self.incoming = {k: sorted(v, key=lambda e: e.get("toPort") or 0)
                         for k, v in incoming.items()}
        self.order = self._topo(nodes)

        # a norm with no size takes the size flowing into it
        sizes = self._incoming_sizes(nodes, edges)
        for n in nodes:
            if n.get("kind") == "norm" and n.get("size") is None and n["id"] in sizes:
                n["size"] = sizes[n["id"]]
        self.node_list = nodes

        for n in nodes:
            self.boxes[n["id"]] = Box(n, seed=seed, parent_path=self.path,
                                      frozen=frozen, off=off, loader=self._bind, dtype=dtype)

    # ---- compile helpers ---------------------------------------------------
    def _bind(self, node):
        """Load a box's bound GGUF tensor(s) when a file is in reach; a
        `...bias` binding becomes the layer's bias."""
        t = (node.get("t") or "").strip()
        if not t or not self.gguf_file:
            return None
        weight = bias = None
        for name in [x.strip() for x in t.split(",") if x.strip()]:
            if name.endswith(".bias"):
                bias = load_gguf_tensor(self.gguf_file, name)
            elif weight is None:
                weight = load_gguf_tensor(self.gguf_file, name)
        if weight is None:
            return None
        return {"weight": weight, "bias": bias}

    def _incoming_sizes(self, nodes, edges):
        by_id = {n["id"]: n for n in nodes}
        out = {}

        def size_of(n):
            k = n.get("kind")
            if k in ("input", "layer", "embed"):
                return n.get("dout")
            if k == "output":
                return n.get("din")
            if k == "norm":
                return n.get("din")
            return None

        for e in edges:
            src = by_id.get(e["from"])
            if src is not None:
                s = size_of(src)
                if s is not None:
                    out[e["to"]] = s
        return out

    def _topo(self, nodes):
        ids = {n["id"]: n for n in nodes}
        deps = {n["id"]: [e["from"] for e in self.incoming.get(n["id"], [])] for n in nodes}
        order, done = [], set()
        while len(order) < len(nodes):
            fresh = [i for i, d in deps.items() if i not in done and all(x in done for x in d)]
            if not fresh:
                missing = [i for i, d in deps.items() if i not in done and
                           any(x not in ids for x in d)]
                if missing:
                    raise CanvasError(ids[missing[0]].get("name"),
                                      "its input is wired to a box that is not here")
                raise CanvasError("", "the design has a loop the engine cannot run")
            for i in fresh:                      # document order within a wave
                order.append(i)
                done.add(i)
        return order

    def _port_val(self, v, port):
        if isinstance(v, dict):
            if port in v:
                return v[port]
            vals = [x for x in v.values() if x is not None]
            return vals[0] if vals else None
        return v

    # ---- forward -----------------------------------------------------------
    def forward(self, *args, feeds=None):
        """`args` map to the graph's input boxes in port order (how a nested
        box calls its inside); `feeds` is {input box id: tensor}."""
        feeds = dict(feeds or {})
        for i, a in enumerate(args):
            if i < len(self.input_ids):
                feeds[self.input_ids[i]] = a
        vals = {}
        for nid in self.order:
            box = self.boxes[nid]
            if box.loose or box.off:             # drawn, not run / switched off
                vals[nid] = None
                continue
            ins = []
            for e in self.incoming.get(nid, []):
                v = vals.get(e["from"])
                if v is None:
                    if self.boxes[e["from"]].loose:
                        continue                 # an unmapped tensor: ignore its wire
                    raise CanvasError(box.name, "it is fed by a box that did not run")
                ins.append(self._port_val(v, e.get("fromPort") or 0))
            if box.kind == "input":
                if nid not in feeds:
                    raise CanvasError(box.name, "no value was given for this input")
                v = feeds[nid]
            elif box.kind == "custom":
                v = box(*ins)
            else:
                if not ins:
                    raise CanvasError(box.name, "an input dot is not wired")
                v = box(*ins)
            vals[nid] = v
        self.last_vals = vals
        return vals

    def collect(self, prefix=""):
        """Every box's last value, keyed by path — nested levels included."""
        out = {}
        for nid, box in self.boxes.items():
            path = prefix + nid
            v = (self.last_vals or {}).get(nid)
            if box.kind == "custom" and box.inner is not None:
                first = v.get(0) if isinstance(v, dict) else v
                if first is not None:
                    out[path] = first
                out.update(box.inner.collect(path + "/"))
            elif v is not None:
                out[path] = v
        return out


# ---------------------------------------------------------------- describe
def describe(canvas):
    """The pre-flight counts (with the freeze flag read through nested boxes)
    and the top-level input/output boxes."""
    counts = {"params": 0, "trainable": 0, "frozen": 0, "unknown": 0}
    ins, outs = [], []

    def visit(doc, frozen):
        for n in doc.get("nodes", []):
            froz = frozen or bool(n.get("frozen")) or bool(n.get("off"))
            if n.get("kind") == "custom":
                visit(n.get("inner") or {}, froz)
                continue
            if n.get("kind") == "input":
                ins.append({"id": n["id"], "name": n.get("name"), "size": n.get("dout")})
                continue
            if n.get("kind") == "output":
                outs.append({"id": n["id"], "name": n.get("name"), "size": n.get("din")})
                continue
            if n.get("kind") in ("act", "op"):
                continue
            p = None
            if n.get("params") is not None:
                p = int(n["params"])
            elif n.get("kind") == "embed" and n.get("vocab") is not None and n.get("dout") is not None:
                p = int(n["vocab"]) * int(n["dout"])
            elif n.get("kind") == "layer" and n.get("din") is not None and n.get("dout") is not None:
                p = int(n["din"]) * int(n["dout"])
            if p is None:
                counts["unknown"] += 1
                continue
            counts["params"] += p
            if froz:
                counts["frozen"] += p
            else:
                counts["trainable"] += p

    visit(canvas, False)
    return {"counts": counts, "inputs": ins, "outputs": outs}


# ---------------------------------------------------------------- run
def _sample_probs(logits, temperature, top_k, top_p):
    t = max(1e-4, float(temperature or 1.0))
    p = torch.softmax(logits / t, dim=-1)
    if top_k and top_k > 0:
        kth = torch.topk(p, min(int(top_k), p.numel())).values[-1]
        p = torch.where(p < kth, torch.zeros_like(p), p)
    if top_p and top_p < 1.0:
        sorted_p, idx = torch.sort(p, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = (cum - sorted_p) <= float(top_p)
        keep[0] = True
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        p = torch.zeros_like(p).scatter(0, idx, sorted_p)
    s = p.sum()
    return p / s if s > 0 else torch.full_like(p, 1.0 / p.numel())


def run(canvas, feeds, seed=0, device="cpu", generate=0, temperature=0.8,
        top_k=0, top_p=1.0, sample_seed=1, input_id=None, output_id=None,
        max_len=512):
    """One forward pass (or a short generate). Returns per-box values for the
    UI plus the generated token ids."""
    t0 = time.time()
    graph = compile_canvas(canvas, seed=seed, device=device)
    graph.eval()

    def prep(feed_map):
        return {k: (v.to(device) if torch.is_tensor(v) else torch.as_tensor(v, device=device))
                for k, v in (feed_map or {}).items()}

    if generate:
        if not input_id:
            input_id = next((n["id"] for n in canvas.get("nodes", [])
                             if n.get("kind") == "input"), None)
        if not output_id:
            output_id = next((n["id"] for n in canvas.get("nodes", [])
                              if n.get("kind") == "output"), None)
    feeds = prep(feeds)
    generated = []
    with torch.no_grad():
        vals = graph(feeds=feeds)
        if generate and input_id and output_id and input_id in feeds:
            ids = feeds[input_id].reshape(-1).long().cpu()
            gen = torch.Generator(device="cpu").manual_seed(int(sample_seed))
            for _ in range(int(generate)):
                vals = graph(feeds=feeds)
                logits = vals.get(output_id)
                if logits is None:
                    raise CanvasError("output", "the output box is switched off")
                last = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
                nxt = int(torch.multinomial(_sample_probs(last, temperature, top_k, top_p),
                                            1, generator=gen).item())
                generated.append(nxt)
                ids = torch.cat([ids, torch.tensor([nxt], dtype=torch.long)])
                feeds[input_id] = ids[-max_len:].to(device)
    values = {}
    for path, v in graph.collect().items():
        st = tensor_stats(v)
        if st is not None:
            values[path] = st
    return {
        "ok": True,
        "values": values,
        "generated": generated,
        "outputs": [{"id": n["id"], "name": n.get("name"),
                     "stats": tensor_stats(vals.get(n["id"]))}
                    for n in canvas.get("nodes", []) if n.get("kind") == "output"],
        "ms": int((time.time() - t0) * 1000),
    }


def compile_canvas(canvas, seed=0, device="cpu", dtype=torch.float32, gguf_file=None):
    """Build the module for a top-level canvas document."""
    return Graph(dict(canvas), seed=seed, path="", dtype=dtype, gguf_file=gguf_file)


# ---------------------------------------------------------------- data
def build_vocab(text):
    """Char-level vocabulary: the sorted unique characters of the dataset."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi, {i: c for c, i in stoi.items()}


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


def decode(ids, itos):
    return "".join(itos.get(int(i), "") for i in ids)


# ---------------------------------------------------------------- train
class Trainer:
    """Char-level next-token training over every non-frozen weight."""

    def __init__(self, canvas, text, input_id, output_id, cfg, device="cpu",
                 progress=None, should_stop=None):
        self.canvas = canvas
        self.text = text
        self.cfg = cfg or {}
        self.device = device
        self.progress = progress or (lambda **kw: None)
        self.should_stop = should_stop or (lambda: False)
        self.graph = compile_canvas(canvas, seed=int(self.cfg.get("seed") or 0), device=device)
        self.graph.train()
        self.chars, self.stoi, self.itos = build_vocab(text)
        ids = torch.tensor(encode(text, self.stoi), dtype=torch.long)
        if ids.numel() < 16:
            raise CanvasError("", "the dataset is too small (%d characters)" % ids.numel())
        n_val = max(1, int(ids.numel() * 0.05))
        self.train_ids, self.val_ids = ids[:-n_val], ids[-n_val:]
        cfg = dict(self.cfg)
        self.cfg_resolved = {
            "vocab": len(self.chars), "chars": int(ids.numel()),
            "ctx": max(2, int(cfg.get("ctx") or 32)),
            "batch": max(1, int(cfg.get("batch") or 16)),
            "steps": max(1, int(cfg.get("steps") or 500)),
            "lr": float(cfg.get("lr") or 3e-4),
            "optimizer": (cfg.get("optimizer") or "adamw").lower(),
            "seed": int(cfg.get("seed") or 0),
            "eval_every": int(cfg.get("eval_every") or 50),
            "sample_every": int(cfg.get("sample_every") or 100),
            "sample_len": int(cfg.get("sample_len") or 60),
            "temperature": float(cfg.get("temperature") or 0.8),
            "max_minutes": float(cfg.get("max_minutes") or 0),
        }
        self.input_id, self.output_id = input_id, output_id
        out_node = _find(canvas, output_id)
        width = (out_node or {}).get("din")
        if width is not None and int(width) != len(self.chars):
            raise CanvasError((out_node or {}).get("name") or "output",
                              "is %s wide but the data has %d characters — resize it, or use "
                              "data with the same character set" % (width, len(self.chars)))
        if self.train_ids.numel() <= self.cfg_resolved["ctx"] + 1:
            raise CanvasError("", "the dataset is too small for a context of %d"
                              % self.cfg_resolved["ctx"])
        trainable = [p for p in self.graph.parameters() if p.requires_grad]
        self.opt = None
        if self.cfg_resolved["optimizer"] != "none" and trainable:
            if self.cfg_resolved["optimizer"] == "sgd":
                self.opt = torch.optim.SGD(trainable, lr=self.cfg_resolved["lr"])
            else:
                self.opt = torch.optim.AdamW(trainable, lr=self.cfg_resolved["lr"])
        self.trainable_params = sum(p.numel() for p in trainable)
        self.params_total = sum(p.numel() for p in self.graph.parameters())
        self.evals, self.samples, self.losses = [], [], []
        self.tps = self.elapsed = 0.0
        self.checkpoint = None

    def _batch(self, source, gen):
        ctx, batch = self.cfg_resolved["ctx"], self.cfg_resolved["batch"]
        hi = max(1, source.numel() - ctx - 1)
        starts = torch.randint(0, hi, (batch,), generator=gen)
        x = torch.stack([source[int(s):int(s) + ctx] for s in starts]).to(self.device)
        y = torch.stack([source[int(s) + 1:int(s) + ctx + 1] for s in starts]).to(self.device)
        return x, y

    def _forward_loss(self, x, y):
        vals = self.graph(feeds={self.input_id: x})
        logits = vals.get(self.output_id)
        if isinstance(logits, dict):
            logits = next((v for v in logits.values() if v is not None), None)
        if logits is None:
            raise CanvasError("", "the output box is switched off")
        v = len(self.chars)
        lg = logits.reshape(-1, v)
        tg = y.reshape(-1)
        if lg.shape[0] != tg.numel() and lg.shape[0] == x.shape[0]:
            tg = y[:, -1]                       # one prediction per window
        if lg.shape[0] != tg.numel():
            raise CanvasError("", "the output (%d values) does not line up with the tokens (%d)"
                              % (lg.shape[0], tg.numel()))
        return F.cross_entropy(lg, tg)

    def step(self, gen):
        x, y = self._batch(self.train_ids, gen)
        loss = self._forward_loss(x, y)
        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def eval_loss(self):
        x, y = self._batch(self.val_ids, torch.Generator().manual_seed(1234))
        return float(self._forward_loss(x, y).detach())

    @torch.no_grad()
    def sample(self, n):
        seed = encode(self.text[:1], self.stoi)[:1] or [0]
        ids = torch.tensor([seed], dtype=torch.long)          # [1, 1]
        out = []
        for _ in range(int(n)):
            vals = self.graph(feeds={self.input_id: ids.to(self.device)})
            logits = vals.get(self.output_id)
            if isinstance(logits, dict):
                logits = next((v for v in logits.values() if v is not None), None)
            if logits is None:
                break
            last = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
            p = _sample_probs(last, self.cfg_resolved["temperature"], 0, 1.0)
            nxt = int(torch.multinomial(p, 1).item())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=-1)[:, -self.cfg_resolved["ctx"]:]
        return decode(out, self.itos)

    def train(self):
        cfg = self.cfg_resolved
        gen = torch.Generator().manual_seed(cfg["seed"] or 1)
        t0 = time.time()
        self.progress(state="running", step=0, steps=cfg["steps"],
                      vocab=cfg["vocab"], chars=cfg["chars"],
                      trainable_params=self.trainable_params, params=self.params_total,
                      frozen_params=self.params_total - self.trainable_params)
        state = "done"
        for i in range(cfg["steps"]):
            if self.should_stop():
                state = "stopped"
                break
            if cfg["max_minutes"] and (time.time() - t0) / 60.0 > cfg["max_minutes"]:
                state = "capped"
                break
            loss = self.step(gen)
            self.losses.append(loss)
            self.tps = (i + 1) * cfg["batch"] * cfg["ctx"] / max(1e-6, time.time() - t0)
            self.elapsed = time.time() - t0
            ev = sample = None
            if cfg["eval_every"] and (i + 1) % cfg["eval_every"] == 0:
                ev = self.eval_loss()
                self.evals.append({"step": i + 1, "loss": ev})
            if cfg["sample_every"] and (i + 1) % cfg["sample_every"] == 0:
                sample = self.sample(cfg["sample_len"])
                self.samples.append({"step": i + 1, "text": sample})
            self.progress(state="running", step=i + 1, steps=cfg["steps"], loss=loss,
                          eval=ev, sample=sample, tps=self.tps, elapsed=self.elapsed)
        self.progress(state=state, step=len(self.losses), steps=cfg["steps"],
                      tps=self.tps, elapsed=self.elapsed)
        return {
            "state": state, "step": len(self.losses),
            "trainable_params": self.trainable_params, "params": self.params_total,
            "frozen_params": self.params_total - self.trainable_params,
            "evals": self.evals, "samples": self.samples,
            "loss": self.losses[-1] if self.losses else None,
            "losses": self.losses, "elapsed": self.elapsed,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.graph.state_dict(), "vocab": self.chars,
                    "canvas": self.canvas, "config": dict(self.cfg_resolved)}, path)
        self.checkpoint = path
        return path


def train(canvas, text, input_id, output_id, cfg=None, device="cpu",
          progress=None, should_stop=None, checkpoint_path=None):
    """Train a canvas: char-level next token, every non-frozen weight."""
    tr = Trainer(canvas, text, input_id, output_id, cfg or {}, device=device,
                 progress=progress, should_stop=should_stop)
    result = tr.train()
    if checkpoint_path:
        try:
            result["checkpoint"] = tr.save(checkpoint_path)
        except Exception as e:  # noqa: BLE001 — a failed save must not lose the run
            result["checkpoint_error"] = str(e)
    return result


def _find(doc, nid):
    for n in doc.get("nodes", []):
        if n.get("id") == nid:
            return n
        if n.get("kind") == "custom" and n.get("inner"):
            hit = _find(n["inner"], nid)
            if hit:
                return hit
    return None
