#!/usr/bin/env python3
"""nnstudio's runner: a canvas design, executed in torch.

The canvas JSON is a dataflow graph, so it compiles straight into a module
tree: every box is a module, every wire is an argument, and a nested box is a
sub-graph whose input/output boxes are its ports (top-to-bottom, exactly the
order the UI reads its dots in). On top of that:

  * `run`   — one forward pass; every box's output is captured (shape + stats)
              so the UI can put the value on the card. An optional generate
              loop samples ids from an output box. `mode` decides how the
              weights are held: "inference" (the default) asks for no
              gradients at all; "training" lets the non-frozen weights keep
              them and reads the design's dataset boxes.
  * `train` — backprop through every weight that is not frozen (the existing
              `frozen` flag, read through nested boxes), AdamW/SGD, char-level
              next-token cross-entropy, eval, samples, a checkpoint.
  * GGUF   — a bound box reads its tensor(s) through `load_gguf_tensor`:
              quantized blocks are dequantized with gguf.quants.dequantize,
              and `tensor.data` is already the logical array in torch's
              orientation ([out, in] for a linear, [vocab, d] for a table).
  * safetensors / pt — `load_weight_tensor` dispatches by extension; a box
              carries `file` + `t` and gets its real weights instead of the
              random init. `file_tensors` reads the metadata the app shows,
              `export_weights` writes a whole design back out as one file.

The same module runs locally (nn_server) and on Modal (nnstudio_modal) —
nothing here knows where it runs.
"""
import json
import math
import base64
import csv
import io
import os
import re
import shutil
import struct
import threading
import time
import zlib
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                          # gguf is only needed when a box
    from gguf import GGUFReader               # is bound to a real GGUF file
    from gguf.quants import dequantize as gguf_dequantize
except Exception:  # noqa: BLE001 — the agent runs hand-built designs without it
    GGUFReader = None
    gguf_dequantize = None

import nn_tokenizers as nn_hf                  # huggingface tokenizers: stdlib until used

try:                                          # native quantized-layer kernels —
    import nn_quant                           # optional, detection-only (triton
except Exception:  # noqa: BLE001            # ships with torch; nothing installs)
    nn_quant = None


class CanvasError(Exception):
    """A design that cannot execute — with the box it is about (and its node
    id, once a graph knows which box raised it)."""

    def __init__(self, box, reason):
        self.box = box or ""
        self.reason = reason
        self.nid = None
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
    orientation. The values are copied off the mmap (writable, trainable);
    a quantized tensor already comes back as a fresh array, so only a plain
    mmap needs the extra copy."""
    if GGUFReader is None:
        raise CanvasError(name, "this box is bound to a GGUF file — pip install gguf")
    path = _resolve(path)
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
    if raw.dtype != np.float32 or not raw.flags.writeable:
        raw = np.array(raw, dtype=np.float32, copy=True)
    return torch.from_numpy(raw).to(want_dtype)


def bound_weight(bound):
    """The weight a binding carries: already loaded, or the dequantized
    fallback for a native block set the box did not take."""
    w = bound.get("weight")
    if w is None and bound.get("native"):
        nat = bound["native"]
        w = load_gguf_tensor(nat["path"], nat["name"], None)
    return w


def load_gguf_native(path, name):
    """A quantized GGUF tensor's raw blocks + type, when a native kernel for
    it exists on this machine — else None (the caller dequantizes as usual).
    The blocks stay in ggml's layout; `shape` is the logical [out, in]."""
    if nn_quant is None or GGUFReader is None:
        return None
    path = _resolve(path)
    if not path or not os.path.isfile(path):
        return None
    hit = next((t for t in _reader(path).tensors if t.name == name), None)
    if hit is None:
        return None
    meta = nn_quant.native_gguf_tensor(hit)
    if meta is None:
        return None
    meta["path"] = path
    meta["name"] = name
    return meta


# ---------------------------------------------------------------- weight files
# The engine reads weights from GGUF, safetensors and torch checkpoints. The
# app reads the same files' *metadata* (file_tensors) so a block can be pointed
# at the tensor that belongs to it; the design carries {file, t} and the run
# loads it instead of the random init.
_SAFE_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
                "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
                "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
                "BOOL": torch.bool}
for _f8name, _f8attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    _f8type = getattr(torch, _f8attr, None)
    if _f8type is not None:
        _SAFE_DTYPES[_f8name] = _f8type
_F8_DTYPES = tuple(d for n, d in _SAFE_DTYPES.items() if n.startswith("F8_"))
_SAFE_NAMES = {v: k for k, v in _SAFE_DTYPES.items()}
_WEIGHT_EXTS = (".gguf", ".safetensors", ".pt", ".pth", ".bin")


def _resolve(path):
    """A relative path in a design ('weights/x.pt', 'models/y.gguf') resolves
    against the repo, wherever the agent was started from."""
    p = str(path or "")
    if p and not os.path.isabs(p):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root, p)
        if os.path.exists(cand):
            return cand
    return p


def read_safetensors(path):
    """(header, data_start) — the JSON header of a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CanvasError("", "not a safetensors file: %s" % os.path.basename(path))
        n = struct.unpack("<Q", raw)[0]
        try:
            header = json.loads(f.read(n).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise CanvasError("", "bad safetensors header in %s: %s"
                              % (os.path.basename(path), e))
    if not isinstance(header, dict):
        raise CanvasError("", "bad safetensors header in %s" % os.path.basename(path))
    return header, 8 + n


def load_safetensors_tensor(path, name, want_dtype=None):
    """One tensor out of a safetensors file, read by hand — no dependency.
    With no `want_dtype` it comes back exactly as stored (fp8 and packed
    uint8 included); with one, floating tensors are cast."""
    path = _resolve(path)
    if not os.path.isfile(path):
        raise CanvasError(name, "the weight file is not on disk: %s" % (path or "?"))
    header, start = read_safetensors(path)
    ent = header.get(name)
    if not isinstance(ent, dict):
        raise CanvasError(name, "tensor not found in %s" % os.path.basename(path))
    dtype = _SAFE_DTYPES.get(str(ent.get("dtype") or "").upper())
    if dtype is None:
        raise CanvasError(name, "unsupported safetensors dtype '%s'" % ent.get("dtype"))
    off = ent.get("data_offsets") or [0, 0]
    with open(path, "rb") as f:
        f.seek(start + int(off[0]))
        raw = f.read(int(off[1]) - int(off[0]))
    t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(list(ent.get("shape") or [])).clone()
    if want_dtype is None or not t.is_floating_point():
        return t
    return t.to(want_dtype)


_SAFE_HEADER_CACHE = {}


def safetensors_header(path):
    """A safetensors file's header (tensor names included), cached — a big
    shard's header is read once, not once per bound block."""
    p = hf_hub_file(path) if str(path or "").startswith("hf:") else _resolve(path)
    try:
        key = (p, os.path.getmtime(p), os.path.getsize(p))
    except OSError:
        return {}
    head = _SAFE_HEADER_CACHE.get(key)
    if head is None:
        try:
            head, _start = read_safetensors(p)
        except CanvasError:
            head = {}
        _SAFE_HEADER_CACHE.clear()             # one shard at a time is plenty
        _SAFE_HEADER_CACHE[key] = head
    return head


def write_safetensors(path, tensors, metadata=None):
    """Write {name: tensor} as one safetensors file (header + raw bytes)."""
    header = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs, offset = {}, 0
    for key, t in tensors.items():
        t = t.detach().to("cpu").contiguous()
        dt = _SAFE_NAMES.get(t.dtype)
        if dt is None:
            t = t.to(torch.float32)
            dt = "F32"
        if t.dtype in _F8_DTYPES:
            raw = t.view(torch.uint8).numpy().tobytes()
        elif t.dtype != torch.bfloat16:
            raw = t.numpy().tobytes()
        else:
            raw = t.view(torch.uint16).numpy().tobytes()
        header[key] = {"dtype": dt, "shape": list(t.shape),
                       "data_offsets": [offset, offset + len(raw)]}
        blobs[key] = raw
        offset += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for key in tensors:
            f.write(blobs[key])
    return len(hb) + 8 + offset


def _state_dict(obj):
    """The tensors inside whatever torch.load returned: a state_dict key, a
    dict of tensors, a nested one, or the tensor entries of a mixed dict."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    for v in obj.values():
        if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
            return v
    mixed = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    return mixed or None


def load_pt_tensor(path, name, want_dtype=torch.float32):
    """One tensor out of a .pt/.pth/.bin checkpoint (weights_only: plain
    tensors and state dicts, no pickle code runs)."""
    path = _resolve(path)
    if not os.path.isfile(path):
        raise CanvasError(name, "the weight file is not on disk: %s" % (path or "?"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — refuse rather than run pickle code
        raise CanvasError(name, "cannot read %s safely (weights_only): %s — "
                          "re-save it, or export/safetensors instead"
                          % (os.path.basename(path), e))
    state = _state_dict(obj)
    if state is None or name not in state:
        raise CanvasError(name, "tensor not found in %s" % os.path.basename(path))
    t = state[name].detach().clone()
    return t.to(want_dtype) if t.is_floating_point() else t


def hf_hub_file(path):
    """The local file behind an `hf:<repo>/<file>` reference — downloaded once
    (huggingface_hub when present, a plain HTTPS get otherwise), cached under
    ~/.cache/nnstudio/hf/<repo>/."""
    spec = str(path)[len("hf:"):].strip("/")
    repo, _, filename = spec.rpartition("/")       # a repo id itself has a slash
    if not repo or not filename:
        raise CanvasError("", "bad huggingface file reference: %s" % path)
    cache = os.path.join(os.path.expanduser("~"), ".cache", "nnstudio", "hf",
                         repo, filename)
    if os.path.isfile(cache):
        return cache                          # already fetched: no network
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=repo, filename=filename,
                              token=os.environ.get("HF_TOKEN") or None)
        if got and os.path.isfile(got):
            return got
    except Exception:  # noqa: BLE001 — no hub library: fetch it by hand
        pass
    import urllib.request
    req = urllib.request.Request("https://huggingface.co/%s/resolve/main/%s"
                                 % (repo, filename))
    req.add_header("User-Agent", "nnstudio/1.0")   # the CDN likes a caller
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    tmp = cache + ".part"
    try:
        with urllib.request.urlopen(req, timeout=3600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp, cache)
    except Exception as e:  # noqa: BLE001 — the app shows the reason
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise CanvasError("", "could not download %s from huggingface: %s"
                          % (spec, str(e)[:160]))
    return cache


def load_weight_tensor(path, name, want_dtype=torch.float32):
    """Any supported weight file -> one tensor, [out, in] as torch likes it.
    `hf:<repo>/<shard>` fetches the shard first (once). `want_dtype=None`
    keeps the stored dtype where the file has one (fp8, packed uint8)."""
    if str(path or "").startswith("hf:"):
        path = hf_hub_file(path)
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext == ".gguf":
        return load_gguf_tensor(path, name, want_dtype or torch.float32)
    if ext == ".safetensors":
        return load_safetensors_tensor(path, name, want_dtype)
    if ext in (".pt", ".pth", ".bin"):
        return load_pt_tensor(path, name, want_dtype or torch.float32)
    raise CanvasError(name, "unsupported weight file '%s' — gguf, safetensors or pt"
                      % os.path.basename(str(path or "")))


def file_tensors(path):
    """A weight file's metadata: kind, tensor names, shapes, dtypes, sizes.
    This is what the app shows so a block can be pointed at its tensor."""
    path = _resolve(path)
    if not path or not os.path.isfile(path):
        raise CanvasError("", "the weight file is not on disk: %s" % (path or "?"))
    ext = os.path.splitext(path)[1].lower()
    out = {"kind": ext.lstrip("."), "file": path, "metadata": None, "tensors": []}
    if ext == ".gguf":
        if GGUFReader is None:
            raise CanvasError("", "reading GGUF needs the gguf package")
        for t in _reader(path).tensors:
            shape = [int(x) for x in reversed(list(t.shape))]     # logical order
            params = 1
            for d in shape:
                params *= int(d)
            out["tensors"].append({"name": t.name, "shape": shape,
                                   "dtype": t.tensor_type.name,
                                   "params": params,
                                   "bytes": int(getattr(t, "n_bytes", 0) or 0)})
    elif ext == ".safetensors":
        header, _ = read_safetensors(path)
        meta = header.pop("__metadata__", None)
        if isinstance(meta, dict):
            out["metadata"] = meta
        for name, ent in header.items():
            if not isinstance(ent, dict):
                continue
            shape = [int(x) for x in ent.get("shape") or []]
            params = 1
            for d in shape:
                params *= int(d)
            off = ent.get("data_offsets") or [0, 0]
            out["tensors"].append({"name": name, "shape": shape,
                                   "dtype": ent.get("dtype"),
                                   "params": params, "bytes": int(off[1]) - int(off[0])})
    elif ext in (".pt", ".pth", ".bin"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise CanvasError("", "cannot read %s safely (weights_only): %s"
                              % (os.path.basename(path), e))
        state = _state_dict(obj)
        if state is None:
            raise CanvasError("", "no tensors in %s (expected a state dict)"
                              % os.path.basename(path))
        for name, t in state.items():
            out["tensors"].append({"name": name, "shape": [int(x) for x in t.shape],
                                   "dtype": str(t.dtype).replace("torch.", ""),
                                   "params": int(t.numel()),
                                   "bytes": int(t.numel() * t.element_size())})
    else:
        raise CanvasError("", "unsupported weight file '%s' — gguf, safetensors or pt"
                          % os.path.basename(path))
    out["tensors"].sort(key=lambda t: t["name"])
    out["count"] = len(out["tensors"])
    out["bytes"] = sum(t["bytes"] for t in out["tensors"])
    return out


def export_weights(canvas, path, fmt="safetensors"):
    """Every weight box's tensor, saved as one file, and a manifest that names
    which box each key belongs to — so the design can be pointed at it."""
    graph = compile_canvas(canvas, device="cpu")
    graph.eval()
    tensors, manifest, used = {}, [], set()

    def visit(g, prefix):
        for box in g.boxes.values():
            base = prefix + (box.name or box.kind or "box")
            if box.kind == "custom" and box.inner is not None:
                visit(box.inner, base + "/")
                continue
            pairs = []
            if torch.is_tensor(getattr(box, "w", None)):
                pairs.append(("weight", box.w))
            if torch.is_tensor(getattr(box, "bias", None)):
                pairs.append(("bias", box.bias))
            for suffix, tensor in pairs:
                key = base + "." + suffix
                n = 2
                while key in used:
                    key = "%s (%d).%s" % (base, n, suffix)
                    n += 1
                used.add(key)
                t = tensor.detach().to("cpu").contiguous()
                tensors[key] = t
                manifest.append({"id": box.node.get("id"), "key": key,
                                 "name": box.name, "suffix": suffix,
                                 "shape": [int(x) for x in t.shape],
                                 "dtype": str(t.dtype).replace("torch.", ""),
                                 "params": int(t.numel())})

    visit(graph, "")
    if not tensors:
        raise CanvasError("", "the design has no weights to export yet")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if fmt == "pt":
        torch.save(tensors, path)
        size = os.path.getsize(path)
    else:
        size = write_safetensors(path, tensors,
                                 metadata={"app": "nnstudio", "boxes": len(manifest)})
    return {"file": path, "format": "pt" if fmt == "pt" else "safetensors",
            "bytes": size, "tensors": manifest, "keys": len(manifest)}


# ---------------------------------------------------------------- caches
def offload():
    """Let go of every model the engine is holding — cached GGUF readers,
    loaded tokenizers — then collect and hand the pages back. The next use
    simply loads again."""
    with _READER_LOCK:
        _READERS.clear()
    with _TEXT_TOKENS_LOCK:
        _TEXT_TOKENS.clear()
    with _HF_LOCK:
        _HF_MODELS.clear()                        # whole transformer models too
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)   # glibc: give freed arenas back
    except Exception:  # noqa: BLE001 — not glibc, not fatal
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001 — a machine without torch still frees the caches
        pass
    return True


def parse_dims(text):
    """'128x40', '128 40', '128,40', '-1 8 4' -> [128, 40]; empty/none -> [].
    A `-1` is kept: torch's reshape infers that one dimension."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        out = []
        for x in text:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                return []
        return out
    return [int(p) for p in re.findall(r"-?\d+", str(text))]


# ---------------------------------------------------------------- stats
def _stat(v):
    """A rounded number for the card — None for NaN/±inf, which are not JSON:
    Python writes them as Infinity, and the browser's JSON.parse stops dead on
    the minus sign ("No number after minus sign in JSON")."""
    v = float(v)
    return round(v, 4) if math.isfinite(v) else None


# torch's storage types, in the picker's own short names (ui/index.html)
_DTYPE_SHORT = {
    "float64": "f64", "float32": "f32", "bfloat16": "bf16", "float16": "f16",
    "float8_e4m3fn": "f8_e4m3", "float8_e5m2": "f8_e5m2",
    "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8",
    "uint8": "u8", "bool": "bool",
}


def dtype_short(dt):
    """A torch dtype in the app's short names: f32, i64 … (anything else as-is,
    so a new torch type still reads)."""
    s = str(dt).replace("torch.", "")
    return _DTYPE_SHORT.get(s, s)


def tensor_stats(t, peek=8):
    """The little summary a card pill knows: shape, dtype, three numbers, a peek."""
    if t is None:
        return None
    if isinstance(t, dict):                     # a nested box's port values
        t = next((v for v in t.values() if v is not None), None)
        if t is None:
            return None
    if not torch.is_tensor(t):
        return None                             # a de-tokenizer's text has no stats
    dt = dtype_short(t.dtype)                   # the value's own type, before the cast
    with torch.no_grad():
        x = t.detach().float()
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return {"shape": list(x.shape), "dtype": dt, "mean": None, "min": None,
                    "max": None, "scalar": None, "peek": []}
        return {
            "shape": list(x.shape),
            "dtype": dt,
            "mean": _stat(x.mean()),
            "min": _stat(x.min()),
            "max": _stat(x.max()),
            "scalar": _stat(flat[0]) if flat.numel() == 1 else None,
            "peek": [_stat(v) for v in flat[:peek].tolist()],
        }


def image_story(canvas, collected):
    """Which of the run's values are pictures: {id(tensor): {w, h, blob, name}},
    keyed by the tensor the run itself produced — a wire that passed an image
    on unchanged hands over the same object, so an output box can show the
    picture it received. Image inputs are matched by their path, nested levels
    included."""
    out = {}

    def walk(nodes, prefix):
        for n in (nodes or []):
            nid = n.get("id") or ""
            path = prefix + nid
            if n.get("kind") == "custom":
                walk((n.get("inner") or {}).get("nodes"), path + "/")
                continue
            if n.get("kind") != "input" or n.get("input_mode") != "image":
                continue
            v = (collected or {}).get(path)
            if not torch.is_tensor(v) or v.dim() != 3 or int(v.shape[-1]) != 3:
                continue
            out[id(v)] = {"w": int(v.shape[1]), "h": int(v.shape[0]),
                          "blob": n.get("image") or "",
                          "name": n.get("name") or "image"}

    walk((canvas or {}).get("nodes"), "")
    return out


def _thumb(x, max_side=96):
    """A small RGB copy of a picture, base64 — what the app draws when the
    value is a matrix the backend *determined* to be an image (no blob of its
    own). Downscaled so a result never carries a full picture."""
    h, w = int(x.shape[0]), int(x.shape[1])
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        x = F.interpolate(x.permute(2, 0, 1)[None], size=(nh, nw),
                          mode="bilinear", antialias=True)[0].permute(1, 2, 0)
        h, w = nh, nw
    px = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().cpu().numpy()
    return {"w": w, "h": h, "rgb": base64.b64encode(px.tobytes()).decode()}


def image_view(v):
    """Does this value read as a picture? An RGB matrix in 0..1: [h, w, 3], or
    [n, 3] with n a square (a pixel list). Answers {w, h, thumb} — the data
    drawn as it is — or None when the matrix is something else."""
    if not torch.is_tensor(v) or not v.is_floating_point():
        return None
    x = v.detach().float()
    if x.dim() == 3 and int(x.shape[-1]) == 3:
        h, w = int(x.shape[0]), int(x.shape[1])
    elif x.dim() == 2 and int(x.shape[-1]) == 3:
        n = int(x.shape[0])
        side = int(math.isqrt(n))
        if side * side != n:
            return None                          # not a grid: not a picture
        h = w = side
        x = x.reshape(h, w, 3)
    else:
        return None
    if h < 2 or w < 2 or h * w < 4:
        return None                              # a 2x2 grid is the smallest picture
    lo, hi = float(x.min()), float(x.max())
    if lo < -0.02 or hi > 1.02:
        return None                              # not pixel values
    return {"w": w, "h": h, "blob": "", "thumb": _thumb(x)}


_HEAT_SIGMAS = 2.0                               # the ramp covers +/- 2 robust sigmas
_HEAT_FLOOR_ULPS = 32.0                          # …and never scales below 32 ulps
_HEAT_STOPS = ((0.00, 0.03, 0.02, 0.22),     # deep blue
               (0.25, 0.05, 0.52, 0.76),     # cyan
               (0.50, 0.10, 0.72, 0.34),     # green
               (0.75, 0.95, 0.84, 0.16),     # yellow
               (1.00, 0.82, 0.10, 0.10))     # red


def _heat_grid(n):
    """(rows, cols, pad) for n values: the exact factor pair closest to square
    when one exists, a near-square grid with a padded tail otherwise (a prime
    has no pair, and a picture wants at least 2x2)."""
    r = int(math.isqrt(n))
    while r > 1 and n % r:
        r -= 1
    rows, cols = n // r, r
    if rows >= 2 and cols >= 2:
        return rows, cols, 0
    cols = max(2, int(math.ceil(math.sqrt(n))))
    rows = max(2, int(math.ceil(n / float(cols))))
    return rows, cols, rows * cols - n


def _heatmap(x, box=""):
    """A vector drawn as one picture: the values flattened and mapped through
    a five-stop colour ramp, centred on their median and spread by their
    robust deviation (1.4826 x the median absolute deviation — the ramp then
    covers +/- two of those). One value moving moves one cell, never the whole
    picture, and a far outlier clips to the ramp's end instead of squashing
    everything else flat. Returns [h, w, 3] in 0..1 — a padded tail reads as
    neutral gray, and a (near-)constant vector as one mid colour."""
    x = torch.nan_to_num(x.float()).reshape(-1)
    n = int(x.numel())
    if n == 0:
        raise CanvasError(box or "heatmap", "the vector is empty - nothing to draw")
    rows, cols, _pad = _heat_grid(n)
    c = x.median()
    spread = float((x - c).abs().median()) * 1.4826
    # the scale is floored a few dozen ulps under the vector's own magnitude:
    # float dust reads flat instead of stretching into a rainbow
    anchor = float(x.abs().max())
    floor = _HEAT_FLOOR_ULPS * float(torch.finfo(x.dtype).eps) * anchor
    scale = _HEAT_SIGMAS * max(spread, floor)
    if scale <= 0.0:
        v = x.new_full((n,), 0.5)                # the all-zero vector: one flat colour
    else:
        v = (((x - c) / scale).clamp(-1.0, 1.0) + 1.0) * 0.5
    pos = v * (len(_HEAT_STOPS) - 1)
    idx = pos.floor().clamp(0, len(_HEAT_STOPS) - 2).long()
    t = (pos - idx.to(pos.dtype)).unsqueeze(-1)
    rgb = torch.tensor([s[1:] for s in _HEAT_STOPS], device=x.device, dtype=x.dtype)
    col = rgb[idx] + (rgb[idx + 1] - rgb[idx]) * t
    img = x.new_full((rows * cols, 3), 0.16)   # the padded cells: neutral gray
    img[:n] = col
    return img.reshape(rows, cols, 3)


def view_paths(canvas):
    """Every viewable node's path — an output box, or a box that draws a
    picture of its own (the heatmap) — nested levels included."""
    out = set()

    def walk(nodes, prefix):
        for n in (nodes or []):
            path = prefix + (n.get("id") or "")
            if n.get("kind") == "custom":
                walk((n.get("inner") or {}).get("nodes"), path + "/")
                continue
            if n.get("kind") in ("output", "heatmap"):
                out.add(path)

    walk((canvas or {}).get("nodes"), "")
    return out


def _stats_with_image(v, story, view=False):
    """One value's stats, with the picture attached when it is one: the marked
    story (an image input's own, blob and all) always; a matrix the backend
    determines to be a picture only where it will be shown (`view`), as a
    small thumbnail."""
    st = tensor_stats(v)
    if st is None:
        return None
    meta = story.get(id(v)) if story else None
    if meta is None and view:
        meta = image_view(v)
    if meta:
        st["image"] = meta
    return st


# ---------------------------------------------------------------- boxes
# What a box's own weights live in: its declared dtype when torch knows it,
# anything quantised (dequantised at load) in f16, else the compile default.
# Computation still happens in f32 — only the stored tensors are cast.
_DTYPE_NAMES = {
    "f64": torch.float64, "f32": torch.float32, "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f8_e4m3": getattr(torch, "float8_e4m3fn", None),
    "f8_e5m2": getattr(torch, "float8_e5m2", None),
    "i64": torch.int64, "i32": torch.int32, "i16": torch.int16,
    "i8": torch.int8, "u8": torch.uint8, "bool": torch.bool,
}
_DTYPE_NAMES = {k: v for k, v in _DTYPE_NAMES.items() if v is not None}


def param_grad(dtype, frozen, train=False):
    """Only float tensors can ask for gradients — ints and bools never do.
    And only in a training graph: an inference run never backprops, so its
    weights ask for nothing and hold nothing."""
    return train and (not frozen) and (dtype.is_floating_point or dtype.is_complex)


def _rand(shape, dtype, gen, uniform=None, mean=0.0, std=0.02):
    """A random tensor of any storage dtype: init in f32, then cast (the
    `uniform_`/`normal_` fills are float-only)."""
    t = torch.empty(shape, dtype=torch.float32)
    if uniform is None:
        t.normal_(mean, std, generator=gen)
    else:
        t.uniform_(uniform[0], uniform[1], generator=gen)
    return t.to(dtype)


def box_dtype(node, fallback=torch.float32):
    name = str((node or {}).get("dtype") or "").strip().lower()
    if name in _DTYPE_NAMES:
        return _DTYPE_NAMES[name]
    if name:                                    # q4_k, iq4_xs, mixed, ...
        return torch.float16
    return fallback


_DTYPE_BYTES = {"f64": 8, "f32": 4, "f16": 2, "bf16": 2,
                "f8_e4m3": 1, "f8_e5m2": 1,
                "i64": 8, "i32": 4, "i16": 2, "i8": 1, "u8": 1, "bool": 1}


def estimate_node(node, native=False):
    """(params, bytes) one box will hold — from the JSON alone, no tensor
    touched. A bound box counts the bytes the file actually stores when the
    engine keeps that storage: packed/fp8 tensors, or a gguf quant with a
    native kernel wherever the box runs (a gpu fuses the dequant into the
    matmul; off it, the forward dequantizes per pass, still packed at rest).
    Everything else — a gguf quant without a kernel, and every embedding
    table — counts the f16 copy it dequantizes to. Unset dtypes take f32."""
    k = (node or {}).get("kind")
    params = None
    if k == "layer":
        params = (node.get("din") or 0) * (node.get("dout") or 0)
    elif k == "embed":
        params = (node.get("vocab") or 0) * (node.get("dout") or 0)
    elif k == "norm":
        params = node.get("size") or 0
    elif k == "op" and node.get("op") == "scale":
        params = 1
    if not params:
        params = int(node.get("params") or 0)
    name = str(node.get("dtype") or "").strip().lower()
    stored = int(node.get("wbytes") or 0)
    keep = ((k != "embed" and name in _DTYPE_BYTES)
            or (native and nn_quant is not None
                and nn_quant.can(node.get("dtype"))))
    if stored and keep:
        # bound and stored as-is: the file's own bytes are the footprint
        return params, stored
    if not params:
        return 0, 0
    per = _DTYPE_BYTES.get(name)
    if per is None:
        per = 2 if node.get("dtype") else 4
    return params, params * per


def estimate_canvas(canvas, default_device="cpu"):
    """What compiling a design would hold, per device, from the JSON alone —
    the backend refuses an impossible load before spending minutes on it.
    Nested boxes inherit the nearest `place`; unset boxes take `default_device`.
    A gguf layer whose type has a native kernel stays packed wherever it runs;
    everything that dequantizes counts its f16 copy."""
    base = "system" if str(default_device) in ("", "cpu", "system", "none") else "gpu"
    native_ok = bool(nn_quant is not None and nn_quant.gpu_ok())
    out = {"params": 0, "bytes": 0, "gpu_bytes": 0, "system_bytes": 0, "boxes": 0,
           "shared": 0}
    seen = set()                              # one load per namespace, not per box
    refs = {}                                 # namespace -> boxes that bind it

    def walk(nodes, where):
        for n in nodes or []:
            if n.get("kind") == "custom":
                w = where
                if n.get("place") == "system":
                    w = "system"
                elif n.get("place") == "gpu":
                    w = "gpu"
                walk((n.get("inner") or {}).get("nodes"), w)
                continue
            if n.get("place") == "system":
                w = "system"
            elif n.get("place") == "gpu":
                w = "gpu"
            else:
                w = where
            params, b = estimate_node(n, native=native_ok)
            if not b:
                continue
            ns = (n.get("file"), str(n.get("t") or "").split("::")[-1])
            if n.get("t"):
                refs[ns] = refs.get(ns, 0) + 1
                if ns in seen:                # a second box on one tensor:
                    continue                  # one copy, counted once
                seen.add(ns)
            out["params"] += params
            out["bytes"] += b
            out["boxes"] += 1
            out[w + "_bytes"] += b

    walk((canvas or {}).get("nodes"), base)
    out["shared"] = sum(1 for c in refs.values() if c > 1)
    return out


class Box(nn.Module):
    """One canvas box. Weight-bearing kinds own parameters; a nested box owns
    a Graph. `frozen`/`off` are inherited from the enclosing levels; `train`
    says the graph will backprop — only then do its weights ask for grads."""

    def __init__(self, node, seed=0, parent_path="", frozen=False, off=False,
                 loader=None, dtype=torch.float32, gguf_file=None,
                 device=None, on_box=None, native=True, train=False, store=None):
        super().__init__()
        self.node = node
        self.kind = node.get("kind")
        self.name = node.get("name") or self.kind
        self.off = off or bool(node.get("off"))
        self.frozen = frozen or bool(node.get("frozen"))
        self.native = bool(native)      # frozen gguf layers may keep raw blocks
        self.loose = bool(node.get("loose"))
        self.place = node.get("place") if node.get("place") in ("gpu", "system") else None
        # where this box calls home: its own `place`, else the level's device
        want = device or "cpu"
        if self.place == "system":
            want = "cpu"
        elif self.place == "gpu":
            want = device or "cpu"
        self.device = str(want)                 # where it runs
        self.inner = None
        self.w = None
        self.w_blocks = None                    # native gguf blocks at rest
        self.w_quant = None                     # {"qname", "shape"} for them
        self.bias = None
        self.w_scale = None                     # fp4 block scales / fp8 channel scales
        self.w_global = None                    # compressed-tensors' stored divisor
        self.w_packed = False                   # two E2M1 nibbles per stored byte
        self.extras = nn.ParameterList()        # sibling tensors (dt_bias, …)
        self.last = None
        gen = torch.Generator(device="cpu").manual_seed(_seed_for(seed, node.get("id") or self.name))
        dtype = box_dtype(node, dtype)          # its weights, its own dtype
        grad = param_grad(dtype, self.frozen, train)   # ints/bools (and inference) never ask

        if self.kind == "layer":
            din, dout = node.get("din"), node.get("dout")
            if din is None or dout is None:
                raise CanvasError(self.name, "needs both an in and an out size")
            bound = loader(node) if loader else None
            if bound is not None:
                if bound.get("blocks") is not None:
                    # blocks are the weight (native quant): shared through the
                    # store, already on the box's device
                    self.w_blocks = bound["blocks"]
                    self.w_quant = bound["quant"]
                else:
                    self.w = bound["weight"]        # shared Parameter, final
                    self.w_packed = bool(bound.get("packed"))
                    if bound.get("scale") is not None:
                        self.w_scale = bound["scale"]
                    if bound.get("global_scale") is not None:
                        self.w_global = float(bound["global_scale"])
                if bound.get("bias") is not None:
                    self.bias = bound["bias"]
            else:
                k = 1.0 / math.sqrt(max(1, int(din)))
                self.w = nn.Parameter(
                    _rand((int(dout), int(din)), dtype, gen,
                          uniform=(-math.sqrt(3) * k, math.sqrt(3) * k)),
                    requires_grad=grad)
        elif self.kind == "embed":
            vocab, dout = node.get("vocab"), node.get("dout")
            if vocab is None or dout is None:
                raise CanvasError(self.name, "needs the table's vocab and its out size")
            bound = loader(node) if loader else None
            if bound is not None and bound.get("blocks") is not None:
                # the table stays packed (shared): the forward gathers and
                # dequantizes only the rows the batch asks for
                self.w_blocks = bound["blocks"]
                self.w_quant = bound["quant"]
            elif bound is not None:
                self.w = bound["weight"]
            else:
                self.w = nn.Parameter(_rand((int(vocab), int(dout)), dtype, gen),
                                      requires_grad=grad)
        elif self.kind == "norm":
            bound = loader(node) if loader else None
            if bound is not None:
                self.w = bound["weight"]          # shared, already (-1,) at dtype
            elif node.get("size") is not None:
                self.w = nn.Parameter(torch.ones(int(node["size"]), dtype=dtype),
                                      requires_grad=grad)
            else:
                hint = ("the model's file is not set — reload the GGUF into its box"
                        if node.get("t") else "wire it or bind its tensor")
                raise CanvasError(self.name, "its size is not known — " + hint)
        elif self.kind == "op" and node.get("op") == "scale":
            bound = loader(node) if loader else None
            w = bound["weight"].reshape(-1)[:1] if bound is not None else torch.ones(1)
            self.w = nn.Parameter(w.to(dtype), requires_grad=grad)
        elif self.kind == "op" and node.get("op") in ("conv1d", "scan"):
            # a real kernel: the short causal conv / the gated delta rule, with
            # its sibling tensors (A_log, dt_bias) riding along — all shared
            bound = loader(node) if loader else None
            if bound is not None:
                self.w = bound["weight"]
                rides = list(bound.get("extras") or [])
                if bound.get("bias") is not None:
                    rides.append(bound["bias"])      # the scan's dt_bias rides here
                if rides:
                    self.extras = nn.ParameterList(rides)
        elif self.kind == "custom":
            self.inner = Graph(node.get("inner") or {"nodes": [], "edges": []}, seed=seed,
                               path=parent_path + "/" + (node.get("id") or ""),
                               frozen=self.frozen, off=self.off, dtype=dtype,
                               gguf_file=node.get("file") or gguf_file,   # the file reads through
                               device=want, on_box=on_box, native=self.native,
                               train=train, store=store)
        elif self.kind == "vision":
            self.tower = VisionTower(node, train=train, device=want)
        if self.kind != "custom":
            self.to(want)                       # its device, right as it is built: a
                                                # gpu-bound weight never stages on the cpu

    def __call__(self, *xs):
        try:                                     # its own forward's errors carry its id
            return super().__call__(*xs)
        except CanvasError as e:
            if e.nid is None:
                e.nid = self.node.get("id")
            raise

    def forward(self, *xs):
        if self.off or self.kind in ("agent", "train"):
            self.last = None               # a config box runs no maths of its own
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
            if self.w_blocks is not None:
                # native gguf blocks at rest: the dequant is fused into the
                # matmul. Offloaded to system RAM — or a graph that must carry
                # gradients through this box — dequantize host-side instead.
                if (self.w_blocks.device.type == "cuda"
                        and not (torch.is_grad_enabled() and xs[0].requires_grad)):
                    out = nn_quant.linear(xs[0].float(), self.w_blocks, self.w_quant)
                else:
                    out = _linear(xs[0].float(),
                                  nn_quant.dequant_host(self.w_blocks, self.w_quant),
                                  None)
                if self.bias is not None:
                    out = out + self.bias.float()
            else:
                # stored at its own dtype (packed/fp8 included), computed in f32
                # without doubling it: one output slice dequantized at a time
                out = _linear(xs[0].float(), self.w, self.bias,
                              scale=self.w_scale, gscale=self.w_global,
                              packed=self.w_packed)
        elif k == "embed":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            if self.w_blocks is not None:
                out = nn_quant.embed(xs[0].long(), self.w_blocks, self.w_quant)
            else:
                out = F.embedding(xs[0].long(), self.w).float()
            # a scaled word embedding (Gemma): the table times sqrt(d_model)
            scale = self.node.get("embed_scale")
            if scale is not None:
                out = out * float(scale)
        elif k == "dataset":
            # its labels ride to the loss box; a training run sets `batch`
            out = getattr(self, "batch", None)
            if out is None:
                raise CanvasError(self.name, "no rows were given to train on — "
                                             "it carries the labels a loss box reads")
        elif k == "loss":
            if len(xs) < 2:
                raise CanvasError(self.name, "wants the logits and the dataset's labels")
            if xs[1].dim() == 1:
                # 1-D labels are class indices, one per sample: the dataset
                # left-pads X, so the class is picked at the sequence's end
                lg = xs[0].float()
                last = lg[:, -1, :] if lg.dim() == 3 else \
                    lg.reshape(-1, lg.shape[-1])[-xs[1].shape[0]:]
                out = F.cross_entropy(last, xs[1].long())
                with torch.no_grad():
                    self.last_acc = (last.argmax(-1) == xs[1].long()).float().mean()
            else:
                out = _cross_entropy(xs[0], xs[1])
        elif k == "logprobs":
            if len(xs) < 2:
                raise CanvasError(self.name, "wants the logits and the picked ids")
            out = _gather_logprobs(xs[0], xs[1])
        elif k == "norm":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            x = xs[0].float()
            if self.node.get("norm") == "layer":
                out = F.layer_norm(x, (x.shape[-1],), self.w.float(), None,
                                   float(self.node.get("eps") or 1e-5))
            else:
                w = self.w.float()
                size = int(w.numel())
                xr = x
                if size and x.shape[-1] != size and x.shape[-1] % size == 0:
                    xr = x.reshape(*x.shape[:-1], -1, size)   # per-head groups
                v = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) +
                                     float(self.node.get("eps") or 1e-6))
                if self.node.get("zc"):
                    # zero-centered: the weight is the *delta* from 1, so the
                    # stored zero means "the identity" (Qwen3_5's norms)
                    out = v * (1.0 + w)
                else:
                    out = v * w
                if xr is not x:
                    out = out.reshape(*x.shape)
        elif k == "act":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            x = xs[0]
            gain = self.node.get("gain")
            if gain is not None:          # e.g. attention scores: 1/sqrt(d)
                x = x * float(gain)
            if self.node.get("causal"):   # attention scores: the future is masked
                n = int(x.shape[-1])
                tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
                x = x.masked_fill(tri, float("-inf"))
            out = _activate(self.node.get("act") or "none", x, self.name)
        elif k == "op":
            out = _operate(self, xs)
        elif k == "custom":
            if self.inner is None:
                raise CanvasError(self.name, "the nested box has no inside")
            vals = self.inner(*xs, on_node=getattr(self, "_on_node", None))
            out = {i: vals.get(oid) for i, oid in enumerate(self.inner.output_ids)}
            self.last = out
            return out
        elif k == "detokenize":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            if not str(self.node.get("vocab") or "") and not str(self.node.get("repo") or ""):
                raise CanvasError(self.name, "its vocabulary is empty (and no huggingface repo)")
            out = detokenize_ids(self.node, xs[0])
        elif k == "sample":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = sample_one(xs[0], self.node)
        elif k == "hf":
            src = xs[0] if xs else None
            if src is None:
                src = self.node.get("text") or ""   # its own prompt, when no wire
            out = hf_next_token(self.node, src, self.device)
        elif k == "heatmap":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = _heatmap(xs[0], self.name)
        elif k == "vision":
            if not xs:
                raise CanvasError(self.name, "an input dot is not wired")
            out = self.tower(xs[0])
        else:
            raise CanvasError(self.name, "unknown box kind '%s'" % k)
        self.last = out
        return out


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _dequant(wi, scale, gscale, packed):
    """One output-row slice, dequantized to f32: packed NVFP4 is E2M1 nibbles
    (low first) x the per-16 block scale / compressed-tensors' stored global
    divisor; fp8 is x its per-channel scale."""
    if packed:
        lo = (wi & 0x0F).to(torch.int64)
        hi = ((wi >> 4) & 0x0F).to(torch.int64)
        code = torch.stack([lo, hi], dim=-1).reshape(wi.shape[0], -1)
        w = _E2M1.to(wi.device)[code]
        if scale is not None:
            sc = scale.to(torch.float32)
            if sc.shape[-1] not in (1, w.shape[-1]):
                sc = sc.repeat_interleave(max(1, w.shape[-1] // int(sc.shape[-1])),
                                          dim=-1)
            w = w * sc
        if gscale:
            w = w / float(gscale)
        return w
    w = wi.to(torch.float32)
    if scale is not None:
        w = w * scale.to(torch.float32)
    return w


def _linear(x, w, bias=None, tile=1 << 23, scale=None, gscale=None, packed=False):
    """`F.linear` with the weight dequantized/upsized to f32 a slice at a
    time: the maths stays f32 without ever holding a second full copy of the
    weight (a 50k² f16 weight would otherwise need +9.3 GB for its f32 twin;
    a packed 4-bit weight twice its size in f32)."""
    quant = packed or w.dtype in _F8_DTYPES
    if w.dtype == torch.float32 and not quant:
        return F.linear(x, w, bias)
    cols = max(1, int(w.shape[1]))
    rows = max(1, int(tile) // cols)
    if rows >= int(w.shape[0]) and not quant:
        b = None if bias is None else bias.to(torch.float32)
        return F.linear(x, w.to(torch.float32), b)
    outs = []
    for i in range(0, int(w.shape[0]), rows):
        wi = _dequant(w[i:i + rows],
                      None if scale is None else scale[i:i + rows],
                      gscale, packed)
        bi = None if bias is None else bias[i:i + rows].to(torch.float32)
        outs.append(F.linear(x, wi, bi))
    return torch.cat(outs, dim=-1)


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
    if name in ("gelu_pytorch_tanh", "gelu_new", "gelu tanh", "gelu approx tanh"):
        # transformers' GELUActivation(approximate="tanh") — Gemma's MLP
        return F.gelu(x, approximate="tanh")
    if name == "silu":
        return F.silu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "softmax":
        return F.softmax(x.float(), dim=-1).to(x.dtype)
    raise CanvasError(box, "unknown activation '%s'" % name)


def _causal_conv1d(box, a):
    """Depthwise causal convolution over time — Qwen3.5's short conv: left pad
    by k-1, one grouped conv, truncated back to the input length. The input is
    time-major, the way every box speaks: [T, C], or [B, T, C]."""
    if box.w is None:
        raise CanvasError(box.name, "its kernel is not bound — reload the model")
    w = box.w.float()
    if w.dim() == 2:                             # [channels, k]
        w = w.unsqueeze(1)
    if w.dim() != 3 or a.dim() not in (2, 3):
        raise CanvasError(box.name, "wants [T, C] (or a batch) and a [C, 1, k] kernel")
    k = int(w.shape[-1])
    channels = int(w.shape[0])
    x = a.float()
    batchless = x.dim() == 2
    if batchless:
        x = x.unsqueeze(0)
    if x.shape[-1] == channels:                  # time-major: [B, T, C]
        time_major = True
        x = x.transpose(1, 2)                    # [B, C, T] for the kernel
    elif x.shape[1] == channels:                 # legacy [B, C, T] / [C, T]
        time_major = False
    else:
        raise CanvasError(box.name, "its input is %d wide, not %d channels"
                          % (x.shape[-1], channels))
    t = int(x.shape[1])
    y = F.conv1d(F.pad(x, (k - 1, 0)), w, groups=channels)[..., :t]
    if time_major:
        y = y.transpose(1, 2)                    # back to [B, T, C]
    if batchless:
        y = y.squeeze(0)
    # a separate act box arrives as its own node; a bare conv is silu, the
    # library's short-conv activation
    return _activate(box.node.get("act") or "silu", y, box.name)


def _gated_delta_rule(box, xs):
    """Qwen3.5's linear attention, token by token — the same math as
    transformers' `torch_recurrent_gated_delta_rule`: a decayed recurrent
    state, the delta update with beta, and the output read with q (both q and
    k l2-normalized, q scaled by 1/sqrt(head_k_dim))."""
    node = box.node
    if len(xs) < 3 or box.w is None:
        if len(getattr(box, "extras", [])) > 1:
            raise CanvasError(box.name, "this design was built before the alpha/beta "
                                        "inputs existed \u2014 reload the GGUF into its box "
                                        "(Load GGUF \u2026) and run again")
        raise CanvasError(box.name, "wants the convolved qkv, b and a in, "
                                    "with A_log and dt_bias bound")
    conv, b_in, a_in = xs[0].float(), xs[1].float(), xs[2].float()
    batchless = conv.dim() == 2                       # the canvas runs [T, …]
    if batchless:
        conv, b_in, a_in = (x.unsqueeze(0) for x in (conv, b_in, a_in))
    try:
        nk, dk = int(node["k_heads"]), int(node["k_dim"])
        nv, dv = int(node["heads"]), int(node["head_dim"])
    except (KeyError, TypeError, ValueError):
        raise CanvasError(box.name, "needs k_heads/k_dim/heads/head_dim — this canvas "
                                    "was built by an older spec; load the repo into "
                                    "the box again to refresh it")
    key_dim, value_dim = nk * dk, nv * dv
    if conv.shape[-1] != key_dim * 2 + value_dim:
        raise CanvasError(box.name, "its input is %d wide, not %d key + %d value"
                          % (conv.shape[-1], key_dim * 2, value_dim))
    batch, seq, _ = conv.shape
    q = conv[..., :key_dim].reshape(batch, seq, nk, dk)
    k = conv[..., key_dim:2 * key_dim].reshape(batch, seq, nk, dk)
    v = conv[..., 2 * key_dim:].reshape(batch, seq, nv, dv)
    if not len(box.extras):
        raise CanvasError(box.name, "its dt_bias is not bound")
    dt = box.extras[0].float().reshape(-1)
    if nk <= 0 or nv % nk:
        raise CanvasError(box.name, "%d value heads do not divide %d key heads" % (nv, nk))
    beta = torch.sigmoid(b_in)
    # `box.w` is the decay scale, and its meaning depends on where it came
    # from: an HF checkpoint binds the raw `A_log`, so the decay log is
    # `-exp(A_log) * softplus(a + dt)` (the spec marks that binding
    # `a_log`); a GGUF binds llama.cpp's `ssm_a`, which already holds
    # `-exp(A_log)` and is multiplied in as-is.
    a_w = box.w.float().reshape(-1)
    if box.node.get("a_log"):
        g = -a_w.exp() * F.softplus(a_in + dt)
    else:
        g = a_w * F.softplus(a_in + dt)
    rep = nv // nk
    if rep > 1:
        # a GGUF stores the V heads tiled (llama.cpp's converter reorders
        # them so its broadcast repeat lines up) — repeat q/k tiled too; an
        # HF checkpoint stores them grouped, matching repeat_interleave
        if box.node.get("v_tiled"):
            q = q.repeat(1, 1, rep, 1)
            k = k.repeat(1, 1, rep, 1)
        else:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))     # [B, heads, T, dim]
    q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q / (dk ** 0.5)
    decay = g.transpose(1, 2).exp()                      # [B, heads, T]
    beta = beta.transpose(1, 2)
    state = torch.zeros(batch, nv, dk, dv, dtype=torch.float32, device=conv.device)
    out = torch.empty_like(v)
    for t in range(seq):
        k_t, q_t, v_t = k[:, :, t], q[:, :, t], v[:, :, t]
        state = state * decay[:, :, t].unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous()               # [B, T, heads, dim]
    return out.squeeze(0) if batchless else out


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
        # every input is an addend: a residual add may take the live branch's
        # output and the zero its elided twin would have contributed
        live = [x for x in xs if x is not None]
        out = live[0]
        for x in live[1:]:
            out = out + x
        return out
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
        keep = int(box.node.get("keep") or 0)    # leading dims kept as they come
        if keep and a.dim() >= keep:
            dims = list(a.shape[:keep]) + dims
        if not dims:
            return a
        known = 1
        for d in dims:
            if d > 0:
                known *= d
        if all(d > 0 for d in dims) and a.numel() != known:
            raise CanvasError(box.name, "reshape: %d values cannot become %s"
                              % (a.numel(), "×".join(str(d) for d in dims)))
        if any(d == -1 for d in dims) and known and a.numel() % known:
            raise CanvasError(box.name, "reshape: %d values cannot become %s"
                              % (a.numel(), "×".join(str(d) for d in dims)))
        return a.reshape(dims)
    if op == "permute":
        axes = parse_dims(box.node.get("axes"))
        if not axes:
            return a
        if sorted(axes) != list(range(a.dim())) and \
                sorted(x % a.dim() for x in axes) != list(range(a.dim())):
            raise CanvasError(box.name, "permute: %r is not a permutation of %d dimensions"
                              % (axes, a.dim()))
        return a.permute(*axes)
    if op == "repeat":
        repeats = parse_dims(box.node.get("repeats"))
        n = repeats[0] if repeats else 1
        dim = int(box.node.get("dim") or 0)
        return a.repeat_interleave(max(1, n), dim=dim)
    if op == "mask":
        # what a position may attend to: everything before it (`causal`), or
        # the last `window` keys of that (`causal` + window) — fill -inf, add
        n = a.shape[-1]
        row = torch.arange(n, device=a.device).unsqueeze(1)      # query position
        col = torch.arange(n, device=a.device).unsqueeze(0)      # key position
        keep = col <= row
        win = int(box.node.get("window") or 0)
        if win > 0:
            keep = keep & (col > row - win)
        floor = torch.finfo(a.dtype).min if a.is_floating_point() else -1e9
        return a.masked_fill(~keep, floor)
    if op == "scale":
        factor = box.node.get("factor")
        if factor is not None:
            return a.float() * float(factor)      # 1/sqrt(head_dim), typically
        return a.float() * box.w.float()
    if op == "rope":
        return _rope(a, float(box.node.get("theta") or 10000.0),
                     box.node.get("dim"), box.node.get("partial"))
    if op == "slice":
        # a contiguous selection: `index` picks one slot along `axis` (the
        # dimension disappears), else `start`/`stop` keep a range of it
        axis = int(box.node.get("axis") or 0)
        idx = box.node.get("index")
        if idx is not None:
            return a.select(axis, int(idx))
        n = a.shape[axis]
        start = int(box.node.get("start") or 0)
        stop = box.node.get("stop")
        stop = n if stop is None else int(stop)
        if stop < 0:
            stop += n
        return a.narrow(axis, start, max(0, stop - start))
    if op == "zeros":
        # a branch that is elided for this layer still has a live consumer;
        # a real zero of the right shape keeps that sum exact
        return torch.zeros_like(a) if a is not None else None
    if op == "topk":
        k = 1
        m = re.search(r"(\d+)", str(box.name))
        if m:
            k = int(m.group(1))
        vals, idx = torch.topk(a.float(), min(k, a.shape[-1]), dim=-1)
        out = torch.zeros_like(a.float())
        out.scatter_(-1, idx, vals)
        return out.to(a.dtype)
    if op == "conv1d":
        return _causal_conv1d(box, a)
    if op == "scan":
        return _gated_delta_rule(box, xs)
    raise CanvasError(box.name, "unknown operation '%s'" % op)


def _rope(x, theta, rotary_dim=None, partial=None):
    """Rotate half-split pairs over the last dim; positions run along the
    last-but-one dim (a lone vector sits at position 0). `rotary_dim` rotates
    only the leading part of the last dim and passes the rest through, the way
    a partial-rotary family (Qwen3_5) keeps a non-rotated tail.

    `partial` is transformers' *proportional* RoPE (Gemma 4's full-attention
    layers): the whole head dim rotates, but only the leading
    `int(partial * head_dim / 2)` angle pairs carry a frequency — the rest have
    `inv_freq == 0`, which leaves those pairs at identity. The denominator is
    the full head dim, unlike the plain partial-RoPE branch above."""
    d = x.shape[-1]
    part = None
    try:
        if partial is not None:
            part = float(partial)
    except (TypeError, ValueError):
        part = None
    proportional = part is not None and 0.0 < part < 1.0
    if rotary_dim is None and not proportional:
        if d % 2 or d < 2:
            return x
        rot = d
    else:
        rot = d if rotary_dim is None else max(2, min(d, int(rotary_dim)))
    xr, xp = x[..., :rot], x[..., rot:]
    positions = torch.arange(x.shape[-2], device=x.device) if x.dim() >= 2 \
        else torch.zeros(1, device=x.device)
    half = rot // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device).float() * 2.0 / rot)
    if proportional:
        # the library's inv_freq uses head_dim, not the (whole) rotary width
        freqs = theta ** (-torch.arange(0, half, device=x.device).float() * 2.0 / d)
        n = int(part * d // 2)
        if n < half:
            freqs = freqs.clone()
            freqs[n:] = 0.0
    ang = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)      # [T, half]
    x1, x2 = xr[..., :half], xr[..., half:]
    while ang.dim() < x1.dim():
        ang = ang.unsqueeze(0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if xp.shape[-1]:
        out = torch.cat([out, xp], dim=-1)
    return out.to(x.dtype)


def _flow_path(prefix, nid):
    """A box's full path ('gg002/gg017') — how the app tells this box from a
    namesake in another block while it lights the data flow."""
    prefix = (prefix or "").strip("/")
    return (prefix + "/" + nid) if prefix else nid


def ns_ref(node, ref, gguf_file):
    """One binding -> (path, tensor). A namespace `file::tensor` names its own
    file; a plain name resolves against the box's file, then the enclosing
    model's."""
    ref = str(ref or "").strip()
    if "::" in ref:
        path, _, tensor = ref.partition("::")
        return (path.strip() or node.get("file") or gguf_file), tensor.strip()
    return (node.get("file") or gguf_file), ref


def _node_device(node, fallback):
    """Where a box runs: its own `place`, else the level's device."""
    want = fallback or "cpu"
    if node.get("place") == "system":
        return "cpu"
    return want


class WeightStore:
    """Loaded tensors by namespace (`file::tensor`): every box that binds the
    same namespace shares ONE copy of its storage — two boxes on one tensor
    cost one weight, not two. A materialisation is shared per
    (namespace, shape, dtype, grad, device): the same Parameter object lands
    in every box that asks for it. Anything that differs (a box offloaded to
    the system, a second dtype) gets its own copy — counted, and said out
    loud in the report."""

    def __init__(self):
        self._raw = {}                          # (path, tensor) -> file-dtype tensor
        self._native = {}                       # (path, tensor) -> native block set
        self._params = {}                       # materialisation key -> Parameter
        self._refs = {}                         # (path, tensor) -> bindings that use it

    def raw(self, path, tensor):
        key = (path, tensor)
        hit = self._raw.get(key)
        if hit is None:
            hit = load_weight_tensor(path, tensor, None)   # file dtype, cpu
            self._raw[key] = hit
        return hit

    def native_blocks(self, path, tensor):
        key = (path, tensor)
        hit = self._native.get(key)
        if hit is None:
            hit = load_gguf_native(path, tensor)
            if hit is None:
                return None
            self._native[key] = hit
        return hit

    def param(self, path, tensor, shape=None, dtype=None,
              requires_grad=False, device=None):
        """One shared Parameter per materialisation — same namespace, same
        shape/dtype/grad/device, same object."""
        key = (path, tensor, tuple(int(s) for s in shape) if shape else None,
               str(dtype), bool(requires_grad), str(device))
        self._refs[(path, tensor)] = self._refs.get((path, tensor), 0) + 1
        hit = self._params.get(key)
        if hit is not None:
            return hit
        t = self.raw(path, tensor)
        if dtype is not None:
            t = t.to(dtype)
        t = t.to(device or "cpu")
        if shape:
            t = t.reshape(shape)
        hit = nn.Parameter(t, requires_grad=bool(requires_grad))
        self._params[key] = hit
        return hit

    def blocks_param(self, path, tensor, device=None):
        """The native quant blocks of a namespace, shared the same way."""
        key = ("blocks", path, tensor, str(device))
        self._refs[(path, tensor)] = self._refs.get((path, tensor), 0) + 1
        hit = self._params.get(key)
        if hit is not None:
            return hit
        nat = self.native_blocks(path, tensor)
        if nat is None:
            return None
        hit = nn.Parameter(
            torch.from_numpy(nat["blocks"]).contiguous().to(device or "cpu"),
            requires_grad=False)
        self._params[key] = hit
        return hit

    def report(self):
        shared = sum(1 for n in self._refs.values() if n > 1)
        return {"namespaces": len(self._raw) + len(self._native),
                "shared": shared}


# ---------------------------------------------------------------- the graph
class Graph(nn.Module):
    """One design: a whole canvas, or the inside of a nested box."""

    def __init__(self, doc, seed=0, path="", frozen=False, off=False,
                 dtype=torch.float32, gguf_file=None, device=None, on_box=None,
                 native=True, train=False, store=None):
        super().__init__()
        self.path = path
        self.gguf_file = gguf_file
        self.native = bool(native)
        self.store = store if store is not None else WeightStore()
        self._ctx_dtype = dtype
        self._ctx_frozen = bool(frozen)
        self._ctx_train = bool(train)
        self._ctx_device = device
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
        # a dataset wired only into training dots (an output's training tab), or
        # one that only a training host names: its rows are for a training run —
        # an inference run stands it down
        self.idle_datasets = set()
        if not train:
            outs, by_id = {}, {n.get("id"): n for n in nodes}
            bound = set()
            for n in nodes:
                if (n.get("kind") == "train"
                        or (n.get("kind") == "output" and n.get("train"))):
                    for key in ("dataset", "test_dataset"):
                        if n.get(key):
                            bound.add(n[key])
            for e in edges:
                outs.setdefault(e.get("from"), []).append(e)
            for n in nodes:
                if n.get("kind") != "dataset":
                    continue
                es = outs.get(n.get("id")) or []
                if es and all(self._train_dot(by_id, e) for e in es):
                    self.idle_datasets.add(n.get("id"))
                elif not es and n.get("id") in bound:
                    self.idle_datasets.add(n.get("id"))
        self.order = self._topo(nodes)

        # a norm with no size takes the size flowing into it
        sizes = self._incoming_sizes(nodes, edges)
        for n in nodes:
            if n.get("kind") == "norm" and n.get("size") is None and n["id"] in sizes:
                n["size"] = sizes[n["id"]]
        self.node_list = nodes

        for n in nodes:
            try:
                self.boxes[n["id"]] = Box(n, seed=seed, parent_path=self.path,
                                          frozen=frozen, off=off, loader=self._bind, dtype=dtype,
                                          gguf_file=self.gguf_file, device=device, on_box=on_box,
                                          native=self.native, train=train, store=self.store)
                if on_box is not None:
                    on_box(n)               # compile progress: every box, any depth
            except CanvasError as e:             # the app marks this box on the canvas
                if e.nid is None:
                    e.nid = n["id"]
                raise

    # ---- compile helpers ---------------------------------------------------
    def _bind(self, node):
        """Bind one box's tensors — or fail naming the box: a namespace that
        cannot load or whose shape the box cannot wear is that box's problem,
        said the way every other canvas failure is said."""
        try:
            return self._bind_all(node)
        except CanvasError as e:
            name = node.get("name") or node.get("kind") or ""
            if e.box == name:
                raise
            raise CanvasError(name, e.reason) from None

    def _bind_all(self, node):
        """Load a box's bound tensor(s) when a weight file is in reach: the
        box's own `file` first (external gguf / safetensors / pt), then the
        enclosing GGUF box's. Bindings are namespaces (`file::tensor`) or
        plain tensor names; every tensor goes through the graph's
        `WeightStore`, so boxes bound to one namespace share ONE loaded copy.
        A `...bias` binding becomes the layer's bias; a quantized weight also
        picks up its siblings from the same file: `…weight_scale` (fp8 per
        channel, fp4 per 16) and `…weight_global_scale` (the stored divisor).
        Everything comes back already on the box's device, at its dtype —
        and the same object for every box that asks for the same thing."""
        t = (node.get("t") or "").strip()
        if not t:
            return None
        kind = node.get("kind")
        want = _node_device(node, self._ctx_device)
        dtype = box_dtype(node, self._ctx_dtype)
        grad = param_grad(dtype, self._ctx_frozen or bool(node.get("frozen")),
                          self._ctx_train)
        store = self.store
        weight = blocks = bias = None
        quant = None
        packed = False
        weight_name = weight_path = None
        extras = []
        for ref in [x.strip() for x in t.split(",") if x.strip()]:
            path, tensor = ns_ref(node, ref, self.gguf_file)
            if not path or not tensor:
                raise CanvasError(node.get("name") or kind,
                                  "its binding %r has no file to read" % ref)
            if tensor.endswith(".bias"):
                bias = store.param(path, tensor, shape=(-1,), dtype=dtype,
                                   requires_grad=grad, device=want)
            elif weight is None and blocks is None:
                weight_name, weight_path = tensor, path
                if (str(path).lower().endswith(".gguf")
                        and kind in ("layer", "embed") and self.native
                        and nn_quant is not None):
                    nat = store.native_blocks(path, tensor)
                    if nat is not None and nn_quant.can(nat["qname"], nat["shape"][1]):
                        # blocks are the weight: on the gpu the kernel
                        # dequantizes inside the matmul, off it the forward
                        # dequantizes host-side. Training compiles with
                        # native=False, so no run needs an f16 copy at rest.
                        blocks = store.blocks_param(path, tensor, device=want)
                        quant = {"qname": nat["qname"],
                                 "shape": [int(nat["shape"][0]), int(nat["shape"][1])]}
                        continue
                raw = store.raw(path, tensor)
                if kind == "layer":
                    din, dout = node.get("din"), node.get("dout")
                    packed = (raw.dtype == torch.uint8 and raw.dim() == 2
                              and list(raw.shape) == [int(dout), int(din) // 2])
                    f8 = raw.dtype in _F8_DTYPES
                    if packed or f8:
                        want_shape = [int(dout), int(din) // 2] if packed \
                            else [int(dout), int(din)]
                        if list(raw.shape) != want_shape:
                            raise CanvasError(node.get("name") or "layer",
                                              "its stored shape %s is not the %s weight %s"
                                              % (list(raw.shape),
                                                 "packed" if packed else "fp8",
                                                 want_shape))
                        weight = store.param(path, tensor, dtype=None,
                                             requires_grad=False, device=want)
                    else:
                        weight = store.param(path, tensor,
                                             shape=(int(dout), int(din)), dtype=dtype,
                                             requires_grad=grad, device=want)
                elif kind == "embed":
                    weight = store.param(path, tensor,
                                         shape=(int(node.get("vocab")),
                                                int(node.get("dout"))),
                                         dtype=dtype, requires_grad=grad,
                                         device=want)
                elif kind == "norm":
                    weight = store.param(path, tensor, shape=(-1,), dtype=dtype,
                                         requires_grad=grad, device=want)
                else:                                # op scale / conv1d / scan
                    weight = store.param(path, tensor, dtype=torch.float32,
                                         requires_grad=False, device=want)
            else:                                # a second bound tensor (dt_bias, …)
                extras.append(store.param(path, tensor, shape=(-1,),
                                          dtype=torch.float32, requires_grad=False,
                                          device=want))
        if weight is None and blocks is None:
            return None
        out = {"weight": weight, "blocks": blocks, "quant": quant,
               "bias": bias, "packed": packed}
        if extras:
            out["extras"] = extras
        if weight_name and str(weight_path).endswith(".safetensors"):
            head = safetensors_header(weight_path)
            base = (weight_name[:-len("_packed")]
                    if weight_name.endswith("_packed") else weight_name)
            if base + "_scale" in head:
                out["scale"] = store.param(weight_path, base + "_scale", shape=None,
                                           dtype=torch.float32, requires_grad=False,
                                           device=want)
            if base + "_global_scale" in head:
                out["global_scale"] = float(
                    store.param(weight_path, base + "_global_scale", dtype=None,
                                requires_grad=False, device="cpu").reshape(-1)[0])
        return out

    @staticmethod
    def _train_dot(by_id, e):
        """The edge lands on an output box's training dot (its dataset port)."""
        dst = by_id.get(e.get("to")) or {}
        return (dst.get("kind") == "output" and bool(dst.get("train"))
                and int(e.get("toPort") or 0) >= 1)

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

    def _fail(self, nid, why):
        """Raise about the node being run — carrying its id for the canvas."""
        err = CanvasError(self.boxes[nid].name, why)
        if err.nid is None:
            err.nid = nid
        raise err

    def _port_val(self, v, port):
        if isinstance(v, dict):
            if port in v:
                return v[port]
            vals = [x for x in v.values() if x is not None]
            return vals[0] if vals else None
        return v

    # ---- forward -----------------------------------------------------------
    def forward(self, *args, feeds=None, on_node=None):
        """`args` map to the graph's input boxes in port order (how a nested
        box calls its inside); `feeds` is {input box id: tensor}. `on_node`
        hears every box's full path **as its value lands**, data-flow order —
        a nested box after its inside, so a marked block really is finished."""
        self.on_node = on_node
        feeds = dict(feeds or {})
        for i, a in enumerate(args):
            if i < len(self.input_ids):
                feeds[self.input_ids[i]] = a
        vals = {}
        for nid in self.order:
            box = self.boxes[nid]
            if box.loose or box.off or box.kind in ("agent", "reward", "train") \
                    or nid in self.idle_datasets:
                vals[nid] = None               # drawn, not run / off / a config box
                continue
            ins = []
            for e in self.incoming.get(nid, []):
                v = vals.get(e["from"])
                if v is None:
                    if self.boxes[e["from"]].loose:
                        continue                 # an unmapped tensor: ignore its wire
                    if e["from"] in self.idle_datasets:
                        continue                 # a training dot's dataset: not this run's
                    self._fail(nid, "it is fed by a box that did not run")
                ins.append(self._port_val(v, e.get("fromPort") or 0))
            dev = box.device or "cpu"            # each box runs where it was placed
            if box.kind == "tokenize":
                src = ins[0] if ins else feeds.get(nid)    # the wire wins, then its own text
                if src is None:
                    self._fail(nid, "no text was given — wire an input into it, "
                                    "or type a text")
                try:
                    if not ins and torch.is_tensor(src):   # a generate grew them: ids pass
                        v = _token_ids(src)
                        if v is None:
                            self._fail(nid, "its fed ids are not token ids")
                    else:
                        v = tokenize_text(box.node, src)
                except CanvasError as e:
                    if e.nid is None:
                        e.nid = nid
                    raise
                if torch.is_tensor(v):
                    v = v.to(dev)
            elif box.kind == "hf":
                src = ins[0] if ins else feeds.get(nid)    # the prompt, from either
                try:
                    v = box(*([src] if src is not None else []))
                except CanvasError as e:
                    if e.nid is None:
                        e.nid = nid
                    raise
            elif box.kind == "input":
                if nid not in feeds:
                    self._fail(nid, "no value was given for this input \u2014 "
                                    "type its matrix on the box")
                v = feeds[nid]
                if torch.is_tensor(v):
                    v = v.to(dev)
            elif box.kind == "dataset":        # its labels are the run's, not a wire's
                try:
                    v = box()
                except CanvasError as e:
                    if e.nid is None:
                        e.nid = nid
                    raise
            elif box.kind == "custom":
                box._on_node = self.on_node    # the inside reports its own nodes too
                v = box(*ins)                    # its inside moves what it needs
            else:
                if not ins:
                    self._fail(nid, "an input dot is not wired")
                ins = [x.to(dev) if torch.is_tensor(x) else x for x in ins]
                v = box(*ins)
            vals[nid] = v
            if self.on_node:                   # a mark means done: a nested box's
                self.on_node(_flow_path(self.path, nid))   # comes after its inside
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

    # ---- device placement --------------------------------------------------
    def apply_placement(self, default_device):
        """Every box runs where it was told: `place: "system"` keeps it on the
        machine's RAM, `place: "gpu"` (or nothing) takes the compile device.
        Reads through nested boxes; the innermost choice wins. A custom box
        passes its choice down — the leaves move themselves, never whole
        subtrees (a round trip would stage the model twice)."""
        for box in self.boxes.values():
            want = default_device
            if box.place == "system":
                want = "cpu"
            elif box.place == "gpu":
                want = default_device
            if box.kind == "custom" and box.inner is not None:
                box.device = str(want)
                box.inner.apply_placement(want)  # inner choices override the level
                continue
            box.to(want)
            box.device = str(want)
        return self

    def weight_report(self):
        """Where the weights actually sit, in bytes — what a `load` costs."""
        rep = {"params": 0, "bytes": 0, "on_gpu_bytes": 0, "on_system_bytes": 0,
               "boxes": 0}
        seen = set()

        def count(box):
            own = False
            for p in box.parameters():
                key = (p.data_ptr(), p.numel())      # a shared tensor counts once
                if key in seen:
                    continue
                seen.add(key)
                n = p.numel()
                b = n * p.element_size()
                quant = getattr(box, "w_quant", None)
                if (quant is not None and getattr(box, "w_blocks", None) is not None
                        and p is box.w_blocks):
                    # native blocks: bytes are the stored ones, params logical
                    rep["params"] += int(quant["shape"][0]) * int(quant["shape"][1])
                else:
                    rep["params"] += n
                rep["bytes"] += b
                if str(p.device).startswith("cuda"):
                    rep["on_gpu_bytes"] += b
                else:
                    rep["on_system_bytes"] += b
                own = True
            if own:
                rep["boxes"] += 1

        def visit(graph):
            for box in graph.boxes.values():
                if box.kind == "custom" and box.inner is not None:
                    visit(box.inner)                 # nested weights count at their place
                else:
                    count(box)

        visit(self)
        rep.update(self.store.report())          # namespaces loaded, shared once
        return rep


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


# ---------------------------------------------------------------- text boxes
# A tokenizer feeds ids from text; a de-tokenizer turns ids (or logits) back
# into text. A box may carry a huggingface repo — its tokenizer.json is
# downloaded once and loaded here — or a plain character vocabulary, where the
# index of a char in it is the id (chars outside it are skipped).
_TEXT_TOKENS = {}
_TEXT_TOKENS_LOCK = threading.Lock()


def hf_text_tokenizer(repo, name=""):
    """The repo's tokenizer.json, loaded once per process."""
    key = str(repo or "").strip()
    with _TEXT_TOKENS_LOCK:
        hit = _TEXT_TOKENS.get(key)
    if hit is not None:
        return hit
    try:
        from tokenizers import Tokenizer
    except Exception:  # noqa: BLE001
        raise CanvasError(name, "loading a huggingface tokenizer needs `pip install tokenizers`")
    try:
        tok = Tokenizer.from_file(nn_hf.tokenizer_path(repo))
    except Exception as e:  # noqa: BLE001
        raise CanvasError(name, "cannot load the tokenizer of %s: %s" % (repo, e))
    with _TEXT_TOKENS_LOCK:
        _TEXT_TOKENS[key] = tok
    return tok


def tokenize_text(node, text):
    """Text -> ids: the box's huggingface repo when one is set, else its
    character vocabulary. The text comes from the wire (an input's text) or
    from the box's own text."""
    name = node.get("name") or "tokenizer"
    if torch.is_tensor(text):
        raise CanvasError(name, "its input must be text — give the input box the text mode")
    repo = str(node.get("repo") or "").strip()
    if repo:
        tok = hf_text_tokenizer(repo, name)
        return torch.tensor(tok.encode(str(text)).ids, dtype=torch.long)
    chars = list(str(node.get("vocab") or ""))
    look = {ch: i for i, ch in enumerate(chars)}
    ids = [look[ch] for ch in str(text) if ch in look]
    return torch.tensor(ids, dtype=torch.long)


def sample_one(logits, node):
    """One id out of logits: the **last position's** distribution is sampled —
    softmax -> temperature -> top-k/top-p -> multinomial (temperature 0 is
    greedy, the seed decides otherwise)."""
    name = node.get("name") or "sample"
    if not torch.is_tensor(logits) or not logits.is_floating_point() or logits.dim() < 1:
        raise CanvasError(name, "it needs logits — a floating tensor whose last dim "
                                "is the vocabulary")
    if logits.dim() == 1:
        row = logits.float().cpu()             # a single position's logits
    else:
        row = logits.reshape(-1, int(logits.shape[-1]))[-1].float().cpu()   # the last one
    temp = 0.0 if node.get("temperature") is None else float(node.get("temperature"))
    # greedy by default: one prompt, one answer. Set a temperature to sample.
    gen = torch.Generator(device="cpu").manual_seed(
        _seed_for(int(node.get("sample_seed") or 1), node.get("id") or name))
    if temp <= 0:                              # greedy: the most likely id
        return torch.tensor([int(torch.argmax(row))], dtype=torch.long)
    p = _sample_probs(row, temp, int(node.get("top_k") or 0),
                      float(node.get("top_p") if node.get("top_p") is not None else 1.0))
    return torch.tensor([int(torch.multinomial(p, 1, generator=gen).item())],
                        dtype=torch.long)


def detokenize_ids(node, t):
    """Token ids -> text. The de-tokenizer takes **ids only** — a model's
    logits go through a Sample box first, which picks one."""
    name = node.get("name") or "detokenize"
    if torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0:
        raise CanvasError(name, "it takes token ids — put a sample box between "
                                "the logits and it")
    repo = str(node.get("repo") or "").strip()
    chars = list(str(node.get("vocab") or ""))
    ids = torch.as_tensor(t).reshape(-1).long().tolist()
    if repo:
        tok = hf_text_tokenizer(repo, name)
        return tok.decode(ids, skip_special_tokens=True)
    if not chars:
        raise CanvasError(name, "its vocabulary is empty")
    return "".join(chars[i] if 0 <= i < len(chars) else "" for i in ids)


# ---------------------------------------------------------------- hf models
# One box that is a whole model: text (or ids) in, the **next token's text**
# out. The model comes from a huggingface repo, is cached here while the agent
# lives, and `offload` lets it go like everything else.
_HF_MODELS = {}
_HF_LOCK = threading.Lock()
_HF_DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def hf_loaded(repo):
    """Is a model from this repo already in memory?"""
    repo = str(repo or "")
    with _HF_LOCK:
        return any(k[0] == repo for k in _HF_MODELS)


def hf_model(node, device):
    """(tokenizer, model) for a repo — loaded once per repo/dtype/device."""
    name = node.get("name") or "hf model"
    repo = str(node.get("repo") or "").strip()
    if not repo:
        raise CanvasError(name, "no huggingface repo is set — right-click \u2192 "
                                "set the model \u2026")
    want = _HF_DTYPES.get(str(node.get("dtype") or "").strip().lower())
    if want is None:
        want = "float16" if str(device).startswith("cuda") else "float32"
    key = (repo, want, str(device))
    with _HF_LOCK:
        hit = _HF_MODELS.get(key)
    if hit is not None:
        return hit
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:  # noqa: BLE001
        raise CanvasError(name, "this box needs transformers — pip install transformers")
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=want)
    except Exception as e:  # noqa: BLE001 — the app shows the reason
        raise CanvasError(name, "could not load %s from huggingface: %s"
                          % (repo, str(e)[:200]))
    model.eval()
    model.to(device)
    with _HF_LOCK:
        _HF_MODELS[key] = (tok, model)
    return tok, model


def hf_next_token(node, src, device):
    """The token that follows the text (or the ids) — returned as its text."""
    name = node.get("name") or "hf model"
    tok, model = hf_model(node, device)
    if torch.is_tensor(src):
        ids = src.reshape(1, -1).long().to(device)
    else:
        ids = tok(str(src or ""), return_tensors="pt").input_ids.to(device)
    if ids.numel() == 0 or ids.shape[-1] == 0:
        raise CanvasError(name, "it needs some text to continue")
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float().cpu()
    temp = float(node.get("temperature") or 0)
    if temp <= 0:                               # greedy by default: one next token
        nxt = int(torch.argmax(logits))
    else:
        probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
    return tok.decode([nxt])


# ---------------------------------------------------------------- vision
# A multimodal model's vision half, run as one box: the mmproj file's tower,
# ported from llama.cpp's clip_graph_qwen3vl. The box reads patches (rows
# k = x + grid_w*y) and answers with the merged embeddings [n/4, proj_dim].
_VISION_PROJECTORS = ("qwen3vl_merger",)


class VisionTower(nn.Module):
    """The qwen3vl tower: patch embeds (two summed convs), learned positions
    (bilinear-resized), vision blocks (fused qkv, vision M-RoPE, GELU ffn),
    a post norm and the 2x2 patch merger. Tensor names are the file's own."""

    def __init__(self, node, train=False, device="cpu"):
        super().__init__()
        name = node.get("name") or "vision tower"
        self.name = name
        self.file = str(node.get("file") or "")
        proj = str(node.get("projector") or "")
        if proj not in _VISION_PROJECTORS:
            raise CanvasError(name, "its projector type '%s' is not supported yet "
                                    "(%s is)" % (proj or "?", ", ".join(_VISION_PROJECTORS)))
        self.d_model = int(node.get("d_model") or 0)
        self.heads = int(node.get("heads") or 0)
        self.blocks = int(node.get("blocks") or 0)
        self.patch = int(node.get("patch") or 16)
        self.merge = int(node.get("merge") or 2)
        self.image_size = int(node.get("image_size") or 0)
        self.proj_dim = int(node.get("proj_dim") or 0)
        self.eps = float(node.get("eps") or 1e-6)
        self.image_mean = _image_vec(node.get("image_mean"))
        self.image_std = _image_vec(node.get("image_std"), 1.0)
        if not (self.file and self.d_model and self.heads and self.blocks):
            raise CanvasError(name, "its mmproj is not set \u2014 load the GGUF again")
        grad = bool(train) and not bool(node.get("frozen"))
        self._pos_cache = {}

        def param(tensor_name, want=torch.float16, flat=False):
            t = load_weight_tensor(self.file, tensor_name, want)
            if flat:                                 # a conv kernel, read as a linear
                t = t.reshape(self.d_model, -1)
            return nn.Parameter(t.to(device), requires_grad=grad)

        self.w0 = param("v.patch_embd.weight", flat=True)
        self.w1 = param("v.patch_embd.weight.1", flat=True)
        self.b0 = param("v.patch_embd.bias", torch.float32)
        self.pos = param("v.position_embd.weight", torch.float32)
        self.blk = nn.ModuleList()
        for i in range(self.blocks):
            b = "v.blk.%d." % i
            m = nn.Module()
            m.qkv = param(b + "attn_qkv.weight")
            m.qkv_b = param(b + "attn_qkv.bias", torch.float32)
            m.out = param(b + "attn_out.weight")
            m.out_b = param(b + "attn_out.bias", torch.float32)
            m.up = param(b + "ffn_up.weight")
            m.up_b = param(b + "ffn_up.bias", torch.float32)
            m.down = param(b + "ffn_down.weight")
            m.down_b = param(b + "ffn_down.bias", torch.float32)
            m.ln1_w = param(b + "ln1.weight", torch.float32)
            m.ln1_b = param(b + "ln1.bias", torch.float32)
            m.ln2_w = param(b + "ln2.weight", torch.float32)
            m.ln2_b = param(b + "ln2.bias", torch.float32)
            self.blk.append(m)
        self.post_w = param("v.post_ln.weight", torch.float32)
        self.post_b = param("v.post_ln.bias", torch.float32)
        self.mm0 = param("mm.0.weight")
        self.mm0_b = param("mm.0.bias", torch.float32)
        self.mm2 = param("mm.2.weight")
        self.mm2_b = param("mm.2.bias", torch.float32)

    def _side(self, n):
        side = int(math.isqrt(n))
        if side <= 0 or side * side != n:
            raise CanvasError(self.name, "its %d patches do not make a square grid" % n)
        return side

    def _order(self, x, side):
        """rows k = x + side*y -> the merger's order (2x2 groups, wx fastest)"""
        m = self.merge
        g = x.reshape(side // m, m, side // m, m, x.shape[-1])   # [gy, wy, gx, wx, c]
        return g.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[-1])

    def _positions(self, side, device):
        key = ("pos", side, str(device))
        hit = self._pos_cache.get(key)
        if hit is not None:
            return hit
        m = self.merge
        ys, xs = [], []
        for gy in range(side // m):
            for gx in range(side // m):
                for wy in range(m):
                    for wx in range(m):
                        ys.append(m * gy + wy)
                        xs.append(m * gx + wx)
        out = (torch.tensor(ys, dtype=torch.float32, device=device),
               torch.tensor(xs, dtype=torch.float32, device=device))
        self._pos_cache[key] = out
        return out

    def _position_embedding(self, side, device):
        key = ("emb", side, str(device))
        hit = self._pos_cache.get(key)
        if hit is not None:
            return hit
        n_side = int(round(math.sqrt(self.pos.shape[0])))       # the table's grid
        tab = self.pos.reshape(n_side, n_side, self.d_model)    # [y, x, c]
        t = tab.permute(2, 0, 1)[None].to(torch.float32)        # [1, c, y, x]
        t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=True)
        t = t[0].permute(1, 2, 0).reshape(-1, self.d_model)     # rows k = x + side*y
        out = self._order(t, side)
        self._pos_cache[key] = out
        return out

    def _mrope(self, x, pos_y, pos_x):
        """vision M-RoPE: pairs (e, e+half) rotated; the first section by the
        patch's y, the second by its x (freq base 10000, plain rope)."""
        half = (self.d_model // self.heads) // 2
        sect = half // 2
        e = torch.arange(half, dtype=torch.float32, device=x.device)
        k = torch.where(e < sect, e, e - sect)
        theta = torch.where((e < sect)[None, :], pos_y[:, None], pos_x[:, None])
        theta = theta * (10000.0 ** (-2.0 * k / half))[None, :]
        c, s = torch.cos(theta)[None], torch.sin(theta)[None]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    def _from_image(self, img):
        """[h, w, 3] in 0..1 -> the normalized patch rows the tower reads:
        resized to its own image size when it must be, mean/std applied,
        rows k = x + grid_w*y."""
        size = self.image_size
        h, w = int(img.shape[0]), int(img.shape[1])
        x = img.to(torch.float32)
        if size and (w != size or h != size):
            x = F.interpolate(x.permute(2, 0, 1)[None], size=(size, size),
                              mode="bilinear", antialias=True)[0].permute(1, 2, 0)
            h = w = size
        x = (x - self.image_mean.to(x.device)) / self.image_std.to(x.device)
        p = self.patch
        if p <= 0 or w % p or h % p:
            raise CanvasError(self.name, "its %dx%d image does not split into "
                                         "%d-pixel patches" % (w, h, p))
        npx, npy = w // p, h // p
        x = x.permute(2, 0, 1).reshape(3, npy, p, npx, p)   # [c, y, kh, x, kw]
        return x.permute(1, 3, 0, 2, 4).reshape(npx * npy, 3 * p * p)

    def forward(self, patches):
        if not torch.is_tensor(patches) or not patches.is_floating_point():
            raise CanvasError(self.name, "it needs an image's patches \u2014 "
                                         "wire an image input")
        if patches.dim() == 3 and int(patches.shape[-1]) == 3:
            patches = self._from_image(patches)          # the picture itself
        elif patches.dim() == 2 and int(patches.shape[-1]) == 3:
            side = int(math.isqrt(int(patches.shape[0])))    # an RGB pixel list
            if side * side != int(patches.shape[0]):
                raise CanvasError(self.name, "its %d RGB rows do not make a square "
                                             "picture" % int(patches.shape[0]))
            patches = self._from_image(patches.reshape(side, side, 3))
        elif patches.dim() != 2:
            raise CanvasError(self.name, "it needs an image [h, w, 3] or its "
                                         "patch rows [n, 3*patch*patch]")
        x = patches.to(torch.float32)
        n = int(x.shape[0])
        side = self._side(n)
        d = self.d_model
        x = (x @ self.w0.to(torch.float32).T + x @ self.w1.to(torch.float32).T
             + self.b0.to(torch.float32))
        x = self._order(x, side)
        x = x + self._position_embedding(side, x.device)
        pos_y, pos_x = self._positions(side, x.device)
        heads, dh = self.heads, d // self.heads
        for b in self.blk:
            h = F.layer_norm(x, (d,), b.ln1_w.to(torch.float32),
                             b.ln1_b.to(torch.float32), self.eps)
            qkv = h @ b.qkv.to(torch.float32).T + b.qkv_b.to(torch.float32)
            q, k, v = qkv.split(d, dim=-1)
            q = q.reshape(n, heads, dh).permute(1, 0, 2)
            k = k.reshape(n, heads, dh).permute(1, 0, 2)
            v = v.reshape(n, heads, dh).permute(1, 0, 2)
            q = self._mrope(q, pos_y, pos_x)
            k = self._mrope(k, pos_y, pos_x)
            att = F.softmax(q @ k.transpose(-1, -2) * (1.0 / math.sqrt(dh)), dim=-1)
            o = (att @ v).permute(1, 0, 2).reshape(n, d)
            x = x + (o @ b.out.to(torch.float32).T + b.out_b.to(torch.float32))
            h = F.layer_norm(x, (d,), b.ln2_w.to(torch.float32),
                             b.ln2_b.to(torch.float32), self.eps)
            h = h @ b.up.to(torch.float32).T + b.up_b.to(torch.float32)
            h = F.gelu(h, approximate="tanh")
            x = x + (h @ b.down.to(torch.float32).T + b.down_b.to(torch.float32))
        x = F.layer_norm(x, (d,), self.post_w.to(torch.float32),
                         self.post_b.to(torch.float32), self.eps)
        m = self.merge * self.merge
        emb = x.reshape(-1, m * d)
        emb = emb @ self.mm0.to(torch.float32).T + self.mm0_b.to(torch.float32)
        emb = F.gelu(emb, approximate="tanh")
        emb = emb @ self.mm2.to(torch.float32).T + self.mm2_b.to(torch.float32)
        return emb


def _image_vec(value, default=0.5):
    """A per-channel mean/std from a node field: one value or three."""
    if value is None or str(value).strip() == "":
        return torch.tensor([float(default)] * 3)
    parts = [float(v) for v in str(value).replace(",", " ").split() if v.strip()]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise CanvasError("input", "its image mean/std must be 1 or 3 numbers")
    return torch.tensor(parts)


def image_tensor(node, spec):
    """The pixels the browser sent ({w, h, rgb: base64}) -> the image as data:
    [h, w, 3] in 0..1, RGB, resized to the box's own size when the pixels
    come in at another one. This is what travels the wires — an output box
    shows it, a vision tower normalizes and patchifies it itself."""
    name = node.get("name") or "input"
    if not isinstance(spec, dict):
        raise CanvasError(name, "its image must ride with the run \u2014 pick one "
                                "in the box")
    try:
        w, h = int(spec.get("w") or 0), int(spec.get("h") or 0)
    except (TypeError, ValueError):
        raise CanvasError(name, "its image's size is not a number")
    raw = str(spec.get("rgb") or "")
    if w <= 0 or h <= 0 or not raw:
        raise CanvasError(name, "no image was sent \u2014 pick one in the box")
    try:
        buf = base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        raise CanvasError(name, "its image's bytes are not valid base64")
    if len(buf) != w * h * 3:
        raise CanvasError(name, "its image has %d bytes, expected %d"
                          % (len(buf), w * h * 3))
    x = torch.frombuffer(bytearray(buf), dtype=torch.uint8).float() / 255.0
    x = x.reshape(h, w, 3)                      # [y, x, c]
    size = int(node.get("image_size") or 0)
    if size and (w != size or h != size):
        x = F.interpolate(x.permute(2, 0, 1)[None], size=(size, size),
                          mode="bilinear", antialias=True)[0].permute(1, 2, 0)
    return x


# ---------------------------------------------------------------- run
def build_input(node):
    """One input box's own matrix — the numbers typed on the box, one row per
    line (spaces or commas between them). A single row is a vector; whole
    numbers stay whole (ids), a decimal makes the row a float."""
    name = node.get("name") or "input"
    rows = []
    for line in str(node.get("values") or "").replace(",", " ").splitlines():
        toks = line.split()
        if not toks:
            continue
        nums = []
        for tok in toks:
            try:
                nums.append(int(tok))
            except ValueError:
                try:
                    nums.append(float(tok))
                except ValueError:
                    raise CanvasError(name, "its matrix has a value that is not "
                                            "a number: %s" % tok)
        rows.append(nums)
    if not rows:
        raise CanvasError(name, "its matrix is empty \u2014 type its numbers on the box")
    width = len(rows[0])
    for r in rows:
        if len(r) != width:
            raise CanvasError(name, "its matrix rows do not line up (%d and %d numbers)"
                              % (width, len(r)))
    return torch.as_tensor(rows if len(rows) > 1 else rows[0])


def build_feeds(canvas, feeds=None, seed=0):
    """The run's starting feeds: what the caller passed, plus every top-level
    source's own definition — an input's own matrix or its text, a tokenizer's
    own text. A matrix input's numbers are typed on the box; an image must ride
    with the run."""
    out = dict(feeds or {})
    for node in canvas.get("nodes") or []:
        nid = node.get("id")
        if node.get("kind") == "tokenize":
            if nid not in out and str(node.get("text") or ""):
                out[nid] = node["text"]
            continue
        if node.get("kind") != "input":
            continue
        mode = node.get("input_mode")
        if mode == "image":
            if nid in out:
                if isinstance(out[nid], dict):       # the pixels rode with the run
                    out[nid] = image_tensor(node, out[nid])
            elif node.get("image"):
                raise CanvasError(node.get("name") or "input",
                                  "its image must ride with the run \u2014 the "
                                  "backend cannot read the blob")
            continue
        if mode in ("text", "image"):
            if mode == "text" and nid not in out and str(node.get("text") or ""):
                out[nid] = str(node["text"])
            continue
        if nid not in out and str(node.get("values") or "").strip():
            out[nid] = build_input(node)         # the matrix the box carries
    return out


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


def _pick_one(row, temperature, top_k, top_p, gen):
    """One id from a row of logits — greedy at temperature 0, sampled otherwise."""
    if float(temperature or 0) <= 0:
        return int(torch.argmax(row))
    return int(torch.multinomial(_sample_probs(row, temperature, top_k, top_p),
                                 1, generator=gen).item())


def _token_ids(v):
    """A value read as flat int64 token ids — or None when it is text, empty,
    or plainly a float tensor (logits are not ids)."""
    if v is None or isinstance(v, str):
        return None
    if not torch.is_tensor(v):
        try:
            v = torch.as_tensor(v)
        except Exception:  # noqa: BLE001 — not a tensor at all
            return None
    if not v.numel():
        return None
    if v.is_floating_point() and not bool(torch.all(v == v.round())):
        return None
    return v.reshape(-1).long().cpu()


def _box_name(canvas, nid):
    for n in canvas.get("nodes") or []:
        if n.get("id") == nid:
            return n.get("name") or nid
    return str(nid or "")


def run(canvas, feeds, seed=0, device="cpu", generate=0, temperature=0.8,
        top_k=0, top_p=1.0, sample_seed=1, input_id=None, output_id=None,
        max_len=512, on_node=None, on_step=None, on_box=None, mode="inference"):
    """Compile the design, then run it — one forward pass (or a short
    generate). Returns per-box values for the UI plus the generated ids.
    `mode` is "inference" (the default: the weights need no gradients) or
    "training" (the non-frozen weights keep them; dataset boxes feed it)."""
    train = (mode == "training")
    graph = compile_canvas(canvas, seed=seed, device=device, on_box=on_box,
                           native=not train, train=train)
    return run_graph(graph, canvas, feeds, generate=generate, temperature=temperature,
                     top_k=top_k, top_p=top_p, sample_seed=sample_seed,
                     input_id=input_id, output_id=output_id, max_len=max_len,
                     on_step=on_step, on_node=on_node, train=train)


def run_graph(graph, canvas, feeds, generate=0, temperature=0.8,
              top_k=0, top_p=1.0, sample_seed=1, input_id=None, output_id=None,
              max_len=512, on_node=None, on_step=None, train=False):
    """The forward itself, on an already-compiled graph — the agent keeps one
    loaded, so a re-run pays no compile. `on_node` hears each box's full path
    as its data lands, data-flow order, nested boxes included. `on_step(k)`
    fires before each forward pass — k = 0 is the prompt, then one per token —
    so a watcher can re-walk the graph live instead of showing a stale trail.
    `train` marks a training run: the weights keep gradients, and the
    design's dataset boxes read their rows as the run's batch."""
    t0 = time.time()
    graph.train() if train else graph.eval()
    timing = {"sweeps": 0, "sweep_ms": 0.0}

    def sweep(feed_map):
        """One forward pass of the whole design — the moment the data leaves
        its inputs to the moment it lands on the outputs. Timed here, so the
        run can report the execution apart from the transport and the load."""
        t = time.perf_counter()
        vals = graph(feeds=feed_map, on_node=on_node)
        timing["sweeps"] += 1
        timing["sweep_ms"] += (time.perf_counter() - t) * 1000.0
        return vals

    def prep(feed_map):
        return {k: (v if (torch.is_tensor(v) or isinstance(v, str))
                    else torch.as_tensor(v))
                for k, v in (feed_map or {}).items()}   # each box moves its own

    if generate:
        if not input_id:
            input_id = next((n["id"] for n in canvas.get("nodes", [])
                             if n.get("kind") == "input"), None)
        if not output_id:
            output_id = next((n["id"] for n in canvas.get("nodes", [])
                              if n.get("kind") == "output"), None)
    try:
        feeds = prep(build_feeds(canvas, feeds, sample_seed))
    except CanvasError as e:               # an input's own definition failed: name its box
        if e.nid is None:
            e.nid = next((n.get("id") for n in canvas.get("nodes", [])
                          if n.get("name") == e.box), None)
        raise
    if train:
        # a training host is the hub: it names the dataset that feeds the run
        # — the test split stays idle in the run, the trainer scores it.
        # Without one the first dataset box feeds, as before.
        def _all(nodes):
            for n in nodes or []:
                yield n
                if isinstance(n.get("inner"), dict):
                    for m in _all((n.get("inner") or {}).get("nodes")):
                        yield m

        all_nodes = list(_all(canvas.get("nodes")))
        hub = train_hub(canvas)
        test_id = hub.get("test_dataset")
        if hub.get("dataset"):
            train_dn = next((n for n in all_nodes
                              if n.get("id") == hub.get("dataset")), None)
        else:
            train_dn = next((n for n in all_nodes
                             if n.get("kind") == "dataset" and n.get("id") != test_id),
                            None)
        test_dn = (next((n for n in all_nodes if n.get("id") == test_id), None)
                   if test_id else None)
        if train_dn is None and any(n.get("kind") == "dataset" for n in all_nodes):
            raise CanvasError("run", "training needs a dataset with rows — bind one "
                                  "in the training tab, or wire a dataset into the "
                                  "output's dot")
        paradigm = str(hub.get("paradigm") or "supervised")
        named = dataset_cols(train_dn) if train_dn is not None else []
        x_col = col_indexes(hub.get("x_col"), named) if hub.get("x_col") else [0]
        y_col = col_indexes(hub.get("y_col"), named) if hub.get("y_col") else [1]
        if x_col is None or y_col is None:
            raise CanvasError("run", "a column in the training tab's x or y is not "
                                  "in the dataset \u2014 a name, or a letter like A")
        vector = len(x_col) > 1 or len(y_col) > 1
        if vector and paradigm == "rl":
            raise CanvasError("run", "rl trains on prompt columns \u2014 a vector "
                                  "target is a supervised thing")
        if vector and any(src2.get("kind") == "dataset" and dst.get("kind") == "loss"
                          for src2, dst in _edge_pairs(canvas)):
            raise CanvasError("loss", "a vector target scores with mean squared "
                                  "error \u2014 a wired loss box computes cross entropy")
        label_mode = bool(hub.get("y_class") or hub.get("classes"))
        legacy_classes = list(hub.get("classes") or []) if isinstance(
            hub.get("classes"), (list, tuple)) else []
        if vector and label_mode:
            raise CanvasError("run", "a class label wants a single y column, "
                                  "not a vector target")
        classes = None
        train_labels = None

        def _dbox(dn):
            did = dn.get("id")
            if did in graph.boxes:
                return graph.boxes[did]
            for b in graph.modules():
                if isinstance(b, Box) and b.node.get("id") == did:
                    return b
            return None
        if train_dn is not None:
            rows = dataset_rows(train_dn)
            if not rows:
                raise CanvasError(train_dn.get("name") or "dataset",
                                  "it has no rows to train on — paste some, "
                                  "or point it at a file")
            size = max(1, int(train_dn.get("batch") or 4))
            if label_mode and paradigm != "rl":
                classes = dataset_classes([train_dn, test_dn], y_col,
                                          legacy_classes)
            if paradigm == "rl":
                # the reward box is the signal in rl; a wired loss box is a
                # supervised leftover, and its labels would not fit the
                # growing samples the probe draws
                if any(src.get("kind") == "dataset" and dst.get("kind") == "loss"
                       for src, dst in _edge_pairs(canvas)):
                    raise CanvasError("loss",
                                      "the reward is the signal in rl — a wired "
                                      "loss box is a supervised leftover")
                ids, train_labels = dataset_batch(train_dn, rows, 0, size,
                                                   x_col, x_only=True)
            else:
                ids, train_labels = dataset_batch(train_dn, rows, 0, size,
                                                   x_col, y_col, classes)
            tbox = _dbox(train_dn)
            if tbox is None:
                raise CanvasError(train_dn.get("name") or "dataset",
                                  "it sits inside a box the run cannot reach")
            # every dataset box carries the train batch, so an idle test split
            # cannot fail wanting rows
            for dn in all_nodes:
                if dn.get("kind") == "dataset":
                    b = _dbox(dn)
                    if b is not None:
                        b.batch = train_labels.to(b.device or "cpu")
            iid = _find_kind(canvas, "input")
            if iid:
                feeds[iid] = ids     # the model's input reads the rows
    generated = []

    def passes():
        if on_step:
            on_step(0)                             # the prompt pass
        vals = sweep(feeds)
        if generate and input_id and output_id:
            # grow the ids wherever they are: fed directly, or the ids a
            # tokenizer made of its text — a text pipeline generates the same way
            ids = _token_ids(vals.get(input_id))
            if ids is None:
                ids = _token_ids(feeds.get(input_id))
            if ids is None:
                raise CanvasError(_box_name(canvas, input_id) or "generate",
                                  "it needs token ids to continue — wire a tokenizer "
                                  "into its input")
            gen = torch.Generator(device="cpu").manual_seed(int(sample_seed))
            for _ in range(int(generate)):
                logits = vals.get(output_id)
                if logits is None:
                    raise CanvasError("output", "the output box is switched off")
                last = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
                nxt = _pick_one(last, temperature, top_k, top_p, gen)
                generated.append(nxt)
                ids = torch.cat([ids, torch.tensor([nxt], dtype=torch.long)])
                feeds[input_id] = ids[-max_len:]            # each box moves its own
                if on_step:
                    on_step(len(generated))                # a fresh sweep: token k
                vals = sweep(feeds)                        # the finished text lands too
        return vals

    if train:
        vals = passes()               # gradients on: the weights keep them
    else:
        with torch.no_grad():
            vals = passes()
    collected = graph.collect()
    story = image_story(canvas, collected)
    outs = view_paths(canvas)                    # the viewable nodes
    values = {}
    texts = {}
    for path, v in collected.items():
        if isinstance(v, str):
            # text is a value: the box it reached — the output at the end of a
            # generate, a text input — shows it, not only the box that made it
            values[path] = {"text": v}
            texts[path] = v
            continue
        st = _stats_with_image(v, story, view=(path in outs))
        if st is not None:
            values[path] = st
    out = {
        "ok": True,
        "values": values,
        "generated": generated,
        "texts": texts,
        "outputs": [{"id": n["id"], "name": n.get("name"),
                     "stats": ({"text": vals.get(n["id"])}
                               if isinstance(vals.get(n["id"]), str)
                               else _stats_with_image(vals.get(n["id"]), story, view=True))}
                    for n in canvas.get("nodes", []) if n.get("kind") == "output"],
        "ms": int((time.time() - t0) * 1000),
        "timing": {"sweeps": timing["sweeps"],
                   "input_to_output_ms": int(round(timing["sweep_ms"])),
                   "per_sweep_ms": (int(round(timing["sweep_ms"] / timing["sweeps"]))
                                    if timing["sweeps"] else 0)},
    }
    if train:
        # the probe: this batch's signal, and the test split's — one extra
        # forward each, so the console can show the setup before it's started
        try:
            probe = {"loss": None, "acc": None, "tloss": None, "tacc": None}
            if paradigm == "rl":
                # the rl probe is the policy itself: one sample from the first
                # row, scored by the reward box
                probe.update(_rl_probe(canvas, hub, graph, train_dn,
                                        rows if train_dn is not None else [],
                                        x_col, sample_seed))
            else:
                if train_labels is not None:
                    probe["loss"], probe["acc"] = _probe_loss(canvas, hub, vals,
                                                               train_labels, graph)
                if test_dn is not None:
                    tbox = _dbox(test_dn)
                    trows = dataset_rows(test_dn) if tbox is not None else []
                    if trows:
                        tsize = max(1, int(test_dn.get("batch") or 4))
                        tids, tlabels = dataset_batch(test_dn, trows, 0, tsize,
                                                       x_col, y_col, classes)
                        tbox.batch = tlabels.to(tbox.device or "cpu")
                        tfeeds = dict(feeds)
                        iid = _find_kind(canvas, "input")
                        if iid:
                            tfeeds[iid] = tids
                        with torch.no_grad():
                            tvals = graph(feeds=tfeeds)
                        probe["tloss"], probe["tacc"] = _probe_loss(
                            canvas, hub, tvals, tlabels, graph)
            out["probe"] = probe
        except Exception:
            pass                       # a broken probe must not kill the run
    return out


def compile_canvas(canvas, seed=0, device="cpu", dtype=torch.float32, gguf_file=None,
                   on_box=None, native=True, train=False):
    """Build the module for a top-level canvas document, every box placed
    where its `place` field says (unset boxes take `device`) — boxes move to
    their device as they are built, so nothing stages on the cpu first.
    `on_box(node)` hears every box as it lands, any nesting depth. `native`
    lets frozen gguf layers keep their raw blocks (training compiles with
    native=False: masters stay f16/f32, exactly as before). `train=True`
    marks a training graph: its non-frozen weights ask for gradients; an
    inference graph asks for none."""
    try:
        graph = Graph(dict(canvas), seed=seed, path="", dtype=dtype, gguf_file=gguf_file,
                      device=device or "cpu", on_box=on_box, native=native, train=train)
    except (MemoryError, RuntimeError) as e:
        msg = str(e)
        if isinstance(e, MemoryError) or "allocate" in msg or "out of memory" in msg.lower():
            raise CanvasError("", "not enough memory to load this design — its weights "
                                  "are materialised in full (%s)" % msg)
        raise
    return graph


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
        self.graph = compile_canvas(canvas, seed=int(self.cfg.get("seed") or 0),
                                    device=device, native=False, train=True)
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
        return F.cross_entropy(lg, tg.to(lg.device))

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


# ---------------------------------------------------------------- training
def _dataset_text(node):
    """A dataset's own text: what it carries, else the file it points at."""
    f = node.get("file")
    if f:
        p = _resolve(f)
        if not p or not os.path.isfile(p):
            raise CanvasError(node.get("name") or "dataset",
                              "its rows are not on disk: %s" % (f,))
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    return str(node.get("text") or "")


def _cell(r, col, default):
    """One row's cell, by index — its own cells, else the named prompt/output."""
    cells = r.get("cells")
    if col is not None and cells and 0 <= col < len(cells):
        return str(cells[col])
    return str(default if default is not None else "")


def dataset_classes(nodes, y_col, legacy=None):
    """The class set of a class-label target: every distinct y value, in the
    order a legacy list gives first (older designs keep the indices their
    checkpoints were trained with), then the order the rows show it — the
    train split first, the test after, so a label only the test carries still
    counts. None when there is nothing to pick from."""
    yc = y_col[0] if isinstance(y_col, (list, tuple)) else y_col
    out = []
    for name in (legacy or []):
        s = str(name)
        if s and s not in out:
            out.append(s)
    for node in nodes:
        if not node:
            continue
        for r in dataset_rows(node):
            v = _cell(r, yc, r.get("completion") if r.get("completion") is not None
                      else r.get("output"))
            if v and v not in out:
                out.append(v)
    return out or None


def dataset_cols(node):
    """A dataset's column names, lowercased: its CSV header row. The other
    formats carry none of their own — a letter stands for position."""
    if str(node.get("format") or "") != "csv":
        return []
    try:
        rd = csv.reader(io.StringIO(str(_dataset_text(node))))
        head = next(rd, []) or []
    except (OSError, ValueError):
        return []
    return [str(h).strip().lower() for h in head]


def dataset_rows(node):
    """A dataset box's rows: its own text or its file, one row per line —
    JSON objects ({prompt, completion} or {prompt, chosen, rejected}), or
    `prompt<TAB>completion`. Blank lines are skipped."""
    text = _dataset_text(node)
    if str(node.get("format") or "") == "csv":
        return _csv_rows(node, text)
    rows = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
                continue
            except ValueError:
                pass
        parts = line.split("\t")
        rows.append({"prompt": parts[0],
                     "completion": parts[1] if len(parts) > 1 else "",
                     "cells": parts})          # every column, for the train node's X/Y
    return rows


def _csv_rows(node, text):
    """A dataframe's rows: the header names the columns — prompt/completion by
    name (or output/answer), else the first two. Any other columns (ids of
    images, notes) ride along unread."""
    rd = csv.reader(io.StringIO(str(text)))
    head = [str(h).strip().lower() for h in (next(rd, []) or [])]

    def pick(*names):
        for nm in names:
            if nm in head:
                return head.index(nm)
        return None

    pi = pick("prompt", "input", "question")
    ci = pick("completion", "output", "answer", "response")
    if pi is None:
        pi = 0
    if ci is None:
        ci = 1 if len(head) > 1 else pi
    rows = []
    for r in rd:
        if not r or not any(str(x).strip() for x in r):
            continue

        def get(i):
            return str(r[i]) if i is not None and i < len(r) else ""

        rows.append({"prompt": get(pi), "completion": get(ci),
                     "cells": [str(x) for x in r]})
    return rows


def _ds_encode(node, text):
    """A dataset's own tokenizer: its huggingface repo, else its char vocab."""
    repo = str(node.get("repo") or "").strip()
    if repo:
        return list(hf_text_tokenizer(repo, node.get("name") or "dataset")
                    .encode(str(text)).ids)
    chars = list(str(node.get("vocab") or ""))
    look = {ch: i for i, ch in enumerate(chars)}
    return [look[ch] for ch in str(text) if ch in look]


def _ds_decode(node, ids):
    """`_ds_encode` run backwards: a sample's ids back into text in the
    dataset's own alphabet (its repo's tokenizer, else its char vocab). An id
    outside the alphabet becomes '?', so a drifted sample still reads."""
    repo = str(node.get("repo") or "").strip()
    if repo:
        try:
            return str(hf_text_tokenizer(repo, node.get("name") or "dataset")
                       .decode([int(i) for i in ids]).text)
        except Exception:  # noqa: BLE001 — a broken repo falls to the char vocab
            pass
    chars = list(str(node.get("vocab") or ""))
    out = []
    for i in ids:
        i = int(i)
        out.append(chars[i] if 0 <= i < len(chars) else "?")
    return "".join(out)


def col_letter(i):
    """A 0-based column index as a sheet letter: 1 -> A ... 26 -> Z, 27 -> AA."""
    s = ""
    i = int(i or 0)
    while True:
        s = chr(65 + (i % 26)) + s
        i = i // 26 - 1
        if i < 0:
            return s


def col_indexes(spec, cols=None):
    """One or more sheet columns from a spec: "A" -> [0]; "x0 x1 x2" -> those
    header names' indices, in order (a name wins, case-blind; a letter is a
    position: A=0 ... Z=25, AA=26). None when a token matches nothing."""
    out = []
    for tok in re.split(r"[\s,]+", str(spec or "").strip()):
        if not tok:
            continue
        low = tok.lower()
        idx = None
        for j, name in enumerate(cols or []):
            if name and name == low:
                idx = j
                break
        if idx is None:
            idx = col_index(tok)
        if idx is None:
            return None
        out.append(idx)
    return out or None


def col_index(letter):
    """A sheet column letter to a 0-based index: A->0 … Z->25, AA->26.
    None for an empty or non-letter value."""
    n = 0
    for ch in str(letter or "").strip().upper():
        if not "A" <= ch <= "Z":
            return None
        n = n * 26 + (ord(ch) - 64)
    return n - 1 if n else None


def dataset_batch(node, rows, start, size, x_col=None, y_col=None, classes=None,
                  x_only=False):
    """One batch: ids = X+Y, labels = the same ids with the X span masked to
    -100 (the loss reads Y only), padded right. `x_col`/`y_col` pick the
    conditioning and target columns (0-based; None keeps A/B). With `classes`,
    Y is a class label: the ids carry X only, left-padded so the class falls
    at the sequence's end, and the labels are the class indices (1-D) the
    model must pick at that position. With `x_only` there is no target column
    at all (rl: the dataset hands out prompts, the reward box is the signal) —
    the labels ride the prompt ids, so an idle loss box still reads in-range
    ids."""
    want = rows[start:start + size]
    if not want:
        want = rows[:size]

    def cell(r, col, default):
        return _cell(r, col, default)

    def _cols(c, d):
        if c is None:
            return [d]
        return [int(c)] if not isinstance(c, (list, tuple)) else [int(x) for x in c]
    xs, ys = _cols(x_col, 0), _cols(y_col, 1)
    if not x_only and (len(xs) > 1 or len(ys) > 1):
        # a vector target: the selected cells are numbers — x becomes the
        # batch's input matrix, y the batch's label matrix (mse downstream)
        def num(r, col, side):
            s = cell(r, col, "").strip()
            try:
                return float(s)
            except ValueError:
                raise CanvasError(node.get("name") or "dataset",
                                  "column %s of a row is not a number for the %s "
                                  "vector: %r" % (col_letter(col), side, s))
        ids = torch.tensor([[num(r, c, "x") for c in xs] for r in want],
                           dtype=torch.float32)
        labels = torch.tensor([[num(r, c, "y") for c in ys] for r in want],
                              dtype=torch.float32)
        return ids, labels

    if x_only:
        seqs = [_ds_encode(node, cell(r, xs[0], r.get("prompt"))) for r in want]
        width = max((len(s) for s in seqs), default=1)
        ids = torch.zeros(len(seqs), max(width, 1), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return ids, ids
    if classes:
        seqs, cls = [], []
        for r in want:
            seqs.append(_ds_encode(node, cell(r, xs[0], r.get("prompt"))))
            y = cell(r, ys[0], r.get("completion") if r.get("completion") is not None
                     else r.get("output"))
            if y not in classes:
                raise CanvasError(node.get("name") or "dataset",
                                  "class %r is not among the train node's classes %s"
                                  % (y, ", ".join(map(str, classes))))
            cls.append(classes.index(y))
        width = max((len(s) for s in seqs), default=1)
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad = width - len(s)
            ids[i, pad:pad + len(s)] = torch.tensor(s, dtype=torch.long)
        return ids, torch.tensor(cls, dtype=torch.long)
    seqs, labs = [], []
    for r in want:
        prompt = cell(r, xs[0], r.get("prompt"))
        comp = cell(r, ys[0], r.get("completion") if r.get("completion") is not None
                    else r.get("output"))
        p = _ds_encode(node, prompt)
        c = _ds_encode(node, comp)
        seqs.append(p + c)
        labs.append([-100] * len(p) + list(c))
    width = max((len(s) for s in seqs), default=1)
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    y = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        y[i, :len(l)] = torch.tensor(l, dtype=torch.long)
    return ids, y


def reward_eval(node, prompt, output):
    """A reward box's deterministic score: its python `expr` over the row
    (prompt/output are names), or one of the named recipes. Pure by contract —
    no clock, no randomness: the same pair always earns the same score."""
    kind = str(node.get("reward") or "python")
    if kind == "exact":
        return 1.0 if str(output).strip() == str(node.get("match") or "").strip() else 0.0
    if kind == "regex":
        return 1.0 if re.search(str(node.get("pattern") or ""), str(output)) else 0.0
    if kind == "length":
        n = len(str(output).split())
        cap = int(node.get("limit") or 64)
        return 1.0 if n <= cap else max(0.0, 1.0 - (n - cap) / max(cap, 1))
    expr = str(node.get("expr") or "1.0")
    try:
        return float(eval(expr, {"__builtins__": {}},   # noqa: S307 — the design's own reward
                          {"prompt": prompt, "output": output,
                           "len": len, "abs": abs, "min": min, "max": max,
                           "str": str, "float": float, "int": int}))
    except Exception as e:  # noqa: BLE001
        raise CanvasError(node.get("name") or "reward",
                          "its reward failed to score: %s" % e)


def _vector_loss(logits, labels):
    """A vector target's signal: mean squared error between the target
    output's rows and the batch's label rows — the last position when the
    output is a sequence, the rows themselves when it is per-sample."""
    lg = logits.float()
    pred = lg[:, -1, :] if lg.dim() == 3 else lg.reshape(-1, lg.shape[-1])
    pred = pred[-labels.shape[0]:]
    if pred.shape[-1] != labels.shape[-1]:
        raise CanvasError("output", "the target's output is %d wide, the labels "
                                    "are %d — they must match" % (pred.shape[-1],
                                                                  labels.shape[-1]))
    return F.mse_loss(pred, labels.to(pred.dtype))


def _cross_entropy(logits, labels):
    """Next-token cross entropy, prompt positions masked away (labels carry
    -100 there); the shift lives here so the design stays declarative."""
    lg = logits.float()
    lb = labels.long()
    if lg.dim() == 2:
        lg = lg.unsqueeze(0)
    if lb.dim() == 1:
        lb = lb.unsqueeze(0)
    tgt = torch.full(lb.shape, -100, dtype=torch.long, device=lb.device)
    if lb.shape[-1] > 1:
        tgt[:, :-1] = lb[:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).contiguous(),
                           tgt.reshape(-1), ignore_index=-100)


def _gather_logprobs(logits, ids):
    """log p(id at t | its prefix) for every position — the quantity every
    policy-gradient loss multiplies by its advantage."""
    lg = logits.float()
    lb = ids.long()
    if lg.dim() == 2:
        lg = lg.unsqueeze(0)
    if lb.dim() == 1:
        lb = lb.unsqueeze(0)
    lp = F.log_softmax(lg, dim=-1)
    out = torch.zeros(lb.shape, dtype=torch.float32, device=lb.device)
    if lb.shape[-1] > 1:
        out[:, 1:] = lp[:, :-1].gather(-1, lb[:, 1:].unsqueeze(-1)).squeeze(-1)
    return out


def _edge_pairs(canvas):
    """Every (source, destination) box pair of every edge, at every nesting
    level — the dataset->loss wiring the training signal rides on."""
    pairs = []

    def walk(doc):
        nodes = (doc or {}).get("nodes") or []
        by_id = {n.get("id"): n for n in nodes}
        for e in (doc or {}).get("edges") or []:
            src = by_id.get(e.get("from"))
            dst = by_id.get(e.get("to"))
            if src is not None and dst is not None:
                pairs.append((src, dst))
        for n in nodes:
            if isinstance(n.get("inner"), dict):
                walk(n["inner"])

    walk(canvas)
    return pairs


def train_hub(canvas):
    """The node that carries a training run's setup. An output box with
    `train` set — its training tab — wins: it trains on its own signal, and a
    dataset wired into its dot feeds the run. The train node stays the hub of
    the older designs. Returns the node's fields (with `target` always set
    for an output host), or {} when the design carries no training setup."""
    def _all(nodes):
        for n in nodes or []:
            yield n
            if isinstance(n.get("inner"), dict):
                for m in _all((n.get("inner") or {}).get("nodes")):
                    yield m

    nodes = list(_all((canvas or {}).get("nodes")))
    host = next((n for n in nodes
                 if n.get("kind") == "output" and n.get("train")), None)
    if host is not None:
        hub = dict(host)
        hub["target"] = host.get("id")          # an output trains on itself
        if not hub.get("dataset"):              # a dataset wired into its dot
            for src, dst in _edge_pairs(canvas):    # beats the picker
                if dst.get("id") == host.get("id") and src.get("kind") == "dataset":
                    hub["dataset"] = src.get("id")
                    break
        return hub
    return next((n for n in nodes if n.get("kind") == "train"), None) or {}


def training_setup(canvas, hub):
    """What scores the training: a loss box wired from a dataset box on the
    canvas wins (existing designs keep their wiring); otherwise the hub's
    target output box is the signal and the loss is computed around it.
    Returns the loss box id (or None), the box the logits come from, the
    dataset box that carries the labels, and whether the loss is implicit."""
    hub = hub or {}
    for src, dst in _edge_pairs(canvas):
        if src.get("kind") == "dataset" and dst.get("kind") == "loss":
            return {"loss_id": dst.get("id"), "target_id": None,
                    "carrier_id": src.get("id"), "implicit": False}
    target = hub.get("target")
    return {"loss_id": None, "target_id": target,
            "carrier_id": hub.get("dataset"), "implicit": target is not None}


def _probe_loss(canvas, hub, vals, labels, graph=None):
    """The training signal of one forward: the wired loss box's scalar, or —
    with no loss box — cross entropy of the hub's target output against the
    carried labels. Returns (loss, acc); acc is set for class labels."""
    hub = hub or {}
    loss_id = None
    for src, dst in _edge_pairs(canvas):
        if src.get("kind") == "dataset" and dst.get("kind") == "loss":
            loss_id = dst.get("id")
            break
    if loss_id is not None and vals.get(loss_id) is not None:
        v = vals[loss_id]
        loss = float(v.detach()) if torch.is_tensor(v) else v
        acc = None
        if graph is not None:
            box = graph.boxes.get(loss_id) if loss_id in (graph.boxes or {}) else None
            if box is None:
                for b in graph.modules():
                    if isinstance(b, Box) and b.node.get("id") == loss_id:
                        box = b
                        break
            if box is not None and getattr(box, "last_acc", None) is not None:
                acc = float(box.last_acc)
        return loss, acc
    target = hub.get("target")
    if target and labels is not None and vals.get(target) is not None:
        lg = vals[target]
        if labels.dim() == 1:                       # class labels: the end position
            lgf = lg.float()
            last = lgf[:, -1, :] if lgf.dim() == 3 else \
                lgf.reshape(-1, lgf.shape[-1])[-labels.shape[0]:]
            loss = F.cross_entropy(last.detach(), labels.long())
            with torch.no_grad():
                acc = (last.argmax(-1) == labels.long()).float().mean()
            return float(loss), float(acc)
        if labels.dim() == 2 and labels.is_floating_point():
            return float(_vector_loss(lg, labels).detach()), None   # a vector target
        return float(_cross_entropy(lg, labels).detach()), None
    return None, None


def _rl_probe(canvas, hub, graph, train_dn, rows, x_col, seed):
    """The training readout for an rl design: one sample from the first row,
    scored by the reward box — the console shows the signal before it starts."""
    hub = hub or {}
    out = {"loss": None, "acc": None, "tloss": None, "tacc": None,
           "reward": None, "sample": None}
    if train_dn is None or not rows:
        return out
    reward_node = _node_by_id(canvas, hub.get("reward"))
    iid = _find_kind(canvas, "input")
    target = hub.get("target")
    if reward_node is None or not iid or not target:
        return out
    node = train_dn
    if isinstance(x_col, (list, tuple)):
        x_col = x_col[0] if x_col else 0
    cells = rows[0].get("cells") or []
    text = (cells[x_col] if x_col is not None and x_col < len(cells)
            else str(rows[0].get("prompt") or ""))
    x_ids = _ds_encode(node, text)
    if not x_ids:
        return out
    dboxes = [b for b in graph.modules()
              if isinstance(b, Box) and b.node.get("kind") == "dataset"]
    max_tokens = max(1, int(hub.get("max_tokens") or 32))
    temperature = float(hub.get("temperature") or 1.0)
    try:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            ids = list(x_ids)
            for _ in range(max_tokens):
                for b in dboxes:
                    b.batch = torch.tensor([ids])
                vals = graph(feeds={iid: torch.tensor([ids])})
                logits = vals.get(target)
                if logits is None:
                    break
                last = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
                ids.append(_pick_one(last, temperature, 0, 1.0, gen))
            y_text = _ds_decode(node, ids[len(x_ids):])
            out["reward"] = float(reward_eval(reward_node, text, y_text))
            out["sample"] = y_text[:120]
    except CanvasError as e:
        out["reason"] = str(e)
    return out


class DesignTrainer:
    """Training on a design: a dataset box's rows in, a loss box's scalar out,
    one optimizer step per batch. In rl the rows are prompts instead: each step
    samples outputs from the target box, scores them with the reward box, and
    steps the policy on the advantages against a decaying baseline. Trainable
    boxes are those not `frozen`; the compile is `native=False` (trainable
    masters stay f16/f32). Checkpoints are named-parameter state + optimizer +
    step (and the baseline, in rl) — pause (detach) lands one and exits;
    resume picks up exactly where it stopped."""

    def __init__(self, canvas, cfg=None, device="cpu", checkpoint=None):
        cfg = dict(cfg or {})
        self.canvas = canvas
        self.cfg = cfg
        self.device = device
        self.paradigm = str(cfg.get("paradigm") or "supervised")
        self.graph = compile_canvas(canvas, device=device, native=False, train=True)
        self.graph.train()
        self.input_id = cfg.get("input") or _find_kind(canvas, "input")
        # the signal: a loss box wired on the canvas wins (existing designs),
        # else the hub's target output box is scored and the loss is implicit.
        # In rl the hub's dataset is the carrier, full stop — the reward box is
        # the signal and a wired loss box is a supervised leftover.
        setup = training_setup(canvas, cfg)
        if self.paradigm == "rl":
            if any(src.get("kind") == "dataset" and dst.get("kind") == "loss"
                   for src, dst in _edge_pairs(canvas)):
                raise CanvasError("loss",
                                  "the reward is the signal in rl — a wired "
                                  "loss box is a supervised leftover")
            self.loss_id = None
            self.target_id = cfg.get("target")
            self.implicit = self.target_id is not None
            if not self.target_id:
                raise CanvasError("",
                                  "rl training needs a target output box to "
                                  "sample from — bind one in the train box")
            self.reward_id = cfg.get("reward")
            if not self.reward_id:
                raise CanvasError("", "rl training needs a reward box — bind "
                                      "one in the train box")
            self.reward_node = _node_by_id(canvas, self.reward_id)
            if self.reward_node is None:
                raise CanvasError("", "the train node's reward box "
                                       "is not on the canvas")
            ds_id = (cfg.get("dataset") or _find_kind(canvas, "dataset"))
        else:
            self.loss_id = cfg.get("loss") or setup["loss_id"]
            self.target_id = setup["target_id"]
            self.implicit = self.loss_id is None and self.target_id is not None
            if not self.loss_id and not self.target_id:
                raise CanvasError("", "training needs an output box to score — bind a "
                                      "target in the train box, or wire a loss box")
            self.reward_id = cfg.get("reward") or None
            self.reward_node = None
            ds_id = (setup["carrier_id"] or cfg.get("dataset")
                     or _find_kind(canvas, "dataset"))
        if not self.input_id or not ds_id:
            raise CanvasError("", "training needs an input box the model reads, "
                                  "and a dataset with rows")
        self.dataset_id = ds_id
        self.dataset_box = self._box(ds_id)
        if self.dataset_box is None:
            raise CanvasError("", "the train node's dataset is not on the canvas")
        self.rows = dataset_rows(self.dataset_box.node)
        if not self.rows:
            raise CanvasError("dataset", "it has no rows — paste some, "
                                   "or point it at a file")
        # the hub's columns: a letter like a sheet (A/B as before), or a
        # sequence of names — x0 x1 x2 feeds a 3-wide input vector
        named = dataset_cols(self.dataset_box.node)
        self.x_col = col_indexes(cfg.get("x_col"), named) if cfg.get("x_col") else [0]
        self.y_col = col_indexes(cfg.get("y_col"), named) if cfg.get("y_col") else [1]
        if self.x_col is None or self.y_col is None:
            raise CanvasError("dataset", "a column in the training tab's x or y is "
                                   "not in its rows \u2014 a name, or a letter like A")
        self.vector = len(self.x_col) > 1 or len(self.y_col) > 1
        if self.vector and self.paradigm == "rl":
            raise CanvasError("", "rl trains on prompt columns \u2014 a vector "
                                  "target is a supervised thing")
        # y is a class label: the values it takes are the classes, derived
        # from the column (a legacy list, if the design carries one, fixes
        # the order its checkpoints were trained with)
        self.label_mode = bool(cfg.get("y_class") or cfg.get("classes"))
        self.classes = None
        legacy = (list(cfg.get("classes") or [])
                  if isinstance(cfg.get("classes"), (list, tuple)) else [])
        if self.vector and self.label_mode:
            raise CanvasError("", "a class label wants a single y column, "
                                  "not a vector target")
        if self.vector and self.loss_id:
            raise CanvasError("loss", "a vector target scores with mean squared "
                                  "error \u2014 a wired loss box computes cross entropy")
        # the test split: scored every eval_every steps, never trained —
        # idle in rl, where every sample the policy draws is scored
        self.test_id = cfg.get("test_dataset") or None
        self.test_box = None
        self.test_rows = []
        if self.test_id and self.paradigm != "rl":
            self.test_box = self._box(self.test_id)
            if self.test_box is None:
                raise CanvasError("", "the train node's test dataset "
                                       "is not on the canvas")
            self.test_rows = dataset_rows(self.test_box.node)
        self.classes = (dataset_classes(
            [self.dataset_box.node,
             self.test_box.node if self.test_box else None],
            self.y_col, legacy)
            if self.label_mode and self.paradigm != "rl" else None)
        self.class_mode = bool(self.classes)
        # every dataset box on the graph runs in the forward and each one
        # wants a batch — the carrier wears the split that is being scored,
        # the idle ones ride along on the same labels
        self.dataset_boxes = [self._box(nid) for nid in _find_kinds(canvas, "dataset")]
        self.dataset_boxes = [b for b in self.dataset_boxes if b is not None]
        self._tpos = 0
        self.trainable = [(n, p) for n, p in self.graph.named_parameters()
                          if p.requires_grad]
        if not self.trainable:
            raise CanvasError("", "nothing to train — unfreeze a box first")
        opt = str(cfg.get("optimizer") or "adamw").lower()
        lr = float(cfg.get("lr") or 3e-4)
        wd = float(cfg.get("weight_decay") or 0.0)
        self.optim = (torch.optim.SGD([p for _, p in self.trainable], lr=lr,
                                       momentum=0.9, weight_decay=wd)
                      if opt == "sgd" else
                      torch.optim.AdamW([p for _, p in self.trainable], lr=lr,
                                         weight_decay=wd))
        self.batch_size = max(1, int(cfg.get("batch") or 4))
        self.step = 0
        self.history = []
        self.baseline = None           # rl: the decaying mean of past rewards
        self.checkpoint_root = cfg.get("checkpoints") or "checkpoints"
        self.name = cfg.get("name") or "design"
        self._pos = 0
        if checkpoint:
            self.load(checkpoint)

    def _box(self, nid):
        if nid in self.graph.boxes:
            return self.graph.boxes[nid]
        for b in self.graph.modules():
            if isinstance(b, Box) and b.node.get("id") == nid:
                return b
        return None

    def _batch(self):
        ids, labels = dataset_batch(self.dataset_box.node, self.rows,
                                    self._pos, self.batch_size,
                                    self.x_col, self.y_col, self.classes)
        self._pos = (self._pos + self.batch_size) % max(len(self.rows), 1)
        return ids.to(self.device), labels.to(self.device)

    def _test_batch(self):
        ids, labels = dataset_batch(self.test_box.node, self.test_rows,
                                    self._tpos, self.batch_size,
                                    self.x_col, self.y_col, self.classes)
        self._tpos = (self._tpos + self.batch_size) % max(len(self.test_rows), 1)
        return ids.to(self.device), labels.to(self.device)

    def _loss(self, vals, labels):
        """The signal of one forward: the wired loss box's scalar, or — with
        no loss box on the canvas — the hub's target output scored against
        the batch's labels. Returns (loss, acc); acc is set for classes."""
        if self.loss_id is not None:
            loss = vals.get(self.loss_id)
            if loss is None:
                return None, None
            box = self._box(self.loss_id)
            acc = getattr(box, "last_acc", None) if box is not None else None
            return loss, (float(acc) if acc is not None else None)
        logits = vals.get(self.target_id)
        if logits is None:
            return None, None
        if labels.dim() == 1:                       # class labels: pick at the end
            lg = logits.float()
            last = lg[:, -1, :] if lg.dim() == 3 else \
                lg.reshape(-1, lg.shape[-1])[-labels.shape[0]:]
            loss = F.cross_entropy(last, labels.long())
            with torch.no_grad():
                acc = (last.argmax(-1) == labels.long()).float().mean()
            return loss, float(acc)
        if labels.dim() == 2 and labels.is_floating_point():
            return _vector_loss(logits, labels), None    # a vector target: mse
        return _cross_entropy(logits, labels), None

    def _eval(self):
        """One pass over the test split: its loss (and accuracy) without
        touching the weights. The carrier box wears the test labels for the
        pass, so a wired loss box scores the split too."""
        if self.test_box is None or not self.test_rows:
            return None, None
        ids, labels = self._test_batch()
        for b in self.dataset_boxes:
            b.batch = labels       # the whole graph wears the test labels for the pass
        with torch.no_grad():
            vals = self.graph(feeds={self.input_id: ids})
            return self._loss(vals, labels)

    def _sample(self, x_ids, max_tokens, temperature, gen):
        """One autoregressive draw from the target output box, no gradients.
        Returns the full id sequence — the prompt plus the drawn tail."""
        ids = list(x_ids)
        for _ in range(max_tokens):
            for b in self.dataset_boxes:
                b.batch = torch.tensor([ids]).to(self.device)
            vals = self.graph(feeds={self.input_id: torch.tensor([ids]).to(self.device)})
            logits = vals.get(self.target_id)
            if logits is None:
                raise CanvasError("output", "the output box is switched off")
            last = logits.reshape(-1, logits.shape[-1])[-1].float().cpu()
            ids.append(_pick_one(last, temperature, 0, 1.0, gen))
        return ids

    def _logprob(self, x_ids, ids):
        """log p(sample) under the policy, with gradients: the autoregressive
        walk that drew `ids` replayed, each prefix forward keeping its
        gradient until the caller's backward frees the sample's graph. None
        when the target output is off."""
        total = None
        for t in range(len(x_ids), len(ids)):
            prefix = ids[:t]
            for b in self.dataset_boxes:
                b.batch = torch.tensor([prefix]).to(self.device)
            vals = self.graph(feeds={self.input_id:
                                      torch.tensor([prefix]).to(self.device)})
            logits = vals.get(self.target_id)
            if logits is None:
                return None
            # [-1:] keeps a dimension: the gather below wants a row, not a vector
            last = logits.reshape(-1, logits.shape[-1])[-1:].float()
            term = F.log_softmax(last, dim=-1)[-1, int(ids[t])]
            total = term if total is None else total + term
        return total

    def _rl_step(self, say):
        """One rl step: one row of the hub's dataset, n_samples draws from
        the target box, each scored by the reward box; the policy steps on
        -mean((reward - baseline) * log p(sample)) for the row. The samples
        are drawn without gradients and their log-probabilities recomputed
        with gradients on, one sample's graph at a time, so a step never
        holds more than one draw's worth of forwards. The baseline is a
        decaying mean of past rewards (0.9/0.1) and rides the checkpoint."""
        node = self.dataset_box.node
        row = self.rows[self._pos % max(len(self.rows), 1)]
        self._pos += 1
        cells = row.get("cells") or []
        xc = self.x_col[0] if isinstance(self.x_col, (list, tuple)) else self.x_col
        text = (cells[xc]
                if xc is not None and xc < len(cells)
                else str(row.get("prompt") or ""))
        x_ids = _ds_encode(node, text)
        if not x_ids:
            raise CanvasError(node.get("name") or "dataset",
                               "its row has no prompt to sample from")
        n = max(1, int(self.cfg.get("n_samples") or 1))
        max_tokens = max(1, int(self.cfg.get("max_tokens") or 32))
        temperature = float(self.cfg.get("temperature") or 1.0)
        # pass 1: draw the samples and score them — no gradients
        samples = []
        for i in range(n):
            gen = torch.Generator(device="cpu").manual_seed(self.step * 1000 + i)
            with torch.no_grad():
                ids = self._sample(x_ids, max_tokens, temperature, gen)
            y_text = _ds_decode(node, ids[len(x_ids):])
            reward = float(reward_eval(self.reward_node, text, y_text))
            samples.append((ids, reward, y_text))
        # the baseline: the decaying mean of past rewards
        mean_r = sum(s[1] for s in samples) / len(samples)
        self.baseline = (mean_r if self.baseline is None
                          else self.baseline * 0.9 + mean_r * 0.1)
        # pass 2: the policy gradient, one sample's graph at a time
        self.optim.zero_grad(set_to_none=True)
        loss_val = 0.0
        for ids, reward, _y_text in samples:
            adv = reward - self.baseline
            if adv == 0.0:
                continue
            logp = self._logprob(x_ids, ids)
            if logp is None:
                continue
            loss_i = -adv * logp / n
            loss_val += float(loss_i.detach())
            loss_i.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in self.trainable], float(self.cfg.get("clip") or 1.0))
        self.optim.step()
        self.step += 1
        self.history.append(loss_val)
        self.history = self.history[-400:]
        best = max(samples, key=lambda s: s[1])
        # the reward rides the test slots: tloss is the mean reward, tacc
        # the baseline — the state file keeps its shape
        ev = {"type": "loss", "step": self.step, "loss": loss_val,
              "reward": mean_r, "baseline": self.baseline,
              "sample": best[2][:120],
              "tloss": mean_r, "tacc": self.baseline}
        say(ev)
        if self.step % max(1, int(self.cfg.get("save_every") or 50)) == 0:
            say({"type": "checkpoint", "path": self.save()})

    def run(self, steps=1, on_event=None, command=None):
        """One training session. `command()` may answer 'save' or 'detach';
        detach is graceful — it lands a checkpoint before it returns. Every
        `eval_every` steps the test split is scored and rides the event in
        `tloss`/`tacc`; in rl the mean reward and the baseline ride those
        slots instead."""
        say = on_event or (lambda e: None)
        for _ in range(int(steps)):
            cmd = command() if command else None
            if cmd == "save":
                say({"type": "checkpoint", "path": self.save()})
                continue
            if cmd == "detach":
                path = self.save()
                say({"type": "checkpoint", "path": path, "detached": True})
                return {"state": "detached", "step": self.step, "checkpoint": path}
            if self.paradigm == "rl":
                self._rl_step(say)
                continue
            ids, labels = self._batch()
            for b in self.dataset_boxes:
                b.batch = labels       # every dataset box runs in the forward
            vals = self.graph(feeds={self.input_id: ids})
            loss, acc = self._loss(vals, labels)
            if loss is None:
                raise CanvasError("loss", "the design produced no loss")
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.trainable], float(self.cfg.get("clip") or 1.0))
            self.optim.step()
            self.step += 1
            self.history.append(float(loss.detach()))
            self.history = self.history[-400:]
            ev = {"type": "loss", "step": self.step, "loss": float(loss.detach())}
            if acc is not None:
                ev["acc"] = acc
            if (self.test_box is not None and self.test_rows
                    and self.step % max(1, int(self.cfg.get("eval_every") or 50)) == 0):
                tloss, tacc = self._eval()
                if tloss is not None:
                    ev["tloss"] = float(tloss.detach())
                if tacc is not None:
                    ev["tacc"] = tacc
            say(ev)
            if self.step % max(1, int(self.cfg.get("save_every") or 50)) == 0:
                say({"type": "checkpoint", "path": self.save()})
        return {"state": "ok", "step": self.step,
                "loss": self.history[-1] if self.history else None}

    def save(self, path=None):
        """A named-parameter checkpoint: trainable tensors, the optimizer,
        the step — enough to resume bit-for-bit."""
        path = path or os.path.join(self.checkpoint_root,
                                    self.name, "step-%06d" % self.step)
        os.makedirs(path, exist_ok=True)
        state = {"step": self.step, "config": dict(self.cfg),
                 "trainable": {n: p.detach().cpu() for n, p in self.trainable},
                 "optim": self.optim.state_dict(), "tpos": self._tpos,
                 "baseline": self.baseline}
        torch.save(state, os.path.join(path, "state.pt"))
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"step": self.step,
                       "loss": self.history[-1] if self.history else None,
                       "rows": len(self.rows),
                       "params": sum(p.numel() for _, p in self.trainable),
                       "saved_at": time.time()}, fh)
        latest = os.path.join(self.checkpoint_root, self.name, "latest.json")
        os.makedirs(os.path.dirname(latest), exist_ok=True)
        with open(latest, "w", encoding="utf-8") as fh:
            json.dump({"checkpoint": path, "step": self.step}, fh)
        return path

    def load(self, path):
        """Resume: the trainable tensors, the optimizer and the step come back
        where they were — including the row positions of both splits, and in
        rl the decaying baseline (a row per step, not a batch)."""
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "latest.json")):
            with open(os.path.join(path, "latest.json"), encoding="utf-8") as fh:
                path = json.load(fh).get("checkpoint") or path
        f = os.path.join(path, "state.pt")
        if not os.path.isfile(f):
            raise CanvasError("", "no checkpoint at %s" % path)
        state = torch.load(f, map_location=self.device, weights_only=False)
        own = dict(self.trainable)
        for name, t in (state.get("trainable") or {}).items():
            if name in own:
                own[name].data.copy_(t.to(self.device))
        if state.get("optim"):
            self.optim.load_state_dict(state["optim"])
        self.step = int(state.get("step") or 0)
        self._pos = (self.step * (1 if self.paradigm == "rl" else self.batch_size)
                     % max(len(self.rows), 1))
        self._tpos = int(state.get("tpos") or 0)
        self.baseline = state.get("baseline", self.baseline)
        return path


def apply_checkpoint(graph, path, device="cpu"):
    """A checkpoint's trained tensors onto a fresh graph — the weights of a
    finished run, on the design's live weights, so a trained design can be
    run (or kept warm) without retraining. Names must match the design.
    Returns the step the checkpoint landed at."""
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "latest.json")):
        with open(os.path.join(path, "latest.json"), encoding="utf-8") as fh:
            path = json.load(fh).get("checkpoint") or path
    f = os.path.join(path, "state.pt")
    if not os.path.isfile(f):
        raise CanvasError("", "no checkpoint at %s" % path)
    state = torch.load(f, map_location=device, weights_only=False)
    own = dict(graph.named_parameters())
    hit = 0
    for name, t in (state.get("trainable") or {}).items():
        if name in own:
            with torch.no_grad():
                own[name].data.copy_(t.to(device))
            hit += 1
    if not hit:
        raise CanvasError("", "the checkpoint's tensors match no box of this design")
    return int(state.get("step") or 0)


def _node_by_id(canvas, nid):
    """The node with this id at any nesting level — or None."""
    def walk(nodes):
        for n in nodes or []:
            if n.get("id") == nid:
                return n
            if isinstance(n.get("inner"), dict):
                hit = walk((n.get("inner") or {}).get("nodes"))
                if hit is not None:
                    return hit
        return None
    return walk((canvas or {}).get("nodes"))


def _find_kind(canvas, kind):
    def walk(nodes):
        for n in nodes or []:
            if n.get("kind") == kind:
                return n.get("id")
            if isinstance(n.get("inner"), dict):
                hit = walk((n.get("inner") or {}).get("nodes"))
                if hit:
                    return hit
        return None
    return walk((canvas or {}).get("nodes"))


def _find_kinds(canvas, kind):
    """Every node of a kind, at every nesting level — the id list _find_kind
    truncates."""
    found = []

    def walk(nodes):
        for n in nodes or []:
            if n.get("kind") == kind:
                found.append(n.get("id"))
            if isinstance(n.get("inner"), dict):
                walk((n.get("inner") or {}).get("nodes"))

    walk((canvas or {}).get("nodes"))
    return found
