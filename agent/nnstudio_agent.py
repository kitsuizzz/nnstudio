#!/usr/bin/env python3
"""nnstudio's agent — the user's own backend.

Runs on the machine that will execute the design (ideally with a GPU) and
keeps a **stateful connection** to the nnstudio app: it dials the app's
control port, says hello with this machine's specs, and then stays connected,
answering the app's commands — heartbeat pings, and `run` with a canvas to
execute on this machine. The app dispatches; this runs.

    python nnstudio_agent.py --app 127.0.0.1:8125 [--name my-pc]

Requires: python3, torch (CUDA build recommended), numpy. `gguf` is optional —
only designs bound to real GGUF files need it.

The protocol is newline-delimited JSON over one TCP socket, both directions:

    -> {"type": "hello", "name": ..., "specs": {...}}
    <- {"type": "welcome", "app": "nnstudio"}
    <- {"type": "ping", "t": ...}       -> {"type": "pong", "t": ...}
    <- {"type": "run", "id": "...", "canvas": {...}, "feeds": {...}}
    -> {"type": "result", "id": "...", "ok": true, "result": {...}}
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

# a card that fills up fragments; let the allocator grow segments instead
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

VERSION = 1
_DEVICE = None


def _device():
    """cuda when torch can see one — decided once, not per run."""
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            _DEVICE = "cpu"
    return _DEVICE


# ---------------------------------------------------------------- this machine
def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        q = subprocess.run([exe, "--query-gpu=name,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if q.returncode == 0 and q.stdout.strip():
            name, mib = [p.strip() for p in q.stdout.strip().splitlines()[0].split(",")]
            return name or None, round(float(mib) / 1024, 1)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None, None


_CAPS = None


def native_caps():
    """The native-quant capability manifest (agent/nn_quant), detected once:
    triton/cuda kernels for gguf types, and optional providers present."""
    global _CAPS
    if _CAPS is None:
        try:
            import nn_quant
            _CAPS = nn_quant.capabilities()
        except Exception as e:  # noqa: BLE001 — the app still gets the machine
            _CAPS = {"available": False, "kernels": [], "providers": {},
                     "error": str(e)[:120]}
    return _CAPS


def specs():
    """What the app shows under Backend -> Local."""
    gpu, vram = _nvidia()
    s = {"name": socket.gethostname(), "os": platform.platform(),
         "platform": platform.system(), "arch": platform.machine(),
         "python": platform.python_version(), "cpu": _cpu_model(),
         "cores": os.cpu_count(), "ram_gb": _ram_gb(),
         "gpu": gpu, "vram_gb": vram, "rocm": None, "mps": False,
         "torch": None, "cuda": False, "device": "cpu", "torch_error": None,
         "native_quant": native_caps()}
    try:
        import torch
        s["torch"] = torch.__version__
        s["cuda"] = bool(torch.cuda.is_available())
        s["device"] = "cuda" if s["cuda"] else "cpu"
        s["rocm"] = getattr(torch.version, "hip", None)
        mps = getattr(torch.backends, "mps", None)
        s["mps"] = bool(mps and mps.is_available())
        if s["cuda"]:
            s["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            s["vram_gb"] = round(total / 2 ** 30, 1)
            s["vram_free_gb"] = round(free / 2 ** 30, 1)
    except Exception as e:  # noqa: BLE001 — the app shows the machine anyway
        s["torch_error"] = str(e)[:160]
    return s


def usage():
    """Right now: system RAM and GPU VRAM, the way a task manager reads them.
    Pushed to the app every few seconds, so the Run menu stays current."""
    u = {"ram_used_gb": None, "ram_total_gb": None, "ram_pct": None,
         "vram_used_gb": None, "vram_total_gb": None, "vram_pct": None}
    try:
        total = avail = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        if total and avail is not None:
            used = total - avail
            u["ram_total_gb"] = round(total / 2 ** 30, 1)
            u["ram_used_gb"] = round(used / 2 ** 30, 1)
            u["ram_pct"] = round(100.0 * used / total)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)   # the device, all processes
            used = total - free
            u["vram_total_gb"] = round(total / 2 ** 30, 1)
            u["vram_used_gb"] = round(used / 2 ** 30, 1)
            u["vram_pct"] = round(100.0 * used / total)
    except Exception:  # noqa: BLE001 — a machine without torch still reports RAM
        pass
    return u


def _canvas_key(canvas, device):
    """Is this the design the agent already has loaded? Compared by content."""
    return hashlib.sha1(
        json.dumps([device, canvas], sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


# ---------------------------------------------------------------- the channel
def _send(f, obj):
    f.write((json.dumps(obj) + "\n").encode())
    f.flush()


def _parse_app(text):
    if ":" in text:
        host, port = text.rsplit(":", 1)
        return host, int(port)
    return text, 8125


def _release(e):
    """Let a failing traceback go: its frames' locals can hold the whole
    compiled graph, and a loop as long-lived as this one can keep them
    breathing long after the reply was sent. Clearing the frames drops the
    tensors now; the exception itself, if still referenced, is harmless."""
    tb = getattr(e, "__traceback__", None)
    if tb is not None:
        try:
            import traceback
            traceback.clear_frames(tb)
        except Exception:  # noqa: BLE001
            pass


def _ram_available():
    """The machine's available RAM right now, in bytes (None when unknown)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _fit_or_error(canvas, device):
    """Refuse a design that cannot fit *before* spending minutes dequantising
    it — the estimate reads the design's JSON, so this is instant."""
    try:
        from nn_torch import estimate_canvas
        est = estimate_canvas(canvas, device)
    except Exception:  # noqa: BLE001 — no estimate, no refusal
        return None
    gb = 2 ** 30
    if device == "cuda" and est["gpu_bytes"]:
        free = None
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info(0)
        except Exception:  # noqa: BLE001
            free = None
        if free is not None and est["gpu_bytes"] > free * 0.92:
            return ("its weights need %.1f GB on the gpu, but only %.1f GB is free "
                    "\u2014 raise the cpu offload bar in Run \u25be and press "
                    "load weights" % (est["gpu_bytes"] / gb, free / gb))
    avail = _ram_available()
    if avail and est["system_bytes"] > avail * 0.92:
        return ("its weights need %.1f GB of system ram, but only %.1f GB is available "
                "\u2014 offload the backend, close other apps, or shrink the design"
                % (est["system_bytes"] / gb, avail / gb))
    return None


def _load_progress(rid, total, send, device="cpu"):
    """An `on_box` callback that streams how far a compile is: bytes landed
    of bytes expected, at most every 1.5 s."""
    native_gpu = "cuda" in str(device)
    prog = {"acc": 0, "sent": time.time()}

    def on_box(n):
        try:
            from nn_torch import estimate_node
            _, b = estimate_node(n, native=native_gpu)
        except Exception:  # noqa: BLE001
            b = 0
        prog["acc"] += b
        now = time.time()
        if total and now - prog["sent"] >= 1.5:
            prog["sent"] = now
            try:
                send({"type": "progress", "id": rid, "nodes": [], "done": 0,
                      "load": {"bytes": prog["acc"], "total": total}})
            except OSError:
                pass

    return on_box, prog


def _hf_repos(canvas):
    """Every huggingface model box's repo that is not in memory yet — the run
    is about to download and load it, and that takes a while."""
    try:
        from nn_torch import hf_loaded
    except Exception:  # noqa: BLE001
        return []
    out = []

    def walk(nodes):
        for n in nodes or []:
            if n.get("kind") == "hf" and n.get("repo") and not hf_loaded(n["repo"]):
                out.append(str(n["repo"]))
            if n.get("kind") == "custom":
                walk((n.get("inner") or {}).get("nodes", []))

    walk((canvas or {}).get("nodes"))
    return out


def serve_session(sock, name, log):
    """One connection: hello, then commands until the app hangs up. The
    connection is the state: a `load` stays warm here for the next `run`."""
    f = sock.makefile("rwb")
    lock = threading.Lock()
    stop = threading.Event()

    def send(obj):
        with lock:
            f.write((json.dumps(obj) + "\n").encode())
            f.flush()

    hello = {"type": "hello", "version": VERSION,
             "name": name or socket.gethostname(), "specs": specs()}
    send(hello)
    log("connected — hello sent (%s, %s)" %
        (hello["specs"].get("gpu") or hello["specs"].get("cpu") or "unknown",
         hello["specs"].get("device")))
    send({"type": "usage", "usage": usage()})

    def usage_loop():                            # current RAM/VRAM, every 5 s
        while not stop.wait(5.0):
            try:
                send({"type": "usage", "usage": usage()})
            except OSError:
                break
    threading.Thread(target=usage_loop, daemon=True, name="nnstudio-usage").start()

    loaded = {"key": None, "graph": None}        # the design kept warm on disk/GPU

    def free_loaded():
        loaded["key"] = None
        loaded["graph"] = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    try:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "welcome":
                continue
            if kind == "ping":
                send({"type": "pong", "t": msg.get("t")})
                continue
            if kind == "load":
                rid = msg.get("id")
                t0 = time.time()
                try:
                    from nn_torch import compile_canvas, estimate_canvas  # the bundled engine
                    device = _device()
                    canvas = msg.get("canvas") or {}
                    key = _canvas_key(canvas, device)
                    if (loaded["graph"] is not None and loaded.get("key") == key
                            and loaded.get("report") is not None):
                        rep = dict(loaded["report"])           # already here: no recompile
                        rep["cached"] = True
                        rep["ms"] = int((time.time() - t0) * 1000)
                        log("load %s — already warm (%d params)" % (rid or "?", rep["params"]))
                        send({"type": "usage", "usage": usage()})
                        send({"type": "result", "id": rid, "ok": True, "result": rep})
                        continue
                    graph = None                               # let the old one go first:
                    free_loaded()                              # the fit check must see its memory
                    why = _fit_or_error(canvas, device)
                    if why:
                        from nn_torch import CanvasError
                        raise CanvasError("", why)
                    total = estimate_canvas(canvas, device)["bytes"]
                    on_box, _prog = _load_progress(rid, total, send, device)
                    graph = compile_canvas(canvas, device=device, on_box=on_box)
                    if total:
                        send({"type": "progress", "id": rid, "nodes": [], "done": 0,
                              "load": {"bytes": total, "total": total, "done": True}})
                    graph.eval()
                    rep = graph.weight_report()
                    rep["tensors_total"] = len((canvas.get("weights") or {})) or None
                    rep["device"] = device
                    rep["ms"] = int((time.time() - t0) * 1000)
                    loaded["key"] = key
                    loaded["graph"] = graph
                    loaded["report"] = rep
                    log("load %s — %d params, %.1f GB on %s, %.1f GB on system, %d ms" %
                        (rid or "?", rep["params"],
                         rep["on_gpu_bytes"] / 2 ** 30, device,
                         rep["on_system_bytes"] / 2 ** 30, rep["ms"]))
                    send({"type": "usage", "usage": usage()})
                    send({"type": "result", "id": rid, "ok": True, "result": rep})
                except Exception as e:  # noqa: BLE001 — the app shows the reason
                    _release(e)
                    free_loaded()
                    log("load %s failed: %s" % (rid or "?", e))
                    send({"type": "result", "id": rid, "ok": False,
                          "error": str(e)[:500],
                          "box": getattr(e, "box", None) or None,
                          "nid": getattr(e, "nid", None) or None})
                continue
            if kind == "offload":
                rid = msg.get("id")
                try:
                    before = usage()
                    graph = None                             # the frame's own reference
                    free_loaded()                            # the warm graph, its tensors
                    from nn_torch import offload as engine_offload
                    engine_offload()                         # readers, tokenizers, pages
                    gc.collect()
                    after = usage()
                    log("offload %s \u2014 ram %.1f\u2192%.1f GB, vram %.1f\u2192%.1f GB" %
                        (rid or "?", before["ram_used_gb"] or 0, after["ram_used_gb"] or 0,
                         before["vram_used_gb"] or 0, after["vram_used_gb"] or 0))
                    send({"type": "usage", "usage": after})
                    send({"type": "result", "id": rid, "ok": True,
                          "result": {"before": before, "after": after}})
                except Exception as e:  # noqa: BLE001 — the app shows the reason
                    log("offload %s failed: %s" % (rid or "?", e))
                    send({"type": "result", "id": rid, "ok": False,
                          "error": str(e)[:500]})
                continue
            if kind == "run":
                rid = msg.get("id")
                t0 = time.time()
                failed = True
                try:
                    from nn_torch import compile_canvas, run_graph   # the bundled engine
                    device = _device()
                    canvas = msg.get("canvas") or {}
                    key = _canvas_key(canvas, device)
                    reused = loaded["graph"] is not None and loaded["key"] == key
                    opts = dict(generate=int(msg.get("generate") or 0),
                                temperature=float(0.8 if msg.get("temperature") is None
                                                  else msg.get("temperature")),
                                input_id=msg.get("input"), output_id=msg.get("output"),
                                sample_seed=int(msg.get("seed") or 1))
                    seq = []
                    sent = [0]                 # marks not yet streamed
                    last = [time.time()]

                    def on_node(nid, seq=seq, rid=rid, sent=sent, last=last):
                        seq.append(nid)                  # the data-flow order
                        now = time.time()                # tight sync with the compute:
                        if len(seq) - sent[0] >= 8 or now - last[0] >= 0.3:
                            batch = seq[sent[0]:]        # every 8 marks, or 300 ms
                            sent[0] = len(seq)
                            last[0] = now
                            try:
                                send({"type": "progress", "id": rid,
                                      "nodes": batch, "done": len(seq)})
                            except OSError:
                                pass

                    def on_step(k, rid=rid):
                        try:                             # a new forward pass: the
                            send({"type": "progress", "id": rid, "step": k,
                                  "nodes": [], "done": 0})   # watcher re-walks
                        except OSError:
                            pass

                    for repo in _hf_repos(canvas):       # a download takes a while
                        send({"type": "progress", "id": rid, "nodes": [], "done": 0,
                              "note": "loading %s from huggingface\u2026 "
                                      "(the first run downloads it)" % repo})

                    if reused:
                        result = run_graph(loaded["graph"], canvas,
                                           msg.get("feeds") or {}, on_node=on_node,
                                           on_step=on_step, **opts)
                    else:
                        graph = None                 # this frame outlives the iteration:
                        free_loaded()                # drop its graph before freeing, or the
                        why = _fit_or_error(canvas, device)   # memory is never let go
                        if why:
                            from nn_torch import CanvasError
                            raise CanvasError("", why)
                        from nn_torch import estimate_canvas
                        total = estimate_canvas(canvas, device)["bytes"]
                        on_box, _prog = _load_progress(rid, total, send, device)
                        graph = compile_canvas(canvas, device=device, on_box=on_box)
                        graph.eval()
                        rep = graph.weight_report()
                        rep["tensors_total"] = len((canvas.get("weights") or {})) or None
                        rep["device"] = device
                        rep["ms"] = int((time.time() - t0) * 1000)
                        loaded["key"] = key          # a run compiles warm: the next
                        loaded["graph"] = graph      # run reuses it, no reload
                        loaded["report"] = rep
                        log("run %s — compiled and warm — %d params, %.1f GB on %s, "
                            "%.1f GB on system, %d ms" %
                            (rid or "?", rep["params"], rep["on_gpu_bytes"] / 2 ** 30,
                             device, rep["on_system_bytes"] / 2 ** 30, rep["ms"]))
                        result = run_graph(graph, canvas, msg.get("feeds") or {},
                                           on_node=on_node, on_step=on_step, **opts)
                    result["device"] = device
                    result["loaded"] = reused
                    if seq:
                        result["flow"] = seq             # the whole order, for the app
                    log("run %s — %d boxes, %d ms on %s%s" %
                        (rid or "?", len(result.get("values") or {}),
                         int((time.time() - t0) * 1000), device,
                         " (loaded)" if reused else ""))
                    try:                              # a run diagnostic: what the
                        fsum = {}                     # design actually saw and said
                        for k, v in (msg.get("feeds") or {}).items():
                            fsum[k] = (str(v)[:60] if isinstance(v, str)
                                       else list(v)[:12] if isinstance(v, (list, tuple))
                                       else str(v)[:40])
                        smp = []
                        edges = (msg.get("canvas") or {}).get("edges") or []
                        flags = {"v_tiled": 0, "dim": 0, "gain": 0, "causal": 0, "outs": []}
                        def _walk(nds, where=""):
                            for nn in nds or []:
                                if nn.get("op") == "scan" and nn.get("v_tiled"):
                                    flags["v_tiled"] += 1
                                if nn.get("op") == "rope" and nn.get("dim"):
                                    flags["dim"] += 1
                                if nn.get("kind") == "act" and nn.get("gain") is not None:
                                    flags["gain"] += 1
                                if nn.get("kind") == "act" and nn.get("causal"):
                                    flags["causal"] += 1
                                if nn.get("kind") == "output" and not where:
                                    flags["outs"].append((nn.get("id"), nn.get("name"),
                                                          nn.get("x"), nn.get("y")))
                                if nn.get("kind") == "sample":
                                    srcs = [(e.get("from"), e.get("fromPort"))
                                            for e in edges if e.get("to") == nn.get("id")]
                                    smp.append({kk: nn.get(kk) for kk in
                                                ("id", "name", "temperature", "sample_seed",
                                                 "top_k", "top_p")} | {"from": srcs})
                                if nn.get("kind") == "custom" and isinstance(nn.get("inner"), dict):
                                    _walk((nn.get("inner") or {}).get("nodes"), where or nn.get("id"))
                        _walk((msg.get("canvas") or {}).get("nodes"))
                        picks, ids_, shapes = [], [], []
                        for pth, st in (result.get("values") or {}).items():
                            if not isinstance(st, dict):
                                continue
                            if st.get("scalar") is not None and (st.get("shape") or []) == [1]:
                                picks.append((pth, st["scalar"]))
                            pk = st.get("peek") or []
                            if pk and len(pk) <= 8 and (st.get("shape") or [0])[-1] <= 8 \
                               and "/" not in pth:
                                ids_.append((pth, pk))
                            sh = st.get("shape") or []
                            if len(sh) == 2 and sh[1] > 1000 and pth.count("/") == 0:
                                shapes.append((pth, sh))
                            if pk and len(pk) <= 8 and pth.endswith("/gg001"):
                                ids_.append((pth, pk))
                        log("run %s diag — flags=%r feeds=%r texts=%r "
                            "sample=%r picks=%r ids=%r logits=%r"
                            % (rid or "?", flags, fsum,
                               result.get("texts"),
                               smp, picks, ids_, shapes))
                    except Exception:  # noqa: BLE001 — diagnostics never break a run
                        pass
                    send({"type": "usage", "usage": usage()})
                    send({"type": "result", "id": rid, "ok": True, "result": result})
                    failed = False
                except Exception as e:  # noqa: BLE001 — the app shows the reason
                    _release(e)                  # the frames held the graph
                    log("run %s failed: %s" % (rid or "?", e))
                    send({"type": "result", "id": rid, "ok": False,
                          "error": str(e)[:500],
                          "box": getattr(e, "box", None) or None,
                          "nid": getattr(e, "nid", None) or None})
                if failed:                     # hand the freed pages back now
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                continue
    finally:
        stop.set()
        free_loaded()


def main():
    ap = argparse.ArgumentParser(description="nnstudio's agent — the user's backend")
    ap.add_argument("--app", default="127.0.0.1:8125",
                    help="the nnstudio app's control address (host:port)")
    ap.add_argument("--name", default=None, help="a name for this machine")
    ap.add_argument("--once", action="store_true", help="connect once, do not retry")
    args = ap.parse_args()
    host, port = _parse_app(args.app)

    def log(line):
        print("[nnstudio-agent] " + line, flush=True)

    log("agent v%d for %s:%d" % (VERSION, host, port))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.settimeout(None)
            serve_session(sock, args.name, log)
            log("the app closed the connection")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            log("cannot reach the app at %s:%d — %s" % (host, port, e))
        finally:
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass
        if args.once:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
