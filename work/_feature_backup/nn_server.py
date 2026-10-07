#!/usr/bin/env python3
"""nnstudio's server — the app, its model list, the GGUF -> canvas builder,
and the torch runner behind the canvas' run ribbon.

The design side is stdlib + nn_engine (the UI, the model list, the header-only
builder, the GPU badge); nn_torch is imported lazily, so the page opens
instantly and only a run or a train pays for torch.

    python3 nn_server.py [--host 127.0.0.1] [--port 8124]

Runner endpoints:
    GET  /api/where              what can run where (torch/CUDA, Modal, llama.cpp)
    POST /api/run                one forward pass (or a short generate)
    POST /api/train              train every non-frozen weight; local or Modal
    GET  /api/jobs[/<id>]        job status (losses, evals, samples)
    POST /api/jobs/<id>/stop     stop a job (the checkpoint is saved first)
    GET  /api/jobs/<id>/checkpoint   download the checkpoint
    POST /api/gguf/run           a loaded GGUF runs as-is through llama.cpp
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import nn_engine

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui", "index.html")
MODELS_DIR = os.path.join(HERE, "models")
WORK = os.path.join(HERE, "work")
TRAIN_DIR = os.path.join(WORK, "training")

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
TORCH_LOCK = threading.Lock()              # one torch job at a time (one GPU)
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOB_HISTORY = 40

# Modal list prices, approximate ($/GPU-hour) — the pane labels them "~";
# the Modal dashboard is the source of truth.
MODAL_GPUS = [
    {"name": "T4", "mib": 15360, "usd_hr": 0.59},
    {"name": "L4", "mib": 23034, "usd_hr": 0.80},
    {"name": "A10G", "mib": 23028, "usd_hr": 1.10},
    {"name": "A100-40GB", "mib": 40960, "usd_hr": 2.10},
    {"name": "A100-80GB", "mib": 81920, "usd_hr": 2.50},
    {"name": "H100", "mib": 81920, "usd_hr": 3.95},
    {"name": "H200", "mib": 143771, "usd_hr": 4.54},
]
MODAL_APP = "nnstudio-runner"
MODAL_VOLUME = "nnstudio-work"

# The badge is polled every 15 s; nvidia-smi takes ~50 ms, so cache it for a
# beat to keep the poll free.
_GPU = {"time": 0.0, "data": None}
_GPU_LOCK = threading.Lock()
GPU_TTL = 2.0
_WHERE = {"time": 0.0, "data": None}
_WHERE_LOCK = threading.Lock()
WHERE_TTL = 15.0


def gpu_info():
    """The badge's numbers: name, memory, utilization, temperature."""
    now = time.time()
    with _GPU_LOCK:
        if _GPU["data"] is not None and now - _GPU["time"] < GPU_TTL:
            return _GPU["data"]
    info = {"cuda": False, "name": None, "total_mib": None, "free_mib": None,
            "used_mib": None, "util_pct": None, "temp_c": None}
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if q.returncode == 0 and q.stdout.strip():
            parts = [p.strip() for p in q.stdout.strip().splitlines()[0].split(",")]

            def num(i):
                try:
                    return int(float(parts[i]))
                except (ValueError, IndexError):
                    return None

            info = {"cuda": True, "name": parts[0] or None, "total_mib": num(1),
                    "used_mib": num(2), "free_mib": num(3), "util_pct": num(4),
                    "temp_c": num(5)}
    except (OSError, subprocess.SubprocessError):
        pass
    with _GPU_LOCK:
        _GPU["time"] = now
        _GPU["data"] = info
    return info


def list_models():
    """Local GGUFs, in the shape the UI's GGUF form reads."""
    out = []
    try:
        names = sorted(os.listdir(MODELS_DIR))
    except OSError:
        names = []
    for name in names:
        if not name.lower().endswith(".gguf"):
            continue                      # vision encoders etc. are no canvas
        path = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        out.append({"kind": "gguf", "name": name, "path": "models/" + name,
                    "size_mb": round(size / 2 ** 20)})
    return out


# ---------------------------------------------------------------- the engine
def engine():
    """nn_torch, imported once and only when a run or a train needs it."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            import nn_torch
            _ENGINE = nn_torch
        return _ENGINE


def where_info():
    """What can run where: local torch/CUDA, Modal, llama.cpp — cached."""
    now = time.time()
    with _WHERE_LOCK:
        if _WHERE["data"] is not None and now - _WHERE["time"] < WHERE_TTL:
            return _WHERE["data"]
    local = {"torch": False, "cuda": False, "name": None, "free_mib": None,
             "total_mib": None, "device": "cpu"}
    try:
        import torch
        local["torch"] = True
        if torch.cuda.is_available():
            local["cuda"] = True
            local["name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            local["free_mib"], local["total_mib"] = int(free / 2 ** 20), int(total / 2 ** 20)
            local["device"] = "cuda"
    except Exception as e:  # noqa: BLE001 — torch missing or a broken CUDA
        local["error"] = str(e)[:200]
    g = gpu_info()
    if local["free_mib"] is None and g.get("free_mib") is not None:
        local["free_mib"], local["total_mib"] = g["free_mib"], g["total_mib"]
        local["name"] = local["name"] or g.get("name")
    data = {"ok": True, "local": local, "modal": modal_status(),
            "llama_cli": llama_cli(), "gpus": MODAL_GPUS}
    with _WHERE_LOCK:
        _WHERE["time"] = now
        _WHERE["data"] = data
    return data


def modal_status():
    """Is Modal installed, authenticated, is the app deployed?"""
    st = {"installed": False, "token": False, "deployed": False, "app": MODAL_APP,
          "volume": MODAL_VOLUME, "note": None}
    try:
        import modal
        st["installed"] = True
        st["version"] = getattr(modal, "__version__", "?")
    except Exception:  # noqa: BLE001
        st["note"] = "the modal package is not installed in the server's venv"
        return st
    token = (bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
             or os.path.exists(os.path.expanduser("~/.modal.toml")))
    st["token"] = token
    if not token:
        st["note"] = "run  .venv/bin/modal token new  once, then reload"
        return st
    try:
        modal.Function.from_name(MODAL_APP, "train")
        st["deployed"] = True
    except Exception as e:  # noqa: BLE001
        st["note"] = "not deployed yet (the first remote job deploys it): %s" % str(e)[:160]
    return st


def llama_cli():
    """The llama.cpp CLI the as-is run uses (CUDA build preferred)."""
    env = os.environ.get("NNSTUDIO_LLAMA_CLI")
    if env and os.path.isfile(env):
        return env
    cand = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
    if os.path.isfile(cand):
        return cand
    return shutil.which("llama-cli")


# ---------------------------------------------------------------- jobs
def new_job(where, canvas):
    job = {"id": "j" + uuid.uuid4().hex[:10], "state": "queued", "where": where,
           "created": time.time(), "step": 0, "steps": 0, "losses": [], "evals": [],
           "samples": [], "name": canvas_name(canvas), "stop": False,
           "checkpoint": None, "error": None, "gpu": None, "device": None,
           "max_minutes": 0, "max_usd": 0, "cost_usd": None, "modal_call": None}
    with _JOBS_LOCK:
        _JOBS[job["id"]] = job
        if len(_JOBS) > _MAX_JOB_HISTORY:
            for k in sorted(_JOBS, key=lambda k: _JOBS[k]["created"])[:-_MAX_JOB_HISTORY]:
                if _JOBS[k]["state"] in ("done", "stopped", "error", "capped"):
                    _JOBS.pop(k, None)
    return job


def canvas_name(canvas):
    return (canvas or {}).get("name") or "canvas"


def job_public(job, full=True):
    out = {k: v for k, v in job.items() if k not in ("modal_call", "thread")}
    if not full:
        for k in ("losses", "evals", "samples"):
            out.pop(k, None)
    return out


def job_progress(job, **kw):
    """The engine's progress callback: fields land on the job, lists within caps."""
    with _JOBS_LOCK:
        for k, v in kw.items():
            if v is None:
                continue
            if k == "loss":
                job["loss"] = v
                job["losses"].append(v)
                if len(job["losses"]) > 5000:
                    del job["losses"][:-5000]
            elif k == "eval":
                job["eval"] = v
                job["evals"].append({"step": job.get("step", 0), "loss": v})
                if len(job["evals"]) > 1000:
                    del job["evals"][:-1000]
            elif k == "sample":
                job["samples"].append({"step": job.get("step", 0), "text": v})
                if len(job["samples"]) > 50:
                    del job["samples"][:-50]
            else:
                job[k] = v


def _write_job_files(jobdir, canvas, text, config):
    try:
        with open(os.path.join(jobdir, "canvas.json"), "w") as f:
            json.dump(canvas, f)
        with open(os.path.join(jobdir, "data.txt"), "w") as f:
            f.write(text)
        with open(os.path.join(jobdir, "config.json"), "w") as f:
            json.dump(config, f)
    except OSError:
        pass


def start_local(job, canvas, text, config):
    nn = engine()
    params = nn.describe(canvas)["counts"]["params"]
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available() and params > 5_000_000:
            device = "cuda"
    except Exception:  # noqa: BLE001
        pass
    job["device"] = device
    jobdir = os.path.join(TRAIN_DIR, job["id"])

    def run():
        try:
            os.makedirs(jobdir, exist_ok=True)
            _write_job_files(jobdir, canvas, text, config)
            result = nn.train(canvas, text, config.get("input"), config.get("output"),
                              config, device=device,
                              progress=lambda **kw: job_progress(job, **kw),
                              should_stop=lambda: job["stop"],
                              checkpoint_path=os.path.join(jobdir, "nnstudio.pt"))
            job_progress(job, state=result.get("state", "done"), loss=result.get("loss"),
                         tps=result.get("tps", 0), elapsed=result.get("elapsed", 0),
                         trainable_params=result.get("trainable_params"),
                         params=result.get("params"),
                         frozen_params=result.get("frozen_params"),
                         checkpoint=result.get("checkpoint"))
            with _JOBS_LOCK:
                job["finished"] = time.time()
        except Exception as e:  # noqa: BLE001 — the UI shows the reason
            with _JOBS_LOCK:
                job["state"] = "error"
                job["error"] = str(e)
                job["finished"] = time.time()

    job["state"] = "running"
    t = threading.Thread(target=run, daemon=True, name="train-" + job["id"])
    job["thread"] = t
    t.start()
    return job


# ---------------------------------------------------------------- modal jobs
def _modal_volume():
    import modal
    return modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)


def _vol_read_all(volume, path):
    import asyncio

    async def read():
        chunks = []
        async for c in volume.read_file(path):
            chunks.append(c)
        return b"".join(chunks)

    return asyncio.run(read())


def _vol_put(volume, local_path, remote_path):
    import asyncio

    async def put():
        async with volume.batch_upload(force=True) as batch:
            batch.put_file(local_path, remote_path)

    asyncio.run(put())


def ensure_modal_deployed():
    """Deploy the runner once per nnstudio_modal.py revision."""
    os.makedirs(WORK, exist_ok=True)
    marker = os.path.join(WORK, ".modal_deployed")
    src = os.path.join(HERE, "nnstudio_modal.py")
    sig = "%s:%s" % (os.path.getmtime(src), os.path.getsize(src))
    try:
        with open(marker) as f:
            if f.read().strip() == sig:
                return None
    except OSError:
        pass
    q = subprocess.run([sys.executable, "-m", "modal", "deploy", src],
                       cwd=HERE, capture_output=True, text=True, timeout=900)
    if q.returncode != 0:
        raise RuntimeError("modal deploy failed: %s" % ((q.stderr or q.stdout).strip()[-400:]))
    with open(marker, "w") as f:
        f.write(sig)
    return (q.stdout or "").strip()[-200:]


def _walk_custom(doc, cb, inherited=None):
    for n in doc.get("nodes", []):
        if n.get("kind") == "custom":
            cb(n, inherited)
            if n.get("inner"):
                _walk_custom(n["inner"], cb, n.get("file") or inherited)


def _modal_map_files(canvas):
    """Upload every GGUF the design references; return the canvas with each
    custom box's `file` replaced by its volume path."""
    volume = _modal_volume()
    mapped = json.loads(json.dumps(canvas))
    uploads = {}

    def cb(box, inherited):
        src = box.get("file") or inherited
        target = uploads.get(src) if src else None
        if src and target is None:
            if not os.path.isfile(src):
                raise RuntimeError("the model file is not on disk: %s" % src)
            st = os.stat(src)
            key = "%s-%s-%x" % (os.path.basename(src).replace(" ", "_"),
                                st.st_size, int(st.st_mtime))
            target = "/work/gguf/" + key
            uploads[src] = target
            _vol_put(volume, src, target)
        if target:
            box["file"] = target

    _walk_custom(mapped, cb)
    return mapped


def choose_gpu(needed_bytes, asked=None):
    """An explicit GPU when it fits, else the smallest that does — or None
    when the run needs more than one card (Phase D)."""
    if asked:
        hit = next((g for g in MODAL_GPUS if g["name"] == asked), None)
        if hit is None:
            raise RuntimeError("unknown Modal GPU '%s'" % asked)
        if needed_bytes > hit["mib"] * 2 ** 20 * 0.85:
            raise RuntimeError("this run needs more than %s offers — pick a bigger card"
                               % hit["name"])
        return hit
    for g in MODAL_GPUS:
        if needed_bytes <= g["mib"] * 2 ** 20 * 0.85:
            return g
    return None


def start_modal(job, canvas, text, config):
    nn = engine()
    counts = nn.describe(canvas)["counts"]
    trainable = max(0, counts["params"] - counts["frozen"])
    needed = max(1 << 30, int(trainable * 20))     # bf16 w + fp32 g + adam + master
    gpu = choose_gpu(needed, config.get("gpu"))
    if gpu is None:
        raise RuntimeError(
            "this design needs more than one 80 GB GPU (about %.1f GiB of weights, "
            "gradients and optimizer) — multi-GPU runs are not wired yet; freeze some "
            "boxes or lower the scope" % (needed / 2 ** 30))
    job["gpu"] = gpu["name"]
    job["state"] = "preparing"
    ensure_modal_deployed()

    def run():
        import modal
        try:
            vol = _modal_volume()
            mapped = _modal_map_files(canvas)
            jobdir = os.path.join(TRAIN_DIR, job["id"])
            os.makedirs(jobdir, exist_ok=True)
            _write_job_files(jobdir, canvas, text, config)
            payload = {"job_id": job["id"], "canvas": mapped, "text": text,
                       "config": config, "input": config.get("input"),
                       "output": config.get("output")}
            fn = modal.Function.from_name(MODAL_APP, "train")
            fn = fn.with_options(gpu=gpu["name"],
                                 timeout=max(900, int(config.get("max_minutes") or 30) * 60 + 600))
            call = fn.spawn(payload)
            with _JOBS_LOCK:
                job["modal_call"] = call
                job["state"] = "running"
            _modal_poll(job, vol, call, gpu)
        except Exception as e:  # noqa: BLE001
            with _JOBS_LOCK:
                job["state"] = "error"
                job["error"] = str(e)[:500]
                job["finished"] = time.time()

    t = threading.Thread(target=run, daemon=True, name="mtrain-" + job["id"])
    job["thread"] = t
    t.start()
    return job


def _modal_poll(job, vol, call, gpu):
    """Mirror the volume's status.json into the job and enforce the caps."""
    import io
    status_path = "jobs/%s/status.json" % job["id"]
    t0 = time.time()
    last = None
    done = False
    while not done:
        if job["stop"] and not job.get("_stop_sent"):
            job["_stop_sent"] = True
            try:
                import asyncio

                async def put_stop():
                    async with vol.batch_upload(force=True) as b:
                        b.put_file(io.BytesIO(b"1"), "jobs/%s/stop" % job["id"])
                asyncio.run(put_stop())
            except Exception:  # noqa: BLE001
                pass
        try:
            st = json.loads(_vol_read_all(vol, status_path).decode())
            last = st
            with _JOBS_LOCK:
                for k in ("state", "step", "steps", "loss", "eval", "tps", "elapsed",
                          "trainable_params", "params", "frozen_params", "vocab",
                          "chars", "error", "checkpoint"):
                    if st.get(k) is not None:
                        job[k] = st[k]
                if "losses" in st:
                    job["losses"] = st["losses"]
                if "evals" in st:
                    job["evals"] = st["evals"]
                if "samples" in st:
                    job["samples"] = st["samples"]
        except Exception:  # noqa: BLE001 — no status yet
            pass
        # caps: exceeding one is a hard stop (cancel); a user's stop is soft
        # (the flag file lets the remote run save its checkpoint first)
        elapsed = (last or {}).get("elapsed") if isinstance(last, dict) else None
        hours = (elapsed or (time.time() - t0)) / 3600.0
        cost = hours * gpu["usd_hr"]
        with _JOBS_LOCK:
            job["cost_usd"] = round(cost, 3)
        cap_usd = float(job.get("max_usd") or 0)
        cap_min = float(job.get("max_minutes") or 0)
        hard = (cap_usd and cost > cap_usd) or \
               (cap_min and (time.time() - t0) / 60.0 > cap_min + 2)
        if hard:
            job["stop"] = True
            job_progress(job, state="capped")
            try:
                call.cancel()
            except Exception:  # noqa: BLE001
                pass
            done = True
        if not done:
            try:
                result = call.get(timeout=0.01)
                done = True
                if isinstance(result, dict) and result.get("ok") is False:
                    job_progress(job, state="error", error=result.get("error"))
                elif last and last.get("state") in ("done", "stopped", "capped"):
                    job_progress(job, state=last["state"])
            except Exception as e:  # noqa: BLE001 — still running, or it blew up
                name = type(e).__name__.lower()
                if "timeout" in name or "timeout" in str(e).lower():
                    done = False
                else:
                    done = True
                    job_progress(job, state="error", error=str(e)[:500])
        if not done:
            time.sleep(2)
    try:                                            # pull the checkpoint home
        data = _vol_read_all(vol, "jobs/%s/nnstudio.pt" % job["id"])
        if data:
            jobdir = os.path.join(TRAIN_DIR, job["id"])
            os.makedirs(jobdir, exist_ok=True)
            ckpt = os.path.join(jobdir, "nnstudio.pt")
            with open(ckpt, "wb") as f:
                f.write(data)
            job_progress(job, checkpoint=ckpt)
    except Exception:  # noqa: BLE001
        pass
    with _JOBS_LOCK:
        if job["state"] not in ("error", "capped", "stopped"):
            job["state"] = "done"
        job["finished"] = time.time()


# ---------------------------------------------------------------- request bodies
def _training_request(data):
    """Validate a /api/train body into (canvas, text, config)."""
    canvas = data.get("canvas")
    if not isinstance(canvas, dict) or not canvas.get("nodes"):
        raise ValueError("no canvas in the request")
    text = data.get("text")
    if not isinstance(text, str) or len(text) < 16:
        raise ValueError("the dataset is empty — paste some text to train on")
    if len(text) > 4 * 1024 * 1024:
        raise ValueError("the dataset is larger than 4 MiB — trim it")
    cfg = data.get("config") or {}
    d = engine().describe(canvas)
    if cfg.get("input") not in {i["id"] for i in d["inputs"]}:
        raise ValueError("the training input is not a top-level input box")
    if cfg.get("output") not in {o["id"] for o in d["outputs"]}:
        raise ValueError("the training output is not a top-level output box")
    return canvas, text, cfg


def _num_list(v, what):
    """A feed value from the UI: a number or a list of numbers."""
    if isinstance(v, (int, float)):
        return [v]
    if isinstance(v, str):
        parts = [p for p in v.replace(",", " ").split() if p]
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError("%s: '%s' is not a number" % (what, v)) from None
    if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
        return v
    raise ValueError("%s: expected numbers" % what)


# ---------------------------------------------------------------- the handler
class Handler(BaseHTTPRequestHandler):
    server_version = "nnstudio"

    def log_message(self, fmt, *args):    # keep the journal quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                          # the client hung up mid-answer: fine

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                with open(UI, "rb") as f:
                    body = f.read()
            except OSError as e:
                self._json({"ok": False, "error": str(e)}, 500)
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/models":
            self._json({"models": list_models()})
            return
        if path == "/api/gpu":
            self._json(gpu_info())
            return
        if path == "/api/where":
            self._json(where_info())
            return
        if path == "/api/jobs":
            with _JOBS_LOCK:
                jobs = [job_public(j, full=False) for j in
                        sorted(_JOBS.values(), key=lambda j: j["created"])]
            self._json({"jobs": jobs})
            return
        if path.startswith("/api/jobs/"):
            bits = path.split("/")
            with _JOBS_LOCK:
                job = _JOBS.get(bits[3]) if len(bits) > 3 else None
            if not job:
                self._json({"ok": False, "error": "no such job"}, 404)
                return
            if len(bits) > 4 and bits[4] == "checkpoint":
                ckpt = job.get("checkpoint")
                if not ckpt or not os.path.isfile(ckpt):
                    self._json({"ok": False, "error": "this job has no checkpoint yet"}, 404)
                    return
                with open(ckpt, "rb") as f:
                    blob = f.read()
                self._send(200, blob, "application/octet-stream",
                           {"Content-Disposition": 'attachment; filename="%s"'
                                                   % os.path.basename(ckpt)})
                return
            self._json({"ok": True, "job": job_public(job)})
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._body()
        if path == "/api/gguf/canvas":
            model = str(data.get("model") or "").strip()
            if not model:
                self._json({"ok": False, "error": "model is required"}, 400)
                return
            try:
                self._json(nn_engine.gguf_canvas(model))     # header only, no lock
            except Exception as e:  # noqa: BLE001 — the form shows the message
                self._json({"ok": False, "error": str(e)}, 400)
            return
        if path == "/api/run":
            self._run(data)
            return
        if path == "/api/train":
            self._train(data)
            return
        if path.startswith("/api/jobs/") or path.startswith("/api/train/"):
            bits = path.split("/")
            with _JOBS_LOCK:
                job = _JOBS.get(bits[3]) if len(bits) > 3 else None
            if not job:
                self._json({"ok": False, "error": "no such job"}, 404)
                return
            if len(bits) > 4 and bits[4] == "stop":
                job["stop"] = True
                if job["state"] in ("queued", "preparing"):
                    job["state"] = "stopped"
                self._json({"ok": True, "job": job_public(job, full=False)})
                return
        if path == "/api/gguf/run":
            self._gguf_run(data)
            return
        self._json({"ok": False, "error": "not found"}, 404)

    # ---- run ---------------------------------------------------------------
    def _run(self, data):
        canvas = data.get("canvas")
        if not isinstance(canvas, dict) or not canvas.get("nodes"):
            self._json({"ok": False, "error": "no canvas in the request"}, 400)
            return
        feeds = {}
        try:
            for k, v in (data.get("feeds") or {}).items():
                feeds[k] = _num_list(v, "input")
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
            return
        with _JOBS_LOCK:
            busy = next((j for j in _JOBS.values()
                         if j["state"] == "running" and j.get("device") == "cuda"), None)
        if busy:
            self._json({"ok": False,
                        "error": "a training job is using the GPU (%s) — stop it to run"
                                 % busy["id"]}, 409)
            return
        try:
            nn = engine()
            with TORCH_LOCK:
                res = nn.run(canvas, feeds,
                             generate=int(data.get("generate") or 0),
                             temperature=float(data.get("temperature") or 0.8),
                             top_k=int(data.get("top_k") or 0),
                             top_p=float(data.get("top_p") if data.get("top_p") is not None
                                         else 1.0),
                             sample_seed=int(data.get("seed") or 1),
                             input_id=data.get("input"), output_id=data.get("output"))
            self._json(res)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e), "box": getattr(e, "box", None)}, 400)

    # ---- train -------------------------------------------------------------
    def _train(self, data):
        try:
            canvas, text, config = _training_request(data)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e)}, 400)
            return
        with _JOBS_LOCK:
            busy = next((j for j in _JOBS.values()
                         if j["state"] in ("running", "queued", "preparing")), None)
        if busy:
            self._json({"ok": False, "job": busy["id"],
                        "error": "a job is already running (%s) — stop it first"
                                 % busy["id"]}, 409)
            return
        where = (data.get("where") or "local").lower()
        try:
            job = new_job(where, canvas)
            job["max_minutes"] = float(config.get("max_minutes") or 0)
            job["max_usd"] = float(config.get("max_usd") or 0)
            if where == "modal":
                start_modal(job, canvas, text, config)
            else:
                start_local(job, canvas, text, config)
            self._json({"ok": True, "job": job_public(job, full=False)})
        except Exception as e:  # noqa: BLE001 — bad request or Modal setup
            self._json({"ok": False, "error": str(e)}, 400)

    # ---- gguf as-is --------------------------------------------------------
    def _gguf_run(self, data):
        model = str(data.get("model") or "").strip()
        path = model if os.path.isabs(model) else os.path.join(HERE, model)
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(MODELS_DIR) + os.sep) or not os.path.isfile(real):
            cand = os.path.join(MODELS_DIR, os.path.basename(model))
            if not model.lower().endswith(".gguf") or not os.path.isfile(cand):
                self._json({"ok": False, "error": "pick a GGUF from the local models"}, 400)
                return
            real = cand
        cli = llama_cli()
        if not cli:
            self._json({"ok": False, "error": "llama.cpp was not found — set NNSTUDIO_LLAMA_CLI"}, 400)
            return
        # llama-completion is the plain-completion tool; llama-cli is the
        # conversational one and always needs -no-cnv (otherwise it reads
        # stdin forever and floods the pipe). stdin is /dev/null either way.
        tool = os.path.join(os.path.dirname(cli), "llama-completion")
        if not os.path.isfile(tool):
            tool = cli
        prompt = str(data.get("prompt") or "")
        if not prompt:
            self._json({"ok": False, "error": "a prompt is needed"}, 400)
            return
        n = max(1, min(512, int(data.get("n") or 64)))
        cmd = [tool, "-m", real, "-p", prompt[:8000], "-n", str(n), "-ngl", "999", "-no-cnv",
               "--temp", str(float(data.get("temperature") or 0.8)),
               "--top-k", str(int(data.get("top_k") or 40)),
               "--top-p", str(float(data.get("top_p") if data.get("top_p") is not None else 0.95)),
               "-s", str(int(data.get("seed") if data.get("seed") is not None else -1)),
               "--no-display-prompt", "--no-warmup"]
        t0 = time.time()
        try:
            q = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            self._json({"ok": False, "error": "llama.cpp timed out after 10 minutes"}, 400)
            return
        if q.returncode != 0 and not q.stdout.strip():
            self._json({"ok": False, "error": (q.stderr or "llama.cpp failed").strip()[-400:]}, 400)
            return
        self._json({"ok": True, "text": q.stdout.strip()[:65536],
                    "ms": int((time.time() - t0) * 1000),
                    "model": os.path.basename(real), "prompt": prompt[:8000]})


def main():
    ap = argparse.ArgumentParser(description="nnstudio — the training canvas")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("nnstudio on http://%s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
