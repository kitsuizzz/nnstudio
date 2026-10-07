#!/usr/bin/env python3
"""nnstudio's Modal runner — the remote half of the run/train switch.

Deploy once (the server does it lazily on the first remote job, or by hand):

    .venv/bin/modal deploy nnstudio_modal.py

One shared volume, `nnstudio-work`:

    /work/gguf/<key>                  local GGUFs uploaded for a job
    /work/jobs/<job id>/canvas.json   the design the job runs
    /work/jobs/<job id>/status.json   live progress (read by the server)
    /work/jobs/<job id>/stop          the stop flag the server writes
    /work/jobs/<job id>/nnstudio.pt   the checkpoint

The loop itself is `nn_torch` — the same module the local server imports —
so a remote run and a local one are the same code.
"""
import json
import os
import time

import modal

APP = "nnstudio-runner"
VOLUME = "nnstudio-work"

app = modal.App(APP)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu129")
    .pip_install("numpy", "gguf==0.19.0")
    .add_local_python_source("nn_torch")
)

vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/work": vol})
def train(payload: dict) -> dict:
    """One training job: every non-frozen weight, char-level next token."""
    import torch  # noqa: F401 — proves the CUDA build on the rented card
    import nn_torch as T

    job = payload["job_id"]
    jdir = "/work/jobs/%s" % job
    os.makedirs(jdir, exist_ok=True)
    status_path = os.path.join(jdir, "status.json")
    status = {"state": "running", "where": "modal", "step": 0,
              "steps": int(payload["config"].get("steps") or 0), "created": time.time()}
    throttle = {"last": 0.0}
    last_reload = [0.0]

    def write(force=False):
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f)
        os.replace(tmp, status_path)
        now = time.time()
        if force or now - throttle["last"] > 1.5:
            vol.commit()
            throttle["last"] = now

    def progress(**kw):
        for k, v in kw.items():
            if v is None:
                continue
            if k == "loss":
                status["loss"] = v
                status.setdefault("losses", []).append(v)
                del status["losses"][:-4000]
            elif k == "eval":
                status.setdefault("evals", []).append({"step": status.get("step", 0), "loss": v})
            elif k == "sample":
                status.setdefault("samples", []).append({"step": status.get("step", 0), "text": v})
                del status["samples"][:-50]
            else:
                status[k] = v
        write(force=kw.get("state") != "running")

    def should_stop():
        now = time.time()
        if now - last_reload[0] > 5:
            last_reload[0] = now
            try:
                vol.reload()                      # so the driver's stop file is visible
            except Exception:  # noqa: BLE001
                pass
        return os.path.exists(os.path.join(jdir, "stop"))

    try:
        result = T.train(payload["canvas"], payload["text"],
                         payload["input"], payload["output"], payload["config"],
                         device="cuda", progress=progress, should_stop=should_stop,
                         checkpoint_path=os.path.join(jdir, "nnstudio.pt"))
        for k in ("state", "step", "trainable_params", "params", "frozen_params",
                  "loss", "elapsed"):
            if result.get(k) is not None:
                status[k] = result[k]
        status["evals"] = result.get("evals", status.get("evals", []))
        status["samples"] = result.get("samples", status.get("samples", []))
        status["losses"] = result.get("losses", status.get("losses", []))[-4000:]
        status["checkpoint"] = os.path.join(jdir, "nnstudio.pt")
        write(force=True)
        vol.commit()
        return {"ok": True, "state": status["state"], "checkpoint": status["checkpoint"]}
    except Exception as e:  # noqa: BLE001 — the driver shows the reason
        status["state"] = "error"
        status["error"] = str(e)
        write(force=True)
        vol.commit()
        return {"ok": False, "error": str(e)}


@app.function(image=image, gpu="T4", timeout=600)
def smoke() -> dict:
    """The one-time setup check: a GPU, a torch that sees it, the volume."""
    import torch
    vol.reload()
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "volume": True}
