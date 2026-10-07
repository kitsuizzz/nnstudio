#!/usr/bin/env python3
"""nnstudio's server — the app, its model list and the GGUF -> canvas builder.

Small on purpose: stdlib + nn_engine. The canvas itself lives in the browser,
so the server only has to
  * serve the one-page UI (`/`),
  * list the local GGUFs (`GET /api/models`) — the GGUF node's form reads it,
  * build a canvas from a model header (`POST /api/gguf/canvas`, lock-free),
  * answer the GPU badge (`GET /api/gpu`, straight from nvidia-smi).

    python3 nn_server.py [--host 127.0.0.1] [--port 8124]
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import nn_engine
import nn_arc
import nn_tasks                # agent_tasks: what the AI agents are doing

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui", "index.html")
MODELS_DIR = os.path.join(HERE, "models")
AGENT_DIR = os.path.join(HERE, "agent")
WEIGHTS_DIR = os.path.join(HERE, "weights")
WEIGHT_EXTS = (".gguf", ".safetensors", ".pt", ".pth", ".bin")


def json_safe(obj):
    """What the browser receives: a NaN/±inf float becomes null. They are not
    JSON — Python writes them as Infinity, and JSON.parse refuses at the minus
    sign of -Infinity ("No number after minus sign in JSON at position …")."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

# nn_torch — the agent's engine — is imported lazily and only for weight
# metadata and weight export (it drags torch in; the app itself stays light).
_ENGINE_MOD = None
_ENGINE_LOCK = threading.Lock()


def weight_engine():
    global _ENGINE_MOD
    with _ENGINE_LOCK:
        if _ENGINE_MOD is None:
            if AGENT_DIR not in sys.path:
                sys.path.insert(0, AGENT_DIR)
            import nn_torch
            _ENGINE_MOD = nn_torch
        return _ENGINE_MOD


def hf_tokenizer_meta(repo):
    """nn_tokenizers' metadata — stdlib only, so no torch is dragged in."""
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)
    import nn_tokenizers
    return nn_tokenizers.tokenizer_meta(repo)


def _weight_path(text):
    """A design's relative path ('weights/x.pt') resolves against the repo."""
    p = str(text or "").strip()
    if p and not os.path.isabs(p):
        p = os.path.join(HERE, p)
    return p


def list_weight_files():
    """Every weight file the app can offer: models/ and weights/."""
    out = []
    for base, label in ((MODELS_DIR, "models"), (WEIGHTS_DIR, "weights")):
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith(WEIGHT_EXTS):
                continue
            full = os.path.join(base, name)
            if not os.path.isfile(full):
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            out.append({"name": name, "path": label + "/" + name,
                        "kind": os.path.splitext(name)[1].lstrip(".").lower(),
                        "size_mb": round(size / 2 ** 20)})
    return out


def _safe_name(text, fallback="nnstudio-weights"):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-._")
    return s[:60] or fallback

# The badge is polled every 15 s; nvidia-smi takes ~50 ms, so cache it for a
# beat to keep the poll free.
_GPU = {"time": 0.0, "data": None}
_GPU_LOCK = threading.Lock()
GPU_TTL = 2.0


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


# ---------------------------------------------------------------- the agents
# The user's backend dials in here and stays: one stateful TCP channel per
# agent, newline-delimited JSON both ways. The agent says hello with the
# machine's specs; the app heartbeats it and dispatches `run` commands.
_AGENTS = {}
_AGENT_LOCK = threading.Lock()
AGENT_TIMEOUT = 45          # no word for this long: the connection is stale


class Agent:
    def __init__(self, conn, addr, name=None):
        self.conn = conn
        self.addr = addr
        self.f = conn.makefile("rwb")
        self.cid = uuid.uuid4().hex[:12]  # this connection, not this machine
        self.name = name
        self.specs = {}
        self.usage = None                # RAM/VRAM, pushed by the agent
        self.progress = None             # the running design's data-flow order
        self.since = time.time()
        self.last_seen = time.time()
        self.busy = 0                    # a load or a run is in flight
        self._pending = {}
        self._lock = threading.Lock()
        self.alive = True

    def send(self, obj):
        with self._lock:
            self.f.write((json.dumps(obj) + "\n").encode())
            self.f.flush()

    def request(self, obj, timeout=1800):
        """Send one command and wait for its result (runs in an HTTP thread).
        While in flight the agent is `busy`: the heartbeat will not drop a
        connection that is simply thinking."""
        rid = uuid.uuid4().hex[:12]
        box = {"event": threading.Event(), "reply": None}
        self._pending[rid] = box
        self.busy += 1
        try:
            try:
                self.send(dict(obj, id=rid))
            except (OSError, ValueError):
                self._pending.pop(rid, None)
                return None
            box["event"].wait(timeout)
            self._pending.pop(rid, None)
            return box["reply"]
        finally:
            self.busy -= 1

    def on_message(self, msg):
        self.last_seen = time.time()
        kind = msg.get("type")
        if kind == "usage":
            self.usage = msg.get("usage") or None
            return
        if kind == "progress":
            if self.progress is None or self.progress.get("id") != msg.get("id"):
                self.progress = {"id": msg.get("id"), "nodes": [], "done": 0}
            if msg.get("step") is not None:
                step = int(msg["step"])
                if self.progress.get("step") != step:
                    self.progress["step"] = step          # a new forward pass
                    self.progress["nodes"] = []           # re-walks: the marks restart
            self.progress["nodes"].extend(msg.get("nodes") or [])
            if msg.get("load"):
                self.progress["load"] = msg["load"]      # a compile's bytes landed
            if msg.get("note"):
                self.progress["note"] = msg["note"]      # e.g. a huggingface download
            self.progress["done"] = int(msg.get("done") or len(self.progress["nodes"]))
            return
        if kind == "result" and msg.get("id") in self._pending:
            box = self._pending[msg["id"]]
            box["reply"] = msg
            box["event"].set()

    def close(self):
        self.alive = False
        try:
            self.conn.shutdown(socket.SHUT_RDWR)     # wake the reader thread
        except OSError:
            pass
        try:
            self.f.close()
        except (OSError, ValueError):
            pass
        try:
            self.conn.close()
        except OSError:
            pass


# ---------------------------------------------------------------- reverse jobs
# `build canvas` on a repo without an `hf_arc` row starts the reverse-model
# agent in the background (tools/reverse_job.py), keyed by the repo; the app
# polls the status file and paints the checklist on the box. Only the job's
# own gates land a row in the database — the agent's say-so never does.
def reverse_status(repo):
    """The job's status for one repo, or None while it has not started."""
    try:
        with open(nn_arc.state_path(repo, ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def reverse_find(family):
    """The live job of a family, for callers that only know the family."""
    try:
        names = os.listdir(nn_arc.state_dir())
    except OSError:
        return None
    for name in sorted(names):
        if not name.endswith(".json") or name.endswith(".todos.json"):
            continue
        try:
            with open(os.path.join(nn_arc.state_dir(), name), encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        if st.get("family") == family and not st.get("done"):
            return st
    return None


def _spawn_detached(cmd, log, slug):
    """Run one job so it outlives the app.

    A detached child is not enough when the app runs under systemd: restarting
    the service kills every process in its cgroup (KillMode=control-group),
    even one in its own session — and only the task being deleted may stop a
    job. A transient unit is its own cgroup, so the job keeps going; without
    systemd, fall back to a plain detached child.
    """
    os.makedirs(os.path.dirname(log), exist_ok=True)
    if os.environ.get("INVOCATION_ID") and shutil.which("systemd-run"):
        unit = "nnstudio-%s-%d" % ((slug or "job")[:40], int(time.time()))
        runner = ["systemd-run", "--user", "--collect", "--quiet",
                  "--unit", unit, "--working-directory", HERE,
                  "--property=StandardOutput=null",
                  "--property=StandardError=append:%s" % log]
        for key in ("NNSTUDIO_REVERSE_AGENT", "NNSTUDIO_REVERSE_MODEL",
                    "NNSTUDIO_REVERSE_BASE_URL", "NNSTUDIO_REVERSE_API_KEY",
                    "NNSTUDIO_ARC_STATE_DIR", "NNSTUDIO_ARC_DB",
                    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(key):
                runner += ["--setenv=%s=%s" % (key, os.environ[key])]
        rc = 1
        try:
            p = subprocess.Popen(runner + ["--"] + cmd, cwd=HERE,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            rc = p.wait(timeout=10)                 # systemd-run exits once up
        except subprocess.TimeoutExpired:
            return                                  # starting; assume it is up
        except OSError:
            rc = 1
        if rc == 0:
            return                                  # the transient unit is up
    with open(log, "ab") as out:
        subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)


def _spawn_reverse(repo, family, task_id=None):
    """The reverse driver: research one HF repo, gated by the job's own gates."""
    cmd = [sys.executable, os.path.join(HERE, "tools", "reverse_job.py"),
           repo, "--family", family]
    log = nn_arc.state_path(repo, ".log")
    if task_id:
        cmd += ["--task", task_id]
        row = nn_tasks.get(task_id)          # the durable stream the box tails
        if row and row.get("log"):
            log = row["log"]
    _spawn_detached(cmd, log, "reverse-" + nn_arc.repo_slug(repo))


def _signal_task(task, sig):
    """Signal everything that runs one task: the wrapper's process group first
    (it owns its agent), then the agent's own session group."""
    for key in ("pid", "agent_pid"):
        pid = (task or {}).get(key)
        if not pid:
            continue
        try:
            os.killpg(os.getpgid(int(pid)), sig)
        except OSError:
            try:
                os.kill(int(pid), sig)
            except OSError:
                pass


def _fire_reverse(task):
    """Turn one task row into a running job — the only firing path there is.

    The family is resolved **inside the wrapper**, never here: whatever goes
    wrong (a gated repo without a token, an unreachable hub) then lands in the
    task's own stream, where the box shows it — not in a pre-spawn refusal
    nobody can see."""
    repo = nn_arc.canonical(str((task.get("params") or {}).get("repo") or ""))
    family = str((task.get("meta") or {}).get("family") or "").strip()
    try:
        have = bool(repo and nn_arc.get(repo))     # the library already answered
    except Exception:  # noqa: BLE001 — a broken row must not block the ask
        have = False
    if have:
        nn_tasks.update(task["id"], status="done", outcome="ok", finished=time.time(),
                        phase="done",
                        meta=dict(task.get("meta") or {}, arc=True,
                                  note="already in the library — nothing to reverse"))
        return True, None
    try:
        _spawn_reverse(repo, family, task_id=task["id"])
    except OSError as e:
        nn_tasks.update(task["id"], status="done", outcome="failed",
                        finished=time.time(),
                        meta=dict(task.get("meta") or {},
                                  error="could not start the job: %s" % e))
        return False, "could not start the job: %s" % e
    return True, None


def _fire_agent(task):
    """The general-purpose driver: one prompt, one agent, its stream teed."""
    prompt = str((task.get("params") or {}).get("prompt") or "").strip()
    if not prompt:
        nn_tasks.update(task["id"], status="done", outcome="failed",
                        finished=time.time(),
                        meta=dict(task.get("meta") or {}, error="the task has no prompt"))
        return False, "the task has no prompt"
    log = task.get("log") or nn_tasks.log_file(task["id"])
    cmd = [sys.executable, os.path.join(HERE, "tools", "agent_job.py"),
           "--task", task["id"], "--prompt", prompt]
    try:
        _spawn_detached(cmd, log, "agent-" + task["id"])
    except OSError as e:
        nn_tasks.update(task["id"], status="done", outcome="failed",
                        finished=time.time(),
                        meta=dict(task.get("meta") or {},
                                  error="could not start the job: %s" % e))
        return False, "could not start the job: %s" % e
    return True, None


def _fire_task(task):
    """Turn one task row into a running job — the only firing path there is."""
    t = (task or {}).get("type")
    if t == "reverse":
        return _fire_reverse(task)
    if t == "python":
        return _fire_agent(task)
    return False, "unknown task type: %s" % t


def agent_public(a):
    return {"name": a.name, "remote": "%s:%s" % a.addr, "since": int(a.since),
            "last_seen": int(a.last_seen), "specs": a.specs, "usage": a.usage}


def local_agent():
    """The backend run/load goes to the one that connected last.

    A freshly attached agent takes over — starting one next to the systemd
    service means *that* one answers, and when it exits the service (still
    alive) answers again. Usage pings keep every agent's last_seen fresh, so
    connection time, not activity, decides."""
    with _AGENT_LOCK:
        alive = [a for a in _AGENTS.values() if a.alive]
    return max(alive, key=lambda a: a.since) if alive else None


def backend_info():
    a = local_agent()
    return {"ok": True,
            "local": {"connected": a is not None,
                      "agent": agent_public(a) if a else None}}


def heartbeat_loop():
    while True:
        time.sleep(15)
        now = time.time()
        for key, a in list(_AGENTS.items()):
            if a.busy <= 0 and now - a.last_seen > AGENT_TIMEOUT:
                a.close()                            # its reader wakes on shutdown
                with _AGENT_LOCK:
                    _AGENTS.pop(key, None)
                continue
            try:
                a.send({"type": "ping", "t": now})
            except (OSError, ValueError):
                a.close()


class ControlHandler(socketserver.BaseRequestHandler):
    """One agent connection: hello, register, then read until it hangs up."""

    def handle(self):
        self.request.settimeout(15)
        agent = Agent(self.request, self.client_address)
        try:
            line = agent.f.readline()
            if not line:
                return
            hello = json.loads(line)
        except (OSError, ValueError):
            agent.close()
            return
        if hello.get("type") != "hello":
            agent.close()
            return
        agent.name = str(hello.get("name") or self.client_address[0])
        agent.specs = hello.get("specs") or {}
        self.request.settimeout(None)
        with _AGENT_LOCK:
            _AGENTS[agent.cid] = agent              # one entry per connection:
                                                    # same-named machines coexist
        try:
            agent.send({"type": "welcome", "app": "nnstudio"})
            while True:
                line = agent.f.readline()
                if not line:
                    break
                try:
                    agent.on_message(json.loads(line))
                except ValueError:
                    continue
        except (OSError, ValueError):
            pass
        finally:
            with _AGENT_LOCK:
                _AGENTS.pop(agent.cid, None)
            agent.close()


class ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "nnstudio"

    def log_message(self, fmt, *args):    # keep the journal quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                          # the client hung up mid-answer: fine

    def _json(self, obj, code=200):
        self._send(code, json.dumps(json_safe(obj)))

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
        if path == "/api/backend":
            self._json(backend_info())
            return
        if path == "/api/run/progress":
            agent = local_agent()
            self._json({"ok": True, "progress": agent.progress if agent else None})
            return
        if path == "/api/tasks":
            self._tasks_list()
            return
        if path == "/api/tasks/log":
            q = parse_qs(urlparse(self.path).query)
            self._tasks_log(str((q.get("task") or [""])[0]).strip(),
                            int((q.get("pos") or ["0"])[0] or 0))
            return
        if path == "/api/reverse/status":
            q = parse_qs(urlparse(self.path).query)
            repo = nn_arc.canonical(str((q.get("repo") or [""])[0]).strip())
            family = str((q.get("family") or [""])[0]).strip()
            st = reverse_status(repo) if repo else None
            if st is None and family:
                st = reverse_find(family)
            if st is None and repo:
                arc = bool(nn_arc.get(repo))          # never started: the DB answers
                st = {"repo": repo, "family": family or None,
                      "phase": "done" if arc else "none",
                      "done": arc, "ok": arc, "arc": arc}
            if st is None:
                self._json({"ok": False, "error": "repo or family is required"}, 400)
                return
            if "arc" not in st:
                st["arc"] = bool(nn_arc.get(st.get("repo") or repo or ""))
            if not st.get("done") and not st.get("arc"):
                pid = st.get("pid")
                gone = False
                if pid:
                    try:
                        os.kill(int(pid), 0)       # the wrapper still alive?
                    except (OSError, ValueError):
                        gone = True
                elif time.time() - (st.get("updated") or 0) > 300:
                    gone = True
                if gone and time.time() - (st.get("updated") or 0) > 30:
                    st = dict(st, stale=True, done=True, ok=False,
                              detail="the job is gone \u2014 load the repo again to retry")
            self._json({"ok": True, "repo": st.get("repo"), "family": st.get("family"),
                        "status": st})
            return
        if path == "/api/weights":
            self._json({"ok": True, "files": list_weight_files()})
            return
        if path == "/api/weights/meta":
            q = parse_qs(urlparse(self.path).query)
            target = _weight_path((q.get("file") or [""])[0])
            try:
                out = weight_engine().file_tensors(target)
                out["ok"] = True
                self._json(out)
            except Exception as e:  # noqa: BLE001 — the form shows the message
                self._json({"ok": False, "error": str(e)}, 400)
            return
        if path == "/api/tokenizer":
            q = parse_qs(urlparse(self.path).query)
            repo = (q.get("repo") or [""])[0]
            try:
                out = hf_tokenizer_meta(repo)
                out["ok"] = True
                self._json(out)
            except Exception as e:  # noqa: BLE001 — the form shows the message
                self._json({"ok": False, "error": str(e)}, 400)
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            data = {}
        if path == "/api/gguf/canvas":
            model = str(data.get("model") or "").strip()
            repo = str(data.get("repo") or "").strip()
            if not model and not repo:
                self._json({"ok": False, "error": "model or repo is required"}, 400)
                return
            try:
                out = nn_engine.hf_canvas(repo) if repo else nn_engine.gguf_canvas(model)
                self._json(out)                              # headers only, no weights
            except Exception as e:  # noqa: BLE001 — the form shows the message
                self._json({"ok": False, "error": str(e)}, 400)
            return
        if path == "/api/reverse":            # compat alias: the one firing path
            self._tasks_create({"type": "reverse",
                                "params": {"repo": data.get("repo"),
                                           "family": data.get("family")}})
            return
        if path == "/api/reverse/cancel":     # compat alias, by repo
            repo = nn_arc.canonical(str(data.get("repo") or "").strip())
            if not repo:
                self._json({"ok": False, "error": "a repo id is required"}, 400)
                return
            task = nn_tasks.find_active("reverse", {"repo": repo}) if repo else None
            if not task:
                self._json({"ok": True, "cancelled": False})
                return
            self._tasks_cmd({"task": task["id"], "cmd": "cancel",
                             "reason": str(data.get("reason") or "stopped")})
            return
        if path == "/api/tasks":
            self._tasks_create(data)
            return
        if path == "/api/tasks/cmd":
            self._tasks_cmd(data)
            return
        if path == "/api/run":
            self._run(data)
            return
        if path == "/api/load":
            self._load(data)
            return
        if path == "/api/offload":
            self._offload()
            return
        if path == "/api/weights/export":
            self._weights_export(data)
            return
        self._json({"ok": False, "error": "not found"}, 404)

    # ---- the whole design, on the user's backend ---------------------------
    def _run(self, data):
        canvas = data.get("canvas")
        if not isinstance(canvas, dict) or not canvas.get("nodes"):
            self._json({"ok": False, "error": "no canvas in the request"}, 400)
            return
        agent = local_agent()
        if agent is None:
            self._json({"ok": False,
                        "error": "no backend connected \u2014 Backend \u2192 Local"}, 409)
            return
        agent.progress = None                    # a fresh data-flow for this run
        reply = agent.request({"type": "run", "canvas": canvas,
                               "feeds": data.get("feeds") or {},
                               "generate": int(data.get("generate") or 0),
                               "temperature": float(0.8 if data.get("temperature") is None
                                                    else data.get("temperature")),
                               "input": data.get("input"), "output": data.get("output"),
                               "seed": int(data.get("seed") or 1)})
        if reply is None:
            self._json({"ok": False,
                        "error": "the backend did not answer (or is gone)"}, 504)
            return
        if not reply.get("ok"):
            self._json({"ok": False, "error": reply.get("error") or "the run failed",
                        "box": reply.get("box"), "nid": reply.get("nid")}, 400)
            return
        out = reply.get("result") or {}
        out["backend"] = agent.name
        self._json(out)

    # ---- load the weights: the whole design, kept warm on the backend -------
    def _load(self, data):
        canvas = data.get("canvas")
        if not isinstance(canvas, dict) or not canvas.get("nodes"):
            self._json({"ok": False, "error": "no canvas in the request"}, 400)
            return
        agent = local_agent()
        if agent is None:
            self._json({"ok": False,
                        "error": "no backend connected \u2014 Backend \u2192 Local"}, 409)
            return
        reply = agent.request({"type": "load", "canvas": canvas}, timeout=3600)
        if reply is None:
            self._json({"ok": False,
                        "error": "the backend did not answer (or is gone)"}, 504)
            return
        if not reply.get("ok"):
            self._json({"ok": False, "error": reply.get("error") or "the load failed",
                        "box": reply.get("box"), "nid": reply.get("nid")}, 400)
            return
        out = reply.get("result") or {}
        out["ok"] = True
        out["backend"] = agent.name
        self._json(out)

    # ---- offload: let the backend drop every model and cache ----------------
    def _offload(self):
        agent = local_agent()
        if agent is None:
            self._json({"ok": False,
                        "error": "no backend connected \u2014 Backend \u2192 Local"}, 409)
            return
        reply = agent.request({"type": "offload"}, timeout=300)
        if reply is None:
            self._json({"ok": False,
                        "error": "the backend did not answer (or is gone)"}, 504)
            return
        if not reply.get("ok"):
            self._json({"ok": False, "error": reply.get("error") or "the offload failed"}, 400)
            return
        out = reply.get("result") or {}
        out["ok"] = True
        out["backend"] = agent.name
        # the app host's own lazy engine may hold readers from metadata reads
        if _ENGINE_MOD is not None:
            try:
                _ENGINE_MOD.offload()
            except Exception:  # noqa: BLE001 — clearing caches is best effort
                pass
        self._json(out)


    # ---- agent tasks: the AI agents' work, as the app sees it --------------
    # The task is the entity — created only by an agent box, managed only by
    # agent boxes. Nothing else in the app fires or stops one.
    def _tasks_list(self):
        try:
            nn_tasks.reconcile()
            tasks = nn_tasks.list()
        except Exception as e:  # noqa: BLE001 — the list must never explode
            self._json({"ok": False, "error": str(e)}, 500)
            return
        self._json({"ok": True, "tasks": tasks})

    def _tasks_log(self, tid, pos):
        task = nn_tasks.get(tid) if tid else None
        if not task:
            self._json({"ok": False, "error": "no such task"}, 404)
            return
        path = task.get("log") or nn_tasks.log_file(tid)
        items, end = [], max(0, int(pos or 0))
        try:
            size = os.path.getsize(path)
            end = min(end, size)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(end)
                data = f.read()
            lines = data.split("\n")
            lines.pop()                      # the last line may be half-written
            end += sum(len(l) + 1 for l in lines)
            items = nn_tasks.fold_log(lines, uid=str(end))[:400]
        except OSError:
            pass
        self._json({"ok": True, "task": tid, "items": items, "pos": end})

    def _tasks_create(self, data):
        t = str(data.get("type") or "reverse").strip()
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        if t == "reverse":
            repo = nn_arc.canonical(str(params.get("repo") or "").strip())
            if not repo:
                self._json({"ok": False, "error": "a repo id is required"}, 400)
                return
            params = dict(params, repo=repo)
        try:
            joined = nn_tasks.find_active(t, params)   # one active ask per repo
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e)}, 500)
            return
        if joined:
            self._json({"ok": True, "task": joined, "joined": True})
            return
        prior = nn_tasks.find_finished(t, params)   # an earlier attempt: retry it
        if prior:
            meta = dict(prior.get("meta") or {})
            meta.pop("error", None)
            meta.pop("reason", None)
            task = nn_tasks.update(prior["id"], status="running", outcome=None,
                                   phase="queued", finished=None, pid=None,
                                   agent_pid=None, todo=[], meta=meta)
            try:                                     # the stream starts over clean
                open(task["log"], "w", encoding="utf-8").close()
            except OSError:
                pass
            ok, why = _fire_task(task)
            if not ok:
                self._json({"ok": False, "error": why,
                            "task": nn_tasks.get(task["id"])}, 400)
                return
            self._json({"ok": True, "task": nn_tasks.get(task["id"]), "joined": True})
            return
        try:
            task = nn_tasks.create(t, params)
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
            return
        ok, why = _fire_task(task)           # the one firing path
        if not ok:
            self._json({"ok": False, "error": why, "task": nn_tasks.get(task["id"])}, 400)
            return
        self._json({"ok": True, "task": nn_tasks.get(task["id"])})

    def _tasks_cmd(self, data):
        tid = str(data.get("task") or "").strip()
        cmd = str(data.get("cmd") or "").strip()
        task = nn_tasks.get(tid)
        if not task:
            self._json({"ok": False, "error": "no such task"}, 404)
            return
        live = task.get("status") in ("running", "paused")
        meta = dict(task.get("meta") or {})
        if cmd == "pause":
            if task.get("status") == "running":
                _signal_task(task, signal.SIGSTOP)
                meta["pause_at"] = time.time()
                task = nn_tasks.update(tid, status="paused", meta=meta)
        elif cmd == "resume":
            if task.get("status") == "paused":
                _signal_task(task, signal.SIGCONT)
                meta.pop("pause_at", None)
                task = nn_tasks.update(tid, status="running", meta=meta)
        elif cmd == "cancel":
            if live:
                _signal_task(task, signal.SIGCONT)   # a stopped task must wake to die
                _signal_task(task, signal.SIGTERM)
            meta["reason"] = str(data.get("reason") or "stopped by you")
            task = nn_tasks.update(tid, status="cancelled", meta=meta,
                                   finished=time.time())
        elif cmd == "restart":
            if live:
                _signal_task(task, signal.SIGCONT)
                _signal_task(task, signal.SIGTERM)
            meta.pop("reason", None)
            meta.pop("pause_at", None)
            meta.pop("error", None)
            task = nn_tasks.update(tid, status="running", outcome=None,
                                   phase="queued", finished=None,
                                   pid=None, agent_pid=None, todo=[], meta=meta)
            ok, why = _fire_task(task)
            if not ok:
                self._json({"ok": False, "error": why, "task": nn_tasks.get(tid)}, 400)
                return
            task = nn_tasks.get(tid)
        elif cmd == "delete":
            if live:
                _signal_task(task, signal.SIGCONT)
                _signal_task(task, signal.SIGTERM)
            nn_tasks.delete(tid, purge=bool(data.get("purge")))
            self._json({"ok": True, "deleted": tid})
            return
        else:
            self._json({"ok": False, "error": "unknown command: %s" % cmd}, 400)
            return
        self._json({"ok": True, "task": task, "changed": True})

    # ---- export the design's weights: one pt / safetensors file ------------
    def _weights_export(self, data):
        canvas = data.get("canvas")
        if not isinstance(canvas, dict) or not canvas.get("nodes"):
            self._json({"ok": False, "error": "no canvas in the request"}, 400)
            return
        fmt = "pt" if str(data.get("format") or "").lower() in ("pt", "pytorch", ".pt") \
            else "safetensors"
        name = _safe_name(data.get("name"))
        try:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            path = os.path.join(WEIGHTS_DIR, name + "." + fmt)
            t0 = time.time()
            out = weight_engine().export_weights(canvas, path, fmt)
            out["ok"] = True
            out["ms"] = int((time.time() - t0) * 1000)
            out["path"] = "weights/" + name + "." + fmt     # what the design stores
            self._json(out)
        except Exception as e:  # noqa: BLE001 — the form shows the message
            self._json({"ok": False, "error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="nnstudio — the training canvas")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--control-port", type=int, default=8125)
    args = ap.parse_args()
    try:
        nn_tasks.reconcile()               # a task whose process is gone is done
    except Exception:  # noqa: BLE001 — the list repairs itself on read too
        pass
    ctl = ControlServer((args.host, args.control_port), ControlHandler)
    threading.Thread(target=ctl.serve_forever, daemon=True,
                     name="nnstudio-control").start()
    threading.Thread(target=heartbeat_loop, daemon=True,
                     name="nnstudio-heartbeat").start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("nnstudio on http://%s:%d (agents on %s:%d)"
          % (args.host, args.port, args.host, args.control_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
