#!/usr/bin/env python3
"""One general-purpose agent task: a prompt, an agent, its stream — landed.

This is the driver behind an agent task of type `python` (and any future type
that is "ask the agent to do a thing"): it runs the same agent CLI the reverse
job uses, tees its JSONL stdout into the task's log (the agent box streams it
token by token), mirrors progress into the `agent_tasks` row, and lands the
row's status when the agent exits.

    python tools/agent_job.py --task t_123 --prompt "plot the training loss"

Only the agent's own work decides the outcome — there are no gates here; a
non-zero exit or an error event lands `failed`, a clean exit lands `done`.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nn_arc  # noqa: E402
import nn_tasks  # noqa: E402
from reverse_job import provider_config, PROVIDER_ID  # noqa: E402

CANCELLED = {"flag": False}


def _term(_signum, _frame):
    CANCELLED["flag"] = True


def mirror(task, **cols):
    """Keep the task row's live progress honest; never die over the list."""
    try:
        nn_tasks.update(task["id"], **cols)
    except Exception:  # noqa: BLE001
        pass


def land(task, ok, detail):
    import time as _t
    mirror(task, status="done", outcome=None if ok else "failed",
           phase="done", finished=_t.time(),
           meta=dict(task.get("meta") or {}, detail=detail))


def run_agent(task, prompt, timeout):
    """The agent CLI, with the stream teed to the task's log."""
    log_path = task.get("log") or nn_tasks.log_file(task["id"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    model = (os.environ.get("NNSTUDIO_REVERSE_MODEL") or
             os.environ.get("OPENCODE_MODEL") or "").strip()
    base = (os.environ.get("NNSTUDIO_REVERSE_BASE_URL") or "").strip()
    key = (os.environ.get("NNSTUDIO_REVERSE_API_KEY") or "").strip()
    direct = bool(base and key and model)
    child_env = dict(os.environ)
    if direct:
        child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps(provider_config(base, model))
    override = os.environ.get("NNSTUDIO_REVERSE_AGENT") or ""
    if override:
        cmd = shlex.split(override) + [prompt]
    else:
        exe = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")
        cmd = [exe, "run", "--format", "json", "--auto"]
        if direct:
            cmd += ["--standalone", "--model", "%s/main" % PROVIDER_ID]
        elif model:
            cmd += ["--model", model]
        cmd += ["--title", "agent task %s" % task["id"], prompt]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd[:6]) + "\n")
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, env=child_env)
    except OSError as e:
        return False, "the agent could not start: %s" % e
    mirror(task, pid=os.getpid(), agent_pid=proc.pid, phase="agent")

    stop = threading.Event()

    def read_lines():
        with open(log_path, "a", encoding="utf-8") as log:
            for raw in iter(proc.stdout.readline, ""):
                log.write(raw)
                log.flush()
        stop.set()

    threading.Thread(target=read_lines, daemon=True).start()
    failed = False
    while proc.poll() is None:
        if CANCELLED["flag"]:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                proc.terminate()
            return False, "cancelled"
        if time.time() - task["started"] > timeout:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                proc.terminate()
            return False, "timed out after %ds" % timeout
        time.sleep(1)
    stop.wait(timeout=5)
    rc = proc.returncode or 0
    if rc != 0:
        failed = True
    return (not failed), ("the agent exited with %d" % rc) if failed else "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="the agent_tasks row it serves")
    ap.add_argument("--prompt", required=True, help="what the user asked for")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    task = nn_tasks.get(args.task)
    if not task:
        print("no such task: %s" % args.task)
        return 2
    mirror(task, pid=os.getpid(), phase="agent")
    try:
        ok, detail = run_agent(task, args.prompt, args.timeout)
    except Exception as e:  # noqa: BLE001 — never die without a status
        import traceback
        with open(task.get("log") or nn_tasks.log_file(task["id"]), "a") as f:
            f.write("# the task crashed:\n" + traceback.format_exc())
        ok, detail = False, "the task crashed: %s" % e
    land(task, ok, detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
