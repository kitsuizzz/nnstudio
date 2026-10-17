#!/usr/bin/env python3
"""One conversational turn with the reverse-engineering agent.

This is the driver behind a task of type `chat` — the AI agent tab's every
message. A turn runs the agent CLI (the same OpenCode the gated jobs use)
with the user's message, tees its JSONL stdout into the task's log (the tab
streams it token by token) and lands the row's status when the agent exits.

The **session** is what makes it a conversation. The first turn's prompt
points the agent at `.opencode/commands/reverse-model.md` — the workflow the
reverse job follows — and the CLI's session id is mirrored into the task's
meta as it appears, so the next turn starts with `--session <id>` and the
agent remembers everything said before.

    python tools/chat_job.py --task t_123 --message "reverse engineer org/x"
    python tools/chat_job.py --task t_124 --message "why two norms?" --session ses_a

There are no gates here: the agent's own tools (tools/reverse_model.py) land
what they land, and a non-zero exit lands the turn `failed`.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
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
    """The turn's exit — unless a stop already landed it (its word stands)."""
    now = nn_tasks.get(task["id"]) or task
    if now.get("status") == "cancelled":
        return
    mirror(task, status="done", outcome=None if ok else "failed",
           phase="done", finished=time.time(),
           meta=dict(now.get("meta") or {}, detail=detail))


def note_session(task, sid):
    """The CLI's session id, into the task's meta: the next turn continues it."""
    sid = str(sid or "").strip()
    if not sid:
        return
    meta = dict(task.get("meta") or {})
    if meta.get("session") == sid:
        return
    meta["session"] = sid
    task["meta"] = meta
    mirror(task, meta=meta)


def first_prompt(message, repo):
    """The turn that opens a conversation: who the agent is, and the ask."""
    lines = [
        "You are nnstudio's reverse-engineering agent — the AI behind this "
        "canvas app. When the user asks you to reverse engineer a model, "
        "follow `.opencode/commands/reverse-model.md` exactly: read the "
        "family's source (`python tools/reverse_model.py sources <family>`), "
        "inspect the repo's tensors (`tools/reverse_model.py inspect <repo>`), "
        "write or update `specs/<family>.json`, and run "
        "`tools/reverse_model.py check <family>` until every stage is green. "
        "The user reads your stream in the app's AI agent tab — say what you "
        "are doing, in short sentences, and do not ask questions the tools "
        "can answer.",
    ]
    if repo:
        lines.append("The repo of interest: " + repo)
    lines.append("The user's message:\n" + message)
    return "\n\n".join(lines)


def run_agent(task, message, repo, session, timeout):
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
    prompt = message if session else first_prompt(message, repo)
    override = os.environ.get("NNSTUDIO_REVERSE_AGENT") or ""
    if override:
        cmd = shlex.split(override)
        if session:
            cmd += ["--session", session]
        cmd += [prompt]
    else:
        exe = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")
        cmd = [exe, "run", "--format", "json", "--auto"]
        if direct:
            cmd += ["--standalone", "--model", "%s/main" % PROVIDER_ID]
        elif model:
            cmd += ["--model", model]
        if session:
            cmd += ["--session", session]
        else:
            cmd += ["--title", "nnstudio chat %s" % task["id"]]
        cmd += [prompt]
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
                if '"sessionID"' in raw:         # the CLI's own session id
                    try:
                        ev = json.loads(raw)
                    except ValueError:
                        ev = None
                    if isinstance(ev, dict) and ev.get("sessionID"):
                        note_session(task, ev["sessionID"])
        stop.set()

    threading.Thread(target=read_lines, daemon=True).start()
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
    return (rc == 0), ("done" if rc == 0 else "the agent exited with %d" % rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="the agent_tasks row this turn serves")
    ap.add_argument("--message", required=True, help="what the user said")
    ap.add_argument("--session", default="", help="the conversation's CLI session id")
    ap.add_argument("--repo", default="", help="the repo of interest, when named")
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
        ok, detail = run_agent(task, args.message, args.repo, args.session, args.timeout)
    except Exception as e:  # noqa: BLE001 — never die without a status
        import traceback
        with open(task.get("log") or nn_tasks.log_file(task["id"]), "a") as f:
            f.write("# the turn crashed:\n" + traceback.format_exc())
        ok, detail = False, "the turn crashed: %s" % e
    land(task, ok, detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
