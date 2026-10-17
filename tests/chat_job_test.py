#!/usr/bin/env python3
"""The chat job: one conversational turn — the stream teed, the session kept.

The agent is stubbed (a script that prints OpenCode-shaped JSONL and records
its argv), so this tests the runner: the first-turn prompt, the session id
mirrored into the task's meta, the follow-up started with `--session`, and
the log folding into the tab's items. Nothing here talks to a model.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = sys.executable
JOB = os.path.join(ROOT, "tools", "chat_job.py")

STUB = r'''
import json, os, sys
argv = sys.argv[1:]
if os.environ.get("STUB_ARGV"):
    with open(os.environ["STUB_ARGV"], "w") as f:
        f.write(json.dumps(argv))
sid = os.environ.get("STUB_SESSION", "ses_stub_1")
print(json.dumps({"sessionID": sid, "type": "message_started"}), flush=True)
print(json.dumps({"part": {"id": "p1", "type": "text", "text": "reading "}}), flush=True)
print(json.dumps({"part": {"id": "p1", "type": "text", "delta": "the repo"}}), flush=True)
print(json.dumps({"part": {"id": "c1", "type": "tool",
                           "tool": "read",
                           "state": {"input": {"path": "nn_engine.py"}}}}), flush=True)
sys.exit(0)
'''

fails = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        import traceback
        fails.append(name)
        print("  FAIL " + name + " — " + str(e))
        if os.environ.get("STACK"):
            traceback.print_exc()


with tempfile.TemporaryDirectory() as d:
    stub = os.path.join(d, "stub_agent.py")
    with open(stub, "w", encoding="utf-8") as f:
        f.write(STUB)
    os.environ["NNSTUDIO_TASKS_DB"] = os.path.join(d, "tasks.db")
    os.environ["NNSTUDIO_ARC_DB"] = os.path.join(d, "hf_arc.db")
    import nn_tasks  # noqa: E402  (after the dbs are pointed away)

    def run_turn(task, message, session="", repo=""):
        e = dict(os.environ)
        e["NNSTUDIO_REVERSE_AGENT"] = PY + " " + stub
        e["STUB_ARGV"] = os.path.join(d, "argv.json")
        if os.path.exists(e["STUB_ARGV"]):
            os.remove(e["STUB_ARGV"])
        cmd = [PY, JOB, "--task", task["id"], "--message", message]
        if session:
            cmd += ["--session", session]
        if repo:
            cmd += ["--repo", repo]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           env=e, timeout=120)
        argv = []
        if os.path.exists(e["STUB_ARGV"]):
            with open(e["STUB_ARGV"], encoding="utf-8") as f:
                argv = json.load(f)
        return p, argv

    def first_turn():
        task = nn_tasks.create("chat", {"message": "reverse engineer org/x",
                                        "repo": "org/x"})
        p, argv = run_turn(task, "reverse engineer org/x", repo="org/x")
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        t = nn_tasks.get(task["id"])
        assert t["status"] == "done", t
        assert (t.get("meta") or {}).get("session") == "ses_stub_1", t.get("meta")
        assert "--session" not in argv, argv
        prompt = argv[-1]
        assert "reverse-model.md" in prompt and "reverse engineer org/x" in prompt, prompt
        with open(t["log"], encoding="utf-8") as f:
            log = f.read()
        assert "reverse engineer org/x" in log and "$ " in log, log[:200]
        items = nn_tasks.fold_log(log.split("\n")[:-1] or log.split("\n"), uid="x")
        kinds = [it["kind"] for it in items]
        assert "say" in kinds and "do" in kinds, items
        said = "".join(it["text"] for it in items if it["kind"] == "say")
        assert said == "reading the repo", said          # deltas fold in place
        print("  the first turn: prompt carries the workflow, session kept (%s)"
              % (t.get("meta") or {}).get("session"))

    def follow_up():
        prev = nn_tasks.create("chat", {"message": "why two norms?",
                                        "session": "ses_stub_1"})
        p, argv = run_turn(prev, "why two norms?", session="ses_stub_1")
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        assert argv[:2] == ["--session", "ses_stub_1"], argv
        assert argv[-1] == "why two norms?", argv           # the session carries the rest
        t = nn_tasks.get(prev["id"])
        assert t["status"] == "done", t
        print("  the follow-up: --session rides, the message goes as it is")

    def turn_history_is_not_joined():
        # the same words twice make two turns — a conversation, not a retry
        a = nn_tasks.create("chat", {"message": "hello"})
        assert nn_tasks.find_active("chat", {"message": "hello"}) is not None
        assert nn_tasks.find_finished("chat", {"message": "bye"}) is None
        nn_tasks.update(a["id"], status="done", outcome=None, finished=time.time())
        assert nn_tasks.find_finished("chat", {"message": "hello"}) is not None
        assert a["type"] == "chat"
        print("  turns are rows of their own; a chat row is a normal task")

    def title_reads_the_message():
        t = {"type": "chat", "params": {"message": "reverse engineer org/y"}}
        assert nn_tasks.title(t) == "reverse engineer chat \u00b7 reverse engineer org/y", \
            nn_tasks.title(t)
        assert nn_tasks.status_word({"status": "running"}) == "In-Progress"
        print("  a chat task names itself by its message")

    check("the first turn opens the conversation", first_turn)
    check("a follow-up continues the session", follow_up)
    check("turns are history, not retries", turn_history_is_not_joined)
    check("the chat title", title_reads_the_message)

if fails:
    print("chat_job_test: %d failure(s): %s" % (len(fails), ", ".join(fails)))
    raise SystemExit(1)
print("chat_job_test: all assertions passed")
