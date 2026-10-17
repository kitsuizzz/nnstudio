#!/usr/bin/env python3
"""agent_tasks: the task rows in hf_arc's database, and the agent's stream.

A task says which agent runs and what the user asked for (a HF repo id for
reverse engineering, a prompt for the python agent); the app's AI box is the
only surface that fires and manages one. The stream fold is what the box
renders: the model's words, its tool actions, step markers and errors — with a
growing text part updating in place, token by token.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nn_arc      # noqa: E402
import nn_tasks    # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print("  FAIL " + name + " \u2014 " + str(e))


WORK = tempfile.mkdtemp(prefix="nnstudio-tasks-")
os.environ["NNSTUDIO_ARC_DB"] = os.path.join(WORK, "hf_arc.db")
os.environ["NNSTUDIO_TASKS_DB"] = os.path.join(WORK, "tasks.db")
os.environ["NNSTUDIO_ARC_STATE_DIR"] = os.path.join(WORK, "state")


def crud():
    t = nn_tasks.create("reverse", {"repo": "org/name"})
    assert t["type"] == "reverse" and t["params"] == {"repo": "org/name"}, t
    assert t["status"] == "running" and t["phase"] == "queued", t
    assert t["log"].endswith(t["id"] + ".log"), t["log"]
    got = nn_tasks.get(t["id"])
    assert got and got["id"] == t["id"], got
    assert [x["id"] for x in nn_tasks.list()] == [t["id"]], nn_tasks.list()
    nn_tasks.update(t["id"], phase="agent", pid=os.getpid(),
                    todo=[{"key": "a", "title": "reading", "state": "done"},
                          {"key": "b", "title": "reconstruct", "state": "active"}],
                    meta={"family": "qwen3_5"})
    got = nn_tasks.get(t["id"])
    assert got["phase"] == "agent" and got["meta"]["family"] == "qwen3_5", got
    assert nn_tasks.todo_summary(got) == (1, 2, "reconstruct"), got["todo"]
    assert nn_tasks.delete(t["id"]) and nn_tasks.get(t["id"]) is None


check("a task row: create, mirror progress, land, delete", crud)


def the_task_explains_itself():
    t = {"id": "t1", "type": "reverse", "params": {"repo": "unsloth/Qwen3.8-27B-NVFP4"},
         "status": "running", "todo": []}
    assert nn_tasks.title(t) == "reverse engineer repo \u00b7 unsloth/Qwen3.8-27B-NVFP4"
    assert nn_tasks.status_word(t) == "In-Progress"
    assert nn_tasks.status_word(dict(t, status="paused")) == "Paused"
    assert nn_tasks.status_word(dict(t, status="done")) == "Done"
    assert nn_tasks.status_word(dict(t, status="cancelled")) == "Cancelled"
    py = {"id": "t2", "type": "python", "params": {"prompt": "plot it"}, "status": "running"}
    assert "python agent" in nn_tasks.title(py) and "plot it" in nn_tasks.title(py)


check("the task labels itself: type, ask, chip", the_task_explains_itself)


def one_active_ask_per_repo():
    a = nn_tasks.create("reverse", {"repo": "org/dup"})
    hit = nn_tasks.find_active("reverse", {"repo": "org/dup"})
    assert hit and hit["id"] == a["id"], hit
    nn_tasks.update(a["id"], status="done", outcome="exact")
    assert nn_tasks.find_active("reverse", {"repo": "org/dup"}) is None
    nn_tasks.delete(a["id"])


check("one active task per ask — and a done one is free again", one_active_ask_per_repo)


def a_start_reuses_settled_attempts():
    """Starting the same ask again retries the settled row instead of piling
    up another failed one."""
    t = nn_tasks.create("reverse", {"repo": "org/retry"})
    nn_tasks.update(t["id"], status="done", outcome="failed")
    assert nn_tasks.find_active("reverse", {"repo": "org/retry"}) is None
    again = nn_tasks.find_finished("reverse", {"repo": "org/retry"})
    assert again and again["id"] == t["id"], again
    nn_tasks.delete(t["id"])


check("a start reuses the settled attempt for the same ask",
      a_start_reuses_settled_attempts)


def a_dead_process_is_not_running():
    import time
    t = nn_tasks.create("python", {"prompt": "x"})
    assert nn_tasks.reconcile() == [], "a just-fired row is not killed mid-spawn"
    nn_tasks.update(t["id"], pid=os.getpid())       # this very process: alive
    assert nn_tasks.reconcile() == []
    nn_tasks.update(t["id"], pid=99999999)
    assert nn_tasks.reconcile() == [t["id"]]
    got = nn_tasks.get(t["id"])
    assert got["status"] == "done" and got["outcome"] == "failed", got
    nn_tasks.delete(t["id"])
    old = nn_tasks.create("python", {"prompt": "y"})   # no pid, and long stale
    conn = nn_tasks.connect()
    try:
        with conn:
            conn.execute("UPDATE agent_tasks SET updated = ? WHERE id = ?",
                         (time.time() - 600, old["id"]))
    finally:
        conn.close()
    assert nn_tasks.reconcile() == [old["id"]], \
        "a stale pid-less row still lands (a reboot mid-task)"
    nn_tasks.delete(old["id"])


check("reconcile lands a task whose process is gone", a_dead_process_is_not_running)


def the_stream_folds_token_by_token():
    lines = [
        "# the job begins",                                  # the wrapper's own note
        json.dumps({"type": "text", "part": {"type": "text", "partID": "p1",
                                             "text": "Thinking."}}),
        json.dumps({"type": "tool_use", "part": {"type": "tool", "id": "c1",
                                                 "tool": "read",
                                                 "state": {"status": "completed",
                                                           "input": {"path": "nn.py",
                                                                     "limit": 10},
                                                           "output": "1: x"}}}),
        json.dumps({"type": "step_finish",
                    "part": {"id": "s1", "type": "step-finish",
                             "tokens": {"input": 100, "output": 50}}}),
        json.dumps({"type": "text", "part": {"type": "text", "partID": "p1",
                                             "text": "Thinking. So the answer"}}),
    ]
    items = nn_tasks.fold_log(lines, uid="100")
    kinds = [i["kind"] for i in items]
    assert kinds == ["note", "say", "do", "step"], kinds
    assert items[1]["text"] == "Thinking. So the answer", items[1]   # the part grew
    assert items[2]["text"] == "read \u00b7 nn.py" and "output" in items[2]["detail"]
    assert items[3]["text"] == "step \u00b7 150 tokens"
    assert all(i.get("raw") for i in items), "every item keeps its raw event"
    # token by token: the same part arrives as deltas, and they accumulate
    deltas = [
        json.dumps({"type": "text", "part": {"type": "text", "partID": "p2",
                                             "delta": "Hel"}}),
        json.dumps({"type": "text", "part": {"type": "text", "partID": "p2",
                                             "delta": "lo"}}),
    ]
    got = nn_tasks.fold_log(deltas, uid="200")
    assert [(i["id"], i["text"], i.get("append")) for i in got] == \
        [("p2", "Hello", True)], got
    # and a later poll's delta keeps appending the same part
    later = nn_tasks.fold_log([json.dumps(
        {"type": "text", "part": {"type": "text", "partID": "p2", "delta": " there"}})],
        uid="300")
    assert later[0]["text"] == " there" and later[0]["append"], later


check("the stream folds: parts grow in place, deltas accumulate",
      the_stream_folds_token_by_token)


def errors_and_unique_notes():
    items = nn_tasks.fold_log([
        json.dumps({"type": "error", "error": "the provider said 429"}),
        "plain note",
    ], uid="7")
    assert [i["kind"] for i in items] == ["error", "note"], items
    assert "429" in items[0]["text"]
    a = nn_tasks.fold_log(["plain note"], uid="7")
    b = nn_tasks.fold_log(["plain note"], uid="8")
    assert a[0]["id"] != b[0]["id"], "notes are unique per position"


check("errors surface, and notes never collide", errors_and_unique_notes)


def deleting_keeps_the_log_unless_purged():
    t = nn_tasks.create("python", {"prompt": "x"})
    path = t["log"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# hi\n")
    assert nn_tasks.delete(t["id"]) and os.path.isfile(path), "the log stays"
    t = nn_tasks.create("python", {"prompt": "y"})
    with open(t["log"], "w", encoding="utf-8") as f:
        f.write("# hi\n")
    assert nn_tasks.delete(t["id"], purge=True) and not os.path.isfile(t["log"])


check("a deleted task keeps its log unless purged", deleting_keeps_the_log_unless_purged)


def the_audit_ignores_agent_boxes():
    import nn_engine
    c = {"type": "canvas", "version": 1,
         "nodes": [{"id": "i", "kind": "input", "name": "x", "x": 0, "y": 0, "dout": 2},
                   {"id": "ag", "kind": "agent", "name": "agent", "x": 160, "y": 0},
                   {"id": "l", "kind": "layer", "name": "fc", "x": 160, "y": 120,
                    "din": 2, "dout": 2},
                   {"id": "o", "kind": "output", "name": "y", "x": 320, "y": 0, "din": 2}],
         "edges": [{"from": "i", "to": "l"}, {"from": "l", "to": "o"}]}
    audit = nn_engine.audit_canvas(c)
    hits = [i for i in audit.get("issues") or [] if i.get("nid") == "ag"]
    assert hits == [], hits
    assert audit.get("ok") is True, audit


check("the audit ignores the agent box — it runs no maths", the_audit_ignores_agent_boxes)

if FAILED:
    print("nn_tasks: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    raise SystemExit(1)
print("nn_tasks: all assertions passed")
