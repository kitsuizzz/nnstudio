#!/usr/bin/env python3
"""agent_tasks — what the AI agents are doing, as rows of their own database.

The task is the entity: it says which agent runs it (`type`), what the user
 asked for (`params`, e.g. `{"repo": "org/name"}` for a reverse job), and where
it stands (`status`/`outcome`). The canvas' agent boxes are only views of these
rows — they select, start, pause and stop them; nothing else in the app may
fire a task.

    agent_tasks(
      id TEXT PRIMARY KEY,      -- "t_<hex>"
      type TEXT,                -- the agent type ("reverse")
      params TEXT,              -- JSON: what the user asked for
      status TEXT,              -- running | paused | cancelled | done
      outcome TEXT,             -- exact | approximate | failed | NULL
      phase TEXT,               -- the wrapper's live phase
      todo TEXT,                -- the checklist as JSON
      pid INTEGER,              -- the wrapper's process
      agent_pid INTEGER,        -- the model agent's process
      unit TEXT,                -- the transient systemd unit, if any
      log TEXT,                 -- the stream file (agents/<id>.log)
      meta TEXT,                -- JSON: family, provider, cancel reason …
      created REAL, started REAL, updated REAL, finished REAL)

It lives in **its own database**, `agents/tasks.db` (`NNSTUDIO_TASKS_DB`) —
git-ignored, like everything under `agents/`, because a row holds what the user
asked for and the machine paths it runs on. The tracked `hf_arc.db` keeps only
researched specs. The table is created on first use and grows additively
(`ensure` adds missing columns), so new agent types can add fields without
breaking older databases.

Writers keep to their own columns: the job wrapper writes progress
(`phase`, `todo`, `pids`, `finished`), the app/server writes control state
(`status`, `meta`), and both only ever UPDATE the columns they own.

The agent's live stream is its own log file — the CLI's JSONL stdout, teed
verbatim. `fold_log` turns raw lines into display items for the agent box:
the model's words, its tool actions, step markers and errors, each keyed by a
stable id so a growing text part updates in place as the tokens arrive.
"""
import json
import os
import re
import sqlite3
import time

import nn_arc

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS_DB = os.path.join(ROOT, "agents", "tasks.db")   # its own, git-ignored file
TASK_TYPES = ("reverse", "python", "train", "chat")   # what the app offers
STATUSES = ("running", "paused", "cancelled", "done")
_OUTCOMES = ("exact", "approximate", "failed")

_TASK_COLS = """(
  id TEXT PRIMARY KEY,
  type TEXT,
  params TEXT,
  status TEXT,
  outcome TEXT,
  phase TEXT,
  todo TEXT,
  pid INTEGER,
  agent_pid INTEGER,
  unit TEXT,
  log TEXT,
  meta TEXT,
  created REAL,
  started REAL,
  updated REAL,
  finished REAL)"""
_ADD_COLS = (("agent_pid", "INTEGER"), ("unit", "TEXT"), ("meta", "TEXT"),
             ("log", "TEXT"), ("outcome", "TEXT"), ("phase", "TEXT"),
             ("todo", "TEXT"), ("pid", "INTEGER"), ("started", "REAL"),
             ("finished", "REAL"))


# ------------------------------------------------------------- the table
def db_path(path=None):
    """The tasks' database — never the tracked hf_arc.db, so a user's prompts
    and machine paths can't ride into git with the researched specs."""
    return path or os.environ.get("NNSTUDIO_TASKS_DB") or TASKS_DB


def connect(path=None):
    p = db_path(path)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except OSError:
        pass
    return nn_arc.connect(p)


def ensure(path=None):
    """The task table in hf_arc's database — created, then grown additively."""
    conn = connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS agent_tasks" + _TASK_COLS)
        have = {r[1] for r in conn.execute("PRAGMA table_info(agent_tasks)")}
        with conn:
            for name, decl in _ADD_COLS:
                if name not in have:
                    conn.execute("ALTER TABLE agent_tasks ADD COLUMN %s %s" % (name, decl))
    finally:
        conn.close()


def _row(conn, rid):
    r = conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (rid,)).fetchone()
    return None if r is None else dict(zip([c[0] for c in conn.execute(
        "SELECT * FROM agent_tasks LIMIT 0").description], r))


def _plain(row):
    """One task as the app sees it: JSON fields parsed, times kept raw."""
    if not row:
        return None
    out = dict(row)
    for k in ("params", "todo", "meta"):
        try:
            out[k] = json.loads(out.get(k) or ("[]" if k == "todo" else "{}"))
        except (TypeError, ValueError):
            out[k] = [] if k == "todo" else {}
    return out


# ------------------------------------------------------------- the rows
def new_id():
    return "t_" + os.urandom(6).hex()


def create(task_type, params, path=None, log_dir=None):
    """A new task row — the only way a task exists. `params` is what the user
    asked for; for a reverse task it carries the HF repo id."""
    if task_type not in TASK_TYPES:
        raise ValueError("unknown task type: %s" % task_type)
    params = dict(params or {})
    rid = new_id()
    now = time.time()
    row = dict(id=rid, type=task_type, params=json.dumps(params, separators=(",", ":")),
               status="running", outcome=None, phase="queued", todo="[]",
               pid=None, agent_pid=None, unit=None,
               log=log_file(rid, log_dir), meta="{}", created=now, started=now,
               updated=now, finished=None)
    ensure(path)
    conn = connect(path)
    try:
        with conn:
            conn.execute("INSERT INTO agent_tasks (%s) VALUES (%s)"
                         % (",".join(row), ",".join("?" * len(row))),
                         tuple(row.values()))
    finally:
        conn.close()
    return get(rid, path)


def get(rid, path=None):
    ensure(path)
    conn = connect(path)
    try:
        return _plain(_row(conn, rid))
    finally:
        conn.close()


def list(path=None):
    """Every task, newest first — the agent box's list."""
    ensure(path)
    conn = connect(path)
    try:
        rows = conn.execute("SELECT * FROM agent_tasks ORDER BY created DESC").fetchall()
        names = [c[0] for c in conn.execute("SELECT * FROM agent_tasks LIMIT 0").description]
        return [_plain(dict(zip(names, r))) for r in rows]
    finally:
        conn.close()


def update(rid, path=None, **cols):
    """Update only the given columns — writers never clobber each other."""
    if not cols:
        return get(rid, path)
    cols = dict(cols)
    for k in ("params", "todo", "meta"):            # JSON columns arrive as dicts
        if k in cols and not isinstance(cols[k], str):
            cols[k] = json.dumps(cols[k], separators=(",", ":"))
    if "updated" not in cols:
        cols["updated"] = time.time()
    sets = ", ".join("%s = ?" % k for k in cols)
    ensure(path)
    conn = connect(path)
    try:
        with conn:
            conn.execute("UPDATE agent_tasks SET %s WHERE id = ?" % sets,
                         tuple(cols.values()) + (rid,))
    finally:
        conn.close()
    return get(rid, path)


def delete(rid, purge=False, path=None, log_dir=None):
    """Remove a task row (its log stays unless `purge`) — the task is gone."""
    row = get(rid, path)
    if not row:
        return False
    conn = connect(path)
    try:
        with conn:
            conn.execute("DELETE FROM agent_tasks WHERE id = ?", (rid,))
    finally:
        conn.close()
    if purge:
        try:
            os.remove(row.get("log") or log_file(rid, log_dir))
        except OSError:
            pass
    return True


def find_active(task_type, params, path=None):
    """The live task that already asks for exactly this (e.g. the same repo)."""
    return _find(task_type, params, ("running", "paused"), path)


def find_finished(task_type, params, path=None):
    """The most recent settled task for the same ask — a start reuses it
    instead of piling up rows, and its stream starts over."""
    return _find(task_type, params, ("done", "cancelled"), path)


def _find(task_type, params, statuses, path=None):
    want = json.dumps(dict(params or {}), separators=(",", ":"), sort_keys=True)
    for t in list(path):
        if t["type"] != task_type or t["status"] not in statuses:
            continue
        if json.dumps(t["params"], separators=(",", ":"), sort_keys=True) == want:
            return t
    return None


def title(task):
    """A task explains itself: its type, and what the user asked for."""
    t = (task or {}).get("type") or "?"
    params = (task or {}).get("params") or {}
    label = {"reverse": "reverse engineer repo",
             "python": "python agent",
             "chat": "reverse engineer chat"}.get(t, t)
    who = (params.get("repo") or params.get("source") or params.get("prompt")
           or params.get("message") or "")
    return (label + " \u00b7 " + who).rstrip(" \u00b7")


def status_word(task):
    """The chip: In-Progress / Paused / Done / Cancelled."""
    st = (task or {}).get("status")
    return {"running": "In-Progress", "paused": "Paused",
            "done": "Done", "cancelled": "Cancelled"}.get(st, st or "?")


def todo_summary(task):
    """(done, total, active item) from the checklist, for the list line."""
    items = (task or {}).get("todo") or []
    if not items:
        return 0, 0, None
    done = sum(1 for it in items if it.get("state") == "done")
    act = next((it.get("title") for it in items if it.get("state") == "active"), None)
    return done, len(items), act


def log_file(rid, log_dir=None):
    """The agent's live stream: its CLI's stdout, kept next to the database."""
    base = log_dir or os.path.join(ROOT, "agents")
    return os.path.join(base, "%s.log" % rid)


def reconcile(path=None):
    """A task whose process is gone is not running: land it honestly. (The
    wrapper usually lands itself; this covers a machine reboot mid-task.)"""
    changed = []
    for t in list(path):
        if t["status"] not in ("running", "paused"):
            continue
        pid = t.get("pid") or t.get("agent_pid")
        alive = False
        if pid:
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                alive = False
        if not alive:
            if pid is None and time.time() - (t.get("updated") or 0) < 15:
                continue                      # fired a breath ago: the wrapper is
                                              # still starting — its pid lands next
            update(t["id"], path, status="done", outcome="failed",
                   finished=time.time(),
                   meta=dict(t.get("meta") or {},
                             note="the agent process is gone"))
            changed.append(t["id"])
    return changed


# ------------------------------------------------------------- the stream
_JSON = re.compile(r"^\s*\{")


def _tool_summary(tool, inp):
    """One compact line for a tool action: what it touched."""
    tool = str(tool or "tool")
    for key in ("path", "command", "pattern", "file", "url", "prompt", "query"):
        if isinstance(inp, dict) and inp.get(key):
            v = str(inp[key]).replace("\n", " ")
            return "%s \u00b7 %s" % (tool, v[:120] if len(v) <= 120 else v[:117] + "\u2026")
    return tool


def _clip(text, n):
    text = str(text or "")
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def fold_log(lines, uid=""):
    """Raw agent-CLI lines -> display items `{id, kind, text, detail?, raw?}`.

    `kind` is one of say (the model's words), do (a tool action), step (a step
    marker with its tokens), error, note (the wrapper's own milestones). Items
    are keyed by the CLI's own part/call id, so a text part that grows token by
    token updates in place instead of piling up; `uid` (the tail's byte offset)
    keeps the keyless wrapper lines unique between polls. The caller only ever
    folds new lines, so an item is returned once — updated or not."""
    items = []
    at = {}
    for seq, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        ev = None
        if _JSON.match(line):
            try:
                ev = json.loads(line)
            except ValueError:
                ev = None
        if not isinstance(ev, dict):
            it = {"id": "note-%s-%d" % (uid, seq), "kind": "note",
                  "text": _clip(line, 400)}
            it["raw"] = _clip(line, 2000)
            items.append(it)
            continue
        part = ev.get("part") or {}
        ptype = part.get("type") or ""
        rid = part.get("partID") or part.get("id") or ev.get("id")
        delta = part.get("delta") if isinstance(part.get("delta"), str) else None
        if (ptype == "text" or delta is not None) and (part.get("text") or delta):
            if delta is not None:            # token by token: the same part grows
                it = {"id": str(rid), "kind": "say", "text": delta, "append": True}
            else:
                it = {"id": str(rid), "kind": "say", "text": str(part["text"])}
        elif ptype == "tool":
            tool = part.get("tool") or "tool"
            state = part.get("state") or {}
            inp = state.get("input") or {}
            it = {"id": str(part.get("id") or rid), "kind": "do",
                  "text": _tool_summary(tool, inp)}
            detail = "input:\n%s" % _clip(json.dumps(inp, indent=1, default=str), 2000)
            if state.get("output"):
                detail += "\noutput:\n%s" % _clip(str(state["output"]), 2000)
            it["detail"] = detail
        elif ptype == "step-finish" or ev.get("type") == "step_finish":
            tk = (part.get("tokens") or ev.get("tokens") or {})
            tot = (tk.get("input") or 0) + (tk.get("output") or 0)
            it = {"id": str(part.get("id") or rid or ("step-%s-%d" % (uid, seq))),
                  "kind": "step",
                  "text": ("step \u00b7 %s tokens" % tot) if tot else "step"}
        elif ev.get("type") == "error":
            it = {"id": "err-%s" % (rid or seq), "kind": "error",
                  "text": _clip(str(ev.get("error") or ev.get("message") or ev), 600)}
        else:
            continue                      # snapshots, message frames: nothing to say
        it["raw"] = _clip(line, 2000)
        if it["id"] in at:                # the same part again: grow or replace
            old = items[at[it["id"]]]
            if it.get("append"):           # a delta: the part grows token by token
                items[at[it["id"]]] = {"id": it["id"], "kind": it["kind"],
                                       "text": (old.get("text") or "") + it["text"],
                                       "append": True, "raw": it["raw"]}
            else:
                items[at[it["id"]]] = it
            continue
        at[it["id"]] = len(items)
        items.append(it)
    return items
