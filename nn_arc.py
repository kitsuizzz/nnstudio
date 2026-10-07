#!/usr/bin/env python3
"""hf_arc — the repo architecture database, and the reverse job's shared state.

The reverse-model agent researches how one HF repo really runs on raw torch and
draws it with nnstudio's own native boxes; the result lands here, keyed by the
repo id: the serialized nested-box canvas the app renders. Loading a repo
checks this table first — a researched architecture is instant, exact, and
never pays for research twice.

    hf_arc(repo_id TEXT PRIMARY KEY,   -- the HF repo id ("org/name")
           canvas TEXT NOT NULL,       -- the nested box, as JSON
           created_at REAL,
           transformers TEXT,          -- the library version it was proven on
           note TEXT)

The table lives next to the app (`hf_arc.db`; override `NNSTUDIO_ARC_DB`). The
app and the background job are separate processes, and a job must outlive an
app restart, so both talk to this SQLite file directly — no live service in
the middle.

The same module keeps the job's file state: `<slug>.json` (progress the box
polls), `<slug>.todos.json` (the checklist the agent fills as it works),
`<slug>.pipeline.json` (the agent's model-to-model wiring for a multi-model
repo). `NNSTUDIO_ARC_STATE_DIR` overrides the directory (tests use it).
"""
import json
import os
import re
import sqlite3
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE_DIR = "/tmp/opencode/reverse"


# ------------------------------------------------------------- the database
def db_path(path=None):
    return path or os.environ.get("NNSTUDIO_ARC_DB") or os.path.join(ROOT, "hf_arc.db")


def connect(path=None):
    """A connection to the hf_arc table (created if it is not there yet)."""
    conn = sqlite3.connect(db_path(path), timeout=15)
    conn.execute("""CREATE TABLE IF NOT EXISTS hf_arc (
        repo_id TEXT PRIMARY KEY,
        canvas TEXT NOT NULL,
        created_at REAL,
        transformers TEXT,
        note TEXT)""")
    return conn


def put(repo_id, canvas, transformers=None, note=None, path=None):
    """Store one repo's canvas — the verified architecture, as JSON."""
    repo_id = canonical(repo_id)
    if not repo_id or not canvas:
        raise ValueError("repo_id and canvas are required")
    conn = connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO hf_arc (repo_id, canvas, created_at, transformers, note) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(repo_id) DO UPDATE SET canvas=excluded.canvas, "
                "created_at=excluded.created_at, transformers=excluded.transformers, "
                "note=excluded.note",
                (repo_id, json.dumps(canvas, separators=(",", ":")),
                 time.time(), transformers, note))
    finally:
        conn.close()


def get(repo_id, path=None):
    """The repo's stored canvas, or None — a fresh dict every call."""
    repo_id = canonical(repo_id)
    if not repo_id:
        return None
    try:
        conn = connect(path)
        try:
            row = conn.execute("SELECT canvas FROM hf_arc WHERE repo_id = ?",
                               (repo_id,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None                  # a locked/corrupt db is a miss, not a crash
    if not row:
        return None
    try:
        return json.loads(row[0])
    except ValueError:
        return None


def delete(repo_id, path=None):
    repo_id = canonical(repo_id)
    if not repo_id:
        return False
    conn = connect(path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM hf_arc WHERE repo_id = ?", (repo_id,))
        return cur.rowcount > 0
    finally:
        conn.close()


def rows(path=None):
    """Every row, newest first — the app's cache overview."""
    try:
        conn = connect(path)
        try:
            cur = conn.execute("SELECT repo_id, created_at, transformers, note, "
                               "length(canvas) FROM hf_arc ORDER BY created_at DESC")
            return [{"repo_id": r[0], "created_at": r[1], "transformers": r[2],
                     "note": r[3], "bytes": r[4]} for r in cur.fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


# ------------------------------------------------------------- job state
def canonical(repo):
    """A repo id as the table and the job files key it: URLs unwrapped, no
    trailing slash. Local paths pass through untouched."""
    s = str(repo or "").strip()
    if s.startswith("http"):
        s = s.rstrip("/")
        if "huggingface.co/" in s:
            s = s.split("huggingface.co/", 1)[1]
        if s.endswith("/tree/main"):
            s = s[:-len("/tree/main")]
        return s.strip("/")
    if s.startswith("/") or s.startswith("./") or s.startswith("../"):
        return s.rstrip("/") or s
    return s.strip("/")


def repo_slug(repo):
    """A filesystem-safe key for one repo, readable in /tmp: the owner and
    name joined with `__`, everything odd as `-`."""
    s = canonical(repo).replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")
    return (s or "repo")[:96]


def state_dir(path=None):
    return path or os.environ.get("NNSTUDIO_ARC_STATE_DIR") or DEFAULT_STATE_DIR


def state_path(repo, ext=".json", path=None):
    return os.path.join(state_dir(path), repo_slug(repo) + ext)


def todos_path(repo, path=None):
    return state_path(repo, ".todos.json", path)


def pipeline_path(repo, path=None):
    return state_path(repo, ".pipeline.json", path)


def atomic_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------- the checklist
def todo_items(models=None):
    """The checklist for one repo, exactly the workflow the user watches:

        1  Reading Repository                       (once)
        2  Understanding Architecture — <model>     (one per model)
        3  Re-constructing Boxes — <model>          (one per model)
        4  Connecting the models — pipeline box     (once)
        5  Done                                     (the job's own gate)

    States move `wait` -> `active` -> `done` (or `failed`)."""
    items = [{"key": "reading", "title": "Reading Repository", "state": "wait"}]
    for name in models or []:
        items.append({"key": "understand:%s" % name, "state": "wait",
                      "title": "Understanding Architecture \u2014 %s" % name})
        items.append({"key": "reconstruct:%s" % name, "state": "wait",
                      "title": "Re-constructing Boxes \u2014 %s" % name})
    items.append({"key": "pipeline", "state": "wait",
                  "title": "Connecting the models \u2014 pipeline box"})
    items.append({"key": "done", "state": "wait", "title": "Done"})
    return items


def initial_todos(repo, path=None):
    """What the box shows the moment the job starts: reading, then done."""
    doc = {"repo": canonical(repo), "items": todo_items(None)}
    doc["items"][0]["state"] = "active"
    return write_todos(repo, doc, path)


def read_todos(repo, path=None):
    try:
        with open(todos_path(repo, path), encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def write_todos(repo, doc, path=None):
    doc = dict(doc or {})
    doc["repo"] = canonical(doc.get("repo") or repo)
    doc["updated"] = time.time()
    atomic_json(todos_path(repo, path), doc)
    return doc


def todo_find(items, key):
    for item in items or []:
        if item.get("key") == key:
            return item
    return None
