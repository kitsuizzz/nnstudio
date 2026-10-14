#!/usr/bin/env python3
"""Accounts, and the designs they save.

Guests can do everything the app does — draw, wire, run, train — but nothing
is kept for them: saving a canvas is what an account is for. Creating one
always goes through Cloudflare's bot check: the browser solves a Turnstile
challenge, the **client** generates its own token (Cryptographically random —
never a password), and the register endpoint verifies the challenge with
Cloudflare before the row exists. Logging in is that same token, presented
again.

The database is its own file (`users.db`, git-ignored): the token's sha256,
the account's name, and the designs. The raw token is never written down, so
a leaked file cannot be replayed.
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
USERS_DB = os.path.join(ROOT, "users.db")

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{1,31}$")
NAME_MAX = 32
CANVAS_MAX = 4 * 1024 * 1024                  # one saved design, bytes
CANVAS_NAME = re.compile(r"^[^\x00-\x1f]{1,64}$")

SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
# Cloudflare's own "always passes" pair. The app is usable out of the box, and
# the dialog says these are the test keys until the env pair is set.
TEST_SITE_KEY = "1x00000000000000000000AA"
TEST_SECRET_KEY = "1x0000000000000000000000000000000AA"


def load_env(path=None):
    """The repo's `.env` seeds os.environ at startup — KEY=VALUE lines, a real
    environment always wins. It is where the optional keys live (the Turnstile
    pair among them): the file is ignored, and never holds anything tracked."""
    path = path or os.path.join(ROOT, ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if key.strip():
                    os.environ.setdefault(key.strip(), val)
    except OSError:
        pass


def db_path(path=None):
    """The accounts' own file — never a tracked one."""
    return path or os.environ.get("NNSTUDIO_USERS_DB") or USERS_DB


def connect(path=None):
    p = db_path(path)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except OSError:
        pass
    conn = sqlite3.connect(p, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def ensure(path=None):
    with connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE COLLATE NOCASE,
            token_hash TEXT NOT NULL,
            created REAL NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS canvases (
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            json TEXT NOT NULL,
            updated REAL NOT NULL,
            PRIMARY KEY (user_id, name))""")


# ---------------------------------------------------------------- tokens
def hash_token(token):
    """What is stored for a token: its sha256. 32 random bytes make brute
    force a non-question, so no slow KDF is needed."""
    return hashlib.sha256(str(token or "").encode()).hexdigest()


def new_token():
    """A fresh client-side token — the one the browser makes. Kept here so
    tests (and the curl recipes in the docs) can mint one without WebCrypto."""
    return secrets.token_urlsafe(32)


def _user(row):
    return {"id": row["id"], "name": row["name"], "created": row["created"]} if row else None


def user_for(token, path=None):
    """The account a presented token belongs to — None for a guest."""
    if not token:
        return None
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE token_hash = ?",
                           (hash_token(token),)).fetchone()
        return _user(row)


def register(name, token, path=None):
    """Create the account. Raises ValueError with the reason it is refused."""
    name = str(name or "").strip()
    token = str(token or "")
    if not NAME_RE.match(name):
        raise ValueError("a name is 2–32 letters, digits or _ . - (it starts with one)")
    if len(token) < 16:
        raise ValueError("the token is too short — 32 random bytes, please")
    ensure(path)
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM users WHERE name = ? COLLATE NOCASE",
                        (name,)).fetchone():
            raise ValueError("that name is taken")
        uid = "u_" + secrets.token_hex(8)
        conn.execute("INSERT INTO users (id, name, token_hash, created) VALUES (?,?,?,?)",
                     (uid, name, hash_token(token), time.time()))
    return {"id": uid, "name": name, "created": time.time()}


# ---------------------------------------------------------------- bot check
def site_keys(site=None, secret=None):
    """(site key, secret, are they the test pair). The environment wins; the
    test pair keeps the app working — and the dialog says so — until then."""
    site = site if site is not None else os.environ.get("NNSTUDIO_TURNSTILE_SITE_KEY")
    secret = secret if secret is not None else os.environ.get("NNSTUDIO_TURNSTILE_SECRET_KEY")
    test = not (site and secret)
    return (site or TEST_SITE_KEY, secret or TEST_SECRET_KEY, test)


def _post_form(url, fields, fetch=None, timeout=15):
    body = urllib.parse.urlencode(fields).encode()
    if fetch is not None:                      # the tests hand a fake in
        return fetch(url, body)
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def verify_turnstile(secret, response, ip=None, fetch=None):
    """Ask Cloudflare whether the browser's challenge token is real. Returns
    (ok, error) — the registration only happens when ok."""
    response = str(response or "").strip()
    if not response:
        return False, "the bot check did not run"
    fields = {"secret": str(secret or ""), "response": response}
    if ip:
        fields["remoteip"] = str(ip)
    try:
        data = _post_form(SITEVERIFY, fields, fetch=fetch)
    except Exception as e:  # noqa: BLE001 — the network is the caller's problem
        return False, "the bot check could not be reached (%s)" % (e,)
    if not isinstance(data, dict):
        return False, "the bot check answered nothing"
    if data.get("success"):
        return True, None
    codes = data.get("error-codes") or []
    if "invalid-input-response" in codes or "timeout-or-duplicate" in codes:
        return False, "the bot check expired — try again"
    return False, "the bot check refused (%s)" % (", ".join(str(c) for c in codes) or "no reason")


# ---------------------------------------------------------------- the designs
def _canvas_name(name):
    name = str(name or "").strip()
    if not name or not CANVAS_NAME.match(name):
        raise ValueError("give the design a name (up to 64 characters)")
    return name


def save_canvas(user_id, name, design, path=None):
    """One design per account per name: saving over one replaces it."""
    name = _canvas_name(name)
    if not user_id:
        raise ValueError("an account is needed to save")
    text = json.dumps(design)
    if len(text) > CANVAS_MAX:
        raise ValueError("that design is too big to save (%.1f MB)" % (len(text) / 2 ** 20))
    ensure(path)
    with connect(path) as conn:
        conn.execute("""INSERT INTO canvases (user_id, name, json, updated)
                        VALUES (?,?,?,?)
                        ON CONFLICT(user_id, name)
                        DO UPDATE SET json = excluded.json, updated = excluded.updated""",
                     (user_id, name, text, time.time()))
    return {"name": name, "bytes": len(text)}


def list_canvases(user_id, path=None):
    if not user_id:
        return []
    ensure(path)
    with connect(path) as conn:
        rows = conn.execute("SELECT name, json, updated FROM canvases "
                            "WHERE user_id = ? ORDER BY updated DESC", (user_id,)).fetchall()
        return [{"name": r["name"], "updated": r["updated"], "bytes": len(r["json"])}
                for r in rows]


def get_canvas(user_id, name, path=None):
    if not user_id:
        return None
    ensure(path)
    with connect(path) as conn:
        row = conn.execute("SELECT json, updated FROM canvases WHERE user_id = ? AND name = ?",
                           (user_id, str(name or "").strip())).fetchone()
        return {"design": json.loads(row["json"]), "updated": row["updated"]} if row else None


def drop_canvas(user_id, name, path=None):
    if not user_id:
        return False
    ensure(path)
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM canvases WHERE user_id = ? AND name = ?",
                           (user_id, str(name or "").strip()))
        return cur.rowcount > 0
