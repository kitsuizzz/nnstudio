#!/usr/bin/env python3
"""Accounts, and the designs they save — with the server kept in the dark.

Guests can do everything the app does — draw, wire, run, train — but nothing
is kept for them: saving a canvas is what an account is for. Creating one
always goes through Cloudflare's bot check: the register endpoint verifies the
challenge with Cloudflare before the row exists.

The token the browser makes during sign-up is **never sent anywhere**. The
client derives three values from it, each with its own label:

  * the **auth credential** — a sha256 the server sees on every request and
    stores only hashed again; enough to prove "this is the same token", useless
    for reading anything;
  * the **name tag** — a keyed hash of the account (or design) name; it lets
    the server key rows and refuse duplicates without learning the names;
  * the **data key** — a sha256 the browser turns into an AES-GCM key, which
    names and designs are sealed with *before* leaving the browser.

So this file never holds a plaintext name, design, or token: `users` keeps the
tag, the sealed account name and the auth hash; `canvases` keeps the tag, the
sealed design name and the sealed design. A stolen database is a pile of
ciphertext — and losing the token loses the designs for good, server included.
What remains visible is only the metadata: that an account exists, how many
designs it has, their sizes and timestamps.
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

TAG_RE = re.compile(r"^[0-9a-f]{64}$")         # a 32-byte sha256, hex
B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
IV_MAX = 64                                    # a GCM nonce, base64
NAME_CT_MAX = 1024                             # a sealed name, base64 chars
SESSION_MAX = 64 * 1024                        # a sealed session: the open tabs
EMAIL_CT_MAX = 1024                            # a sealed email, base64 chars
BLOB_MAX = 6 * 1024 * 1024                     # a sealed design, base64 chars

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
    p = db_path(path)
    conn = connect(path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    finally:
        conn.close()
    if cols and "name_tag" not in cols:            # the pre-sealing schema: keep it aside
        os.replace(p, p + ".pre-e2e")
    with connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name_tag TEXT UNIQUE,
            name_ct TEXT NOT NULL,
            auth_hash TEXT NOT NULL,
            created REAL NOT NULL,
            email_ct TEXT)""")
        # an older users.db: the optional email joins the row in place
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
        if "email_ct" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN email_ct TEXT")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
        if "sesh_ct" not in cols:              # the last screen: which saved tabs
            conn.execute("ALTER TABLE users ADD COLUMN sesh_ct TEXT")   # were open
        conn.execute("""CREATE TABLE IF NOT EXISTS canvases (
            user_id TEXT NOT NULL,
            name_tag TEXT NOT NULL,
            name_ct TEXT NOT NULL,
            blob TEXT NOT NULL,
            updated REAL NOT NULL,
            PRIMARY KEY (user_id, name_tag))""")


# ---------------------------------------------------------------- the secrets
def hash_auth(auth):
    """What is stored for an auth credential: its sha256. The credential is
    256 bits of derived randomness, so no slow KDF is needed."""
    return hashlib.sha256(str(auth or "").encode()).hexdigest()


def new_token():
    """A fresh client-side token — the one the browser makes. Kept here so the
    tests (and the curl recipes) can mint one without WebCrypto."""
    return secrets.token_urlsafe(32)


def new_auth():
    """A fresh client-side auth credential (what new_token() would derive)."""
    return secrets.token_hex(32)


def _tag(value, what):
    value = str(value or "").strip().lower()
    if not TAG_RE.match(value):
        raise ValueError("the %s is not a sha256 tag" % what)
    return value


def _sealed(value, what, cap):
    """A client-sealed value: {"iv": b64, "ct": b64}. The server can neither
    open it nor tell what is inside — only that it is shaped right."""
    if not isinstance(value, dict):
        raise ValueError("the %s must be sealed" % what)
    iv = str(value.get("iv") or "")
    ct = str(value.get("ct") or "")
    if not iv or len(iv) > IV_MAX or not B64_RE.match(iv):
        raise ValueError("the %s has no proper nonce" % what)
    if not ct or len(ct) > cap or not B64_RE.match(ct):
        raise ValueError("the %s is malformed or too big" % what)
    return json.dumps({"iv": iv, "ct": ct})


def _unpack(text):
    return json.loads(text) if text else None


def _user(row):
    if not row:
        return None
    keys = row.keys()
    return {"id": row["id"], "name_ct": _unpack(row["name_ct"]),
            "email_ct": _unpack(row["email_ct"]) if "email_ct" in keys else None,
            "created": row["created"]}


def user_for(auth, path=None):
    """The account a presented credential belongs to — None for a guest."""
    if not auth:
        return None
    ensure(path)
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE auth_hash = ?",
                           (hash_auth(auth),)).fetchone()
        return _user(row)


def register(name_tag, name_ct, auth, email_ct=None, path=None):
    """Create the account. Raises ValueError with the reason it is refused.
    The email is optional, and sealed like the name: for the server it is
    ciphertext either way."""
    tag = _tag(name_tag, "account tag")
    sealed = _sealed(name_ct, "account name", NAME_CT_MAX)
    sealed_mail = _sealed(email_ct, "account email", EMAIL_CT_MAX) if email_ct else None
    auth = str(auth or "")
    if not TAG_RE.match(auth):
        raise ValueError("the auth credential is not a sha256")
    ensure(path)
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM users WHERE name_tag = ?", (tag,)).fetchone():
            raise ValueError("that name is taken")
        uid = "u_" + secrets.token_hex(8)
        conn.execute("INSERT INTO users (id, name_tag, name_ct, auth_hash, created,"
                     " email_ct) VALUES (?,?,?,?,?,?)",
                     (uid, tag, sealed, hash_auth(auth), time.time(), sealed_mail))
    return {"id": uid, "name_ct": _unpack(sealed), "email_ct": _unpack(sealed_mail),
            "created": time.time()}


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
def save_canvas(user_id, name_tag, name_ct, blob, path=None):
    """One design per account per name tag: sealing over one replaces it."""
    tag = _tag(name_tag, "design tag")
    sealed_name = _sealed(name_ct, "design name", NAME_CT_MAX)
    sealed_blob = _sealed(blob, "design", BLOB_MAX)
    if not user_id:
        raise ValueError("an account is needed to save")
    ensure(path)
    with connect(path) as conn:
        conn.execute("""INSERT INTO canvases (user_id, name_tag, name_ct, blob, updated)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(user_id, name_tag)
                        DO UPDATE SET name_ct = excluded.name_ct,
                                      blob = excluded.blob,
                                      updated = excluded.updated""",
                     (user_id, tag, sealed_name, sealed_blob, time.time()))
    return {"name_tag": tag, "bytes": len(sealed_blob)}


def list_canvases(user_id, path=None):
    if not user_id:
        return []
    ensure(path)
    with connect(path) as conn:
        rows = conn.execute("SELECT name_tag, name_ct, blob, updated FROM canvases "
                            "WHERE user_id = ? ORDER BY updated DESC", (user_id,)).fetchall()
        return [{"name_tag": r["name_tag"], "name_ct": _unpack(r["name_ct"]),
                 "bytes": len(r["blob"]), "updated": r["updated"]} for r in rows]


def get_canvas(user_id, name_tag, path=None):
    if not user_id:
        return None
    ensure(path)
    with connect(path) as conn:
        row = conn.execute("SELECT name_ct, blob, updated FROM canvases "
                           "WHERE user_id = ? AND name_tag = ?",
                           (user_id, str(name_tag or "").strip().lower())).fetchone()
        return ({"name_ct": _unpack(row["name_ct"]), "blob": _unpack(row["blob"]),
                 "updated": row["updated"]} if row else None)


def save_session(user_id, blob, path=None):
    """The browser's last screen, sealed: which saved canvases were open as
    tabs, and which was active — one per account, replaced each time."""
    sealed = _sealed(blob, "session", SESSION_MAX)
    if not user_id:
        raise ValueError("an account is needed to remember a session")
    ensure(path)
    with connect(path) as conn:
        conn.execute("UPDATE users SET sesh_ct = ? WHERE id = ?", (sealed, user_id))
    return {"bytes": len(sealed)}


def get_session(user_id, path=None):
    if not user_id:
        return None
    ensure(path)
    with connect(path) as conn:
        row = conn.execute("SELECT sesh_ct FROM users WHERE id = ?",
                           (user_id,)).fetchone()
    return _unpack(row["sesh_ct"]) if row and row["sesh_ct"] else None


def drop_canvas(user_id, name_tag, path=None):
    if not user_id:
        return False
    ensure(path)
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM canvases WHERE user_id = ? AND name_tag = ?",
                           (user_id, str(name_tag or "").strip().lower()))
        return cur.rowcount > 0
