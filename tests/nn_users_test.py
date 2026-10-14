#!/usr/bin/env python3
"""Accounts, sealed: the server stores a keyed tag per name, the sealed name,
the sealed design — and only the auth credential's hash. Never a plaintext
name, design or credential. (Plus the bot check's answers and the .env loader.)
"""
import base64
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nn_users  # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print("  FAIL " + name + " \u2014 " + str(e))


WORK = tempfile.mkdtemp(prefix="nnstudio-users-")
DB = os.path.join(WORK, "users.db")


def sealed(text, iv=b"0123456789ab"):
    """What the client sends: a nonce and ciphertext, base64. The server only
    ever checks the shape — it cannot open one of these."""
    return {"iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(str(text).encode()).decode()}


def tag(ch):
    return ch * 64


def registering():
    auth = nn_users.new_auth()
    u = nn_users.register(tag("a"), sealed("tester"), auth, path=DB)
    assert u["id"].startswith("u_"), u
    row = sqlite3.connect(DB).execute(
        "select name_tag, name_ct, auth_hash from users").fetchone()
    assert row[2] == nn_users.hash_auth(auth) and auth not in row[2], \
        "the credential is stored as its hash"
    assert row[0] == tag("a") and "tester" not in row[1], "the name is sealed: %r" % (row,)
    blob = open(DB, "rb").read()
    assert b"tester" not in blob, "not one plaintext name in the whole file"
    assert nn_users.user_for(auth, path=DB)["id"] == u["id"], "the credential logs in"
    assert nn_users.user_for(auth + "0", path=DB) is None, "a tampered credential is nobody"
    assert nn_users.user_for("", path=DB) is None, "no credential: a guest"
    try:
        nn_users.register(tag("a"), sealed("someone else"), nn_users.new_auth(), path=DB)
        raise AssertionError("a taken tag was accepted")
    except ValueError as e:
        assert "taken" in str(e), e
    for bad_tag in ("", "zz" * 32, "a" * 63):
        try:
            nn_users.register(bad_tag, sealed("x"), nn_users.new_auth(), path=DB)
            raise AssertionError("accepted the tag %r" % bad_tag)
        except ValueError:
            pass
    try:
        nn_users.register(tag("b"), sealed("x"), "not-a-sha256", path=DB)
        raise AssertionError("a bad credential was accepted")
    except ValueError as e:
        assert "auth" in str(e), e
    for bad in (None, {}, {"iv": "", "ct": "AA=="}, {"iv": "AA==", "ct": ""},
                {"iv": "AA==", "ct": "x" * (nn_users.NAME_CT_MAX + 2)}):
        try:
            nn_users.register(tag("c"), bad, nn_users.new_auth(), path=DB)
            raise AssertionError("accepted a bad sealed name: %r" % (bad,))
        except ValueError:
            pass


def the_bot_check():
    seen = {}

    def ok(url, body):
        seen["url"] = url
        seen["fields"] = dict(__import__("urllib.parse", fromlist=["parse_qsl"])
                              .parse_qsl(body.decode()))
        return {"success": True}

    good, err = nn_users.verify_turnstile("s3cret", "cf-token", ip="1.2.3.4", fetch=ok)
    assert good and err is None, (good, err)
    assert seen["url"] == nn_users.SITEVERIFY, seen["url"]
    assert seen["fields"] == {"secret": "s3cret", "response": "cf-token",
                              "remoteip": "1.2.3.4"}, seen["fields"]
    for answer, word in (({"success": False, "error-codes": ["invalid-input-response"]}, "expired"),
                         ({"success": False, "error-codes": ["bad-request"]}, "refused"),
                         ({"success": False}, "refused")):
        bad, why = nn_users.verify_turnstile("s", "r", fetch=lambda u, b, a=answer: a)
        assert not bad and word in why, (answer, why)
    assert nn_users.verify_turnstile("s", "")[0] is False, "no challenge token, no account"

    def boom(url, body):
        raise OSError("no route to Cloudflare")

    bad, why = nn_users.verify_turnstile("s", "r", fetch=boom)
    assert not bad and "reached" in why, why
    site, secret, test = nn_users.site_keys()
    assert site and secret and test, "unset keys fall back to Cloudflare's test pair"
    assert nn_users.site_keys("site-x", "secret-y") == ("site-x", "secret-y", False), \
        "the environment's keys win"


def the_env_file():
    key = "NNSTUDIO_TEST_ENV_KEY"
    os.environ.pop(key, None)
    p = os.path.join(WORK, "dotenv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# a comment\n\nNNSTUDIO_TEST_ENV_KEY = 'a value'\nnot a line\n")
    nn_users.load_env(p)
    assert os.environ.get(key) == "a value", os.environ.get(key)   # quotes and spacing go
    os.environ[key] = "already set"
    with open(p, "w", encoding="utf-8") as f:
        f.write("NNSTUDIO_TEST_ENV_KEY=from the file\n")
    nn_users.load_env(p)
    assert os.environ.get(key) == "already set", "a real environment variable always wins"
    os.environ.pop(key, None)


def the_old_store():
    p = os.path.join(WORK, "old.db")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT UNIQUE, "
                 "token_hash TEXT, created REAL)")
    conn.execute("INSERT INTO users VALUES ('u_1', 'old', 'deadbeef', 0)")
    conn.commit()
    conn.close()
    nn_users.ensure(p)                         # the pre-sealing schema steps aside
    assert os.path.exists(p + ".pre-e2e"), "the old file is kept, not deleted"
    assert os.path.exists(p), "a fresh store takes its place"
    cols = [r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(users)")]
    assert "name_tag" in cols and "auth_hash" in cols, cols
    assert nn_users.user_for(nn_users.new_auth(), path=p) is None, \
        "and no old row can log in"


def designs():
    u = nn_users.register(tag("d"), sealed("saver"), nn_users.new_auth(), path=DB)
    a = nn_users.save_canvas(u["id"], tag("1"), sealed("client"),
                             sealed('{"nodes":[]}'), path=DB)
    assert a["name_tag"] == tag("1") and a["bytes"] > 0, a
    nn_users.save_canvas(u["id"], tag("1"), sealed("client"),
                         sealed('{"nodes":[1]}'), path=DB)
    got = nn_users.get_canvas(u["id"], tag("1"), path=DB)
    assert got["blob"] == sealed('{"nodes":[1]}'), "saving over one replaces it"
    nn_users.save_canvas(u["id"], tag("2"), sealed("other"), sealed("{}"), path=DB)
    tags = sorted(c["name_tag"] for c in nn_users.list_canvases(u["id"], path=DB))
    assert tags == [tag("1"), tag("2")], tags
    assert nn_users.drop_canvas(u["id"], tag("2"), path=DB) is True, "dropping one works"
    assert [c["name_tag"] for c in nn_users.list_canvases(u["id"], path=DB)] == [tag("1")]
    other = nn_users.register(tag("e"), sealed("other"), nn_users.new_auth(), path=DB)
    assert nn_users.get_canvas(other["id"], tag("1"), path=DB) is None, "nobody else's design"
    assert nn_users.get_canvas(u["id"], "not-a-tag", path=DB) is None
    for bad in (None, {"iv": "AA==", "ct": "x" * (nn_users.BLOB_MAX + 2)}):
        try:
            nn_users.save_canvas(u["id"], tag("9"), sealed("big"), bad, path=DB)
            raise AssertionError("accepted a bad sealed design")
        except ValueError as e:
            assert "design" in str(e), e
    try:
        nn_users.save_canvas(None, tag("1"), sealed("x"), sealed("{}"), path=DB)
        raise AssertionError("a guest saved a design")
    except ValueError as e:
        assert "account" in str(e), e


check("an account: a hashed credential, a sealed name, nothing plaintext", registering)
check("the bot check: Cloudflare's answer decides", the_bot_check)
check("the .env next to the code seeds the keys — an env var still wins", the_env_file)
check("an old store steps aside for the sealed one, nothing deleted", the_old_store)
check("a sealed design: one per tag, nobody else's", designs)

if FAILED:
    print("nn_users: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    raise SystemExit(1)
print("nn_users: all assertions passed")
