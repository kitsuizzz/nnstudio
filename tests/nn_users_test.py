#!/usr/bin/env python3
"""Accounts: what is stored for a token (its hash, never the token), the name
rules, Cloudflare's bot-check answers, and one design per name per account."""
import os
import sqlite3
import sys
import tempfile
import urllib.parse

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


def registering():
    tok = nn_users.new_token()
    u = nn_users.register("tester", tok, path=DB)
    assert u["name"] == "tester" and u["id"].startswith("u_"), u
    row = sqlite3.connect(DB).execute("select token_hash from users").fetchone()
    assert row[0] == nn_users.hash_token(tok) and tok not in row[0], "the raw token must not be stored"
    assert nn_users.user_for(tok, path=DB)["id"] == u["id"], "the token logs in"
    assert nn_users.user_for(tok + "x", path=DB) is None, "a tampered token is nobody"
    assert nn_users.user_for("", path=DB) is None, "no token: a guest"
    for bad in ("", "a", "x" * 33, "has/slash"):
        try:
            nn_users.register(bad, nn_users.new_token(), path=DB)
            raise AssertionError("accepted the name %r" % bad)
        except ValueError:
            pass
    try:
        nn_users.register("TESTER", nn_users.new_token(), path=DB)
        raise AssertionError("a taken name was accepted")
    except ValueError as e:
        assert "taken" in str(e), e
    try:
        nn_users.register("shorty", "abc", path=DB)
        raise AssertionError("a short token was accepted")
    except ValueError as e:
        assert "short" in str(e), e


def the_bot_check():
    seen = {}

    def ok(url, body):
        seen["url"] = url
        seen["fields"] = dict(urllib.parse.parse_qsl(body.decode()))
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


def designs():
    u = nn_users.register("saver", nn_users.new_token(), path=DB)
    a = nn_users.save_canvas(u["id"], "client", {"type": "canvas", "nodes": []}, path=DB)
    assert a["name"] == "client" and a["bytes"] > 0, a
    nn_users.save_canvas(u["id"], "client", {"nodes": [1]}, path=DB)
    got = nn_users.get_canvas(u["id"], "client", path=DB)
    assert got["design"]["nodes"] == [1], "saving over one replaces it"
    nn_users.save_canvas(u["id"], "other", {"nodes": []}, path=DB)
    names = sorted(c["name"] for c in nn_users.list_canvases(u["id"], path=DB))
    assert names == ["client", "other"], names
    assert nn_users.drop_canvas(u["id"], "other", path=DB) is True, "dropping one works"
    assert [c["name"] for c in nn_users.list_canvases(u["id"], path=DB)] == ["client"]
    other = nn_users.register("other", nn_users.new_token(), path=DB)
    assert nn_users.get_canvas(other["id"], "client", path=DB) is None, "nobody else's design"
    for name in ("", "x" * 65):
        try:
            nn_users.save_canvas(u["id"], name, {}, path=DB)
            raise AssertionError("saved the name %r" % name)
        except ValueError as e:
            assert "name" in str(e), e
    try:
        nn_users.save_canvas(u["id"], "big", {"blob": "x" * (nn_users.CANVAS_MAX + 10)}, path=DB)
        raise AssertionError("an oversized design was saved")
    except ValueError as e:
        assert "too big" in str(e), e
    try:
        nn_users.save_canvas(None, "client", {}, path=DB)
        raise AssertionError("a guest saved a design")
    except ValueError as e:
        assert "account" in str(e), e


check("an account: the token's hash is kept, and that token logs in", registering)
check("the bot check: Cloudflare's answer decides", the_bot_check)
check("the .env next to the code seeds the keys — an env var still wins", the_env_file)
check("a saved design: one per name, nobody else's", designs)

if FAILED:
    print("nn_users: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    raise SystemExit(1)
print("nn_users: all assertions passed")
