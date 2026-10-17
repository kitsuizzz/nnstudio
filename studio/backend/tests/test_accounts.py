"""Account API: one-click sign-up (client-generated key), sign-in, sign-out.

Accounts are key-based and the key is generated IN THE BROWSER: the client
sends only SHA-256(key) (the "credential"), and the server stores only
SHA-256(credential) — the key itself never reaches the gateway. Sign-in
accepts the credential. The suite also covers the E2E store (/auth/store,
/auth/store/sync), which keeps conversations and studios as opaque blobs.

Also the Turnstile robot check: with the check on, sign-up accepts a captcha
token in addition to the credential. In "test" mode tokens are accepted
locally (the offline dev path — no siteverify call at all); in "live" mode
the gateway POSTs the token to Cloudflare's siteverify endpoint and 400s
when it is rejected, 503 when the endpoint is unreachable. Sign-in is never
gated. The suite stays offline: the Cloudflare call is intercepted with an
httpx.MockTransport (the dev install has no respx).
"""

import secrets
import sqlite3

import httpx
import pymysql
import pytest

from app import accounts, migrate_sqlite
from app.main import app


@pytest.fixture(autouse=True)
def clean_db(fake):
    """Every account test starts from an empty database.

    The schema itself is created once per process (accounts._connect); this
    wipes the rows so ids start at 1 and nothing leaks between tests. Needs
    the compose stack's MariaDB: `docker compose up -d mariadb`.
    """
    try:
        conn = accounts._connect()
    except pymysql.MySQLError as err:
        pytest.fail(
            f"MariaDB is not reachable ({err}) — start it with "
            "`docker compose up -d mariadb`"
        )
    try:
        for table in ("e2e_records", "admin_challenges", "tokens", "profiles"):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM users")
        conn.execute("ALTER TABLE users AUTO_INCREMENT = 1")
        conn.commit()
    finally:
        conn.close()


def auth_headers(token):
    # The client keeps the token in the aistudio.account cookie and reads it
    # into an Authorization header itself; the backend authenticates on that.
    return {"Authorization": f"Bearer {token}"}


def new_key() -> str:
    """A fresh access key, exactly as the browser mints it."""
    return accounts.KEY_PREFIX + secrets.token_urlsafe(32)


def signup_body(key: str, captcha: str = "") -> dict:
    """The sign-up body the client sends for `key` (never the key itself)."""
    raw = key[len(accounts.KEY_PREFIX) :]
    return {
        "credential": accounts.credential_from_key(key),
        "handle": raw[:6],
        "captcha": captcha,
    }


def signin_body(key: str) -> dict:
    return {"credential": accounts.credential_from_key(key)}


def _form_body(request: httpx.Request) -> dict:
    """The url-encoded form a siteverify POST carries."""
    return dict(
        line.split("=", 1)
        for line in request.content.decode().split("&")
        if "=" in line
    )


def _patch_cloudflare(monkeypatch, respond, seen: list):
    """Intercept the gateway's outbound Cloudflare client.

    Every siteverify request the gateway would send is appended to `seen`;
    `respond(request)` is the scripted answer (it may raise, to simulate an
    outage). Patching httpx.AsyncClient on the shared module reaches
    accounts.py, which looks the attribute up at call time.
    """
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return respond(request)

        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_signup_creates_an_account_with_a_client_key(client):
    # Robot check off (the default). The client generated the key; the server
    # only ever saw its credential, and echoes neither back.
    key = new_key()
    r = await client.post("/auth/signup", json=signup_body(key))
    assert r.status_code == 201
    body = r.json()
    assert "key" not in body
    assert "credential" not in body
    assert body["token"]
    assert body["user"]["id"] == 1
    assert body["user"]["name"] == key[len(accounts.KEY_PREFIX) :][:6]


async def test_signup_stores_only_a_hash_of_the_credential(client, db):
    key = new_key()
    credential = accounts.credential_from_key(key)
    await client.post("/auth/signup", json=signup_body(key))
    (stored,) = db.execute("SELECT credential_hash FROM users WHERE id = 1").fetchone()
    # Neither the key nor the credential is in the database — only the hash.
    assert key not in stored
    assert credential not in stored
    assert stored == accounts._hash_credential(credential)


async def test_signup_rejects_a_reused_credential(client):
    key = new_key()
    assert (await client.post("/auth/signup", json=signup_body(key))).status_code == 201
    again = await client.post("/auth/signup", json=signup_body(key))
    assert again.status_code == 409


async def test_signup_rejects_a_malformed_credential_or_handle(client):
    assert (
        await client.post(
            "/auth/signup", json={"credential": "not-hex", "handle": "abc123"}
        )
    ).status_code == 400
    key = new_key()
    assert (
        await client.post(
            "/auth/signup",
            json={"credential": accounts.credential_from_key(key), "handle": ""},
        )
    ).status_code == 400
    assert (
        await client.post(
            "/auth/signup",
            json={
                "credential": accounts.credential_from_key(key),
                "handle": "bad handle!",
            },
        )
    ).status_code == 400


async def test_me_requires_a_valid_token(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    assert (await client.get("/auth/me")).status_code == 401
    assert (
        await client.get("/auth/me", headers=auth_headers("nope"))
    ).status_code == 401
    r = await client.get("/auth/me", headers=auth_headers(token))
    assert r.status_code == 200
    assert r.json()["user"]["id"] == 1


async def test_me_accepts_the_token_from_the_cookie(client):
    # The browser attaches the aistudio.account cookie to every same-origin
    # request; a raw fetch with no Authorization header must still resolve.
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    r = await client.get("/auth/me", cookies={"aistudio.account": token})
    assert r.status_code == 200


async def test_signin_with_the_access_key(client):
    key = new_key()
    name = (await client.post("/auth/signup", json=signup_body(key))).json()["user"][
        "name"
    ]
    r = await client.post("/auth/signin", json=signin_body(key))
    assert r.status_code == 200
    assert r.json()["user"]["name"] == name
    new_token = r.json()["token"]
    assert new_token
    # A fresh session, so the new cookie resolves through /auth/me...
    assert (
        await client.get("/auth/me", headers=auth_headers(new_token))
    ).status_code == 200
    # ...and the pre-existing one still does.
    assert (
        await client.get("/auth/me", headers=auth_headers(r.json()["token"]))
    ).status_code == 200


async def test_signin_rejects_wrong_or_unknown_keys(client):
    await client.post("/auth/signup", json=signup_body(new_key()))
    wrong = await client.post(
        "/auth/signin", json=signin_body("ast_wrongwrongwrongwrong")
    )
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Incorrect access key."
    assert (
        await client.post("/auth/signin", json={"credential": ""})
    ).status_code == 401
    assert (
        await client.post("/auth/signin", json={"credential": "nope"})
    ).status_code == 401


async def test_no_credential_material_is_ever_returned(client):
    key = new_key()
    r = await client.post("/auth/signup", json=signup_body(key))
    token = r.json()["token"]
    assert "key" not in r.json() and "credential" not in r.json()
    again = await client.post("/auth/signin", json=signin_body(key))
    assert "key" not in again.json() and "credential" not in again.json()
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert "key" not in me.json()["user"] and "credential" not in me.json()["user"]


async def test_signout_invalidates_the_token(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    out = await client.post("/auth/signout", headers=auth_headers(token))
    assert out.status_code == 200
    # The session token no longer resolves to a user.
    assert (
        await client.get("/auth/me", headers=auth_headers(token))
    ).status_code == 401


async def test_the_key_still_signs_in_after_reload(client):
    # The point of the one-time reveal: the key is the account, so the same
    # key signs the user back in any time after.
    key = new_key()
    await client.post("/auth/signup", json=signup_body(key))
    assert (await client.post("/auth/signin", json=signin_body(key))).status_code == 200
    assert (await client.post("/auth/signin", json=signin_body(key))).status_code == 200


def _legacy_db(path, *, server_issued_keys: bool = False):
    """A SQLite accounts database, as the pre-MariaDB gateway wrote it.

    `server_issued_keys` builds the older shape, where users.key held the
    plaintext access key and profiles had no sealed-agents column.
    """
    conn = sqlite3.connect(str(path))
    if server_issued_keys:
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
            "name TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
    else:
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, "
            "credential_hash TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
            "created_at TEXT NOT NULL, admin_verifier TEXT, admin_key TEXT)"
        )
    conn.execute(
        "CREATE TABLE tokens (value TEXT PRIMARY KEY, user_id INTEGER NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE profiles (user_id INTEGER PRIMARY KEY, provider TEXT, "
        "updated_at TEXT)"
        if server_issued_keys
        else "CREATE TABLE profiles (user_id INTEGER PRIMARY KEY, provider TEXT, "
        "agents TEXT, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE admin_challenges (challenge TEXT PRIMARY KEY, "
        "user_id INTEGER NOT NULL, expires_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE e2e_records (user_id INTEGER NOT NULL, kind TEXT NOT NULL, "
        "rec_id TEXT NOT NULL, blob TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "PRIMARY KEY (user_id, kind, rec_id))"
    )
    return conn


async def test_migration_from_a_server_issued_key_database(client, db, tmp_path):
    # A pre-upgrade database stored the plaintext access key. The import
    # rewrites each row as SHA-256(SHA-256(key)) — the plaintext never reaches
    # MariaDB — and the old key still signs in via its credential.
    key = accounts.KEY_PREFIX + "legacykeylegacykey"
    path = tmp_path / "accounts.db"
    conn = _legacy_db(path, server_issued_keys=True)
    conn.execute(
        "INSERT INTO users (key, name, created_at) VALUES (?, ?, ?)",
        (key, "legacy", "2024-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO tokens (value, user_id, created_at) VALUES (?, ?, ?)",
        ("legacy-token", 1, "2024-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO profiles (user_id, provider, updated_at) VALUES (?, ?, ?)",
        (1, '[{"name": "carried"}]', "2024-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    counts = migrate_sqlite.migrate(str(path))
    assert counts["users"] == 1 and counts["tokens"] == 1

    (stored,) = db.execute("SELECT credential_hash FROM users WHERE id = 1").fetchone()
    assert stored == accounts._hash_credential(accounts.credential_from_key(key))
    assert key not in stored
    # The profile gained its sealed-agents column with nothing in it.
    (provider, agents) = db.execute(
        "SELECT provider, agents FROM profiles WHERE user_id = 1"
    ).fetchone()
    assert provider == '[{"name": "carried"}]'
    assert agents is None

    # The migrated session works as-is, and the key still signs in.
    r = await client.get("/auth/me", headers=auth_headers("legacy-token"))
    assert r.status_code == 200
    assert r.json()["user"]["name"] == "legacy"
    assert (await client.post("/auth/signin", json=signin_body(key))).status_code == 200


async def test_migration_copies_a_current_database(client, db, tmp_path):
    # Credential-hash era: the account, its profile, challenge and encrypted
    # records all come across, and a re-run replaces the same rows (upserts).
    key = new_key()
    credential = accounts.credential_from_key(key)
    path = tmp_path / "accounts.db"
    conn = _legacy_db(path)
    conn.execute(
        "INSERT INTO users (id, credential_hash, name, created_at, admin_verifier, "
        "admin_key) VALUES (?, ?, ?, ?, ?, ?)",
        (
            1,
            accounts._hash_credential(credential),
            "carried",
            "2024-01-01T00:00:00+00:00",
            None,
            "SEALED-ADMIN-COPY",
        ),
    )
    conn.execute(
        "INSERT INTO tokens (value, user_id, created_at) VALUES (?, ?, ?)",
        ("carried-token", 1, "2024-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO profiles (user_id, provider, agents, updated_at) VALUES (?, ?, ?, ?)",
        (1, '[{"name": "p"}]', "SEALED-AGENTS", "2024-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO admin_challenges (challenge, user_id, expires_at) VALUES (?, ?, ?)",
        ("carried-challenge", 1, "2999-01-01T00:00:00+00:00"),
    )
    for rec_id, blob, when in (
        ("1", "OPAQUE-1", "2024-01-01T00:00:00+00:00"),
        ("spot", "OPAQUE-SPOT", "2024-01-01T00:00:01+00:00"),
    ):
        conn.execute(
            "INSERT INTO e2e_records (user_id, kind, rec_id, blob, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, "view" if rec_id == "spot" else "conversation", rec_id, blob, when),
        )
    conn.commit()
    conn.close()

    counts = migrate_sqlite.migrate(str(path))
    assert counts == {
        "users": 1,
        "tokens": 1,
        "profiles": 1,
        "admin_challenges": 1,
        "e2e_records": 2,
    }
    # A verifier-less row gains fresh noise; the sealed copies come across as-is.
    (verifier,) = db.execute("SELECT admin_verifier FROM users WHERE id = 1").fetchone()
    assert verifier and len(verifier) == 64
    body = (await client.post("/auth/signin", json={"credential": credential})).json()
    assert body["providers"] == [{"name": "p"}]
    assert body["agents"] == "SEALED-AGENTS"
    assert body["adminKey"] == "SEALED-ADMIN-COPY"
    store = (
        await client.get("/auth/store", headers=auth_headers(body["token"]))
    ).json()
    assert [c["blob"] for c in store["conversations"]] == ["OPAQUE-1"]
    assert [v["blob"] for v in store["views"]] == ["OPAQUE-SPOT"]

    # Re-running the import replaces the same rows; it never doubles them.
    assert migrate_sqlite.migrate(str(path))["users"] == 1
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM e2e_records").fetchone()[0] == 2


def test_migration_refuses_the_pre_key_schema(db, tmp_path):
    # Email + password accounts never got an access key: nothing to carry over.
    path = tmp_path / "accounts.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, "
        "password_hash TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit):
        migrate_sqlite.migrate(str(path))


# --- Robot check (Cloudflare Turnstile) -------------------------------------
#
# The captcha_mode fixture turns the check on (test or live) on the isolated
# test settings; the outbound Cloudflare call is intercepted with an
# httpx.MockTransport. Sign-in is covered too: it must never trigger a
# siteverify call, whatever the mode.


async def test_signup_requires_the_robot_check(fake, captcha_mode, client, monkeypatch):
    captcha_mode("live")
    seen: list = []
    _patch_cloudflare(
        monkeypatch, lambda request: httpx.Response(200, json={"success": True}), seen
    )
    # No token at all -> 400, and it short-circuits before any outbound call.
    r = await client.post("/auth/signup", json=signup_body(new_key()))
    assert r.status_code == 400
    assert r.json()["detail"] == "Please complete the robot check."
    assert seen == []


async def test_signup_test_mode_passes_locally_without_siteverify(
    fake, captcha_mode, client, monkeypatch
):
    # "test" is the offline dev path: the documented test tokens are accepted
    # locally, so no siteverify round-trip happens at all.
    captcha_mode("test")
    seen: list = []
    _patch_cloudflare(
        monkeypatch, lambda request: httpx.Response(200, json={"success": True}), seen
    )
    r = await client.post(
        "/auth/signup", json=signup_body(new_key(), captcha="XXXX.DUMMY.TOKEN.XXXX")
    )
    assert r.status_code == 201
    assert seen == []


async def test_signup_accepts_a_passed_robot_check(
    fake, captcha_mode, client, monkeypatch
):
    captcha_mode("live")
    seen: list = []
    _patch_cloudflare(
        monkeypatch,
        lambda request: httpx.Response(200, json={"success": True, "hostname": "test"}),
        seen,
    )
    r = await client.post(
        "/auth/signup", json=signup_body(new_key(), captcha="token-from-a-human")
    )
    assert r.status_code == 201
    assert len(seen) == 1
    # The token went to siteverify with the configured secret.
    body = _form_body(seen[0])
    assert body["secret"] == "captcha-secret"
    assert body["response"] == "token-from-a-human"


async def test_signup_rejects_a_failed_robot_check(
    fake, captcha_mode, client, monkeypatch, db
):
    captcha_mode("live")
    seen: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"success": False, "error-codes": ["invalid-input-response"]}
        )

    _patch_cloudflare(monkeypatch, respond, seen)
    r = await client.post(
        "/auth/signup", json=signup_body(new_key(), captcha="token-from-a-bot")
    )
    assert r.status_code == 400
    assert "invalid-input-response" in r.json()["detail"]
    # A bot was not created: the check fails before the database is opened.
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


async def test_signup_503_when_cloudflare_is_unreachable(
    fake, captcha_mode, client, monkeypatch
):
    captcha_mode("live")
    seen: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated Cloudflare outage")

    _patch_cloudflare(monkeypatch, respond, seen)
    r = await client.post(
        "/auth/signup", json=signup_body(new_key(), captcha="token-from-a-bot")
    )
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


async def test_signup_live_mode_sends_the_request_ip(
    fake, captcha_mode, client, monkeypatch
):
    # siteverify can't tell a bot from a misconfigured reverse proxy without
    # the client address; live mode sends the first X-Forwarded-For hop.
    captcha_mode("live")
    seen: list = []
    _patch_cloudflare(
        monkeypatch, lambda request: httpx.Response(200, json={"success": True}), seen
    )
    r = await client.post(
        "/auth/signup",
        json=signup_body(new_key(), captcha="token-from-a-human"),
        headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"},
    )
    assert r.status_code == 201
    assert len(seen) == 1
    body = _form_body(seen[0])
    assert body["secret"] == "captcha-secret"
    assert body["remoteip"] == "203.0.113.7"


async def test_signin_never_calls_siteverify(fake, captcha_mode, client, monkeypatch):
    # Sign-up (check on, live mode) does call siteverify; sign-in must not.
    captcha_mode("live")
    seen: list = []
    _patch_cloudflare(
        monkeypatch, lambda request: httpx.Response(200, json={"success": True}), seen
    )
    key = new_key()
    r = await client.post(
        "/auth/signup", json=signup_body(key, captcha="token-from-a-human")
    )
    assert r.status_code == 201
    assert len(seen) == 1
    seen.clear()
    # A wrong credential 401s straight away — no siteverify round-trip.
    assert (
        await client.post("/auth/signin", json=signin_body("ast_nope"))
    ).status_code == 401
    assert (await client.post("/auth/signin", json=signin_body(key))).status_code == 200
    assert seen == []


# --- Per-account provider profile -------------------------------------------
#
# The one place the server holds a provider key: a signed-in account stores
# its chosen LLM provider in the profiles table, and the gateway returns it to
# the owner on /auth/me and /auth/signin so a returning user resumes straight
# into a conversation. PUT /auth/profile stores or clears it; a session token
# is bound to one account, so a client can only ever reach its own profile.

PROVIDER = {
    "name": "Nova",
    "baseUrl": "https://nova.example/v1",
    "apiKey": "sk-nova-123",
    "model": "nova-4o",
}


async def test_profile_is_absent_until_one_is_saved(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.status_code == 200
    assert me.json()["provider"] is None


async def test_profile_roundtrip_through_me_and_signin(client):
    key = new_key()
    body = (await client.post("/auth/signup", json=signup_body(key))).json()
    token = body["token"]
    put = await client.put(
        "/auth/profile", json={"provider": PROVIDER}, headers=auth_headers(token)
    )
    assert put.status_code == 200
    # /auth/me returns it to the owner...
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["provider"] == PROVIDER
    # ...and /auth/signin returns it on the next sign-in (cross-device resume).
    again = await client.post("/auth/signin", json=signin_body(key))
    assert again.status_code == 200
    assert again.json()["provider"] == PROVIDER


async def test_profile_requires_a_valid_session(client):
    # No token at all, and a token nobody recognises, both 401.
    assert (
        await client.put("/auth/profile", json={"provider": PROVIDER})
    ).status_code == 401
    assert (
        await client.put(
            "/auth/profile", json={"provider": PROVIDER}, headers=auth_headers("bogus")
        )
    ).status_code == 401
    # And a malformed profile body 422s even with a valid session.
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    r = await client.put(
        "/auth/profile",
        json={"provider": {"name": "Nova"}},
        headers=auth_headers(token),
    )
    assert r.status_code == 422


async def test_profile_cannot_reach_another_account(client):
    a = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    b = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    # Account A stores a profile under its own session...
    assert (
        await client.put(
            "/auth/profile",
            json={"provider": PROVIDER},
            headers=auth_headers(a["token"]),
        )
    ).status_code == 200
    # ...which account B (a different session) must not be able to read...
    me_b = await client.get("/auth/me", headers=auth_headers(b["token"]))
    assert me_b.status_code == 200
    assert me_b.json()["provider"] is None
    # ...and B's own write must not clobber A's.
    other = {**PROVIDER, "apiKey": "sk-someone-elses"}
    assert (
        await client.put(
            "/auth/profile", json={"provider": other}, headers=auth_headers(b["token"])
        )
    ).status_code == 200
    assert (await client.get("/auth/me", headers=auth_headers(a["token"]))).json()[
        "provider"
    ] == PROVIDER


async def test_profile_clear_with_null(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    assert (
        await client.put(
            "/auth/profile", json={"provider": PROVIDER}, headers=auth_headers(token)
        )
    ).status_code == 200
    assert (
        await client.put(
            "/auth/profile", json={"provider": None}, headers=auth_headers(token)
        )
    ).status_code == 200
    # A null (or omitted) provider clears the stored config.
    assert (await client.get("/auth/me", headers=auth_headers(token))).json()[
        "provider"
    ] is None


async def test_profile_stores_a_provider_list(client):
    # Two BYOK providers: the list is what /auth/me and /auth/signin return,
    # with the legacy single "provider" echo kept as the first entry.
    key = new_key()
    token = (await client.post("/auth/signup", json=signup_body(key))).json()["token"]
    second = {
        "name": "Helix",
        "baseUrl": "https://helix.example/v1",
        "apiKey": "sk-helix",
        "model": "helix-2",
    }
    put = await client.put(
        "/auth/profile",
        json={"providers": [PROVIDER, second]},
        headers=auth_headers(token),
    )
    assert put.status_code == 200
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["providers"] == [PROVIDER, second]
    assert me.json()["provider"] == PROVIDER
    again = await client.post("/auth/signin", json=signin_body(key))
    assert again.json()["providers"] == [PROVIDER, second]
    assert again.json()["provider"] == PROVIDER


async def test_profile_list_keeps_ids_and_models(client):
    # The stable id (which a model choice references) and the provider's own
    # model list must survive the round-trip; absent optional fields stay
    # absent, so a legacy provider reads back exactly as it was written.
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    tagged = {
        **PROVIDER,
        "id": "p_1",
        "models": ["nova-4o", "nova-4o-mini"],
        "noKey": True,
    }
    assert (
        await client.put(
            "/auth/profile", json={"providers": [tagged]}, headers=auth_headers(token)
        )
    ).status_code == 200
    stored = (await client.get("/auth/me", headers=auth_headers(token))).json()[
        "providers"
    ]
    assert stored == [tagged]
    # A single-provider write (the legacy shape) still reads back as a list.
    assert (
        await client.put(
            "/auth/profile", json={"provider": PROVIDER}, headers=auth_headers(token)
        )
    ).status_code == 200
    me = (await client.get("/auth/me", headers=auth_headers(token))).json()
    assert me["providers"] == [PROVIDER]
    assert me["provider"] == PROVIDER


async def test_profile_clear_with_empty_list(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    assert (
        await client.put(
            "/auth/profile", json={"providers": [PROVIDER]}, headers=auth_headers(token)
        )
    ).status_code == 200
    assert (
        await client.put(
            "/auth/profile", json={"providers": []}, headers=auth_headers(token)
        )
    ).status_code == 200
    me = (await client.get("/auth/me", headers=auth_headers(token))).json()
    assert me["providers"] == []
    assert me["provider"] is None


# --- Agents: sealed by the client, stored beside the providers ---------------
# The agents (Agent Settings) travel as one opaque, client-sealed string. The
# server stores and returns it without reading it, and a write never touches
# the providers: each field a request sends replaces only its own value.

SEALED_AGENTS = "ZXNlYWxlZC1hZ2VudHMtYmxvYg=="


async def test_agents_roundtrip_through_me_and_signin(client):
    key = new_key()
    token = (await client.post("/auth/signup", json=signup_body(key))).json()["token"]
    assert (await client.get("/auth/me", headers=auth_headers(token))).json()[
        "agents"
    ] is None
    put = await client.put(
        "/auth/profile", json={"agents": SEALED_AGENTS}, headers=auth_headers(token)
    )
    assert put.status_code == 200
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["agents"] == SEALED_AGENTS
    again = await client.post("/auth/signin", json=signin_body(key))
    assert again.json()["agents"] == SEALED_AGENTS


async def test_agents_and_providers_are_saved_independently(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    headers = auth_headers(token)
    for body in ({"providers": [PROVIDER]}, {"agents": SEALED_AGENTS}):
        assert (
            await client.put("/auth/profile", json=body, headers=headers)
        ).status_code == 200
    me = (await client.get("/auth/me", headers=headers)).json()
    assert me["providers"] == [PROVIDER]
    assert me["agents"] == SEALED_AGENTS
    # Clearing the providers keeps the agents...
    await client.put("/auth/profile", json={"providers": []}, headers=headers)
    me = (await client.get("/auth/me", headers=headers)).json()
    assert me["providers"] == []
    assert me["agents"] == SEALED_AGENTS
    # ...a body that sends neither field changes nothing...
    await client.put("/auth/profile", json={"providers": [PROVIDER]}, headers=headers)
    await client.put("/auth/profile", json={}, headers=headers)
    me = (await client.get("/auth/me", headers=headers)).json()
    assert me["providers"] == [PROVIDER]
    assert me["agents"] == SEALED_AGENTS
    # ...and clearing the agents keeps the providers.
    await client.put("/auth/profile", json={"agents": ""}, headers=headers)
    me = (await client.get("/auth/me", headers=headers)).json()
    assert me["agents"] is None
    assert me["providers"] == [PROVIDER]


async def test_agents_are_per_account_and_size_capped(client):
    a = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    b = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    assert (
        await client.put(
            "/auth/profile",
            json={"agents": SEALED_AGENTS},
            headers=auth_headers(a["token"]),
        )
    ).status_code == 200
    # B starts with no agents, and B's own write must not clobber A's.
    assert (await client.get("/auth/me", headers=auth_headers(b["token"]))).json()[
        "agents"
    ] is None
    assert (
        await client.put(
            "/auth/profile",
            json={"agents": "b-agents"},
            headers=auth_headers(b["token"]),
        )
    ).status_code == 200
    assert (await client.get("/auth/me", headers=auth_headers(a["token"]))).json()[
        "agents"
    ] == SEALED_AGENTS
    # A runaway blob is refused rather than stored.
    too_big = await client.put(
        "/auth/profile",
        json={"agents": "x" * 300_000},
        headers=auth_headers(a["token"]),
    )
    assert too_big.status_code == 422


# --- Admin: a one-way verifier, proven with a challenge ----------------------
# The user type is not stored readably anywhere: every account carries a
# one-way verifier (a decoy for non-admins), and the server only ever learns
# "admin" when a single-use challenge is answered with the admin key.


def _promoted_key(capsys) -> str:
    """The admin key the promote CLI just printed (it is shown only there)."""
    out = capsys.readouterr().out
    keys = [
        line.strip() for line in out.splitlines() if line.strip().startswith("adm_")
    ]
    assert len(keys) == 1, out
    return keys[0]


async def test_no_readable_user_type_is_stored(client, db):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    me = await client.get("/auth/me", headers=auth_headers(token))
    body = me.json()
    # The API carries no role — the server holds none — and no sealed key yet.
    assert "role" not in body["user"]
    assert body["adminKey"] is None

    # The database has no readable role either: every account carries a
    # one-way verifier (noise, for a non-admin), and no `role` column at all.
    columns = {
        row[0]
        for row in db.execute(
            "SELECT LOWER(column_name) FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND LOWER(table_name) = 'users'"
        ).fetchall()
    }
    (verifier,) = db.execute("SELECT admin_verifier FROM users WHERE id = 1").fetchone()
    assert "role" not in columns
    assert verifier and len(verifier) == 64


async def test_promote_mints_a_key_and_a_challenge_proves_admin(client, capsys):
    from app import accounts

    key = new_key()
    body = (await client.post("/auth/signup", json=signup_body(key))).json()
    token = body["token"]
    handle = body["user"]["name"]
    assert accounts._run_cli(["promote", handle]) == 0
    admin_key = _promoted_key(capsys)

    # A challenge is single-use, and a wrong key consumes it without proving.
    challenge = (
        await client.post("/auth/admin/challenge", headers=auth_headers(token))
    ).json()["challenge"]
    wrong = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": "adm_not-the-key"},
        headers=auth_headers(token),
    )
    assert wrong.status_code == 401
    replay = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": admin_key},
        headers=auth_headers(token),
    )
    assert replay.status_code == 400  # the challenge is spent

    # A fresh challenge, answered with the minted key: the server now knows.
    challenge = (
        await client.post("/auth/admin/challenge", headers=auth_headers(token))
    ).json()["challenge"]
    proven = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": admin_key},
        headers=auth_headers(token),
    )
    assert proven.status_code == 200
    assert proven.json() == {"admin": True}

    # The sealed admin-key copy round-trips to its owner only, and clears.
    sealed = "e2e1:c2VhbGVkLWFkbWluLWtleQ=="
    assert (
        await client.put(
            "/auth/admin/key", json={"key": sealed}, headers=auth_headers(token)
        )
    ).status_code == 200
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["adminKey"] == sealed
    assert (
        await client.put("/auth/admin/key", json={}, headers=auth_headers(token))
    ).status_code == 200
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["adminKey"] is None
    # Nothing admin-shaped works without a session.
    assert (await client.post("/auth/admin/challenge")).status_code == 401
    assert (
        await client.post("/auth/admin/prove", json={"challenge": "x", "key": "y"})
    ).status_code == 401
    assert (
        await client.put("/auth/admin/key", json={"key": sealed})
    ).status_code == 401


async def test_challenge_requires_its_own_session_and_expires(client):
    from app import accounts

    a = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    b = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    challenge = (
        await client.post("/auth/admin/challenge", headers=auth_headers(a["token"]))
    ).json()["challenge"]
    # Another account cannot answer someone else's challenge.
    crossed = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": "adm_x"},
        headers=auth_headers(b["token"]),
    )
    assert crossed.status_code == 400
    # An aged-out challenge is refused (and consumed).
    challenge = (
        await client.post("/auth/admin/challenge", headers=auth_headers(a["token"]))
    ).json()["challenge"]
    conn = accounts._connect()
    conn.execute(
        "UPDATE admin_challenges SET expires_at = %s", ("2000-01-01T00:00:00+00:00",)
    )
    conn.commit()
    conn.close()
    expired = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": "adm_x"},
        headers=auth_headers(a["token"]),
    )
    assert expired.status_code == 400
    assert "expired" in expired.json()["detail"]


async def test_demote_revokes_the_key_and_ambiguity_needs_an_id(client, capsys):
    from app import accounts

    key = new_key()
    body = (await client.post("/auth/signup", json=signup_body(key))).json()
    token = body["token"]
    handle = body["user"]["name"]
    user_id = body["user"]["id"]
    assert accounts._run_cli(["promote", handle]) == 0
    admin_key = _promoted_key(capsys)

    # Handles are not unique: a second account sharing one needs the numeric id.
    other = (await client.post("/auth/signup", json=signup_body(new_key()))).json()
    conn = accounts._connect()
    conn.execute(
        "UPDATE users SET name = %s WHERE id = %s", (handle, other["user"]["id"])
    )
    conn.commit()
    conn.close()
    try:
        accounts._run_cli(["promote", handle])
    except SystemExit as err:
        assert "use the numeric id" in str(err)
    else:
        raise AssertionError("an ambiguous handle must be refused")

    # Demote: the same key no longer proves, and the sealed copy is cleared.
    assert (
        await client.put(
            "/auth/admin/key", json={"key": "e2e1:x"}, headers=auth_headers(token)
        )
    ).status_code == 200
    assert accounts._run_cli(["demote", str(user_id)]) == 0
    challenge = (
        await client.post("/auth/admin/challenge", headers=auth_headers(token))
    ).json()["challenge"]
    proven = await client.post(
        "/auth/admin/prove",
        json={"challenge": challenge, "key": admin_key},
        headers=auth_headers(token),
    )
    assert proven.status_code == 401
    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.json()["adminKey"] is None

    # Re-promoting mints a fresh key: the old one stays dead. (The fresh key
    # is read back from the CLI output; demote printed nothing this time.)
    assert accounts._run_cli(["promote", str(user_id)]) == 0
    fresh = _promoted_key(capsys)
    assert fresh != admin_key


# --- E2E store (encrypted conversations + studios) ---------------------------# --- E2E store (encrypted conversations + studios) ---------------------------
#
# The gateway is a dumb ciphertext store: it keeps the client's opaque blobs
# and never sees plaintext, the access key, or the decryption key. Every row
# is owner-scoped by the session token, so one account can never touch
# another's records.


async def test_store_requires_a_valid_session(client):
    assert (await client.get("/auth/store")).status_code == 401
    assert (await client.post("/auth/store/sync", json={})).status_code == 401
    assert (
        await client.post("/auth/store/sync", json={}, headers=auth_headers("bogus"))
    ).status_code == 401


async def test_store_sync_roundtrip_upsert_and_delete(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    headers = auth_headers(token)
    # Nothing stored yet.
    assert (await client.get("/auth/store", headers=headers)).json() == {
        "conversations": [],
        "studios": [],
        "views": [],
        "last": [],
    }
    # An upsert lands verbatim (the blob is opaque to the server).
    r = await client.post(
        "/auth/store/sync",
        json={
            "conversations": {"upserts": [{"id": "1", "blob": "OPAQUE-CIPHERTEXT-1"}]}
        },
        headers=headers,
    )
    assert r.status_code == 200
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [c["id"] for c in store["conversations"]] == ["1"]
    assert store["conversations"][0]["blob"] == "OPAQUE-CIPHERTEXT-1"
    assert store["conversations"][0]["updatedAt"]
    # A second upsert for the same id replaces the blob.
    await client.post(
        "/auth/store/sync",
        json={
            "conversations": {"upserts": [{"id": "1", "blob": "OPAQUE-CIPHERTEXT-2"}]}
        },
        headers=headers,
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [c["blob"] for c in store["conversations"]] == ["OPAQUE-CIPHERTEXT-2"]
    # Studios live in their own collection.
    await client.post(
        "/auth/store/sync",
        json={"studios": {"upserts": [{"id": "s1", "blob": "STUDIO-BLOB"}]}},
        headers=headers,
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [s["id"] for s in store["studios"]] == ["s1"]
    # A delete drops the row.
    await client.post(
        "/auth/store/sync", json={"conversations": {"deletes": ["1"]}}, headers=headers
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert store["conversations"] == []
    assert [s["id"] for s in store["studios"]] == ["s1"]


async def test_store_is_owner_scoped(client):
    a = (await client.post("/auth/signup", json=signup_body(new_key()))).json()["token"]
    b = (await client.post("/auth/signup", json=signup_body(new_key()))).json()["token"]
    await client.post(
        "/auth/store/sync",
        json={"conversations": {"upserts": [{"id": "1", "blob": "A-ONLY"}]}},
        headers=auth_headers(a),
    )
    # B sees nothing...
    assert (await client.get("/auth/store", headers=auth_headers(b))).json()[
        "conversations"
    ] == []
    # ...and B's delete of the same id must not touch A's record.
    await client.post(
        "/auth/store/sync",
        json={"conversations": {"deletes": ["1"]}},
        headers=auth_headers(b),
    )
    store = (await client.get("/auth/store", headers=auth_headers(a))).json()
    assert [c["blob"] for c in store["conversations"]] == ["A-ONLY"]


async def test_store_rejects_malformed_records(client):
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    headers = auth_headers(token)
    r = await client.post(
        "/auth/store/sync",
        json={"conversations": {"upserts": [{"id": "", "blob": "x"}]}},
        headers=headers,
    )
    assert r.status_code == 400
    r = await client.post(
        "/auth/store/sync",
        json={"conversations": {"upserts": [{"id": "1", "blob": ""}]}},
        headers=headers,
    )
    assert r.status_code == 400


async def test_reading_spots_are_their_own_collection(client):
    # Where the reader left off is kept apart from the conversations: one small
    # record per conversation, opaque like the rest of the store.
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()[
        "token"
    ]
    headers = auth_headers(token)
    r = await client.post(
        "/auth/store/sync",
        json={
            "conversations": {"upserts": [{"id": "1", "blob": "CONVERSATION-1"}]},
            "views": {"upserts": [{"id": "1", "blob": "SPOT-1"}]},
        },
        headers=headers,
    )
    assert r.status_code == 200
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [v["blob"] for v in store["views"]] == ["SPOT-1"]
    assert [c["blob"] for c in store["conversations"]] == ["CONVERSATION-1"]
    # Clearing the spot leaves the conversation alone.
    await client.post(
        "/auth/store/sync", json={"views": {"deletes": ["1"]}}, headers=headers
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert store["views"] == []
    assert [c["blob"] for c in store["conversations"]] == ["CONVERSATION-1"]
    # A runaway record is refused, like any other.
    too_big = "x" * (accounts._MAX_BLOB_CHARS + 1)
    r = await client.post(
        "/auth/store/sync",
        json={"views": {"upserts": [{"id": "1", "blob": too_big}]}},
        headers=headers,
    )
    assert r.status_code == 413


async def test_reading_spots_are_owner_scoped(client):
    a = (await client.post("/auth/signup", json=signup_body(new_key()))).json()["token"]
    b = (await client.post("/auth/signup", json=signup_body(new_key()))).json()["token"]
    await client.post(
        "/auth/store/sync",
        json={"views": {"upserts": [{"id": "1", "blob": "A-SPOT"}]}},
        headers=auth_headers(a),
    )
    assert (await client.get("/auth/store", headers=auth_headers(b))).json()[
        "views"
    ] == []
    # B's delete of the same id must not touch A's spot.
    await client.post(
        "/auth/store/sync",
        json={"views": {"deletes": ["1"]}},
        headers=auth_headers(b),
    )
    store = (await client.get("/auth/store", headers=auth_headers(a))).json()
    assert [v["blob"] for v in store["views"]] == ["A-SPOT"]


# --- MariaDB session semantics -----------------------------------------------


async def test_profile_writes_survive_concurrent_updates(client, db):
    # MariaDB's default REPEATABLE READ raises "record has changed since last
    # read" (1020) when a transaction writes a row that another transaction
    # changed after its snapshot. The profile endpoints are short, concurrent
    # requests (a second device saving providers while agents are saved here),
    # so they run READ COMMITTED: the second writer wins the row, no error.
    key = new_key()
    token = (await client.post("/auth/signup", json=signup_body(key))).json()["token"]
    headers = auth_headers(token)
    provider = {"name": "p", "baseUrl": "http://x", "apiKey": "k", "model": "m"}
    await client.put("/auth/profile", json={"providers": [provider]}, headers=headers)

    # B's transaction sees the profile row...
    other = accounts._connect()
    try:
        other.execute("SELECT provider FROM profiles WHERE user_id = 1").fetchone()
        # ...another request lands...
        await client.put(
            "/auth/profile",
            json={"providers": [{**provider, "name": "p2"}]},
            headers=headers,
        )
        # ...and B's write still lands instead of a 1020.
        other.execute(
            "INSERT INTO profiles (user_id, agents, updated_at) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE agents = VALUES(agents), updated_at = VALUES(updated_at)",
            (1, "SEALED", "2024-01-01T00:00:00+00:00"),
        )
        other.commit()
    finally:
        other.close()

    body = (await client.get("/auth/me", headers=headers)).json()
    assert body["agents"] == "SEALED"
    assert [p["name"] for p in body["providers"]] == ["p2"]


def test_migration_reads_a_read_only_file(db, tmp_path):
    # A docker volume mounted :ro hands the import a file SQLite cannot open
    # for writing; the import copies it aside and reads the copy. (The source
    # itself must stay untouched.)
    key = new_key()
    path = tmp_path / "accounts.db"
    conn = _legacy_db(path)
    conn.execute(
        "INSERT INTO users (credential_hash, name, created_at) VALUES (?, ?, ?)",
        (
            accounts._hash_credential(accounts.credential_from_key(key)),
            "ro",
            "2024-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()
    path.chmod(0o444)
    try:
        counts = migrate_sqlite.migrate(str(path))
        assert counts["users"] == 1
        (stored,) = db.execute("SELECT credential_hash FROM users").fetchone()
        assert stored == accounts._hash_credential(accounts.credential_from_key(key))
    finally:
        path.chmod(0o644)



async def test_last_session_record_is_its_own_collection(client):
    # The conversation to reopen next visit: one small record (`current`),
    # kept apart from the conversations like the reading spots are.
    token = (await client.post("/auth/signup", json=signup_body(new_key()))).json()["token"]
    headers = auth_headers(token)
    r = await client.post(
        "/auth/store/sync",
        json={
            "conversations": {"upserts": [{"id": "1", "blob": "CONVERSATION-1"}]},
            "last": {"upserts": [{"id": "current", "blob": "LAST-1"}]},
        },
        headers=headers,
    )
    assert r.status_code == 200
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [v["blob"] for v in store["last"]] == ["LAST-1"]
    assert [c["blob"] for c in store["conversations"]] == ["CONVERSATION-1"]
    # Moving on replaces the one record in place.
    await client.post(
        "/auth/store/sync",
        json={"last": {"upserts": [{"id": "current", "blob": "LAST-2"}]}},
        headers=headers,
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert [v["blob"] for v in store["last"]] == ["LAST-2"]
    # Clearing it leaves the conversations alone.
    await client.post(
        "/auth/store/sync", json={"last": {"deletes": ["current"]}}, headers=headers
    )
    store = (await client.get("/auth/store", headers=headers)).json()
    assert store["last"] == []
    assert [c["blob"] for c in store["conversations"]] == ["CONVERSATION-1"]
