"""User accounts: signup, sign-in, sign-out, /auth/me, and the E2E store.

A second, app-facing API alongside the OpenAI-compatible pass-through. The
gateway has no other state, so accounts live in a small MariaDB database
(users, tokens, per-account provider profiles, and the encrypted record
store); docker compose ships the server, and the root .env carries its
credentials.

Accounts are key-based, and the key is generated IN THE BROWSER: the client
mints an access key (``ast_`` + 256 random bits), keeps it, and sends only a
one-way credential — ``SHA-256(key)`` — to the server. The server stores
``SHA-256(credential)`` and never sees the key itself, so it cannot derive
the key the client uses to encrypt its data. Sign-in accepts the credential
and issues an opaque session token that /auth/me and /auth/signout accept.
There is no password and no recovery: lose the key, lose the account.

Each account may also store its BYOK provider profiles (name, base URL,
API key, model, and the model ids it keeps) via PUT /auth/profile. Unlike
the access key, the provider configs are returned to the owner by
/auth/me and /auth/signin: they are the account's own settings, and
remembering them is what lets a returning user land straight in a
conversation on any device. (This is the one place the server holds
provider keys — a deliberate, opt-in exception to the "the server never
sees API keys" rule that otherwise governs guests.)

The E2E store (/auth/store, /auth/store/sync) keeps conversations and
studios as opaque ciphertext blobs, encrypted client-side with a key
derived from the access key. The server stores only the blobs (plus the
account they belong to and a timestamp) and can never read them: the
decryption key exists only in the browser that holds the access key.
Metadata (record ids, counts, sizes, timestamps) is still visible to the
server, and the LLM provider still sees plaintext prompts — E2E protects
storage at rest, not inference.

When the robot check is on (TURNSTILE_MODE != "off"), only sign-up is
gated: it requires a Turnstile token in the "captcha" field and exchanges
it with Cloudflare's siteverify endpoint before creating the account.
Turnstile test-mode tokens (``XXXX.DUMMY.TOKEN.XXXX`` and the like) are
accepted without the round-trip. Sign-in/out/me are never gated — a
session token is proof of having passed the check at sign-up, and the
credential is derived from 256 random bits, which no bot is going to guess.

TURNSTILE_SECRET_KEY is a server secret and is never exposed to the
frontend: clients receive only the site key (via VITE_*) and send back the
token the widget produced.

Deployments that predate MariaDB kept their accounts in a SQLite file
(DATA_DIR/accounts.db). `python -m app.migrate_sqlite <path>` imports one:
the server-issued-key era is converted on the way (the plaintext key is
rewritten as a credential hash and never reaches MariaDB), and the pre-key
schema (email + PBKDF2 passwords) is refused — those accounts were never
issued an access key, so there is nothing a user could sign in with.
"""

import hashlib
import hmac
import json
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pymysql
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pymysql.connections import Connection
from pymysql.cursors import Cursor
from pymysql.err import IntegrityError

from . import config


def db_dsn() -> str:
    """Where the accounts database lives (logged at startup; no password)."""
    s = config.settings
    return f"{s.db_user}@{s.db_host}:{s.db_port}/{s.db_name}"


def init(attempts: int = 10, delay_seconds: float = 0.5) -> bool:
    """Open the database once at startup, creating the schema.

    False when the server is unreachable: the caller logs it, and the next
    request retries lazily — so a stack whose database is still coming up is
    not a crash loop.
    """
    for attempt in range(attempts):
        try:
            _connect().close()
            return True
        except pymysql.MySQLError:
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    return False


def auth_router() -> APIRouter:
    """The /auth routes (a factory, as main.py's create_app() expects)."""
    return _router


_router = APIRouter(prefix="/auth")

# A failed robot check is a 400 to the client (it is their input that failed);
# a Cloudflare outage is a 503 (their input may be fine — the check itself is
# down). Two distinct details, so the frontend can tell "try again" from
# "we're down on our end".
ROBOT_CHECK_FAILED = "Robot check failed — please try again."
ROBOT_CHECK_UNAVAILABLE = (
    "Robot check is unavailable right now — please try again in a moment."
)

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
# The token pattern Cloudflare documents for Turnstile's test mode.
_TEST_TOKEN = "XXXX.DUMMY.TOKEN.XXXX"

# How long an admin challenge stays answerable. Single-use either way: the
# first answer (right or wrong) consumes it.
ADMIN_CHALLENGE_TTL_SECONDS = 120

# The prefix of a minted admin key, so it is recognizable at a glance.
ADMIN_KEY_PREFIX = "adm_"

# Access-key format: a self-identifying prefix + 256 random bits, URL-safe.
# The key itself never reaches the server; the client sends only its SHA-256.
KEY_PREFIX = "ast_"

# The client sends SHA-256(key) as hex; the server stores SHA-256(that).
_CREDENTIAL_RE = re.compile(r"[0-9a-f]{64}")
# The display handle is the key's own opening — non-secret, so the client picks it.
_HANDLE_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")

# A single encrypted record's ceiling (base64 chars), a sanity bound on disk use.
_MAX_BLOB_CHARS = 32 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _random_verifier() -> str:
    """A verifier over a secret nobody holds: the decoy every account gets,
    so the table cannot tell admins apart from everyone else."""
    return secrets.token_hex(32)


def _admin_verifier(key: str) -> str:
    """The one-way verifier stored for an admin key (the key itself never
    reaches the server)."""
    return _sha256_hex(key)


def credential_from_key(key: str) -> str:
    """The credential a client sends for an access key (never the key itself)."""
    return _sha256_hex(key)


def _hash_credential(credential: str) -> str:
    """What the server stores for a credential: a one-way hash of it.

    The credential is derived from 256 random bits, so a plain SHA-256 is
    enough (there is nothing to brute-force); hashing it again keeps even the
    credential itself out of the database.
    """
    return _sha256_hex(credential)


TOKEN_COOKIE = "aistudio.account"


def _session_token(request: Request) -> str | None:
    """The session token for this request.

    The client keeps it in the aistudio.account cookie (which the browser
    sends on every same-origin request) and also reads it into an
    Authorization: Bearer header — the header wins when both are present.
    """
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip() or None
    return request.cookies.get(TOKEN_COOKIE)


class _Db:
    """A small handle mirroring the sqlite3 ergonomics this module grew on:
    one execute() per statement, no cursor juggling. PyMySQL is the driver
    and the schema is MariaDB's, so the queries speak its dialect (%s
    placeholders, ON DUPLICATE KEY updates)."""

    def __init__(self, conn: Connection):
        self._conn = conn

    def execute(self, sql: str, args: tuple = ()) -> Cursor:
        cursor = self._conn.cursor()
        cursor.execute(sql, args)
        return cursor

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(db: _Db) -> None:
    """Create the tables (idempotent; run once per process).

    There are no in-place upgrades to walk: this is what every MariaDB
    deployment runs, and a legacy SQLite database is carried over with
    `python -m app.migrate_sqlite` instead. utf8mb4_bin collation makes every
    comparison byte-exact, as the sqlite3 store was — session tokens and
    challenge ids are mixed-case and must not collide case-insensitively.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT NOT NULL AUTO_INCREMENT,
            credential_hash CHAR(64) NOT NULL,
            name VARCHAR(32) NOT NULL,
            created_at VARCHAR(40) NOT NULL,
            admin_verifier VARCHAR(64) NULL,
            admin_key MEDIUMTEXT NULL,
            PRIMARY KEY (id),
            UNIQUE KEY users_credential_hash (credential_hash)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            value VARCHAR(64) NOT NULL,
            user_id INT NOT NULL,
            created_at VARCHAR(40) NOT NULL,
            PRIMARY KEY (value),
            KEY tokens_user (user_id),
            CONSTRAINT tokens_user_fk FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INT NOT NULL,
            provider MEDIUMTEXT NULL,
            agents MEDIUMTEXT NULL,
            updated_at VARCHAR(40) NULL,
            PRIMARY KEY (user_id),
            CONSTRAINT profiles_user_fk FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_challenges (
            challenge VARCHAR(64) NOT NULL,
            user_id INT NOT NULL,
            expires_at VARCHAR(40) NOT NULL,
            PRIMARY KEY (challenge),
            KEY admin_challenges_expiry (expires_at),
            CONSTRAINT admin_challenges_user_fk FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
        """
    )
    # The end-to-end encrypted record store: opaque ciphertext blobs, one row
    # per conversation / studio / reading spot, scoped to their owner.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS e2e_records (
            user_id INT NOT NULL,
            kind VARCHAR(16) NOT NULL,
            rec_id VARCHAR(191) NOT NULL,
            payload LONGTEXT NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            PRIMARY KEY (user_id, kind, rec_id),
            CONSTRAINT e2e_records_user_fk FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
        """
    )
    db.commit()


def _connect() -> _Db:
    """One connection per call, schema ensured once per process."""
    global _schema_ready
    s = config.settings
    conn = pymysql.connect(
        host=s.db_host,
        port=s.db_port,
        user=s.db_user,
        password=s.db_password,
        database=s.db_name,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=60,
        write_timeout=60,
        # MariaDB's default REPEATABLE READ raises "record has changed since
        # last read" (1020) when a short transaction writes a row another one
        # changed after its snapshot — routine here (two devices touching a
        # profile; a sync racing one). These transactions are single-request
        # units with no long reads, so READ COMMITTED is what they want.
        init_command="SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",
        # Records can carry images; the bundled server is told to match
        # (docker-compose.yml passes --max-allowed-packet=64M).
        max_allowed_packet=64 * 1024 * 1024,
    )
    db = _Db(conn)
    try:
        if not _schema_ready:
            with _schema_lock:
                if not _schema_ready:
                    _ensure_schema(db)
                    _schema_ready = True
    except Exception:
        # Never leak a half-initialized handle: the next call retries cleanly.
        conn.close()
        raise
    return db


def _issue_token(conn: _Db, user_id: int) -> str:
    """Mint a session token bound to an account (caller commits)."""
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO tokens (value, user_id, created_at) VALUES (%s, %s, %s)",
        (token, user_id, _now()),
    )
    return token


def _user_for_token(conn: _Db, request: Request) -> int:
    """The account a request's session token belongs to (401 when unknown)."""
    token = _session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    row = conn.execute(
        "SELECT user_id FROM tokens WHERE value = %s", (token,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=401, detail="Session expired. Please sign in again."
        )
    # A session token is bound to exactly one account, so the only records a
    # client can ever reach are its own.
    return row[0]


#: The account's admin verifier. Unlike the E2E-sealed content, the server
#: must hold *something* to verify an admin — but it holds only a one-way hash
#: of the admin key, and every account carries one (a decoy for non-admins),
#: so the database reveals nothing about who administers the deployment. The
#: server learns "admin" only when a challenge is answered (see /admin/prove).
#: Everything private (conversations, studios, provider keys) stays E2E.


def _admin_key_for(conn: _Db, user_id: int) -> str | None:
    """The account's sealed (client-encrypted) admin key copy, if uploaded."""
    row = conn.execute(
        "SELECT admin_key FROM users WHERE id = %s", (user_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


def _get_providers(conn: _Db, user_id: int) -> list[dict]:
    """The account's saved BYOK providers (empty when it has none yet).

    Rows written before this held one provider object rather than a list;
    that shape folds in as a one-element list so old profiles keep working.
    """
    row = conn.execute(
        "SELECT provider FROM profiles WHERE user_id = %s", (user_id,)
    ).fetchone()
    if row is None or not row[0]:
        return []
    try:
        data = json.loads(row[0])
    except ValueError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)]
    return []


def _save_providers(conn: _Db, user_id: int, providers: list[dict]) -> None:
    """Store (or, when empty, clear) the account's provider configs. Caller commits.

    The row stays when cleared: the account's sealed agents live on it too.
    """
    conn.execute(
        "INSERT INTO profiles (user_id, provider, updated_at) VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE provider = VALUES(provider), updated_at = VALUES(updated_at)",
        (user_id, json.dumps(providers) if providers else None, _now()),
    )


def _get_agents(conn: _Db, user_id: int) -> str | None:
    """The account's sealed agents (opaque ciphertext), or None when it has none."""
    row = conn.execute(
        "SELECT agents FROM profiles WHERE user_id = %s", (user_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


def _save_agents(conn: _Db, user_id: int, sealed: str | None) -> None:
    """Store (or, when None, clear) the account's sealed agents. Caller commits."""
    conn.execute(
        "INSERT INTO profiles (user_id, agents, updated_at) VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE agents = VALUES(agents), updated_at = VALUES(updated_at)",
        (user_id, sealed, _now()),
    )


class SignupBody(BaseModel):
    # Turnstile token from the browser widget (required when the robot check is on).
    captcha: str = ""
    # SHA-256 of the client-generated access key, hex. The key itself never ships.
    credential: str
    # The display handle (the key's own opening); non-secret, client-chosen.
    handle: str


def _create_account(credential: str, handle: str) -> tuple[str, int]:
    """Insert the account and mint its first session (blocking; signup threads it)."""
    conn = _connect()
    try:
        try:
            cursor = conn.execute(
                "INSERT INTO users (credential_hash, name, created_at, admin_verifier) "
                "VALUES (%s, %s, %s, %s)",
                (_hash_credential(credential), handle, _now(), _random_verifier()),
            )
        except IntegrityError:
            # The same credential can never own two accounts.
            raise HTTPException(
                status_code=409, detail="This key is already registered."
            )
        user_id = cursor.lastrowid
        token = _issue_token(conn, user_id)
        conn.commit()
        return token, user_id
    finally:
        conn.close()


class SigninBody(BaseModel):
    # SHA-256 of the access key the user saved, hex.
    credential: str


class ProviderConfig(BaseModel):
    # One of the account's BYOK providers, exactly as chosen in the settings UI.
    # Fields at their default are dropped on store, so a legacy provider (no id,
    # no model list) stays exactly as it was written.
    id: str = ""
    name: str
    baseUrl: str
    apiKey: str
    model: str
    models: list[str] = Field(default_factory=list)
    # This provider does not require an API key (local proxies, gateways).
    noKey: bool = False


class ProfileBody(BaseModel):
    # The providers to store for the signed-in account; an empty list (or null)
    # clears them. `provider` is the legacy single-provider shape, still accepted.
    provider: ProviderConfig | None = None
    providers: list[ProviderConfig] | None = None
    # The account's agents, sealed under its key: opaque to the server. An empty
    # string (or null) clears them.
    agents: str | None = Field(default=None, max_length=262_144)


class AdminProveBody(BaseModel):
    # The single-use challenge the server issued, and the admin key answering
    # it (the server compares it against the stored one-way verifier).
    challenge: str
    key: str


class AdminKeyBody(BaseModel):
    # The account's admin key, sealed client-side — the server stores opaque
    # ciphertext it cannot read (null clears the copy).
    key: str | None = None


class StoreRecord(BaseModel):
    # One encrypted record: the client-side id and its opaque ciphertext blob.
    id: str
    blob: str


class StoreDelta(BaseModel):
    # Records to write and ids to drop, as the client's debounced diff reports them.
    upserts: list[StoreRecord] = Field(default_factory=list)
    deletes: list[str] = Field(default_factory=list)


class StoreSyncBody(BaseModel):
    # The two encrypted collections the client keeps (studios ride with their conversations).
    conversations: StoreDelta = Field(default_factory=StoreDelta)
    studios: StoreDelta = Field(default_factory=StoreDelta)
    # Where the reader left off: one small record per conversation.
    views: StoreDelta = Field(default_factory=StoreDelta)
    # The conversation to reopen on the next visit: one small record (`current`).
    last: StoreDelta = Field(default_factory=StoreDelta)


@_router.post("/signup", status_code=201)
async def signup(body: SignupBody, request: Request):
    # The robot check gates sign-up only, and needs the request for the host
    # (live mode) and a network round-trip (hence the async handler). It runs
    # before the database is opened, so a bot never creates a row.
    if config.settings.captcha_enabled:
        if not body.captcha.strip():
            raise HTTPException(
                status_code=400, detail="Please complete the robot check."
            )
        await _verify_captcha(body.captcha.strip(), request)
    credential = body.credential.strip().lower()
    if not _CREDENTIAL_RE.fullmatch(credential):
        raise HTTPException(status_code=400, detail="Malformed credential.")
    handle = body.handle.strip()
    if not _HANDLE_RE.fullmatch(handle):
        raise HTTPException(status_code=400, detail="Malformed handle.")
    token, user_id = await run_in_threadpool(_create_account, credential, handle)
    # No key in the response: the client generated it and already has it, so
    # the one-time reveal shows what the browser minted.
    return {"token": token, "user": {"id": user_id, "name": handle}}


@_router.post("/signin")
def signin(body: SigninBody):
    # Sign-in is never robot-gated: the credential is derived from the access
    # key itself, and a session token is already proof of a check passed at
    # sign-up.
    credential = body.credential.strip().lower()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, name FROM users WHERE credential_hash = %s",
            (_hash_credential(credential),),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Incorrect access key.")
        token = _issue_token(conn, row[0])
        providers = _get_providers(conn, row[0])
        agents = _get_agents(conn, row[0])
        admin_key = _admin_key_for(conn, row[0])
        conn.commit()
    finally:
        conn.close()
    # The providers (with their keys) and the sealed agents ride along so the
    # client can resume straight into a conversation without a second /auth/me.
    return {
        "token": token,
        "user": {"id": row[0], "name": row[1]},
        "providers": providers,
        "provider": providers[0] if providers else None,
        "agents": agents,
        "adminKey": admin_key,
    }


@_router.post("/signout")
def signout(request: Request):
    # The client clears its own cookie; the server just drops the session.
    token = _session_token(request)
    if token:
        conn = _connect()
        try:
            conn.execute("DELETE FROM tokens WHERE value = %s", (token,))
            conn.commit()
        finally:
            conn.close()
    return {"ok": True}


@_router.get("/me")
def me(request: Request):
    token = _session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT u.id, u.name FROM users u JOIN tokens t ON t.user_id = u.id WHERE t.value = %s",
            (token,),
        ).fetchone()
        providers = _get_providers(conn, row[0]) if row else []
        agents = _get_agents(conn, row[0]) if row else None
        admin_key = _admin_key_for(conn, row[0]) if row else None
    finally:
        conn.close()
    if row is None:
        raise HTTPException(
            status_code=401, detail="Session expired. Please sign in again."
        )
    # No credential material in here: it is a secret, not profile data. The
    # providers (including their API keys) are the account's own config,
    # returned to the owner so the client can resume straight into a
    # conversation anywhere; the sealed admin-key copy is opaque ciphertext
    # only its owner can open. No role is returned — the server does not hold
    # one (see /admin/challenge + /admin/prove).
    return {
        "user": {"id": row[0], "name": row[1]},
        "providers": providers,
        "provider": providers[0] if providers else None,
        "agents": agents,
        "adminKey": admin_key,
    }


@_router.post("/admin/challenge")
def admin_challenge(request: Request):
    """Issue a single-use "are you admin?" challenge for the signed-in account.

    Nothing about the account's user type was readable before this; answering
    the challenge with the right admin key (see /admin/prove) is the only way
    the server ever learns it.
    """
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        conn.execute("DELETE FROM admin_challenges WHERE expires_at < %s", (_now(),))
        challenge = secrets.token_urlsafe(32)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ADMIN_CHALLENGE_TTL_SECONDS)
        ).isoformat()
        conn.execute(
            "INSERT INTO admin_challenges (challenge, user_id, expires_at) VALUES (%s, %s, %s)",
            (challenge, user_id, expires_at),
        )
        conn.commit()
    finally:
        conn.close()
    return {"challenge": challenge}


@_router.post("/admin/prove")
def admin_prove(body: AdminProveBody, request: Request):
    """Answer a challenge with the admin key.

    A 400 means the challenge is unknown or expired (ask for a new one); a 401
    means the key does not match the account's verifier. The challenge is
    consumed by the first answer either way, and the key itself never reaches
    the database — only its one-way hash does.
    """
    key = body.key.strip()
    if not key:
        raise HTTPException(status_code=401, detail="Not an admin.")
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        challenge = body.challenge.strip()
        row = conn.execute(
            "SELECT user_id, expires_at FROM admin_challenges WHERE challenge = %s",
            (challenge,),
        ).fetchone()
        if row is None or row[0] != user_id:
            raise HTTPException(
                status_code=400, detail="Unknown challenge — ask for a new one."
            )
        # One answer per challenge, whether or not it is right.
        conn.execute("DELETE FROM admin_challenges WHERE challenge = %s", (challenge,))
        conn.commit()
        if row[1] < _now():
            raise HTTPException(
                status_code=400, detail="The challenge expired — ask for a new one."
            )
        verifier = conn.execute(
            "SELECT admin_verifier FROM users WHERE id = %s", (user_id,)
        ).fetchone()
        stored = verifier[0] if verifier else None
        if not stored or not hmac.compare_digest(_admin_verifier(key), stored):
            raise HTTPException(status_code=401, detail="Not an admin.")
    finally:
        conn.close()
    # Only here does the server know: this session answered with the admin key.
    return {"admin": True}


@_router.put("/admin/key")
def save_admin_key(body: AdminKeyBody, request: Request):
    """Store (or clear) the account's sealed admin-key copy.

    The value arrives sealed under the account key, so the server stores
    opaque ciphertext; it is returned to the owner (via /auth/me), never to
    anyone else, and is what lets admin be proven from another device.
    """
    key = (body.key or "").strip()
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        conn.execute(
            "UPDATE users SET admin_key = %s WHERE id = %s", (key or None, user_id)
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@_router.put("/profile")
def save_profile(body: ProfileBody, request: Request):
    """Store the signed-in account's providers and agents.

    Each field the request sends replaces its stored value (an empty list or
    string clears it); a field it leaves out is kept, so saving one never
    clears the other.
    """
    sent = body.model_fields_set
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        if "provider" in sent or "providers" in sent:
            items = (
                body.providers
                if body.providers is not None
                else ([body.provider] if body.provider else [])
            )
            _save_providers(
                conn, user_id, [p.model_dump(exclude_defaults=True) for p in items]
            )
        if "agents" in sent:
            _save_agents(conn, user_id, body.agents or None)
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@_router.get("/store")
def get_store(request: Request):
    """Every encrypted record the signed-in account owns.

    The blobs are opaque: the server returns exactly what the client stored
    and never attempts to read them.
    """
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        rows = conn.execute(
            "SELECT kind, rec_id, payload, updated_at FROM e2e_records WHERE user_id = %s ORDER BY updated_at",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    conversations: list[dict] = []
    studios: list[dict] = []
    views: list[dict] = []
    last: list[dict] = []
    for kind, rec_id, blob, updated_at in rows:
        item = {"id": rec_id, "blob": blob, "updatedAt": updated_at}
        if kind == "conversation":
            conversations.append(item)
        elif kind == "studio":
            studios.append(item)
        elif kind == "view":
            views.append(item)
        elif kind == "last":
            last.append(item)
    return {"conversations": conversations, "studios": studios, "views": views, "last": last}


@_router.post("/store/sync")
def sync_store(body: StoreSyncBody, request: Request):
    """Apply the client's diff: upsert changed records, drop deleted ones.

    The client sends only what changed (its debounced diff), so unchanged
    ciphertext — which can be large when a message carries an image — is not
    re-uploaded.
    """
    conn = _connect()
    try:
        user_id = _user_for_token(conn, request)
        now = _now()
        for kind, delta in (
            ("conversation", body.conversations),
            ("studio", body.studios),
            ("view", body.views),
            ("last", body.last),
        ):
            for rec in delta.upserts:
                if not rec.id or not rec.blob:
                    raise HTTPException(status_code=400, detail="Malformed record.")
                if len(rec.blob) > _MAX_BLOB_CHARS:
                    raise HTTPException(status_code=413, detail="Record too large.")
                conn.execute(
                    "INSERT INTO e2e_records (user_id, kind, rec_id, payload, updated_at) VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE payload = VALUES(payload), updated_at = VALUES(updated_at)",
                    (user_id, kind, rec.id, rec.blob, now),
                )
            for rec_id in delta.deletes:
                conn.execute(
                    "DELETE FROM e2e_records WHERE user_id = %s AND kind = %s AND rec_id = %s",
                    (user_id, kind, rec_id),
                )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


async def _verify_captcha(token: str, request: Request) -> None:
    """Exchange a Turnstile token with Cloudflare and 400 if it isn't a human.

    Test-mode tokens are accepted locally (they are issued to the
    documented test site key and are useless against the real endpoint, so
    the round-trip is meaningless). Anything else is POSTed to siteverify
    with the secret key and, in live mode, the request's public IP.
    """
    if config.settings.turnstile_mode == "test":
        return
    payload = {
        "secret": config.settings.turnstile_secret_key,
        "response": token,
        # siteverify can't distinguish a bot from a misconfigured reverse
        # proxy without the IP; in live mode that is the whole point.
        "remoteip": _client_ip(request),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(SITEVERIFY_URL, data=payload)
    except httpx.HTTPError:
        # Cloudflare unreachable: fail closed, but with a distinct message
        # (503) than a token that was actually rejected (400).
        raise HTTPException(status_code=503, detail=ROBOT_CHECK_UNAVAILABLE)
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail=ROBOT_CHECK_UNAVAILABLE)
    try:
        result = response.json()
    except ValueError:
        raise HTTPException(status_code=503, detail=ROBOT_CHECK_UNAVAILABLE)
    if not result.get("success"):
        # Cloudflare's documented error codes (invalid-input-response,
        # time-synchronization-required, ...) ride along for the logs.
        errors = result.get("error-codes")
        raise HTTPException(
            status_code=400,
            detail=f"{ROBOT_CHECK_FAILED} {' '.join(errors) if errors else ''}".strip(),
        )


def _client_ip(request: Request) -> str:
    """Best-effort client address, honouring one X-Forwarded-For hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


# --- Admin key maintenance (the operator's CLI) ---------------------------
# Admin-ness is never a readable column: the database stores only a one-way
# verifier, and the key exists only in the hands of whoever promotes (the app
# keeps a sealed copy for its owner). Promotion is out of band — never through
# an API the account itself can call:
#
#   python -m app.accounts list
#   python -m app.accounts promote <id|handle>   # mints a key, shown once
#   python -m app.accounts demote <id|handle>    # revokes it
#
# (Docker: `docker exec aistudio-backend python -m app.accounts promote <id>`.)


def _resolve_user(conn: _Db, target: str) -> tuple[int, str]:
    """The (id, handle) a CLI target names: a numeric id or a handle."""
    if target.isdigit():
        row = conn.execute(
            "SELECT id, name FROM users WHERE id = %s", (int(target),)
        ).fetchone()
    else:
        rows = conn.execute(
            "SELECT id, name FROM users WHERE name = %s ORDER BY id", (target,)
        ).fetchall()
        if len(rows) > 1:
            raise SystemExit(
                f"handle {target!r} matches {len(rows)} accounts — use the numeric id"
            )
        row = rows[0] if rows else None
    if row is None:
        raise SystemExit(f"no account matches {target!r}")
    return row[0], row[1]


def _run_cli(argv: list[str]) -> int:
    usage = "usage: python -m app.accounts [list | promote <id|handle> | demote <id|handle>]"
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage)
        return 0
    command = argv[0]
    if command == "list":
        conn = _connect()
        try:
            rows = conn.execute("SELECT id, name FROM users ORDER BY id").fetchall()
        finally:
            conn.close()
        for user_id, name in rows:
            print(f"{user_id:>4}  {name}")
        return 0
    if command == "promote" and len(argv) == 2:
        key = ADMIN_KEY_PREFIX + secrets.token_urlsafe(32)
        conn = _connect()
        try:
            user_id, name = _resolve_user(conn, argv[1])
            conn.execute(
                "UPDATE users SET admin_verifier = %s, admin_key = NULL WHERE id = %s",
                (_admin_verifier(key), user_id),
            )
            conn.commit()
        finally:
            conn.close()
        print(
            f"{name} (id {user_id}) can prove admin with this key — it is shown once:"
        )
        print(key)
        print("Paste it into Settings → Admin in the app to use it there.")
        return 0
    if command == "demote" and len(argv) == 2:
        conn = _connect()
        try:
            user_id, name = _resolve_user(conn, argv[1])
            conn.execute(
                "UPDATE users SET admin_verifier = %s, admin_key = NULL WHERE id = %s",
                (_random_verifier(), user_id),
            )
            conn.commit()
        finally:
            conn.close()
        print(f"{name} (id {user_id}) can no longer prove admin.")
        return 0
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_run_cli(sys.argv[1:]))
