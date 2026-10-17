"""One-shot import of a legacy SQLite accounts database into MariaDB.

The gateway's accounts used to live in DATA_DIR/accounts.db (SQLite). A
deployment moving to the MariaDB-backed accounts carries its file over once:

    python -m app.migrate_sqlite /path/to/accounts.db

    # in docker (the old volume still holds the SQLite file):
    docker compose run --rm -v aistudio_aistudio-data:/legacy:ro backend \
        python -m app.migrate_sqlite /legacy/accounts.db

Every row is an upsert, so a re-run replaces the same rows instead of
duplicating them. The server-issued-key era stored the plaintext access key in
``users.key``; it is converted on the way — only SHA-256(SHA-256(key)) lands in
MariaDB. Databases from the email+password era are refused: those accounts
were never issued an access key, so there is nothing to carry over.
"""

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from . import accounts

_TABLES = ("users", "tokens", "profiles", "admin_challenges", "e2e_records")


def migrate(path: str) -> dict[str, int]:
    """Copy every table of the SQLite file at `path` into the MariaDB store."""
    src = _open_source(path)
    src.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in src.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "users" not in tables:
            raise SystemExit(f"{path}: no users table — is this an accounts database?")
        user_columns = {row[1] for row in src.execute("PRAGMA table_info(users)")}
        if "credential_hash" not in user_columns and "key" not in user_columns:
            raise SystemExit(
                f"{path}: pre-key schema (email + password accounts) — those "
                "accounts never got an access key, so there is nothing to migrate"
            )
        db = accounts._connect()
        counts: dict[str, int] = {}
        try:
            counts["users"] = _copy_users(src, db, user_columns)
            if "tokens" in tables:
                counts["tokens"] = _copy_tokens(src, db)
            if "profiles" in tables:
                counts["profiles"] = _copy_profiles(src, db)
            if "admin_challenges" in tables:
                counts["admin_challenges"] = _copy_challenges(src, db)
            if "e2e_records" in tables:
                counts["e2e_records"] = _copy_records(src, db)
            db.commit()
        finally:
            db.close()
        return counts
    finally:
        src.close()


def _open_source(path: str) -> sqlite3.Connection:
    """Open the SQLite source for reading.

    A Docker volume mounted read-only can't host SQLite's journal, and an
    engine that can't write refuses even to open it. Fall back to opening a
    copy (with any WAL sidecars) in a temp directory — the source volume stays
    untouched.
    """
    conn = None
    try:
        conn = sqlite3.connect(path)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn
    except sqlite3.OperationalError:
        if conn is not None:
            conn.close()
    temp = Path(tempfile.mkdtemp(prefix="aistudio-migrate-")) / "accounts.db"
    shutil.copyfile(path, temp)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            shutil.copyfile(sidecar, f"{temp}{suffix}")
    return sqlite3.connect(str(temp))


def _copy_users(src: sqlite3.Connection, db: accounts._Db, columns: set[str]) -> int:
    count = 0
    for row in src.execute("SELECT * FROM users ORDER BY id").fetchall():
        if "credential_hash" in columns:
            credential_hash = row["credential_hash"]
        else:
            # The server-issued-key era kept the plaintext key; only the
            # credential hash ever reaches the new database.
            credential_hash = accounts._hash_credential(
                accounts.credential_from_key(row["key"])
            )
        # Every account carries an admin verifier (noise for non-admins); a
        # row from before that column existed gets fresh noise.
        verifier = row["admin_verifier"] if "admin_verifier" in columns else None
        db.execute(
            "INSERT INTO users (id, credential_hash, name, created_at, admin_verifier, admin_key) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE credential_hash = VALUES(credential_hash), name = VALUES(name), "
            "created_at = VALUES(created_at), admin_verifier = VALUES(admin_verifier), "
            "admin_key = VALUES(admin_key)",
            (
                row["id"],
                credential_hash,
                row["name"],
                row["created_at"],
                verifier or accounts._random_verifier(),
                row["admin_key"] if "admin_key" in columns else None,
            ),
        )
        count += 1
    return count


def _copy_tokens(src: sqlite3.Connection, db: accounts._Db) -> int:
    count = 0
    for row in src.execute("SELECT value, user_id, created_at FROM tokens").fetchall():
        db.execute(
            "INSERT INTO tokens (value, user_id, created_at) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), created_at = VALUES(created_at)",
            (row["value"], row["user_id"], row["created_at"]),
        )
        count += 1
    return count


def _copy_profiles(src: sqlite3.Connection, db: accounts._Db) -> int:
    columns = {row[1] for row in src.execute("PRAGMA table_info(profiles)")}
    count = 0
    for row in src.execute("SELECT * FROM profiles").fetchall():
        db.execute(
            "INSERT INTO profiles (user_id, provider, agents, updated_at) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE provider = VALUES(provider), agents = VALUES(agents), "
            "updated_at = VALUES(updated_at)",
            (
                row["user_id"],
                row["provider"] if "provider" in columns else None,
                row["agents"] if "agents" in columns else None,
                row["updated_at"] if "updated_at" in columns else None,
            ),
        )
        count += 1
    return count


def _copy_challenges(src: sqlite3.Connection, db: accounts._Db) -> int:
    count = 0
    for row in src.execute(
        "SELECT challenge, user_id, expires_at FROM admin_challenges"
    ).fetchall():
        db.execute(
            "INSERT INTO admin_challenges (challenge, user_id, expires_at) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), expires_at = VALUES(expires_at)",
            (row["challenge"], row["user_id"], row["expires_at"]),
        )
        count += 1
    return count


def _copy_records(src: sqlite3.Connection, db: accounts._Db) -> int:
    count = 0
    for row in src.execute(
        "SELECT user_id, kind, rec_id, blob, updated_at FROM e2e_records"
    ).fetchall():
        db.execute(
            "INSERT INTO e2e_records (user_id, kind, rec_id, payload, updated_at) VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE payload = VALUES(payload), updated_at = VALUES(updated_at)",
            (
                row["user_id"],
                row["kind"],
                row["rec_id"],
                row["blob"],
                row["updated_at"],
            ),
        )
        count += 1
    return count


def main(argv: list[str]) -> int:
    usage = "usage: python -m app.migrate_sqlite <path/to/accounts.db>"
    if argv and argv[0] in ("-h", "--help"):
        print(usage)
        return 0
    if len(argv) != 1:
        print(usage, file=sys.stderr)
        return 2
    path = argv[0]
    if not Path(path).is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    counts = migrate(path)
    for table in _TABLES:
        if table in counts:
            print(f"{table}: {counts[table]}")
    print(f"migrated into {accounts.db_dsn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
