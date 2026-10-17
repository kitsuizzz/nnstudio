# Database — table of contents

The gateway's entire state lives in **one MariaDB server**: the `mariadb`
service in `docker-compose.yml` (image `mariadb:11`, volume `mariadb-data`).
The schema is created (idempotently) by `_ensure_schema()` in
`backend/app/accounts.py` — that function is the single source of truth; the
queries in the same file are the only readers and writers.

Every table is `InnoDB` with `utf8mb4_bin` collation: comparisons are
byte-exact, which matters because session tokens and challenge ids are
mixed-case. A connection runs `READ COMMITTED` (short request transactions;
two devices can touch the same profile row concurrently).

| Database       | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `aistudio`     | The deployment's data (what this document describes).                  |
| `aistudio_test`| The backend test suite only; wiped before each test. Created by `mariadb/init/01-test-database.sh`. |

Access: user `aistudio`, password `MARIADB_PASSWORD` (= `DB_PASSWORD` in the
backend env, see `.env`); root password `MARIADB_ROOT_PASSWORD`. Published on
`127.0.0.1:3306`, or `mariadb:3306` inside the compose network.

**Contents**

1. [users](#users) — the accounts
2. [tokens](#tokens) — sessions
3. [profiles](#profiles) — per-account BYOK providers and sealed agents
4. [admin_challenges](#admin_challenges) — single-use admin proofs
5. [e2e_records](#e2e_records) — the end-to-end encrypted store
6. [Record kinds](#record-kinds) — what the rows of `e2e_records` mean
7. [Maintenance](#maintenance) — imports, admin CLI, limits

---

## users

One row per account. The access key is minted in the browser and never
reaches the server: only a one-way hash of the client's credential is stored.

| Column            | Type           | Meaning                                                                                     |
| ----------------- | -------------- | ------------------------------------------------------------------------------------------- |
| `id`              | `INT` PK, auto | Internal account id; the id the account sees in API responses.                              |
| `credential_hash` | `CHAR(64)`     | `SHA-256(SHA-256(access key))`, hex. Used for sign-up uniqueness and sign-in lookup.        |
| `name`            | `VARCHAR(32)`  | Display handle (the key's own first six characters).                                        |
| `created_at`      | `VARCHAR(40)`  | ISO-8601 UTC timestamp.                                                                     |
| `admin_verifier`  | `VARCHAR(64)`  | One-way hash of the account's admin key. *Every* account has one (noise for non-admins), so the table reveals nothing about who administers the deployment. Changed by `promote` / `demote`. |
| `admin_key`       | `MEDIUMTEXT`   | Sealed (client-encrypted) copy of the admin key, kept so admin can be proven from another device. Opaque ciphertext. |

Keys: `PRIMARY KEY (id)`, `UNIQUE (credential_hash)`.

## tokens

One row per active session. The client keeps `value` in the `aistudio.account`
cookie (and an `Authorization: Bearer` header); every `/auth/*` call resolves
the session here.

| Column       | Type          | Meaning                                                        |
| ------------ | ------------- | -------------------------------------------------------------- |
| `value`      | `VARCHAR(64)` | Opaque session token (`secrets.token_urlsafe(32)`). Primary key. |
| `user_id`    | `INT`         | The account the session belongs to (`ON DELETE CASCADE`).       |
| `created_at` | `VARCHAR(40)` | ISO-8601 UTC timestamp.                                         |

Keys: `PRIMARY KEY (value)`, `KEY (user_id)`, FK → `users.id` cascade.

## profiles

One row per account that has something to remember. A row present = the
account is BYOK; absent = it rides the platform gateway. The server returns
`provider` / `agents` to their owner but never parses `agents` (opaque to it).

| Column       | Type          | Meaning                                                                                   |
| ------------ | ------------- | ----------------------------------------------------------------------------------------- |
| `user_id`    | `INT`         | The account. Primary key, `ON DELETE CASCADE`.                                             |
| `provider`   | `MEDIUMTEXT`  | JSON array of the account's BYOK providers, exactly as the settings UI sends them (keys sealed under the account key). A pre-list single-object shape folds in as a one-element list. |
| `agents`     | `MEDIUMTEXT`  | The account's agents (and default), sealed client-side: opaque ciphertext.                 |
| `updated_at` | `VARCHAR(40)` | ISO-8601 UTC timestamp.                                                                    |

Keys: `PRIMARY KEY (user_id)`, FK → `users.id` cascade.

## admin_challenges

Transient rows behind the admin proof flow (`/auth/admin/challenge` →
`/auth/admin/prove`). A challenge is single-use (consumed by its first answer,
right or wrong) and expires after 120 seconds; expired rows are swept when the
next challenge is minted. Normally empty.

| Column       | Type          | Meaning                                          |
| ------------ | ------------- | ------------------------------------------------ |
| `challenge`  | `VARCHAR(64)` | Random single-use challenge id. Primary key.      |
| `user_id`    | `INT`         | The account answering it (`ON DELETE CASCADE`).   |
| `expires_at` | `VARCHAR(40)` | ISO-8601 UTC deadline.                            |

Keys: `PRIMARY KEY (challenge)`, `KEY (expires_at)`, `KEY (user_id)`, FK →
`users.id` cascade.

## e2e_records

The end-to-end encrypted store: the browser encrypts each record with an
AES-GCM-256 key derived (HKDF-SHA-256) from the access key; the server stores
only `base64(iv ‖ ciphertext)` and can never read it. One row per record, scoped
to its owner.

| Column       | Type            | Meaning                                                                                     |
| ------------ | --------------- | ------------------------------------------------------------------------------------------- |
| `user_id`    | `INT`           | The owner. Part of the primary key, `ON DELETE CASCADE`.                                     |
| `kind`       | `VARCHAR(16)`   | The record kind: `conversation`, `studio` or `view` (see below). Part of the primary key.    |
| `rec_id`     | `VARCHAR(191)`  | The client-side record id. For `view` rows it is the conversation id. Part of the primary key. |
| `payload`    | `LONGTEXT`      | The opaque envelope: `base64(iv ‖ ciphertext)`. Up to 32 MB per record (64 MB packet limit). |
| `updated_at` | `VARCHAR(40)`   | ISO-8601 UTC timestamp; `GET /auth/store` orders by it.                                      |

Keys: `PRIMARY KEY (user_id, kind, rec_id)`, FK → `users.id` cascade.
Writes come in as diffs via `POST /auth/store/sync` (upsert / delete).

## Record kinds

| Kind           | One row per                          | Contains                                                                 |
| -------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `conversation` | Conversation                         | The chat record: messages, title, agent, token counts, timestamps.        |
| `studio`       | Workspace (generated app)            | The declarative spec, debug log, AI consent flag, capture state.          |
| `view`         | Conversation the reader has opened   | Where the reader left off: at the bottom, or an exchange + offset.        |
| `last`         | Account (one row, id `current`)      | The conversation to reopen next visit (its client-side id).               |

## Maintenance

```bash
# Import a legacy SQLite database (idempotent upserts; read-only mounts OK):
python -m app.migrate_sqlite /path/to/accounts.db
# In Docker, from the repo root (the volume keeps the old file):
docker compose run --rm -v aistudio_aistudio-data:/legacy:ro backend \
    python -m app.migrate_sqlite /legacy/accounts.db

# Admin management (the key is shown once; the DB keeps only the verifier):
python -m app.accounts list
python -m app.accounts promote <id|handle>
python -m app.accounts demote  <id|handle>
# Docker: docker exec aistudio-backend python -m app.accounts promote <id>
```

Notes:

- Deleting a `users` row cascades to its tokens, profile, challenges and
  records — no orphan cleanup needed.
- Backups: everything is in the `mariadb-data` volume; `docker compose exec
  mariadb mariadb-dump ...` or a volume snapshot both work.
- The payload is ciphertext: a database dump reveals account handles, record
  ids/counts/sizes/timestamps — never conversations, prompts or provider keys.
