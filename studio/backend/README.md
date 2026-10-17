# chat.kitsui backend

An OpenAI-compatible gateway that sits between the frontend and any
OpenAI-compatible LLM provider. The frontend keeps its exact wire protocol
(`GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions` with
SSE or JSON) and just points its base URL at this gateway. The real
provider key lives here, server-side.

## Development

```bash
cp .env.example .env            # then edit (upstream URL/key, gateway token)
~/.local/bin/uv sync            # creates .venv with runtime + dev deps
docker compose up -d mariadb    # the accounts database (from the repo root)
~/.local/bin/uv run pytest      # test suite (upstream faked; uses aistudio_test)
~/.local/bin/uv run uvicorn app.main:app --reload   # gateway on :8000
```

Run it together with the frontend (same machine):

```bash
# terminal 1: mock upstream (only needed when UPSTREAM_BASE_URL points at it)
node ../frontend/tmp/mock-provider.mjs
# terminal 2:
~/.local/bin/uv run uvicorn app.main:app --reload
# terminal 3:
cd ../frontend && npm run dev
```

In the app, set Base URL to `http://localhost:1234/v1` (same origin — the
Vite dev proxy forwards `/v1` to this gateway) and any API key
(`GATEWAY_TOKEN` if set). Pointing the app directly at `http://127.0.0.1:8000/v1`
also works; the default CORS allow-list already covers the local dev origins
(plus `https://chat.kitsui.co`).

## Configuration (environment / .env)

| Var                 | Meaning                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `UPSTREAM_BASE_URL` | Provider base URL incl. `/v1` (default: local mock `:9400`)        |
| `UPSTREAM_API_KEY`  | The real provider key (never sent to the browser)                  |
| `GATEWAY_TOKEN`     | Client-facing token; empty = accept any non-empty bearer (dev)     |
| `CONTEXT_TOKENS`    | `model:tokens,...` — fills `max_context_tokens` when upstream omits it |
| `GATEWAY_MODELS`    | Optional comma-separated allow-list for `GET /v1/models`           |
| `CORS_ORIGINS`      | Comma-separated origins allowed cross-origin. Default: `https://chat.kitsui.co` + local dev origins; empty = wide open (`*`) |
| `DB_HOST` … `DB_NAME` | The accounts database (MariaDB). Defaults: `127.0.0.1:3306`, user/database `aistudio`. The docker stack points `DB_HOST` at its `mariadb` service and passes the root `.env`'s `MARIADB_*` values through. |

The accounts database is MariaDB (users, sessions, provider/agent profiles, and
the encrypted record store). A deployment that predates it imports its SQLite
file once: `python -m app.migrate_sqlite /path/to/accounts.db`. The test suite
uses `aistudio_test`, created by the compose init script; the development
database is never touched.

`backend/.env` wins over the repo-root `.env` (used by docker compose).

## Endpoints

- `GET /healthz` — public liveness probe.
- `GET /v1/models` — passthrough (optionally filtered by `GATEWAY_MODELS`).
- `GET /v1/models/{id}` — passthrough; adds `max_context_tokens` from
  `CONTEXT_TOKENS` when the upstream doesn't expose it (real OpenAI's
  single-model endpoint doesn't).
- `POST /v1/chat/completions` — passthrough; `stream: true` responses are
  piped back byte-for-byte as SSE. Upstream 4xx on a stream request is
  returned as a JSON error first, so the client's retry ladder works.

## Docker

`docker compose up --build` from the repo root builds and runs this gateway
(plus the frontend, the mock provider, the searxng search backend and the
MariaDB accounts database). See the root `docker-compose.yml`.
