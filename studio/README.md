# studio — the Chat half (chat.kitsui / AI Studio), vendored

This directory is AI Studio, brought into nnstudio so the one app has both
halves: **Studio** (the canvas, at `/`) and **Chat** (this, at `/chat`). The
sources here are the upstream project's, with the one patch the merge needs —
the `Chat | Studio` switch in `frontend/src/components/TopBar.tsx` and its
styles in `frontend/src/index.css`.

## What runs

| piece | where | how |
|---|---|---|
| the SPA | `frontend/` (React + Vite) | built to `frontend/dist`, served by nnstudio's server at `/chat` |
| the gateway | `backend/` (FastAPI) | OpenAI-compatible `/v1` + accounts `/auth`; nnstudio proxies `/chat/v1` and `/chat/auth` to it, streamed |
| accounts DB | MariaDB (`backend/app/accounts.py`) | the upstream `docker-compose.yml` runs it (with searxng + a mock provider) |

nnstudio's server finds the pieces by convention: `studio/frontend/dist` and
`http://127.0.0.1:8000`. Two environment variables move them:
`NNSTUDIO_STUDIO_WEB` and `NNSTUDIO_STUDIO_GATEWAY`.

## Build the SPA

    cd studio/frontend
    npm install
    VITE_BASE_PATH=chat npm run build      # assets under /chat/assets/…

`VITE_BASE_PATH=chat` is what makes the app address itself (and its gateway)
under the `/chat` prefix; the same-origin gateway default is `/chat/v1`, so no
other variable is needed.

## Run the gateway

The upstream stack (fastapi + mariadb + searxng + a mock provider) is the
project's `docker-compose.yml`, kept in its own repository for now:

    docker compose up -d          # in the AI Studio checkout: gateway on :8000,
                                  # mariadb, searxng, and the mock provider

A host-run gateway works too (`uv run uvicorn app.main:app` in `backend/`,
with `backend/.env` naming the upstream — `UPSTREAM_BASE_URL` +
`UPSTREAM_API_KEY`). While the gateway is down, `/chat` still serves the app
and the chat falls back to its BYOK provider form, by upstream design.

## Tests

The upstream suites live with the upstream project (`backend/tests`, and the
Playwright `shot-*.mjs` gates under `frontend/tmp`). nnstudio's own suite
checks the merge itself: the SPA is served at `/chat` and the gateway answers
through the proxy (`tests/run.sh --live`).
