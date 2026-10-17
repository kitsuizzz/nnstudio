"""chat.kitsui gateway: an OpenAI-compatible pass-through.

The frontend speaks straight to the provider today; this gateway lets it speak
to a same-origin /v1 instead. The real provider key lives here, server-side,
and every supported endpoint is forwarded upstream:

    GET  /v1/models                 (provider onboarding gate + model list)
    GET  /v1/models/{id}            (single model; max_context_tokens is
                                     enriched from CONTEXT_TOKENS when the
                                     upstream doesn't publish it)
    POST /v1/chat/completions       (streamed SSE and plain JSON)
    POST /v1/websearch              (keyless web search for the agents' tools)
    POST /v1/webfetch               (SSRF-safe one-URL fetch, HTML→markdown)

Plus a public GET /healthz for tooling, and an app-facing account API
(/auth/signup, /auth/signin, /auth/signout, /auth/me — see accounts.py) that
lets the browser sign in under a name instead of a raw API key.
"""

import json
import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response as StarletteResponse

from . import accounts, config, upstream, webfetch, websearch

log = logging.getLogger("aistudio.gateway")


class GatewayError(Exception):
    """An error that should surface to the client in the OpenAI error shape."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def require_token(request: Request) -> None:
    # A configured GATEWAY_TOKEN locks the gateway: external consumers and
    # private deployments present it. Unset = the app's open platform gateway —
    # no credential is required, because the browser (guests included) calls
    # /v1 for the platform-provided model and the upstream key stays server-side.
    if not config.settings.gateway_token:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {config.settings.gateway_token}":
        raise GatewayError(401, "Incorrect API key provided.")


def openai_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error"}},
    )


def upstream_unreachable(err: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": f"Upstream provider unreachable: {err}",
                "type": "api_error",
            }
        },
    )


def passthrough(resp: httpx.Response) -> StarletteResponse:
    return StarletteResponse(
        status_code=resp.status_code,
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        s = config.settings
        log.info(
            "upstream: %s (key %s)",
            s.upstream_base_url,
            "configured" if s.upstream_api_key else "not set",
        )
        log.info(
            "gateway token: %s",
            "configured" if s.gateway_token else "any non-empty token (local dev)",
        )
        # Warm the accounts database (schema creation) off the event loop. A
        # database that is still starting is not fatal: /auth retries lazily.
        if await run_in_threadpool(accounts.init):
            log.info("accounts: %s", accounts.db_dsn())
        else:
            log.warning(
                "accounts: %s unreachable — /auth will fail until the database is up",
                accounts.db_dsn(),
            )
        yield
        await upstream.close()

    app = FastAPI(
        title="chat.kitsui gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.settings.cors_origin_list or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        _request: Request, exc: GatewayError
    ) -> JSONResponse:
        return openai_error(exc.status_code, exc.message)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    router = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])

    @router.get("/models")
    async def list_models() -> StarletteResponse:
        try:
            resp = await upstream.client().get(
                upstream.url("/models"), headers=upstream.auth_headers()
            )
        except httpx.HTTPError as err:
            return upstream_unreachable(err)
        if resp.status_code != 200:
            return passthrough(resp)
        body = resp.json()
        allow = config.settings.model_allowlist
        if allow is not None and isinstance(body.get("data"), list):
            body["data"] = [
                m for m in body["data"] if isinstance(m, dict) and m.get("id") in allow
            ]
        return JSONResponse(status_code=200, content=body)

    @router.get("/models/{model_id:path}")
    async def model_detail(model_id: str) -> StarletteResponse:
        try:
            # The router already URL-decoded the id; quote it again for upstream.
            resp = await upstream.client().get(
                upstream.url(f"/models/{quote(model_id, safe='')}"),
                headers=upstream.auth_headers(),
            )
        except httpx.HTTPError as err:
            return upstream_unreachable(err)
        if resp.status_code != 200:
            return passthrough(resp)
        body = resp.json()
        if isinstance(body, dict) and "max_context_tokens" not in body:
            override = config.settings.context_token_map.get(model_id)
            if override is not None:
                body["max_context_tokens"] = override
        return JSONResponse(status_code=200, content=body)

    @router.post("/websearch")
    async def websearch_endpoint(request: Request) -> StarletteResponse:
        try:
            body = json.loads(await request.body())
        except ValueError:
            return openai_error(400, "Request body is not valid JSON.")
        query = str(body.get("query") or "").strip() if isinstance(body, dict) else ""
        if not query:
            return openai_error(400, "Missing search query.")
        try:
            result = await websearch.search(query[:500])
        except websearch.SearchError as err:
            return openai_error(503, f"Web search is unavailable: {err}")
        return JSONResponse(status_code=200, content={"query": query, **result})

    @router.post("/webfetch")
    async def webfetch_endpoint(request: Request) -> StarletteResponse:
        try:
            body = json.loads(await request.body())
        except ValueError:
            return openai_error(400, "Request body is not valid JSON.")
        url = ""
        fmt = "markdown"
        if isinstance(body, dict):
            url = str(body.get("url") or "").strip()
            fmt = str(body.get("format") or "markdown").strip().lower()
        if not url:
            return openai_error(400, "Missing URL.")
        try:
            result = await webfetch.fetch(url, fmt)
        except webfetch.FetchError as err:
            return openai_error(400, f"Could not fetch that page: {err}")
        return JSONResponse(status_code=200, content=result)

    @router.post("/chat/completions")
    async def chat_completions(request: Request) -> StarletteResponse:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            return openai_error(400, "Request body is not valid JSON.")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object.")

        if body.get("stream"):
            # Check the upstream status BEFORE streaming, so a 4xx (e.g. an
            # unsupported stream_options) comes back as a normal JSON error
            # and the client's retry ladder works as it would direct.
            try:
                req = httpx.Request(
                    "POST",
                    upstream.url("/chat/completions"),
                    content=raw,
                    headers={
                        **upstream.auth_headers(),
                        "Content-Type": "application/json",
                    },
                )
                up = await upstream.client().send(req, stream=True)
            except httpx.HTTPError as err:
                return upstream_unreachable(err)
            if up.status_code >= 400:
                err_body = await up.aread()
                await up.aclose()
                return StarletteResponse(
                    status_code=up.status_code,
                    content=err_body,
                    media_type=up.headers.get("content-type", "application/json"),
                )

            async def pipe():
                try:
                    buffer = b""
                    async for chunk in up.aiter_raw():
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            yield line + b"\n"
                    if buffer:
                        yield buffer + b"\n"
                except httpx.HTTPError:
                    return  # upstream dropped mid-stream: end the reply cleanly
                finally:
                    await up.aclose()

            return StreamingResponse(
                pipe(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            resp = await upstream.client().post(
                upstream.url("/chat/completions"),
                content=raw,
                headers={**upstream.auth_headers(), "Content-Type": "application/json"},
            )
        except httpx.HTTPError as err:
            return upstream_unreachable(err)
        return passthrough(resp)

    app.include_router(router)
    # App-facing account API (no gateway token; its own bearer tokens).
    app.include_router(accounts.auth_router())
    return app


app = create_app()
