"""Test fixtures.

The upstream provider is faked with an in-process httpx transport, so the
suite runs fully offline. For the duration of each test the gateway's shared
upstream client is pointed at the fake, and config.settings is swapped to a
fixed, isolated instance (no env-file bleed).
"""

import os

import httpx
import pytest
from httpx import ASGITransport

from app import accounts, config, upstream
from app.main import app

UPSTREAM = "http://upstream.test/v1"
GATEWAY_TOKEN = "gateway-token"


class _Stream(httpx.AsyncByteStream):
    """A minimal async byte stream, for streamed (SSE) upstream responses."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


class FakeUpstream(httpx.AsyncBaseTransport):
    """Stands in for the real provider, keyed by (method, path)."""

    def __init__(self):
        self.routes: dict[tuple[str, str], dict] = {}
        self.requests: list[httpx.Request] = []

    def set(self, method, path, *, status=200, json=None, stream_chunks=None, headers=None):
        self.routes[(method.upper(), path)] = {
            "status": status,
            "json": json,
            "stream_chunks": stream_chunks,
            "headers": headers or {},
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        entry = self.routes.get((request.method, request.url.path))
        if entry is None:
            return httpx.Response(404, json={"error": {"message": f"no fake route for {request.method} {request.url.path}"}})
        if entry["stream_chunks"] is not None:
            headers = {k.lower(): v for k, v in entry["headers"].items()}
            headers.setdefault("content-type", "text/event-stream")
            return httpx.Response(entry["status"], headers=headers, stream=_Stream(entry["stream_chunks"]))
        return httpx.Response(entry["status"], json=entry["json"], headers={"content-type": "application/json", **entry["headers"]})


@pytest.fixture()
async def fake():
    settings = config.Settings(
        _env_file=None,
        upstream_base_url=UPSTREAM,
        upstream_api_key="upstream-secret",
        gateway_token=GATEWAY_TOKEN,
        context_tokens="nova-4o:128000",
        gateway_models="",
        port=8000,
        # The MariaDB the account tests run against: the compose stack publishes
        # it on loopback (`docker compose up -d mariadb`), and the bundled init
        # script creates this test database. The account suite wipes its tables
        # before each test; the development database is never touched.
        db_host=config.settings.db_host,
        db_port=config.settings.db_port,
        db_user=config.settings.db_user,
        db_password=config.settings.db_password,
        db_name=os.environ.get("TEST_DB_NAME", "aistudio_test"),
    )
    previous_settings = config.settings
    config.settings = settings
    previous_client = upstream._client
    fake_upstream = FakeUpstream()
    fake_client = httpx.AsyncClient(transport=fake_upstream)
    upstream._client = fake_client
    yield fake_upstream
    upstream._client = previous_client
    if not fake_client.is_closed:
        await fake_client.aclose()
    config.settings = previous_settings


@pytest.fixture()
async def client(fake):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture()
def db(fake):
    """A MariaDB handle on the test database (the schema is ensured on connect)."""
    conn = accounts._connect()
    yield conn
    conn.close()


@pytest.fixture()
def captcha_mode(client, monkeypatch):
    """Turn the Turnstile robot check ON on the test settings.

    The test calls it with the mode it needs ("test" or "live") after the
    other fixtures are up; the gateway's Cloudflare client is pointed at a
    scripted siteverify answer by the test itself (see test_accounts.py).
    """

    def _set(mode: str) -> None:
        assert mode in ("test", "live"), f"unknown Turnstile mode {mode!r}"
        monkeypatch.setattr(config.settings, "turnstile_mode", mode)
        monkeypatch.setattr(config.settings, "turnstile_secret_key", "captcha-secret")

    return _set
