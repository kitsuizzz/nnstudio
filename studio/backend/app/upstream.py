"""Async client for the upstream provider.

The gateway is a transparent shim: it forwards the client's request to the
upstream with its own (server-side) credentials. URL joining is done
explicitly so the upstream base may or may not carry a trailing slash.
"""

import httpx

from . import config

_client: httpx.AsyncClient | None = None

# Generous read timeout: long generations are normal for a chat upstream.
_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)


def client() -> httpx.AsyncClient:
    """The shared upstream client (lazily created)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


def auth_headers() -> dict[str, str]:
    """Per-request credentials for the upstream (never sent to the client)."""
    if config.settings.upstream_api_key:
        return {"Authorization": f"Bearer {config.settings.upstream_api_key}"}
    return {}


def url(path: str) -> str:
    return config.settings.upstream_base_url.rstrip("/") + path


async def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def reset() -> None:
    """Drop the shared client without awaiting (test helper)."""
    global _client
    _client = None
