"""Web tools: keyless search parsing/fallbacks and the SSRF-safe fetcher."""

import ipaddress
import socket

import httpx
import pytest

from app import webfetch, websearch

from conftest import GATEWAY_TOKEN

AUTH = {"Authorization": f"Bearer {GATEWAY_TOKEN}"}


class WebTransport(httpx.AsyncBaseTransport):
    """Routes requests by (host, path) so one client serves every web tool."""

    def __init__(self):
        self.routes: dict[tuple[str, str], dict] = {}
        self.requests: list[httpx.Request] = []

    def set(self, host, path, *, status=200, json=None, text=None, headers=None):
        self.routes[(host.lower(), path)] = {
            "status": status,
            "json": json,
            "text": text,
            "headers": headers or {},
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        entry = self.routes.get((request.url.host.lower(), request.url.path))
        if entry is None:
            return httpx.Response(
                404, json={"error": f"no fake route for {request.url}"}
            )
        headers = {"content-type": "text/html; charset=utf-8", **entry["headers"]}
        if entry["json"] is not None:
            return httpx.Response(entry["status"], json=entry["json"], headers=headers)
        return httpx.Response(entry["status"], text=entry["text"], headers=headers)


@pytest.fixture()
async def web(fake):
    """Point both web-tool clients at a scripted transport (fake gives isolated settings)."""
    transport = WebTransport()
    client = httpx.AsyncClient(transport=transport)
    previous = (websearch._client, webfetch._client)
    websearch._client = client
    webfetch._client = client
    websearch._cache.clear()
    yield transport
    websearch._client, webfetch._client = previous
    await client.aclose()


DDG_HTML = """
<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">Example <b>A</b></a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Snippet <b>one</b> here</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://example.org/b">Example B</a>
  <td class="result__snippet">Second snippet</td>
</div>
</body></html>
"""


def _public_dns(monkeypatch, mapping):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        address = mapping.get(host)
        if address is None:
            try:
                ipaddress.ip_address(host)  # IP literals resolve to themselves
                address = host
            except ValueError:
                raise socket.gaierror(f"unknown host {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 0))]

    monkeypatch.setattr(webfetch.socket, "getaddrinfo", fake_getaddrinfo)


# ---------------- websearch ----------------


async def test_searxng_results_normalized_and_capped(web, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "searxng_url", "http://searx.test")
    monkeypatch.setattr(config.settings, "search_provider", "auto")
    results = [
        {
            "title": f"Title {i}",
            "url": f"https://example.com/{i}",
            "content": "<b>Snip</b> &amp; text",
        }
        for i in range(12)
    ]
    web.set("searx.test", "/search", json={"results": results})

    out = await websearch.search("hello world")
    assert out["provider"] == "searxng"
    assert len(out["results"]) == websearch.MAX_RESULTS == 8
    assert out["results"][0] == {
        "title": "Title 0",
        "url": "https://example.com/0",
        "content": "Snip & text",
    }
    assert web.requests[0].url.params["format"] == "json"


async def test_duckduckgo_html_parsed_and_links_decoded(web, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "search_provider", "duckduckgo")
    web.set("html.duckduckgo.com", "/html/", text=DDG_HTML)

    out = await websearch.search("anything")
    assert out["provider"] == "duckduckgo"
    assert out["results"] == [
        {
            "title": "Example A",
            "url": "https://example.com/a",
            "content": "Snippet one here",
        },
        {
            "title": "Example B",
            "url": "https://example.org/b",
            "content": "Second snippet",
        },
    ]


async def test_auto_falls_back_from_searxng_to_ddg(web, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "searxng_url", "http://searx.test")
    monkeypatch.setattr(config.settings, "search_provider", "auto")
    web.set("searx.test", "/search", status=500, json={"error": "boom"})
    web.set("html.duckduckgo.com", "/html/", text=DDG_HTML)

    out = await websearch.search("fallback please")
    assert out["provider"] == "duckduckgo"
    assert [r.url.host for r in web.requests] == ["searx.test", "html.duckduckgo.com"]


async def test_search_cached(web, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "search_provider", "duckduckgo")
    web.set("html.duckduckgo.com", "/html/", text=DDG_HTML)

    first = await websearch.search("cache me")
    second = await websearch.search("cache me")
    assert first["cached"] is False
    assert second["cached"] is True
    assert len(web.requests) == 1


async def test_search_without_any_provider_raises(web, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "search_provider", "auto")
    monkeypatch.setattr(config.settings, "searxng_url", "")
    web.set("html.duckduckgo.com", "/html/", status=503, text="blocked")

    with pytest.raises(websearch.SearchError) as err:
        await websearch.search("nothing works")
    assert "duckduckgo" in str(err.value)


# ---------------- webfetch ----------------


async def test_webfetch_html_to_markdown(web, monkeypatch):
    _public_dns(monkeypatch, {"public.test": "93.184.216.34"})
    web.set(
        "public.test",
        "/page",
        text="<html><head><title>My Page</title><script>bad()</script></head>"
        "<body><h1>Hello</h1><p>Some <b>bold</b> text.</p></body></html>",
    )

    out = await webfetch.fetch("http://public.test/page")
    assert out["title"] == "My Page"
    assert "# Hello" in out["content"]
    assert "bold" in out["content"]
    assert "bad()" not in out["content"]
    assert out["truncated"] is False


async def test_webfetch_rejects_non_http(web):
    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("file:///etc/passwd")


async def test_webfetch_rejects_private_addresses(web, monkeypatch):
    _public_dns(monkeypatch, {"private.test": "10.0.0.1"})
    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("http://private.test/page")
    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("http://127.0.0.1:8000/v1/models")
    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("http://169.254.169.254/latest/meta-data/")
    assert web.requests == []


async def test_webfetch_revalidates_redirects(web, monkeypatch):
    _public_dns(
        monkeypatch, {"public.test": "93.184.216.34", "private.test": "192.168.1.10"}
    )
    web.set(
        "public.test",
        "/a",
        status=302,
        text="",
        headers={"location": "http://private.test/next"},
    )

    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("http://public.test/a")


async def test_webfetch_truncates_large_pages(web, monkeypatch):
    _public_dns(monkeypatch, {"public.test": "93.184.216.34"})
    web.set(
        "public.test",
        "/big",
        text="x" * (webfetch.MAX_BYTES + 5000),
        headers={"content-type": "text/plain"},
    )

    out = await webfetch.fetch("http://public.test/big", "text")
    assert out["truncated"] is True
    assert out["content"].endswith("[truncated]")


async def test_webfetch_rejects_unsupported_content_type(web, monkeypatch):
    _public_dns(monkeypatch, {"public.test": "93.184.216.34"})
    web.set("public.test", "/img", text="binary", headers={"content-type": "image/png"})

    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch("http://public.test/img")


# ---------------- routes ----------------


async def test_websearch_route(client, web):
    web.set("html.duckduckgo.com", "/html/", text=DDG_HTML)
    from app import config

    config.settings.search_provider = "duckduckgo"
    r = await client.post("/v1/websearch", json={"query": "route test"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "route test"
    assert body["provider"] == "duckduckgo"
    assert body["results"][0]["url"] == "https://example.com/a"


async def test_websearch_route_needs_query(client, web):
    r = await client.post("/v1/websearch", json={}, headers=AUTH)
    assert r.status_code == 400
    assert "query" in r.json()["error"]["message"].lower()


async def test_webfetch_route_and_ssrf_guard(client, web, monkeypatch):
    _public_dns(monkeypatch, {"public.test": "93.184.216.34"})
    web.set("public.test", "/ok", text="<html><body><p>fine</p></body></html>")

    ok = await client.post(
        "/v1/webfetch", json={"url": "http://public.test/ok"}, headers=AUTH
    )
    assert ok.status_code == 200
    assert "fine" in ok.json()["content"]

    blocked = await client.post(
        "/v1/webfetch", json={"url": "http://127.0.0.1/secret"}, headers=AUTH
    )
    assert blocked.status_code == 400
    assert "non-public" in blocked.json()["error"]["message"]
