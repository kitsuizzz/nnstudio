"""Keyless web search for the gateway: SearXNG (JSON) with a DuckDuckGo HTML
fallback.

The app's agents call ``POST /v1/websearch``; no search-provider key is
involved. ``SEARCH_PROVIDER`` picks the backend (``auto`` = SearXNG when
``SEARXNG_URL`` is configured, else DuckDuckGo), and a failing primary falls
back to the other candidate. Results are normalized to
``{title, url, content}`` and capped.

The HTTP client is a module global so tests can swap in a scripted transport
(see ``tests/test_web.py``), mirroring ``upstream.py``.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse

import httpx

from . import config

MAX_RESULTS = 8
MAX_TITLE_CHARS = 300
MAX_URL_CHARS = 2000
MAX_CONTENT_CHARS = 1000
CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 200
REQUEST_TIMEOUT = 10.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SearchError(Exception):
    """A search that could not be completed (surfaced to the client as JSON)."""


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
    return _client


_client: httpx.AsyncClient | None = None
# query -> (stored_at, provider, results)
_cache: dict[str, tuple[float, str, list[dict]]] = {}


def _clean(text: object) -> str:
    """Strip tags/entities and collapse whitespace from a snippet/title."""
    raw = str(text or "")
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


async def _searxng(query: str) -> list[dict]:
    base = config.settings.searxng_url.strip().rstrip("/")
    if not base:
        raise SearchError("no SearXNG instance is configured")
    resp = await client().get(
        f"{base}/search",
        params={"q": query, "format": "json", "safesearch": "1"},
    )
    if resp.status_code != 200:
        raise SearchError(f"SearXNG answered HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as err:
        raise SearchError(
            "SearXNG did not return JSON (is the json format enabled?)"
        ) from err
    results: list[dict] = []
    for item in (data.get("results") or [])[:MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = _clean(item.get("title"))
        if not url or not title:
            continue
        results.append(
            {
                "title": title[:MAX_TITLE_CHARS],
                "url": url[:MAX_URL_CHARS],
                "content": _clean(item.get("content"))[:MAX_CONTENT_CHARS],
            }
        )
    if not results:
        raise SearchError("SearXNG returned no results")
    return results


_A_TAG = re.compile(r"<a\b[^>]*>", re.I)
_HREF = re.compile(r'href="([^"]+)"', re.I)
_DDG_LINK = re.compile(
    r'<a\b[^>]*class="[^"]*result__a[^"]*"[^>]*>.*?</a>', re.I | re.S
)
_DDG_SNIPPET = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div)>', re.I | re.S
)


def _real_url(url: str) -> str:
    """Decode DuckDuckGo's own /l/?uddg= redirect wrapper to the target URL."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return urllib.parse.unquote(target[0])
    return url


async def _duckduckgo(query: str) -> list[dict]:
    resp = await client().get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    if resp.status_code != 200:
        raise SearchError(f"DuckDuckGo answered HTTP {resp.status_code}")
    body = resp.text
    links = _DDG_LINK.findall(body)
    snippets = _DDG_SNIPPET.findall(body)
    results: list[dict] = []
    for index, link in enumerate(links):
        tag = _A_TAG.search(link)
        href = _HREF.search(tag.group(0)) if tag else None
        if not href:
            continue
        url = _real_url(href.group(1))
        title = _clean(link[tag.end() : link.lower().rfind("</a>")])
        if not url.startswith(("http://", "https://")) or not title:
            continue
        snippet = _clean(snippets[index]) if index < len(snippets) else ""
        results.append(
            {
                "title": title[:MAX_TITLE_CHARS],
                "url": url[:MAX_URL_CHARS],
                "content": snippet[:MAX_CONTENT_CHARS],
            }
        )
        if len(results) >= MAX_RESULTS:
            break
    if not results:
        raise SearchError("DuckDuckGo returned no parseable results")
    return results


def _candidates() -> list[str]:
    mode = (config.settings.search_provider or "auto").strip().lower()
    if mode == "searxng":
        return ["searxng"]
    if mode in ("duckduckgo", "ddg"):
        return ["duckduckgo"]
    ordered = ["searxng"] if config.settings.searxng_url.strip() else []
    ordered.append("duckduckgo")
    return ordered


async def search(query: str) -> dict:
    """Search the web; returns {provider, results, cached}."""
    query = query.strip()
    if not query:
        raise SearchError("the query is empty")

    now = time.time()
    cached = _cache.get(query.lower())
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return {"provider": cached[1], "results": cached[2], "cached": True}

    errors: list[str] = []
    for provider in _candidates():
        try:
            results = await (_searxng if provider == "searxng" else _duckduckgo)(query)
        except SearchError as err:
            errors.append(f"{provider}: {err}")
            continue
        except httpx.HTTPError as err:
            errors.append(f"{provider}: {err}")
            continue
        _cache[query.lower()] = (now, provider, results)
        if len(_cache) > CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        return {"provider": provider, "results": results, "cached": False}

    raise SearchError("; ".join(errors) or "no search provider is available")
