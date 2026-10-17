"""SSRF-safe one-URL fetch with HTML→markdown extraction (``POST /v1/webfetch``).

The app's agents use this to read a page: no keys involved. Safety rules:
http/https only, every hop's host must resolve to public addresses (a redirect
is re-validated), at most 5 redirects, ~20s timeout, a hard response-size cap,
and a small content-type allowlist. HTML is converted to markdown for the model.

The HTTP client is a module global so tests can swap in a scripted transport.
"""

from __future__ import annotations

import html as html_lib
import ipaddress
import re
import socket
import urllib.parse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

MAX_BYTES = 512 * 1024
MAX_CONTENT_CHARS = 200_000
MAX_TITLE_CHARS = 300
TIMEOUT = 20.0
MAX_REDIRECTS = 5
ALLOWED_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xml",
    "text/xml",
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class FetchError(Exception):
    """A fetch that could not be completed (surfaced to the client as JSON)."""


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
    return _client


_client: httpx.AsyncClient | None = None


def _assert_public_host(host: str, port: int | None) -> None:
    """Reject hosts that resolve to anything but a public address."""
    try:
        infos = socket.getaddrinfo(host, port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise FetchError(f"could not resolve {host}") from err
    if not infos:
        raise FetchError(f"could not resolve {host}")
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as err:
            raise FetchError(f"unusable address for {host}") from err
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise FetchError(f"{host} resolves to a non-public address")


def _validate(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("only http and https URLs can be fetched")
    if not parsed.hostname:
        raise FetchError("the URL has no host")
    default_port = 443 if parsed.scheme == "https" else 80
    _assert_public_host(parsed.hostname, parsed.port or default_port)
    return urllib.parse.urlunsplit(parsed)


def _strip_tags(body: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", body)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _title_of(body: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(1))).strip()


_PRUNE_TAGS = ("head", "script", "style", "noscript", "template", "svg", "iframe")


def _prune(body: str) -> str:
    """Drop head and non-content elements before conversion.

    Markdownify keeps some of their text (the title tag especially); the
    title is extracted separately from the raw HTML.
    """
    soup = BeautifulSoup(body, "html.parser")
    for tag in soup.find_all(_PRUNE_TAGS):
        tag.decompose()
    return str(soup)


async def fetch(url: str, fmt: str = "markdown") -> dict:
    """Fetch one URL; returns {url, finalUrl, title, format, content, truncated}."""
    if fmt not in ("markdown", "text", "html"):
        fmt = "markdown"
    current = _validate(url.strip())

    response: httpx.Response | None = None
    for _ in range(MAX_REDIRECTS + 1):
        try:
            request = client().build_request("GET", current)
            response = await client().send(request, stream=True)
        except httpx.HTTPError as err:
            raise FetchError(f"could not fetch {current}: {err}") from err
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            await response.aclose()
            if not location:
                raise FetchError("a redirect arrived without a location")
            current = _validate(urllib.parse.urljoin(current, location))
            continue
        break
    else:
        raise FetchError("too many redirects")

    assert response is not None
    try:
        if response.status_code >= 400:
            raise FetchError(f"{current} answered HTTP {response.status_code}")
        content_type = (
            (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        )
        if not content_type.startswith(ALLOWED_TYPES):
            raise FetchError(f"unsupported content type: {content_type or 'unknown'}")

        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            if total + len(chunk) > MAX_BYTES:
                chunks.append(chunk[: MAX_BYTES - total])
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        body_bytes = b"".join(chunks)
    finally:
        await response.aclose()

    encoding = response.charset_encoding or "utf-8"
    try:
        body = body_bytes.decode(encoding, errors="replace")
    except LookupError:
        body = body_bytes.decode("utf-8", errors="replace")

    if fmt == "html":
        content = body
    elif fmt == "text":
        content = _strip_tags(body)
    else:
        content = html_to_markdown(_prune(body), heading_style="atx")
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

    content = content[:MAX_CONTENT_CHARS]
    if truncated:
        content += "\n\n…[truncated]"
    return {
        "url": url,
        "finalUrl": current,
        "title": _title_of(body)[:MAX_TITLE_CHARS],
        "format": fmt,
        "content": content,
        "truncated": truncated,
    }
