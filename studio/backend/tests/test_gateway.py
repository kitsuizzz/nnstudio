"""Gateway behavior: auth, passthrough, enrichment, and the SSE pipe."""

import json

from conftest import GATEWAY_TOKEN, UPSTREAM

AUTH = {"Authorization": f"Bearer {GATEWAY_TOKEN}"}


async def test_healthz_is_public(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_models_requires_valid_token(client, fake):
    fake.set("GET", "/v1/models", json={"data": [{"id": "nova-4o"}]})

    r = await client.get("/v1/models")
    assert r.status_code == 401
    assert "message" in r.json()["error"]

    r = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert "message" in r.json()["error"]

    # The upstream must never be called for rejected clients.
    assert fake.requests == []


async def test_open_gateway_needs_no_token(client, fake, monkeypatch):
    # Unset GATEWAY_TOKEN = the app's open platform gateway: the browser
    # (guests included) calls /v1 with no credential at all.
    from app import config

    monkeypatch.setattr(config.settings, "gateway_token", "")
    fake.set("GET", "/v1/models", json={"data": [{"id": "nova-4o"}]})

    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"data": [{"id": "nova-4o"}]}
    assert fake.requests[0].headers["authorization"] == "Bearer upstream-secret"


async def test_models_passthrough_with_upstream_key(client, fake):
    fake.set(
        "GET", "/v1/models", json={"data": [{"id": "nova-4o"}, {"id": "nova-4o-mini"}]}
    )

    r = await client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"data": [{"id": "nova-4o"}, {"id": "nova-4o-mini"}]}

    req = fake.requests[0]
    assert str(req.url) == f"{UPSTREAM}/models"
    assert req.headers["authorization"] == "Bearer upstream-secret"


async def test_models_respects_allowlist(client, fake):
    from app import config

    config.settings.gateway_models = "nova-4o"
    fake.set(
        "GET", "/v1/models", json={"data": [{"id": "nova-4o"}, {"id": "nova-4o-mini"}]}
    )

    r = await client.get("/v1/models", headers=AUTH)
    assert [m["id"] for m in r.json()["data"]] == ["nova-4o"]


async def test_models_allowlist_none_exposes_nothing(client, fake):
    # GATEWAY_MODELS="none": the deployment advertises no server models, and
    # the app hides the Server-provided category entirely.
    from app import config

    config.settings.gateway_models = "none"
    fake.set(
        "GET", "/v1/models", json={"data": [{"id": "nova-4o"}, {"id": "nova-4o-mini"}]}
    )

    r = await client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"data": []}


async def test_models_upstream_error_passthrough(client, fake):
    fake.set(
        "GET", "/v1/models", status=401, json={"error": {"message": "bad upstream key"}}
    )

    r = await client.get("/v1/models", headers=AUTH)
    assert r.status_code == 401
    assert r.json()["error"]["message"] == "bad upstream key"


async def test_model_detail_enriches_max_context_tokens(client, fake):
    # The upstream omits max_context_tokens (as real OpenAI does); the
    # CONTEXT_TOKENS map fills it in for the running head's budget display.
    fake.set("GET", "/v1/models/nova-4o", json={"id": "nova-4o", "object": "model"})

    r = await client.get("/v1/models/nova-4o", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["max_context_tokens"] == 128000


async def test_model_detail_keeps_upstream_value(client, fake):
    # When the upstream publishes the field, the map must not clobber it.
    fake.set(
        "GET", "/v1/models/nova-4o", json={"id": "nova-4o", "max_context_tokens": 64000}
    )

    r = await client.get("/v1/models/nova-4o", headers=AUTH)
    assert r.json()["max_context_tokens"] == 64000


async def test_model_detail_upstream_404_passthrough(client, fake):
    fake.set(
        "GET",
        "/v1/models/unknown-model",
        status=404,
        json={"error": {"message": "model unknown-model not found"}},
    )

    r = await client.get("/v1/models/unknown-model", headers=AUTH)
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "model unknown-model not found"


async def test_chat_completions_nonstream(client, fake):
    fake.set(
        "POST",
        "/v1/chat/completions",
        json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
    )
    body = {
        "model": "nova-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    r = await client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"

    # The client's body is forwarded verbatim with the upstream's key.
    assert json.loads(fake.requests[0].content) == body
    assert fake.requests[0].headers["authorization"] == "Bearer upstream-secret"


async def test_chat_completions_streams_sse_verbatim(client, fake):
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    fake.set("POST", "/v1/chat/completions", stream_chunks=[sse])

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "nova-4o", "messages": [], "stream": True},
        headers=AUTH,
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        received = b"".join([c async for c in r.aiter_raw()])
    assert received == sse


async def test_chat_completions_stream_upstream_error(client, fake):
    # A 4xx on a stream request must come back as a JSON error (not SSE),
    # so the client's retry ladder (stream_options, reasoning_effort) works.
    fake.set(
        "POST",
        "/v1/chat/completions",
        status=400,
        json={"error": {"message": "unsupported parameter: stream_options"}},
    )

    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "nova-4o",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers=AUTH,
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "unsupported parameter: stream_options"
    assert "application/json" in r.headers["content-type"]


async def test_chat_completions_invalid_json(client, fake):
    r = await client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={**AUTH, "content-type": "application/json"},
    )
    assert r.status_code == 400
    assert "message" in r.json()["error"]
    assert fake.requests == []
