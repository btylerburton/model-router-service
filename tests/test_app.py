"""End-to-end-ish tests for the proxy app with a FAKE upstream gateway.

We build the app with test Settings, then monkeypatch the shared httpx client's
transport to a MockTransport that stands in for BOTH the upstream gateway and
the judge endpoint. This exercises routing, pin-honor, fail-open, streaming, and
the audit header without any real network.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from model_router_service.app import create_app
from model_router_service.config import Settings


def _settings(**over) -> Settings:
    base = dict(
        upstream_base_url="https://upstream.example/v1",
        upstream_api_key="up-key",
        judge_model="judge-cheap",
        default_model="claude_4_5_sonnet",
        judge_enabled=True,
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def _mock_transport(judge_pick="claude_4_5_haiku", upstream_status=200, record=None):
    """A transport serving /models, judge /chat/completions, and upstream
    /chat/completions. `record` captures the forwarded upstream body."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "claude_4_5_haiku"},
                {"id": "claude_4_5_sonnet"},
                {"id": "gemini-2.5-pro"},
                {"id": "text-embedding-005"},
            ]})
        if url.endswith("/chat/completions"):
            body = json.loads(request.content)
            # The judge call carries the judge system prompt; distinguish it.
            is_judge = any(
                m.get("role") == "system" and "routing judge" in str(m.get("content", "")).lower()
                for m in body.get("messages", [])
            )
            if is_judge:
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps({"model": judge_pick, "why": "test"})}}]
                })
            # Upstream forward.
            if record is not None:
                record["forwarded_model"] = body.get("model")
                record["stream"] = body.get("stream")
            if body.get("stream"):
                # Minimal SSE stream.
                def gen():
                    yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    yield b"data: [DONE]\n\n"
                return httpx.Response(upstream_status, stream=httpx.ByteStream(b"".join(gen())),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(upstream_status, json={
                "id": "cmpl-1",
                "model": body.get("model"),
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            })
        return httpx.Response(404, json={"error": "unexpected path"})

    return httpx.MockTransport(handler)


def _client(settings, transport) -> TestClient:
    app = create_app(settings)
    # Pre-seed the shared client with our MockTransport BEFORE the lifespan
    # startup runs (TestClient enters the lifespan on __enter__). The lifespan
    # honors an already-set client, so startup discovery + all forwards use the
    # fake upstream.
    app.state.client = httpx.AsyncClient(transport=transport)
    return TestClient(app)


def test_routes_simple_prompt_to_cheap_model():
    record = {}
    settings = _settings()
    transport = _mock_transport(judge_pick="claude_4_5_haiku", record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "claude_4_8_opus",  # client asked for opus…
            "messages": [{"role": "user", "content": "fix a quick typo"}],
        })
    assert r.status_code == 200
    assert record["forwarded_model"] == "claude_4_5_haiku"  # …but routed to haiku
    assert "x-model-router-decision" in {k.lower(): v for k, v in r.headers.items()}


def test_bypass_header_honors_pin():
    record = {}
    settings = _settings()
    transport = _mock_transport(record=record)
    with _client(settings, transport) as c:
        r = c.post(
            "/v1/chat/completions",
            headers={"x-model-router-bypass": "1"},
            json={"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "fix a typo"}]},
        )
    assert r.status_code == 200
    assert record["forwarded_model"] == "gemini-2.5-pro"  # untouched


def test_fail_open_to_default_when_judge_returns_garbage():
    record = {}
    settings = _settings(default_model="claude_4_5_sonnet")
    # judge_pick is a hallucinated id → judge raises → scorer fallback (which,
    # for a reasoning task, still yields a covering model). Use a prompt the
    # scorer can also handle so we assert a sane forward happened.
    transport = _mock_transport(judge_pick="not-a-real-model", record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "redesign the architecture"}],
        })
    assert r.status_code == 200
    # Scorer picked a real covering model (not the hallucinated judge id).
    assert record["forwarded_model"] in {"claude_4_5_sonnet", "gemini-2.5-pro"}


def test_streaming_is_proxied():
    record = {}
    settings = _settings()
    transport = _mock_transport(record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "x", "stream": True,
            "messages": [{"role": "user", "content": "fix a typo"}],
        })
    assert r.status_code == 200
    assert record["stream"] is True
    assert b"[DONE]" in r.content


def test_healthz():
    settings = _settings()
    transport = _mock_transport()
    with _client(settings, transport) as c:
        assert c.get("/healthz").json()["status"] == "ok"
