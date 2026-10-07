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
    # judge_pick is a hallucinated id → judge.decide raises → router falls open
    # to the deterministic scorer, which forwards a REAL catalog id (never the
    # hallucinated one, and never an error to the client).
    transport = _mock_transport(judge_pick="not-a-real-model", record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "prove this algorithm correct with formal verification"}],
        })
    assert r.status_code == 200
    # A real catalog id was forwarded (the scorer's floored pick), not the
    # hallucinated judge id and not an error.
    assert record["forwarded_model"] in {
        "claude_4_5_haiku", "claude_4_5_sonnet", "gemini-2.5-pro"
    }
    assert record["forwarded_model"] != "not-a-real-model"


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


# --- Paseo/OpenCode-shaped behavior ----------------------------------------

OPENCODE_SYS = (
    "You are OpenCode, You and the user share the same workspace and collaborate "
    "to achieve the user's goals. You are a deeply pragmatic, effective software "
    "engineer. You communicate efficiently, keeping it short."
)
TITLE_SYS = (
    "You are a title generator. You output ONLY a thread title. Nothing else.\n"
    "Generate a brief title that would help the user find this conversation later."
)


def test_scores_user_message_not_system_persona():
    """Regression (Paseo): the OpenCode system persona must NOT be scored. A hard
    user request behind the fixed persona should route on the USER text — not be
    diluted to band 'none' by the boilerplate preamble (the live-log bug)."""
    record = {}
    settings = _settings(judge_enabled=False)  # deterministic scorer only
    transport = _mock_transport(record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt_5_5_default_v2",
            "messages": [
                {"role": "system", "content": OPENCODE_SYS},
                {"role": "user", "content":
                    "prove this distributed consensus protocol correct with a formal "
                    "verification under adversarial conditions"},
            ],
        })
    assert r.status_code == 200
    # A hard reasoning ask must climb above the cheapest tier (not land on
    # default via band 'none'). With the persona removed from scoring, the
    # scorer grades the proof request and floors UP.
    hdr = {k.lower(): v for k, v in r.headers.items()}["x-model-router-decision"]
    assert "via=default" not in hdr.replace(": ", "=")  # not an abstain-to-default
    assert record["forwarded_model"] != "gpt_5_5_default_v2"  # it rerouted


def test_utility_title_gen_forced_to_cheapest():
    """A harness title-generation call is detected by its system prompt and
    forced to the cheapest model, skipping scoring (via=utility)."""
    record = {}
    settings = _settings(judge_enabled=False)  # cheapest == haiku in this catalog
    transport = _mock_transport(record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gpt_5_5_default_v2",
            "messages": [
                {"role": "system", "content": TITLE_SYS},
                {"role": "user", "content": "what node version is this project using?"},
            ],
        })
    assert r.status_code == 200
    assert record["forwarded_model"] == "claude_4_5_haiku"  # cheapest by cost_rank
    hdr = {k.lower(): v for k, v in r.headers.items()}["x-model-router-decision"]
    assert "utility" in hdr


def test_utility_passthrough_can_be_disabled():
    record = {}
    settings = _settings(judge_enabled=False, passthrough_utility=False)
    transport = _mock_transport(record=record)
    with _client(settings, transport) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [
                {"role": "system", "content": TITLE_SYS},
                {"role": "user", "content": "what node version is this project using?"},
            ],
        })
    assert r.status_code == 200
    hdr = {k.lower(): v for k, v in r.headers.items()}["x-model-router-decision"]
    assert "utility" not in hdr  # routed normally, not forced


def test_decision_log_omits_raw_prompt_by_default(tmp_path, monkeypatch):
    """The on-disk decision log must NOT contain raw prompt text by default —
    only the hash (prompts carry repo contents / PR bodies / paths)."""
    logf = tmp_path / "decisions.jsonl"
    # _DECISION_LOG is module-level; patch it.
    import model_router_service.app as appmod
    monkeypatch.setattr(appmod, "_DECISION_LOG", str(logf))
    record = {}
    settings = _settings(judge_enabled=False)
    transport = _mock_transport(record=record)
    secret_text = "SENSITIVE repo path /home/agent/secret and PR body"
    with _client(settings, transport) as c:
        c.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [
                {"role": "system", "content": OPENCODE_SYS},
                {"role": "user", "content": secret_text},
            ],
        })
    written = logf.read_text()
    assert "prompt_hash" in written
    assert "SENSITIVE" not in written  # raw prompt not persisted
    assert '"prompt"' not in written


def test_decision_log_includes_prompt_when_opted_in(tmp_path, monkeypatch):
    logf = tmp_path / "decisions.jsonl"
    import model_router_service.app as appmod
    monkeypatch.setattr(appmod, "_DECISION_LOG", str(logf))
    settings = _settings(judge_enabled=False, log_prompts=True)
    transport = _mock_transport(record={})
    with _client(settings, transport) as c:
        c.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "DEBUGMARKER fix a typo"}],
        })
    assert "DEBUGMARKER" in logf.read_text()


def test_feedback_disabled_by_default_returns_404():
    settings = _settings(judge_enabled=False)
    transport = _mock_transport()
    with _client(settings, transport) as c:
        r = c.post("/feedback", json={"model": "claude_4_8_opus", "prompt": "x"})
    assert r.status_code == 404
