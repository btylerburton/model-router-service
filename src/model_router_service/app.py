"""app — the OpenAI-compatible routing proxy (FastAPI).

OpenCode (or any OpenAI-compatible client) points its baseURL at THIS service.
Every /v1/chat/completions request is:

  1. Checked for a bypass/pin signal — if set, forwarded UNTOUCHED (honor pins).
  2. Otherwise routed: the router picks the best model for the prompt, we rewrite
     the request's `model` field to that choice, and forward to the real upstream
     gateway. Streaming (SSE) and non-streaming are both proxied faithfully.
  3. The chosen model + rationale is returned in an audit response header and
     logged, so routing is always observable.

FAIL-OPEN: a routing error never fails the turn — the request forwards to the
configured default model. Only an UPSTREAM error (the real gateway) surfaces to
the client, exactly as it would without the proxy.

ENDPOINTS:
  POST /v1/chat/completions   — the routed, streaming-capable proxy
  GET  /v1/models             — passthrough of the upstream model list
  GET  /healthz               — liveness (does not call upstream)
  GET  /readyz                — readiness (confirms candidate profiles loaded)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .catalog import from_catalog_file, from_upstream_ids
from .config import Settings, get_settings
from .judge import Judge
from .router import Router

log = logging.getLogger("model_router_service")


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if value else False


def _extract_request_text(body: dict) -> str:
    """Flatten the chat messages into text the router/judge can read.

    Uses the LAST user message primarily (the current turn), with a little
    preceding context. Handles both string and content-part messages.
    """
    messages = body.get("messages") or []

    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    elif part.get("type") in {"image_url", "image"}:
                        parts.append("[image]")  # a vision signal for the scorer
            return " ".join(parts)
        return ""

    user_msgs = [m for m in messages if m.get("role") == "user"]
    text = _content_to_text(user_msgs[-1]["content"]) if user_msgs else ""
    # Prepend a short slice of the latest system message for intent context.
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if sys_msgs:
        text = (_content_to_text(sys_msgs[-1]["content"])[:300] + "\n" + text).strip()
    return text


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # STARTUP: one shared async client, discover candidates, build the router.
        # A test may pre-seed app.state.client with a MockTransport client; honor
        # it rather than replacing it, so the fake upstream is used.
        if getattr(app.state, "client", None) is None:
            app.state.client = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
        app.state.settings = settings
        profiles = await _load_profiles(settings, app.state.client)
        judge = None
        if settings.judge_enabled:
            judge = Judge(
                base_url=settings.effective_judge_base_url,
                api_key=settings.effective_judge_api_key,
                model=settings.judge_model,
                timeout_s=settings.judge_timeout_s,
                cache_size=settings.judge_cache_size,
            )
        app.state.router = Router(
            profiles=profiles,
            default_model=settings.default_model,
            judge=judge,
        )
        log.info(
            "router ready: %d candidate models, judge=%s, default=%s",
            len(profiles),
            "on" if judge else "off",
            settings.default_model,
        )
        yield
        # SHUTDOWN.
        await app.state.client.aclose()

    app = FastAPI(title="model-router-service", version="0.1.0", lifespan=lifespan)
    app.state.router = None

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        router: Router | None = app.state.router
        if router and router.profiles:
            return JSONResponse({"status": "ready", "candidates": len(router.profiles)})
        return JSONResponse({"status": "not-ready"}, status_code=503)

    @app.get("/v1/models")
    async def models() -> Response:
        # Passthrough so the client sees the real upstream catalog.
        r = await app.state.client.get(
            f"{settings.upstream_base}/models",
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
        )
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return JSONResponse({"error": "malformed JSON body"}, status_code=400)

        requested_model = body.get("model")
        is_stream = bool(body.get("stream"))

        # 1. Honor pins/bypass: forward untouched.
        bypass = _truthy(request.headers.get(settings.bypass_header))
        decision_desc = ""
        if bypass and requested_model:
            chosen = requested_model
            decision_desc = "bypass: pin honored, no routing"
        else:
            # 2. Route. Fail-open to default on any router error.
            router: Router = app.state.router
            request_text = _extract_request_text(body)
            try:
                decision = await router.decide(request_text, client=app.state.client)
                chosen = decision.model
                decision_desc = f"{decision.via}: {decision.reason}"
            except Exception as exc:  # defensive: router.decide already fails open
                chosen = settings.default_model
                decision_desc = f"router error, default: {exc}"
            body["model"] = chosen

        log.info(
            "route: requested=%s chosen=%s stream=%s (%s)",
            requested_model, chosen, is_stream, decision_desc,
        )

        headers = {
            "Authorization": f"Bearer {settings.upstream_api_key}",
            "Content-Type": "application/json",
        }
        # HTTP header values must be latin-1 encodable; the rationale may contain
        # unicode (e.g. the scorer's "→"), so coerce to ASCII for the audit header.
        audit_value = f"{chosen}; {decision_desc}"[:400].encode("ascii", "replace").decode("ascii")
        audit = {settings.decision_header: audit_value}
        url = f"{settings.upstream_base}/chat/completions"
        new_body = json.dumps(body).encode("utf-8")

        if is_stream:
            return await _proxy_stream(app.state.client, url, headers, new_body, audit)
        return await _proxy_once(app.state.client, url, headers, new_body, audit)

    return app


async def _load_profiles(settings: Settings, client: httpx.AsyncClient):
    if settings.catalog_path:
        return from_catalog_file(settings.catalog_path)
    # Discover ids from the upstream /models list.
    try:
        r = await client.get(
            f"{settings.upstream_base}/models",
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
        )
        r.raise_for_status()
        data = r.json()
        ids = [m["id"] for m in data.get("data", data) if isinstance(m, dict) and m.get("id")]
        profiles = from_upstream_ids(ids)
        if profiles:
            return profiles
    except Exception as exc:
        log.warning("could not discover upstream models (%s); starting with empty catalog", exc)
    return []


async def _proxy_once(client, url, headers, body, audit) -> Response:
    r = await client.post(url, headers=headers, content=body)
    out = Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )
    for k, v in audit.items():
        out.headers[k] = v
    return out


async def _proxy_stream(client, url, headers, body, audit) -> Response:
    # Open the upstream stream, relay bytes as they arrive (SSE).
    req = client.build_request("POST", url, headers=headers, content=body)
    r = await client.send(req, stream=True)

    async def _iter():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()

    resp = StreamingResponse(
        _iter(),
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/event-stream"),
    )
    for k, v in audit.items():
        resp.headers[k] = v
    return resp
