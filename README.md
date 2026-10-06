# model-router-service

An **OpenAI-compatible routing proxy** that sits in front of a USAi-style
gateway and **auto-selects the best upstream model for every request**. Point an
OpenAI-compatible client (e.g. OpenCode) at this service instead of the gateway;
each prompt is inspected, routed to the model that best fits it, and forwarded —
transparently, with streaming preserved.

This is the request-path counterpart to the advisory `model-router` MCP kit in
[`agentic-coding-patterns`](https://github.com/GSA-TTS/agentic-coding-patterns):
the MCP tool *recommends* a model, this service *actually switches* it (like the
hermes/jev router), because only something in the request path can do that for
OpenCode.

## How it decides

Two-stage, mirroring hermes/jev's two-call shape — but the engine is yours:

1. **Deterministic scorer** (offline, zero-dependency) reduces the request to the
   capabilities it demands and shortlists candidate models that cover the HARD
   demands (vision, long context).
2. **LLM judge** — one small call to a cheap model (configurable; defaults to the
   same gateway + a cheap model id) ranks the shortlist and picks the cheapest
   capable model. The judge is constrained to the provided ids and its JSON reply
   is validated.

**Fail-open everywhere:** if the judge times out, errors, or returns a bad id,
the service falls back to the scorer's pick; if even that yields nothing, it
forwards to a configured `default_model`. A routing problem never breaks a turn.

**Honors pins:** send the bypass header (`x-model-router-bypass: 1`) with an
explicit `model` and the request is forwarded untouched.

**Auditable:** the chosen model + rationale is returned in the
`x-model-router-decision` response header and logged on every request.

## No hardcoded anything (12-factor)

Every host, URL, model id, and credential is an **environment variable** — the
same build runs locally, in a sandbox, and on cloud.gov; only the env differs.
Required values fail **closed** at startup (the service refuses to start without
an upstream gateway + key + judge model + default model). See `.env.example`.

| Var | Required | Meaning |
|-----|----------|---------|
| `ROUTER_UPSTREAM_BASE_URL` | ✅ | Upstream OpenAI-compatible gateway base URL |
| `ROUTER_UPSTREAM_API_KEY` | ✅ | Bearer key for the gateway (inject as a secret) |
| `ROUTER_JUDGE_MODEL` | ✅ | Cheap model id used to rank candidates |
| `ROUTER_DEFAULT_MODEL` | ✅ | Fail-open landing model |
| `ROUTER_JUDGE_BASE_URL` / `ROUTER_JUDGE_API_KEY` | — | Judge endpoint override (defaults to upstream) |
| `ROUTER_CATALOG_PATH` | — | JSON roster of candidate profiles (else discovered from `/models`) |
| `ROUTER_JUDGE_ENABLED` | — | `false` → scorer-only, no judge call |
| `ROUTER_JUDGE_TIMEOUT_S` / `ROUTER_JUDGE_CACHE_SIZE` | — | Judge call tuning |
| `ROUTER_BYPASS_HEADER` / `ROUTER_DECISION_HEADER` | — | Pin + audit header names |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | The routed, streaming-capable proxy |
| GET | `/v1/models` | Passthrough of the upstream model list |
| GET | `/healthz` | Liveness (no upstream call) |
| GET | `/readyz` | Readiness (candidate profiles loaded) |

## Run locally

```bash
cp .env.example .env            # fill in a real ROUTER_UPSTREAM_API_KEY
pip install -e ".[dev]"
python -m model_router_service   # binds ROUTER_HOST:ROUTER_PORT (default 127.0.0.1:8080)
```

Or with Docker:

```bash
cp .env.example .env
docker compose up --build        # http://localhost:8080
```

Point OpenCode at it (the `usai` provider's `baseURL`):

```jsonc
"provider": {
  "usai": { "options": { "baseURL": "http://localhost:8080/v1" } }
}
```

Every prompt now routes automatically. Watch the chosen model:

```bash
curl -s -D- http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"claude_4_8_opus","messages":[{"role":"user","content":"fix a typo"}]}' \
  | grep -i x-model-router-decision
```

## Deploy to cloud.gov

```bash
# 1. Supply secrets out-of-band (never in the manifest):
cf cups model-router-secrets -p '{"ROUTER_UPSTREAM_API_KEY":"…"}'
# 2. Uncomment the `services:` block in deploy/manifest.yml to bind it.
# 3. Push:
cf push -f deploy/manifest.yml
```

The app binds `$PORT` automatically. Non-secret config lives in the manifest;
keys come from the bound service.

## Test

```bash
pip install -e ".[dev]"
pytest -q
```

Tests cover the scorer, the judge (JSON parsing/validation + a stubbed client),
and the proxy end-to-end against a fake upstream (routing, pin-honor, fail-open,
streaming, audit header) — all offline.

## Security notes

- Put TLS termination (and ideally mTLS or network policy) in front of this
  service; it forwards a bearer key to the upstream and should not be openly
  reachable.
- The `request` text is read only to route; it is forwarded to the upstream as
  the user intended. Avoid logging full request bodies if they may carry
  sensitive content — the service logs decisions + metadata, not bodies.
- Secrets come only from the environment / a bound service; none are committed.

## License

CC0-1.0 (public domain dedication), matching the agentic-coding ecosystem.
