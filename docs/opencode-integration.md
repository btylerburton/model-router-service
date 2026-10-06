# Integrating with OpenCode (and the model-router kit)

There are **two** ways routing reaches an agent, and they are complementary:

| | `model-router` MCP kit | **This proxy service** |
|---|---|---|
| Where it runs | Inside the sandbox (stdio MCP) | In the request path (HTTP, in front of the gateway) |
| What it does | **Recommends** a model via a `model_select` tool | **Auto-switches** the model for every prompt |
| Can it change the active model? | No — advisory only (the agent/you decide) | **Yes** — it rewrites the request's `model` and forwards |
| Needs an LLM call? | No (deterministic scorer) | Yes (cheap judge) + deterministic fallback |
| Best for | Visibility / explainability | Transparent, hands-off auto-routing |

The MCP kit cannot auto-switch because OpenCode gives a tool no hook to change
the active model mid-turn. **This proxy is how you get the hermes/jev-style "all
prompts run through the router and it auto-switches."**

## Wire OpenCode to the proxy

OpenCode talks to a provider via a `baseURL`. Point that at the proxy instead of
the real gateway. The proxy forwards to the real gateway itself (it holds the
upstream key; OpenCode does not need to change its own key handling).

```jsonc
// ~/.config/opencode/opencode.jsonc
"provider": {
  "usai": {
    "npm": "@ai-sdk/openai-compatible",
    "options": {
      // was: https://api.gsa.usai.gov/api/v1
      "baseURL": "http://127.0.0.1:8080/v1"   // the proxy
    }
  }
}
```

That's the only change. Every `/v1/chat/completions` from OpenCode now routes
automatically. The model you *select* in OpenCode becomes a **hint** the proxy
may override; to force a specific model, send the bypass header (below).

### Local dev

```bash
cp .env.example .env            # set a real ROUTER_UPSTREAM_API_KEY
python -m model_router_service   # 127.0.0.1:8080
# then point OpenCode's baseURL at http://127.0.0.1:8080/v1
```

### In a sandbox, pointing at a cloud.gov deployment

Set the proxy URL via env/config, never hardcoded:

```jsonc
"provider": { "usai": { "options": { "baseURL": "{env:MODEL_ROUTER_URL}/v1" } } }
```

```bash
export MODEL_ROUTER_URL="https://model-router-service.app.cloud.gov"
```

The sandbox then needs egress to that host (deny-by-default): add it to the
sandbox's allow-list exactly as you would any provider endpoint. The proxy, not
the sandbox, holds the USAi key.

## Forcing a specific model (bypass)

```bash
curl .../v1/chat/completions \
  -H 'x-model-router-bypass: 1' \
  -d '{"model":"claude_4_8_opus", ...}'    # forwarded untouched
```

## Seeing what it chose

Every response carries the decision header:

```
x-model-router-decision: claude_4_5_haiku; judge: simple edit, cheapest capable
```

and the service logs one line per request with requested vs. chosen model.

## Relationship to the kit's `central-server.md`

The patterns-repo kit documents an *adapter-mode* central **decision** endpoint
(`POST /route` → a recommendation). That is the advisory path at fleet scale.
**This** service is the *proxy* path: it sits on `/v1/chat/completions` and
actually switches the model. They can coexist — the proxy for auto-switching,
the MCP tool for in-agent visibility — but if your goal is "all prompts auto-route
in OpenCode," the proxy is the piece that delivers it.
