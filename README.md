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

The default path is **one stage**: the deterministic graded scorer. An LLM judge
is an **opt-in second stage** (off by default — see "Judge: opt-in" below).

1. **Graded deterministic scorer** (offline, zero-dependency, sub-millisecond,
   **no network call** — this is the DEFAULT engine). The request is
   scored on a continuous **reasoning difficulty** `r ∈ [0,1]` from three signals
   — **depth** (proofs/derivation), **breadth** (cross-file/system scope), and
   **novelty** (design-from-scratch) — combined as `max(weighted_mean,
   strongest_signal)`. `r` maps to a **band** (`none`/`light`/`moderate`/`hard`/
   `extreme`) with a **minimum cost-rank floor**. Capability demands
   (vision/long-context) still gate the candidate set.
2. **LLM judge** *(opt-in; `ROUTER_JUDGE_ENABLED=true`)* — one small call to a
   cheap model (configurable; defaults to the same gateway + a cheap model id)
   ranks the floored shortlist AND may return its own depth/breadth/novelty
   sub-scores. When it does, those are authoritative and can raise the floor. The
   judge is constrained to the provided ids and its JSON reply is validated.

**Judge: opt-in, and why.** The judge adds a *full extra inference round-trip to
the upstream before every prompt* — real added latency on every turn (the earlier
judge-on build timed out in live use). For the ~15-model USAi catalog the
deterministic scorer routes correctly on its own, so the judge is **off by
default** and is best reserved as an escalation after you've *measured* the scorer
misrouting. Turn it on with `ROUTER_JUDGE_ENABLED=true` when the per-turn cost is
acceptable.

**Why graded, not boolean:** a boolean `reasoning` demand made the top tier
*unreachable* — on the USAi catalog every rank-4/5 model (Opus/GPT-5) has the
same capability set as a rank-3 Sonnet, so "cheapest covering reasoning" always
stopped at Sonnet. The per-band rank **floor** lets the hardest work (a
correctness proof, a Paxos formal verification) climb to Opus-tier, while keeping
cost-down behavior *within* a band. Weights and band edges are named constants
(and overridable at runtime — see Steering below), so re-tuning is "change a
number, replay the cache, diff the decisions."

**Fail-open everywhere:** if the judge times out, errors, or returns a bad id,
the service falls back to the graded scorer's floored pick; if even that abstains,
it forwards to a configured `default_model`. A routing problem never breaks a turn.

**Honors pins:** send the bypass header (`x-model-router-bypass: 1`) with an
explicit `model` and the request is forwarded untouched.

**Auditable:** the chosen model, a decision id, and the rationale are returned in
the `x-model-router-decision` response header, and one JSON line per routed turn
(`prompt_hash, r, band, demands, winner`) is written to the decision log.

## Steering the router after a bad route

When the router picks a poor model, you don't edit code — you give feedback and
recalibrate. A correction is just "this prompt should have used `<model>`"; the
service **back-solves** the reasoning target from that model's cost tier, records
it, and an offline **recalibrate** pass searches band edges that satisfy your
corrections without regressing known-good decisions, writing a tuning-overrides
file the scorer loads. Deterministic, diffable, reversible (delete the file).

```bash
# correct the LAST routed turn (reads the decision log), or name a prompt:
model-router feedback --last --model claude_4_8_opus
model-router feedback --prompt "refactor this module and add tests" --model gpt-5.2

# search + apply new band edges (dry-run first to preview):
model-router recalibrate --dry-run
model-router recalibrate

# or from a running proxy / the agent, correct a turn by its decision id
# (from the x-model-router-decision header):
curl -s $ROUTER/feedback -H 'content-type: application/json' \
  -d '{"id":"<decision_id>","model":"claude_4_8_opus","note":"needed opus"}'
```

Corrections are **inert** until `recalibrate` runs — nothing changes routing
mid-session. Point the scorer at the overrides with `MODEL_ROUTER_TUNING`.

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
| `ROUTER_JUDGE_ENABLED` | `false` | `true` adds the per-prompt LLM judge (extra latency); default = scorer-only |
| `ROUTER_JUDGE_TIMEOUT_S` / `ROUTER_JUDGE_CACHE_SIZE` | — | Judge call tuning |
| `ROUTER_BYPASS_HEADER` / `ROUTER_DECISION_HEADER` | — | Pin + audit header names |
| `MODEL_ROUTER_TUNING` | — | Path to a band-edge/weight overrides file (written by `recalibrate`) |
| `MODEL_ROUTER_FEEDBACK` | — | Path to the corrections store (JSONL) |
| `MODEL_ROUTER_DECISION_LOG` | — | Path to the per-turn decision log (JSONL) |
| `ROUTER_CA_BUNDLE` | — | CA bundle for a TLS-inspecting proxy (see TLS section) |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | The routed, streaming-capable proxy |
| GET | `/v1/models` | Passthrough of the upstream model list |
| POST | `/feedback` | Record a routing correction (by prompt or decision id) |
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

### Reaching a host-run service from a sandbox

The default `127.0.0.1` bind is **loopback-only** — a sandbox/container cannot
reach it, even via `host.docker.internal`. To route an in-sandbox OpenCode
through a service on your host, bind an interface the Docker bridge can see:

```bash
scripts/run-local.sh                # bind for sandbox reachability + start
scripts/run-local.sh --print-only   # just show the bind + the sandbox URL
```

The script prefers the narrowest reachable interface (the Docker bridge gateway
if it can detect one) and only falls back to `0.0.0.0` (all interfaces) with an
explicit warning. The sandbox then points at the **host bridge**, never the bind
address: `MODEL_ROUTER_URL=http://host.docker.internal:8080`.

> **Exposure note.** Binding beyond loopback makes the service reachable from
> other hosts on your network, and it forwards your USAi key upstream. Run it
> only on a trusted/firewalled network and stop it when done. **On cloud.gov you
> do none of this** — the buildpack sets `$PORT` and the app binds `0.0.0.0:$PORT`
> automatically (see "Deploy to cloud.gov"); the sandbox reaches it at the public
> app route, not `host.docker.internal`.

## TLS trust (`ROUTER_CA_BUNDLE`) — Zscaler and other inspecting proxies

**Diagnose before you set anything.** The common instinct — "I'm on Zscaler, so
point the service at the Zscaler cert" — is usually **wrong** and will *break* a
connection that would otherwise work. Run the one-line check first.

### Key fact: `ROUTER_CA_BUNDLE` REPLACES the trust store, it does not add to it

Python/httpx verify TLS against the `certifi` bundle (~140 public roots) by
default. Setting `ROUTER_CA_BUNDLE` makes httpx trust **only** the certs in that
file. So pointing it at a single `zscaler-root.pem` means the service trusts
Zscaler and *nothing else* — and any endpoint presenting a normal **public**
chain (e.g. USAi behind Amazon/Starfield roots) then fails with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.

Only set `ROUTER_CA_BUNDLE` when the endpoint's chain is **actually** being
re-signed by a proxy whose root certifi lacks — and even then, use a bundle that
ALSO keeps the public roots (see "combined bundle" below).

### Step 1 — does the default (certifi) already work?

```bash
openssl s_client -connect api.gsa.usai.gov:443 \
  -CAfile "$(python3 -c 'import certifi; print(certifi.where())')" </dev/null 2>/dev/null \
  | grep -i "verify return code"
```

- **`Verify return code: 0 (ok)`** → certifi verifies the chain. **Leave
  `ROUTER_CA_BUNDLE` UNSET.** You're done — no cert wrangling needed. (This is
  the normal case for `api.gsa.usai.gov`, whose chain is public
  Amazon → Amazon Root CA 1 → Starfield, not Zscaler.)
- **Non-zero** (e.g. `20 (unable to get local issuer certificate)`) → the chain
  really is being re-signed by a proxy root certifi doesn't have. Go to Step 2.

To see the actual chain and the root it terminates at:

```bash
openssl s_client -connect api.gsa.usai.gov:443 -showcerts </dev/null 2>/dev/null \
  | grep -E "^\s*[0-9]+ s:|^\s*i:"
# The LAST cert's `i:` (issuer) is the root you must trust. If it says
# "CN=Zscaler Root CA" you are being intercepted; if it says Amazon/Starfield
# you are NOT, and certifi already covers it.
```

### Step 2 — only if Step 1 was non-zero: build a COMBINED bundle

Export the proxy root, then concatenate it **with** certifi so you keep the
public roots too (split-tunnel safe — works whether or not a given host is
intercepted). Verification stays ON; there is deliberately no "disable verify".

```bash
# macOS — export the Zscaler root (it's a public, self-signed root):
security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain > ~/zscaler-root.pem
# (The genuine shared root is self-signed with SHA-256
#  04:F6:1F:1D:13:AA:E1:D1:...:8B:1A:53 — verify with:
#   openssl x509 -in ~/zscaler-root.pem -noout -subject -issuer -fingerprint -sha256 )

# Combine public roots (certifi) + the proxy root into ONE bundle:
cat "$(python3 -c 'import certifi; print(certifi.where())')" ~/zscaler-root.pem > ~/combined-ca.pem
grep -c "BEGIN CERTIFICATE" ~/combined-ca.pem        # expect 100+, NOT 1

# Confirm it verifies the real chain:
openssl s_client -connect api.gsa.usai.gov:443 \
  -CAfile ~/combined-ca.pem </dev/null 2>/dev/null | grep -i "verify return code"
# want: Verify return code: 0 (ok)
```

```bash
# .env — point at the COMBINED bundle (never a lone cert):
ROUTER_CA_BUNDLE=/Users/<you>/combined-ca.pem
```

On a managed **Linux** host the system store already merges public + proxy roots,
so a single path works there: `ROUTER_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`.
On **cloud.gov** / any non-intercepted egress, leave `ROUTER_CA_BUNDLE` unset.

### Step 3 — restart and confirm

```bash
pkill -f model_router_service
<run the service>                 # e.g. uv run python -m model_router_service
curl -s http://127.0.0.1:8080/readyz    # expect {"status":"ready","candidates":15}
```

If `/readyz` reports `candidates > 0` and startup logged no TLS/401 warning,
trust is correct.

> **Note (dev):** a startup `401 Unauthorized` from `/models` is NOT always a
> trust problem — it is also what you get from a stale server started before the
> key was loaded, or from a genuinely bad `ROUTER_UPSTREAM_API_KEY`. Confirm the
> key independently with:
> ```bash
> curl -s -o /dev/null -w "%{http_code}\n" https://api.gsa.usai.gov/api/v1/models \
>   -H "Authorization: Bearer $(grep ^ROUTER_UPSTREAM_API_KEY= .env | cut -d= -f2-)"
> ```
> `200` = key good (so a service-side failure is TLS/trust or a stale process);
> `401` = fix the key.

## Run and point OpenCode at it

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

The service is a config-free app (12-factor): the Python buildpack runs it, it
binds `$PORT` automatically, and **every** host/model/key is an env var. The
`deploy/manifest.yml` holds only non-secret config; the upstream key comes from a
bound user-provided service so it never lives in git or the manifest.

**Prerequisites:** the `cf` CLI, authenticated to your cloud.gov org/space
(`cf login --sso -a api.fr.cloud.gov`), and a real USAi gateway key.

```bash
cd model-router-service

# 1. Create the secrets service (ONE time; holds the upstream key out of band).
#    Add ROUTER_JUDGE_API_KEY too ONLY if the judge uses a different endpoint.
cf cups model-router-secrets -p '{"ROUTER_UPSTREAM_API_KEY":"<your-usai-key>"}'

# 2. Bind it: uncomment the `services:` block at the bottom of deploy/manifest.yml
#       services:
#         - model-router-secrets

# 3. Push (first deploy).
cf push -f deploy/manifest.yml

# 4. Confirm it came up and discovered the catalog.
APP_URL="https://$(cf app model-router-service | awk '/routes:/{print $2}')"
curl -s "$APP_URL/healthz"   # {"status":"ok"}            (liveness, no upstream call)
curl -s "$APP_URL/readyz"    # {"status":"ready","candidates":N}  (catalog loaded)
```

**What ships in the manifest (non-secret):** `ROUTER_UPSTREAM_BASE_URL`,
`ROUTER_JUDGE_MODEL`, `ROUTER_DEFAULT_MODEL`, and **`ROUTER_JUDGE_ENABLED: "false"`**
— the deterministic scorer is the default, so cloud.gov routes with **no
per-prompt upstream judge call** out of the box. To turn the judge on later
without a redeploy:

```bash
cf set-env model-router-service ROUTER_JUDGE_ENABLED true
cf restage model-router-service
```

**Secret rotation:** `cf uups model-router-secrets -p '{"ROUTER_UPSTREAM_API_KEY":"<new>"}'`
then `cf restage model-router-service`.

**TLS trust on cloud.gov:** leave `ROUTER_CA_BUNDLE` **unset** — cloud.gov egress
is not TLS-intercepted, so certifi's default bundle is correct (the Zscaler
`ROUTER_CA_BUNDLE` dance is a *host-only* concern).

**Point OpenCode at it:** once `$APP_URL/readyz` is ready, the `model-router-proxy`
kit (or a manual `model-router-toggle --harness opencode on --url "$APP_URL"`)
flips OpenCode's `baseURL` to `$APP_URL/v1`. The sandbox must be allowed to egress
to that cloud.gov host (add it to your project's egress kit).

> **Notes.** `runtime.txt` pins the buildpack to Python 3.12 (where the bounded
> dependency ranges resolve to prebuilt wheels). `memory: 256M` in the manifest
> is sized for the stdlib scorer path; if you enable the judge and see memory
> pressure under load, bump it. The Dockerfile is provided for container-based
> targets; cloud.gov uses the buildpack + manifest, not the Dockerfile.

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
