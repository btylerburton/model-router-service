"""config — all runtime configuration from the environment (12-factor).

No host, URL, model id, or credential is hardcoded. A deployment is configured
ENTIRELY through environment variables, so the same artifact runs unchanged
locally, in a sandbox, and on cloud.gov — only the env differs.

FAIL-CLOSED on required values: the service refuses to start if the upstream
gateway or its credential is missing, rather than silently defaulting to some
baked-in endpoint. FAIL-OPEN at request time is a separate, deliberate choice
(see router.py) — a judge error forwards to ROUTER_DEFAULT_MODEL rather than
erroring the turn; that is runtime resilience, not a missing-config default.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Upstream gateway (REQUIRED — no default host) ----------------------
    # The OpenAI-compatible endpoint every routed request is forwarded to, e.g.
    # the USAi gateway. Set ROUTER_UPSTREAM_BASE_URL and ROUTER_UPSTREAM_API_KEY.
    upstream_base_url: HttpUrl = Field(
        ...,
        description="Base URL of the upstream OpenAI-compatible gateway (e.g. https://api.gsa.usai.gov/api/v1).",
    )
    upstream_api_key: str = Field(
        ...,
        description="Bearer key for the upstream gateway. Injected as a secret; never commit it.",
    )

    # --- Judge inference (defaults to the SAME upstream) --------------------
    # The LLM judge needs its own inference endpoint. By default it reuses the
    # upstream gateway + key (one credential, one egress target) and only needs a
    # cheap model id. Override base_url/api_key to point the judge elsewhere.
    judge_model: str = Field(
        ...,
        description="A cheap/fast model id (on the judge endpoint) used to RANK candidates, e.g. a haiku/flash tier.",
    )
    judge_base_url: HttpUrl | None = Field(
        default=None,
        description="Judge inference base URL. Defaults to upstream_base_url when unset.",
    )
    judge_api_key: str | None = Field(
        default=None,
        description="Judge inference key. Defaults to upstream_api_key when unset.",
    )

    # --- Routing behavior ---------------------------------------------------
    # The model used when the judge fails/times out (FAIL-OPEN). REQUIRED so a
    # judge outage has a defined, auditable landing model rather than a crash.
    default_model: str = Field(
        ...,
        description="Model to forward to when routing/judging fails (fail-open landing model).",
    )

    # Optional JSON file of candidate model capability profiles (same schema as
    # the model-router kit's catalog). When unset, the service derives candidates
    # from the upstream /models list at startup (ids only) + a built-in family
    # capability map. Never a hardcoded host — a FILE PATH only.
    catalog_path: str | None = Field(
        default=None,
        description="Path to a JSON roster of candidate model capability profiles. Optional.",
    )

    # Judge invocation controls.
    judge_enabled: bool = Field(
        default=True,
        description="If false, routing uses the deterministic scorer only (no LLM judge call).",
    )
    judge_timeout_s: float = Field(
        default=4.0,
        description="Per-call timeout for the judge. On timeout the service fails open to the scorer/default.",
    )
    judge_cache_size: int = Field(
        default=512,
        description="LRU cache size for identical judged prompts (0 disables caching).",
    )

    # Honor an explicit model pin: when a request names a model AND signals
    # 'do not route' (see ROUTER_BYPASS_HEADER), forward it untouched.
    bypass_header: str = Field(
        default="x-model-router-bypass",
        description="If this request header is truthy, pass the request through without routing.",
    )
    # Response header that always reports the model the proxy chose (audit).
    decision_header: str = Field(
        default="x-model-router-decision",
        description="Response header carrying the chosen model + rationale for auditability.",
    )

    # Request timeout for the forwarded (upstream) call. Streaming responses are
    # not bounded by this once headers are received.
    upstream_timeout_s: float = Field(default=600.0)

    # --- TLS trust (for TLS-inspecting proxies like Zscaler) ----------------
    # On a host behind a TLS-intercepting corporate proxy (e.g. Zscaler), the
    # upstream presents a proxy-RE-SIGNED certificate. httpx verifies against
    # certifi's bundle by default, which does NOT contain the corporate root, so
    # the handshake fails — and such a failure often surfaces as a gateway 401 /
    # block page rather than a clean TLS error. Point this at a CA bundle that
    # INCLUDES the proxy root to fix it. This is the service-side equivalent of
    # acq's persistent `zscaler-ca-certificate` kit.
    #
    #   ROUTER_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt   (host system store,
    #     which on a managed host already includes the Zscaler root)
    #   ROUTER_CA_BUNDLE=/path/to/zscaler-root.pem            (just the proxy root)
    #
    # Leave unset to use certifi's default (correct for cloud.gov / any
    # non-intercepted egress). NEVER disable verification — there is deliberately
    # no "verify=false" option here.
    ca_bundle: str | None = Field(
        default=None,
        description="Path to a CA bundle that includes any TLS-inspecting proxy root (e.g. Zscaler). Unset = certifi default.",
    )

    # Server bind. cloud.gov provides $PORT; __main__ maps it to ROUTER_PORT.
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8080)

    log_level: str = Field(default="INFO")

    # --- Derived helpers ----------------------------------------------------
    @property
    def verify(self):
        """httpx `verify` value: a CA-bundle path when configured, else True
        (certifi default). Never False — verification is not disableable."""
        return self.ca_bundle if self.ca_bundle else True
    @property
    def effective_judge_base_url(self) -> str:
        return str(self.judge_base_url or self.upstream_base_url)

    @property
    def effective_judge_api_key(self) -> str:
        return self.judge_api_key or self.upstream_api_key

    @property
    def upstream_base(self) -> str:
        # Normalized, trailing-slash-free base for clean path joins.
        return str(self.upstream_base_url).rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Raises (fail-closed) if a REQUIRED var is missing."""
    return Settings()  # type: ignore[call-arg]
