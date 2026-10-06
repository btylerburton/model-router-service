"""router — decide which upstream model a request should be forwarded to.

Combines the deterministic scorer (pre-filter + fallback) with the LLM judge
(final ranker), honoring explicit pins and ALWAYS failing open:

  1. Pin/bypass → caller handles upstream of this (see app.py); router is only
     invoked when routing is wanted.
  2. Scorer shortlists candidates covering the request's HARD demands.
  3. If the judge is enabled, it ranks the shortlist. On ANY judge failure
     (timeout, bad output, network), fall back to the scorer's pick.
  4. If even the scorer yields nothing, fall back to settings.default_model.

Every path returns a Decision with a human-readable rationale, surfaced in the
audit response header and logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .judge import Judge
from .scorer import ModelProfile, score, shortlist

log = logging.getLogger("model_router_service.router")


@dataclass
class Decision:
    model: str
    reason: str
    via: str  # "judge" | "scorer" | "default"


class Router:
    def __init__(
        self,
        *,
        profiles: list[ModelProfile],
        default_model: str,
        judge: Judge | None,
    ) -> None:
        self._profiles = profiles
        self._default = default_model
        self._judge = judge

    @property
    def profiles(self) -> list[ModelProfile]:
        return self._profiles

    async def decide(self, request_text: str, *, client: httpx.AsyncClient) -> Decision:
        # Scorer first: shortlist + a deterministic fallback pick.
        scored = score(request_text, self._profiles)
        _, candidates = shortlist(request_text, self._profiles)

        if self._judge is not None and candidates:
            try:
                jd = await self._judge.decide(request_text, candidates, client=client)
                return Decision(model=jd.model, reason=jd.why or "selected by judge", via="judge")
            except Exception as exc:  # FAIL OPEN to the scorer
                log.warning("judge failed, falling back to scorer: %s", exc)

        if scored.suggested:
            return Decision(model=scored.suggested, reason=scored.reason, via="scorer")

        # Last resort: configured default (fail-open landing model).
        return Decision(
            model=self._default,
            reason="no candidate from judge or scorer; using default_model",
            via="default",
        )
