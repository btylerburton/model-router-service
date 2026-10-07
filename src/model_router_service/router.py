"""router — decide which upstream model a request should be forwarded to.

Graded flow (see scorer.py for the "why boolean reasoning was unreachable" note):

  1. The scorer grades the request's reasoning (r, band, rank FLOOR) and builds a
     candidate set that (a) covers the HARD capability demands (vision/long
     context) and (b) sits AT OR ABOVE the band floor. The floor is what lets the
     hardest work climb to the top tier.
  2. If the judge is enabled, it ranks WITHIN that floored set and may ALSO
     return its own depth/breadth/novelty sub-scores. When it does, we re-grade
     with the judge's sub-scores (authoritative) — which can RAISE the floor — and
     re-floor the candidate set, then take the judge's pick if it still qualifies,
     else the cheapest at/above the (possibly raised) floor.
  3. On ANY judge failure (timeout, bad output, network) fall open to the
     scorer's deterministic floored pick.
  4. If even the scorer abstains / yields nothing, fall back to default_model.

Every path returns a Decision with a rationale + the graded reasoning dict, both
surfaced in the audit header and the per-turn decision log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .judge import Judge
from .scorer import (
    ModelProfile,
    ReasoningScore,
    grade_from_subscores,
    score,
    shortlist,
)

log = logging.getLogger("model_router_service.router")


@dataclass
class Decision:
    model: str
    reason: str
    via: str  # "judge" | "scorer" | "default"
    reasoning: Optional[dict] = None       # ReasoningScore.to_dict()
    demands: list[str] = field(default_factory=list)


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
        # Cheapest CHAT model in the catalog (profiles are already chat-only — the
        # catalog excludes embeddings/rerankers). Used as the abstain landing
        # model so a routable-but-trivial request (band 'none', r≈0) goes to the
        # cheapest model rather than the mid-tier default_model.
        self._cheapest = (
            min(profiles, key=lambda p: p.cost_rank).id if profiles else default_model
        )

    @property
    def profiles(self) -> list[ModelProfile]:
        return self._profiles

    @property
    def cheapest(self) -> str:
        return self._cheapest

    def _cheapest_at_or_above(self, candidates: list[ModelProfile]) -> Optional[ModelProfile]:
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda p: (p.cost_rank, p.input_cost if p.input_cost is not None else float("inf"), p.id),
        )[0]

    async def decide(self, request_text: str, *, client: httpx.AsyncClient) -> Decision:
        # 1. Heuristic grade + floored candidate set + deterministic fallback pick.
        scored = score(request_text, self._profiles)
        demands, rs, candidates = shortlist(request_text, self._profiles)

        # 2. Judge ranks within the floored set and may re-grade reasoning.
        if self._judge is not None and candidates:
            try:
                jd = await self._judge.decide(request_text, candidates, client=client)

                # If the judge returned sub-scores, they are authoritative — they
                # may RAISE the band/floor, so re-floor the candidate set.
                if jd.has_subscores():
                    rs = grade_from_subscores(jd.depth, jd.breadth, jd.novelty)
                    _, _, candidates = shortlist(
                        request_text, self._profiles, reasoning=rs
                    )

                valid = {p.id for p in candidates}
                if jd.model in valid:
                    chosen = jd.model
                    reason = jd.why or "selected by judge"
                else:
                    # Judge picked below the (possibly raised) floor — enforce it.
                    floored = self._cheapest_at_or_above(candidates)
                    if floored is None:
                        raise RuntimeError("no candidate at/above floor after judge re-grade")
                    chosen = floored.id
                    reason = (
                        f"judge chose {jd.model} below floor {rs.floor}; "
                        f"enforced cheapest at/above floor"
                    )
                return Decision(
                    model=chosen, reason=reason, via="judge",
                    reasoning=rs.to_dict(), demands=sorted(demands),
                )
            except Exception as exc:  # FAIL OPEN to the scorer
                log.warning("judge failed, falling back to scorer: %s", exc)

        # 3. Scorer fallback.
        if scored.suggested:
            return Decision(
                model=scored.suggested, reason=scored.reason, via="scorer",
                reasoning=scored.reasoning, demands=scored.demands,
            )

        # 4. Scorer abstained (band 'none', no hard demand). The request is still
        # a real chat turn — route it to the CHEAPEST chat model, not the
        # mid-tier default_model. (default_model remains the FAIL-OPEN landing for
        # genuine errors, handled by the caller.) This keeps trivial asks cheap
        # ("[1,3,2,12] sort this" → cheapest, not sonnet) while preserving a
        # distinct, auditable "default" path for actual failures.
        return Decision(
            model=self._cheapest,
            reason="scorer abstained (band none); using cheapest chat model",
            via="scorer",
            reasoning=scored.reasoning,
            demands=scored.demands,
        )
