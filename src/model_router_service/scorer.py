"""scorer — deterministic, offline request→model pre-filter and fallback.

Ported from the agentic-coding-patterns model-router kit. Two jobs here:

  1. PRE-FILTER: cheaply reduce the candidate set to the models that can
     actually satisfy a request's hard demands (vision, long context) before the
     LLM judge ranks them — so the judge is never asked to pick a vision model
     for a vision task from a list that includes non-vision models.
  2. FALLBACK: when the judge is disabled, times out, or errors, this scorer
     alone picks the model (fail-open to a sane, cheapest-covering choice rather
     than crashing the turn).

Zero dependencies; pure standard library. Capability semantics live in the
catalog (see catalog.py), not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

CAPABILITIES = frozenset(
    {"reasoning", "coding", "fast", "cheap", "long_context", "vision"}
)

_DEMAND_PATTERNS: dict[str, list[str]] = {
    "reasoning": [
        r"\barchitect(ure|ing)?\b", r"\bdesign\b", r"\brefactor(ing)?\b",
        r"\bdebug(ging)?\b", r"\broot[- ]cause\b", r"\btrade[- ]?off", r"\balgorithm",
        r"\breason(ing)?\b", r"\bprove\b", r"\bcomplex\b", r"\bplan\b",
        r"\bmigrate|migration\b", r"\bthreat model", r"\bsecurity review",
    ],
    "coding": [
        r"\bcode\b", r"\bimplement", r"\bfunction\b", r"\bclass\b", r"\bbug\b",
        r"\bfix\b", r"\bwrite (a |the )?(script|test|unit test)", r"\bedit\b",
        r"\bpatch\b", r"\bcompile", r"\bsyntax", r"\bprogram\b",
    ],
    "fast": [
        r"\btypo\b", r"\brename\b", r"\bcomment\b", r"\bformat(ting)?\b",
        r"\bone[- ]line", r"\bquick\b", r"\bsimple\b", r"\btrivial\b",
        r"\bsmall (fix|change|edit)\b",
    ],
    "long_context": [
        r"\bwhole (repo|repository|codebase)\b", r"\bentire (repo|file|codebase)\b",
        r"\bacross (the )?(many|all) files\b", r"\blarge (file|document|log)\b",
        r"\blong (document|transcript|log|file)\b", r"\b\d{3,}k tokens?\b",
        r"\bmany files\b", r"\bfull context\b",
    ],
    "vision": [
        r"\bimage\b", r"\bscreenshot\b", r"\bdiagram\b", r"\bphoto\b", r"\bpng\b",
        r"\bjpe?g\b", r"\bpicture\b", r"\bmockup\b",
    ],
}


@dataclass
class ModelProfile:
    id: str
    name: str
    capabilities: frozenset[str]
    context: int
    cost_rank: int
    input_cost: Optional[float] = None

    def covers(self, demands: frozenset[str]) -> bool:
        return demands.issubset(self.capabilities)


@dataclass
class ScoreResult:
    suggested: Optional[str]
    demands: list[str] = field(default_factory=list)
    reason: str = ""
    shortlist: list[str] = field(default_factory=list)


def detect_demands(request: str) -> frozenset[str]:
    text = request.lower()
    demands: set[str] = set()
    for capability, patterns in _DEMAND_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            demands.add(capability)
    return frozenset(demands)


def shortlist(
    request: str,
    profiles: list[ModelProfile],
    *,
    max_cost_rank: Optional[int] = None,
) -> tuple[frozenset[str], list[ModelProfile]]:
    """Return (detected demands, candidate profiles covering the HARD demands).

    Hard demands = vision + long_context (a model physically cannot serve a
    request it lacks the modality/window for). Soft demands (reasoning/coding/
    fast) are left to the ranker so the judge can weigh them.
    """
    demands = detect_demands(request)
    hard = demands & {"vision", "long_context"}
    candidates: list[ModelProfile] = []
    for p in profiles:
        if max_cost_rank is not None and p.cost_rank > max_cost_rank:
            continue
        if not hard.issubset(p.capabilities):
            continue
        candidates.append(p)
    if not candidates:
        # Nothing covers the hard demands — don't filter to empty; let the ranker
        # see everything within the cost ceiling rather than fail.
        candidates = [
            p for p in profiles
            if max_cost_rank is None or p.cost_rank <= max_cost_rank
        ]
    return demands, candidates


def score(
    request: str,
    profiles: list[ModelProfile],
    *,
    max_cost_rank: Optional[int] = None,
) -> ScoreResult:
    """Deterministic pick: cheapest model covering all detected demands.

    Used as the fallback when the judge is unavailable. Never raises on normal
    input; returns suggested=None only when the profile list is empty.
    """
    if not profiles:
        return ScoreResult(suggested=None, reason="no candidates")

    demands, candidates = shortlist(request, profiles, max_cost_rank=max_cost_rank)

    covering = [p for p in candidates if p.covers(demands)] or candidates

    def sort_key(p: ModelProfile) -> tuple:
        return (
            p.cost_rank,
            p.input_cost if p.input_cost is not None else float("inf"),
            len(p.capabilities),
            p.id,
        )

    covering.sort(key=sort_key)
    winner = covering[0]
    demand_str = ", ".join(sorted(demands)) if demands else "general"
    return ScoreResult(
        suggested=winner.id,
        demands=sorted(demands),
        reason=f"scorer: demands [{demand_str}] -> cheapest covering model (cost_rank {winner.cost_rank})",
        shortlist=[p.id for p in covering],
    )
