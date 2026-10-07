"""scorer — deterministic, offline request→model pre-filter, grading, and fallback.

Three jobs:

  1. CAPABILITY GATE: reduce the candidate set to models that can physically
     serve a request's HARD demands (vision, long context) before anything ranks
     them. Unchanged by the graded-reasoning work.
  2. GRADED REASONING: score the request's reasoning difficulty on a continuous
     r ∈ [0,1] (depth / breadth / novelty sub-scores), map it to a BAND, and
     derive a minimum cost-rank FLOOR. This is what makes top-tier models
     reachable — see the design note below.
  3. FALLBACK: when the judge is disabled, times out, or errors, this scorer
     alone picks the model (fail-open to the cheapest model AT OR ABOVE the band
     floor that covers the hard demands) rather than crashing the turn.

WHY GRADED REASONING (the bug this fixes):

  The old scorer treated `reasoning` as a BOOLEAN demand and then picked the
  cheapest model COVERING the demand set. On the USAi catalog every rank-4/5
  model (Opus/GPT-5) has a capability set IDENTICAL to a rank-3 model (Sonnet),
  so "cheapest covering reasoning" ALWAYS stopped at Sonnet (rank 3). Opus-tier
  was strictly dominated on cost and UNREACHABLE by construction — a refactor,
  a correctness proof, and a Paxos formal-verification all collapsed to the same
  Sonnet pick. Replacing the boolean with a graded score + per-band rank FLOOR
  lets the hardest work pull the selection up the cost ladder, while keeping
  cost-down behavior WITHIN a band. Inspired by DECRUX9812/typesafe-skill-router
  (continuous scores, thresholds-in-code, replay-and-tune).

Zero dependencies; pure standard library. Capability semantics live in the
catalog (catalog.py); band edges + weights are named constants HERE so re-tuning
is "change a number, replay the cache, diff the decisions" (no re-inference).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

CAPABILITIES = frozenset(
    {"reasoning", "coding", "fast", "cheap", "long_context", "vision"}
)

# ---------------------------------------------------------------------------
# Capability demand detection (hard gates: vision, long_context; soft: coding,
# fast). NOTE: the boolean "reasoning" demand is RETIRED from selection — its
# job is now done by the graded reasoning score below. We keep the keyword list
# only as one input to the `depth`/`novelty` sub-scores, not as a gate.
# ---------------------------------------------------------------------------
_DEMAND_PATTERNS: dict[str, list[str]] = {
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

HARD_DEMANDS = frozenset({"vision", "long_context"})

# ---------------------------------------------------------------------------
# Graded reasoning: three independent signals → r ∈ [0,1].
# Weights and band edges are NAMED CONSTANTS (tune in code, replay the cache).
#
# COMBINE FUNCTION (calibrated against the 8-prompt spectrum test):
#   r = max( weighted_mean , strongest_single_signal_boosted )
# A weighted mean alone can't let a PURE-DEPTH task (a correctness proof with no
# breadth/novelty cues) reach the top band — yet that is exactly the hardest
# work. So a single SATURATED signal is allowed to drive the band up on its own
# (depth is the strongest driver), while multi-signal prompts still benefit from
# breadth + novelty via the mean. Both terms are bounded [0,1].
# ---------------------------------------------------------------------------
W_DEPTH = 0.50
W_BREADTH = 0.30
W_NOVELTY = 0.20

# How much a single saturated signal can contribute on its own (the "dominant
# signal" term). A fully-saturated DEPTH signal reaches 'extreme' alone — a
# correctness proof / formal verification is the hardest work and must be able
# to reach the top tier even with no breadth/novelty cues. breadth/novelty cap
# lower so scope/novelty alone don't demand the most expensive model.
DOMINANT = {"depth": 0.90, "breadth": 0.70, "novelty": 0.66}

# Bonus when the TWO strongest signals are both genuinely firing (corroboration).
# Keeps single-signal prompts where they are, but lets a deep+broad task (a
# thorough PR review) climb a band. Bounded by the product of the two top
# sub-scores, so it only matters when both are substantial.
CORROBORATION = 0.16

# Each entry contributes to a sub-score; a sub-score is min(1.0, hits * step)
# so a few strong cues saturate it. `step` is per-matched-pattern weight.
_DEPTH_PATTERNS = [
    r"\bprove\b", r"\bproof\b", r"\bderive|derivation\b", r"\bformal(ly)? verif",
    r"\bcorrectness\b", r"\binvariant", r"\bcomplexity\b", r"\bnp[- ]hard",
    r"\badversarial\b", r"\brace condition\b", r"\bdeadlock\b", r"\bconcurren",
    r"\bdistributed\b", r"\bconsensus\b", r"\bpaxos\b", r"\braft\b",
    r"\balgorithm", r"\boptimi[sz]e\b", r"\breason through\b", r"\bedge cases?\b",
    r"\bthreat model", r"\bsecurity review", r"\broot[- ]cause\b",
    r"\brefactor(ing)?\b", r"\bdebug(ging)?\b", r"\btrade[- ]?off",
    r"\bmulti[- ]step\b", r"\bstate machine\b", r"\bprotocol\b",
]
_BREADTH_PATTERNS = [
    r"\bentire\b", r"\bwhole (repo|repository|codebase|system)\b",
    r"\bacross (the )?(many|all|\d+) files?\b", r"\ball \d+ ", r"\bmany files\b",
    r"\btrace (the )?data[- ]?flow\b", r"\bend[- ]to[- ]end\b", r"\bcross[- ]module\b",
    r"\bcross[- ]file\b", r"\bwhole[- ]system\b", r"\b\d{3,} lines?\b",
    r"\b\d+ (models?|services?|modules?|components?)\b", r"\bmicroservices?\b",
    r"\bdistributed\b", r"\bsystem[- ]wide\b",
    # Review/audit-class work implies reading a whole change set (a PR diff spans
    # many files; an audit sweeps the codebase) — genuine BREADTH, not just
    # depth. Without this a "review this PR" scored breadth=0 and capped at
    # 'moderate'; these let a review climb toward 'hard' for the right reason.
    r"\breview (this|the|my)? ?(pr|pull request|diff|change ?set|changes)\b",
    r"\bcode review\b", r"\baudit\b", r"\bgo over (the|this|all)\b",
    r"\bpull request\b", r"\bwhat .*(hasn'?t|has not) been (said|covered|mentioned)\b",
    r"\bupgrad(e|able|eable)\b", r"\bdependenc(y|ies)\b", r"\boutdated\b",
    r"\bwhich .*(can|could) be (upgraded|updated|removed)\b",
]
_NOVELTY_PATTERNS = [
    r"\bdesign\b", r"\barchitect(ure|ing)?\b", r"\bfrom scratch\b",
    r"\bgreenfield\b", r"\bnovel\b", r"\binvent\b", r"\bnew (system|service|protocol)\b",
    r"\bpropose (an?|the) (approach|design|architecture)\b", r"\bchoose between\b",
    r"\btrade[- ]?off", r"\bevaluate (options|approaches|designs)\b",
]
# "Apply-known-pattern" / mechanical cues PULL novelty DOWN (clamped at 0).
_NOVELTY_NEGATIVE = [
    r"\brename\b", r"\bfix typo\b", r"\btypo\b", r"\bformat(ting)?\b",
    r"\bone[- ]line", r"\bboilerplate\b", r"\bcopy\b", r"\bget(ter)?/set(ter)?\b",
]

# A moderate-difficulty cue set: ordinary engineering work (refactor, add tests,
# implement a feature) that warrants a mid-tier model but isn't deep reasoning.
# These lift a terse-but-real task out of the abstain gate into 'moderate'.
_EFFORT_PATTERNS = [
    r"\brefactor(ing)?\b", r"\badd (unit )?tests?\b", r"\bimplement\b",
    r"\bmodule\b", r"\bintegrat(e|ion)\b", r"\bmigrat(e|ion)\b", r"\bfeature\b",
    r"\bendpoint\b", r"\bAPI\b", r"\bschema\b", r"\bdebug\b",
]

_DEPTH_STEP = 0.40       # ~2-3 strong depth cues saturate; 1 cue = partial
_BREADTH_STEP = 0.45
_NOVELTY_STEP = 0.45
_EFFORT_STEP = 0.30      # feeds a floor on 'moderate' effort tasks
_NOVELTY_PENALTY = 0.60  # each mechanical cue subtracts this much

# Band edges (lower-inclusive). Below NONE_EDGE → abstain (unless a hard demand
# forces a candidate set anyway). Floors are MIN cost_rank for the band.
BAND_EDGES = (
    ("extreme", 0.90, 5),
    ("hard", 0.75, 4),
    ("moderate", 0.50, 3),
    ("light", 0.28, 1),
    ("none", 0.0, 0),
)
NONE_EDGE = 0.28

# ---------------------------------------------------------------------------
# Tunable overrides (written by the `recalibrate` pass; read here).
# The DEFAULTS above are the committed baseline; an overrides JSON file can
# replace the weights and/or band edges WITHOUT a code edit, so steering the
# engine after a bad route is "change numbers in a file, replay the cache, diff
# the decisions". Set MODEL_ROUTER_TUNING to point at the file.
#
#   { "weights": {"depth":.., "breadth":.., "novelty":..},
#     "band_edges": [["extreme",0.90,5], ["hard",0.75,4], ...] }
#
# Missing keys fall back to the baseline. A malformed/absent file is ignored
# (fail-soft) — the baseline always works.
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402
import os as _os  # noqa: E402
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=1)
def _overrides() -> dict:
    path = _os.environ.get("MODEL_ROUTER_TUNING")
    if not path or not _os.path.isfile(path):
        return {}
    try:
        data = _json.loads(open(path, encoding="utf-8").read())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def reload_overrides() -> None:
    """Drop the cached overrides so a freshly-written tuning file takes effect
    (used by the recalibrate pass and tests)."""
    _overrides.cache_clear()


def _weights() -> dict:
    w = {"depth": W_DEPTH, "breadth": W_BREADTH, "novelty": W_NOVELTY}
    o = _overrides().get("weights")
    if isinstance(o, dict):
        for k in ("depth", "breadth", "novelty"):
            if isinstance(o.get(k), (int, float)):
                w[k] = float(o[k])
    return w


def _band_edges() -> tuple:
    o = _overrides().get("band_edges")
    if isinstance(o, list) and o:
        try:
            edges = tuple((str(n), float(e), int(f)) for n, e, f in o)
            # Must be sorted high→low for band_for's first-match scan.
            return tuple(sorted(edges, key=lambda t: -t[1]))
        except Exception:
            pass
    return BAND_EDGES


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
class ReasoningScore:
    r: float
    depth: float
    breadth: float
    novelty: float
    band: str
    floor: int

    def to_dict(self) -> dict:
        return {
            "r": round(self.r, 3),
            "depth": round(self.depth, 3),
            "breadth": round(self.breadth, 3),
            "novelty": round(self.novelty, 3),
            "band": self.band,
            "floor": self.floor,
        }


@dataclass
class ScoreResult:
    suggested: Optional[str]
    demands: list[str] = field(default_factory=list)
    reason: str = ""
    shortlist: list[str] = field(default_factory=list)
    reasoning: Optional[dict] = None  # ReasoningScore.to_dict() for logging


def detect_demands(request: str) -> frozenset[str]:
    """Capability demands only (vision/long_context/coding/fast). NOT reasoning."""
    text = request.lower()
    demands: set[str] = set()
    for capability, patterns in _DEMAND_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            demands.add(capability)
    return frozenset(demands)


def _subscore(text: str, patterns: list[str], step: float) -> float:
    hits = sum(1 for p in patterns if re.search(p, text))
    return min(1.0, hits * step)


def reasoning_score(request: str) -> ReasoningScore:
    """Grade reasoning difficulty heuristically → r ∈ [0,1], band, rank floor.

    This is the OFFLINE heuristic. The LLM judge may supply its own
    {depth,breadth,novelty} in the same call it already makes; when it does, the
    router uses `grade_from_subscores` with those values instead. This heuristic
    is the always-available fallback.
    """
    text = request.lower()
    depth = _subscore(text, _DEPTH_PATTERNS, _DEPTH_STEP)
    breadth = _subscore(text, _BREADTH_PATTERNS, _BREADTH_STEP)
    novelty = _subscore(text, _NOVELTY_PATTERNS, _NOVELTY_STEP)
    neg = sum(1 for p in _NOVELTY_NEGATIVE if re.search(p, text))
    novelty = max(0.0, novelty - neg * _NOVELTY_PENALTY)
    # Mechanical cues (rename/typo/format) also suppress BREADTH: "rename a var
    # across 3 files" is wide but trivial, not reasoning-heavy. Without this a
    # bulk mechanical edit scores like a cross-file refactor.
    if neg:
        breadth = max(0.0, breadth - neg * _NOVELTY_PENALTY)
    # Ordinary-engineering "effort" cues (refactor, add tests, implement a
    # module) don't score as deep reasoning, but they ARE more than a one-liner.
    # Feed them into depth at a modest level so a terse "refactor module + add
    # tests" clears the abstain gate into 'moderate', while a bare typo stays in
    # 'none'. Mechanical negatives still suppress this.
    effort = _subscore(text, _EFFORT_PATTERNS, _EFFORT_STEP)
    if neg == 0:
        depth = max(depth, min(0.62, effort))
    return grade_from_subscores(depth, breadth, novelty)


def grade_from_subscores(depth: float, breadth: float, novelty: float) -> ReasoningScore:
    """Combine sub-scores → r, band, floor.

    r = max(weighted_mean, dominant_signal). The mean rewards multi-signal
    prompts; the dominant term lets a single SATURATED signal (esp. depth) drive
    the band up on its own, so a pure correctness-proof reaches the top tier even
    with no breadth/novelty cues. Tunable weights come from the overrides file
    via _weights(); band edges via _band_edges().
    """
    depth = _clamp(depth)
    breadth = _clamp(breadth)
    novelty = _clamp(novelty)
    w = _weights()
    mean = w["depth"] * depth + w["breadth"] * breadth + w["novelty"] * novelty
    dominant = max(
        depth * DOMINANT["depth"],
        breadth * DOMINANT["breadth"],
        novelty * DOMINANT["novelty"],
    )
    # Corroboration bonus: TWO strongly-firing signals is harder than one. A deep
    # AND broad task (e.g. a thorough PR review: reason about correctness across a
    # whole diff) should out-rank a single-signal task, but a pure weighted mean
    # drags it down via the zero third signal, and the single-dominant term can't
    # see the second signal at all. Add a bounded bonus when the two highest
    # sub-scores are both strong, so depth+breadth corroboration can reach 'hard'
    # without inflating any single-signal weight (which would over-promote).
    top2 = sorted((depth, breadth, novelty), reverse=True)[:2]
    corroboration = 0.0
    if top2[1] >= 0.5:  # a genuine second signal, not noise
        corroboration = CORROBORATION * top2[0] * top2[1]
    r = min(1.0, max(mean, dominant) + corroboration)
    band, floor = band_for(r)
    return ReasoningScore(r=r, depth=depth, breadth=breadth, novelty=novelty, band=band, floor=floor)


def band_for(r: float) -> tuple[str, int]:
    for name, edge, floor in _band_edges():
        if r >= edge:
            return name, floor
    return "none", 0


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def shortlist(
    request: str,
    profiles: list[ModelProfile],
    *,
    reasoning: Optional[ReasoningScore] = None,
    max_cost_rank: Optional[int] = None,
) -> tuple[frozenset[str], ReasoningScore, list[ModelProfile]]:
    """Return (capability demands, reasoning score, candidate profiles).

    Candidates must (a) cover the HARD capability demands (vision/long_context)
    and (b) sit AT OR ABOVE the reasoning band's rank floor. The band floor is
    what makes the top tier reachable; the judge then ranks within this set.
    """
    demands = detect_demands(request)
    rs = reasoning or reasoning_score(request)
    hard = demands & HARD_DEMANDS

    def eligible(p: ModelProfile) -> bool:
        if max_cost_rank is not None and p.cost_rank > max_cost_rank:
            return False
        if not hard.issubset(p.capabilities):
            return False
        if p.cost_rank < rs.floor:
            return False
        return True

    candidates = [p for p in profiles if eligible(p)]

    # If the floor + hard gate emptied the set (e.g. an extreme band but no
    # rank-5 model exists), relax the FLOOR but keep the hard capability gate —
    # never serve a request a model physically can't handle, but don't abstain
    # just because the ideal tier is absent.
    if not candidates:
        candidates = [
            p for p in profiles
            if (max_cost_rank is None or p.cost_rank <= max_cost_rank)
            and hard.issubset(p.capabilities)
        ]
    return demands, rs, candidates


def score(
    request: str,
    profiles: list[ModelProfile],
    *,
    reasoning: Optional[ReasoningScore] = None,
    max_cost_rank: Optional[int] = None,
) -> ScoreResult:
    """Deterministic graded pick: cheapest model AT/ABOVE the band floor that
    covers the hard demands. Fallback when the judge is unavailable.

    Abstains (suggested=None) only when the band is `none` AND there is no hard
    capability demand forcing a candidate set — matching the reference project's
    "abstain when nothing warrants a switch" gate.
    """
    if not profiles:
        return ScoreResult(suggested=None, reason="no candidates")

    demands, rs, candidates = shortlist(
        request, profiles, reasoning=reasoning, max_cost_rank=max_cost_rank
    )

    hard = demands & HARD_DEMANDS
    if rs.band == "none" and not hard:
        return ScoreResult(
            suggested=None,
            demands=sorted(demands),
            reason=f"reasoning band 'none' (r={rs.r:.2f}) and no hard demand; abstain",
            reasoning=rs.to_dict(),
        )

    if not candidates:
        return ScoreResult(
            suggested=None,
            demands=sorted(demands),
            reason="no model covers the hard demands; abstain",
            reasoning=rs.to_dict(),
        )

    # Cost-DOWN within the floored set: cheapest qualifying model.
    def sort_key(p: ModelProfile) -> tuple:
        return (
            p.cost_rank,
            p.input_cost if p.input_cost is not None else float("inf"),
            len(p.capabilities),
            p.id,
        )

    candidates.sort(key=sort_key)
    winner = candidates[0]
    demand_str = ", ".join(sorted(demands)) if demands else "none"
    return ScoreResult(
        suggested=winner.id,
        demands=sorted(demands),
        reason=(
            f"scorer: reasoning r={rs.r:.2f} band={rs.band} floor={rs.floor}; "
            f"caps [{demand_str}] -> cheapest at/above floor (rank {winner.cost_rank})"
        ),
        shortlist=[p.id for p in candidates],
        reasoning=rs.to_dict(),
    )
